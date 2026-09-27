"""End-to-end runtime F0 feature parity harness (reusable QA gate, not a serving path).

Reference : canonical offline F0 (training/final_hgb.prepare_group_a_training -> 29 features).
Runtime   : Backend-equivalent Contract v1 request -> PredictRequestV1 (serving schema)
            -> runtime_context(model_dump(mode="json"))  [exactly what serving/service.py does]
            -> build_features_from_context -> project_runtime_safe_features.

Backend-equivalent semantics mirror the patched Backend in this branch:
  * history      = selectContractHistory over the vehicle's stored packets: (T-15m, T] + last
                   packet + last strict-valid GPS, order (timestamp, id); id = arrival order
  * VehicleState = epoch seconds (floor), lat/lon null unless location_valid, speed ?? 0
  * target       = classifyTargetActions (docs 5.1): earliest action in (T+10m, T+15m];
                   several at that time -> AMBIGUOUS -> ineligible (no request), unless all
                   are identical in time, geometry and manual_fill (collapsed)
  * deviation    = SCHEDULE_FACT_SOURCE=replay_import -> P1 over facts <= T;
                   SCHEDULE_FACT_SOURCE=none -> 0, status unavailable_no_fact_source
Target values (y) are never used to construct a request.

Variants:
  CODE            offline-identical inputs (fractional times, missing speed kept, P1 deviation,
                  labelled target) fed straight into runtime_context: code-path parity
  BACKEND_REPLAY  Backend mapping + replay-imported facts
  BACKEND_CURRENT Backend mapping + no fact source (current live reality)

Usage:
  MOSTRANSPORT_DATASET=/path/to/dataset python scripts/runtime_parity_harness.py \
      --out-dir OUT [--artifact-dir artifacts/hgb-h0-runtime-safe-v1-group-a-v1]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mostransport_ml.data.official import load_official_split, resolve_dataset_root
from mostransport_ml.data.safe_deviation import load_train_schedule_facts, safe_current_deviation_seconds
from mostransport_ml.features.builder import gps_valid_mask
from mostransport_ml.features.adapters import runtime_context
from mostransport_ml.features.context import build_features_from_context, project_runtime_safe_features
from mostransport_ml.features.schema import RUNTIME_SAFE_FEATURE_NAMES
from mostransport_ml.features.spatial import haversine_m, parse_point_wkt
from mostransport_ml.serving.schemas import PredictRequestV1
from mostransport_ml.training.final_hgb import prepare_group_a_training

_NS = 1_000_000_000
WINDOW_NS = 15 * 60 * _NS
SPEED_FEATURES = {n for n in RUNTIME_SAFE_FEATURE_NAMES if n.startswith(("speed_", "zero_speed"))}


def iso_z(ts: pd.Timestamp, keep_fraction: bool) -> str:
    ts = pd.Timestamp(ts)
    if not keep_fraction:
        ts = ts.floor("s")
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def contract_history(vehicle: pd.DataFrame, ev: np.ndarray, t_ns: int) -> pd.DataFrame:
    """Mirror of backend selectContractHistory; `vehicle` sorted by (event_time, src_seq)."""
    end = int(np.searchsorted(ev, t_ns, "right"))
    past = vehicle.iloc[:end]
    if past.empty:
        return past
    keep = set(np.flatnonzero(ev[:end] > t_ns - WINDOW_NS).tolist())
    keep.add(end - 1)
    valid = np.flatnonzero(gps_valid_mask(past))
    if valid.size:
        keep.add(int(valid[-1]))
    return past.iloc[sorted(keep)]


def classify_target(plan_v: pd.DataFrame, t: pd.Timestamp):
    """Mirror of backend classifyTargetActions: ties collapse only if identical in time, geometry, manual_fill."""
    lo, hi = t + pd.Timedelta(minutes=10), t + pd.Timedelta(minutes=15)
    h = plan_v[(plan_v.time_begin > lo) & (plan_v.time_begin <= hi)]
    if h.empty:
        return "none", None
    at = h[h.time_begin == h.time_begin.min()].sort_values("tt_action_item_id", key=lambda c: c.astype("int64"))
    if len(at[["lat", "lon", "manual_fill"]].drop_duplicates()) > 1:
        return "ambiguous", None
    return "ok", at.iloc[0]


def packets(h: pd.DataFrame, *, backend: bool, speed_null_to_zero: bool, truncate: bool) -> list[dict]:
    out = []
    for r in h.itertuples(index=False):
        valid = bool(r.location_valid) if pd.notna(r.location_valid) else False
        coords = valid and pd.notna(r.lat) and pd.notna(r.lon)
        speed = r.speed if pd.notna(r.speed) else (0.0 if speed_null_to_zero else None)
        out.append(dict(event_time=iso_z(r.event_time, keep_fraction=not truncate),
                        lat=float(r.lat) if (coords if backend else pd.notna(r.lat)) else None,
                        lon=float(r.lon) if (coords if backend else pd.notna(r.lon)) else None,
                        location_valid=valid if backend else (None if pd.isna(r.location_valid) else bool(r.location_valid)),
                        speed=None if speed is None else float(speed)))
    return out


def features_from_request(req: dict, *, validate_schema: bool, keep: dict | None = None) -> pd.Series:
    body = PredictRequestV1.model_validate(req).model_dump(mode="json") if validate_schema else req
    batch = runtime_context(body)
    if keep is not None:
        keep["batch"] = batch
    return project_runtime_safe_features(build_features_from_context(batch)).iloc[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--artifact-dir", default=None, type=Path)
    args = ap.parse_args(argv)
    root = resolve_dataset_root(args.dataset_root)
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)

    data = prepare_group_a_training(root, expected_rows=1141, expected_tr_ids=13)
    X0, pts = data.X, data.points.reset_index(drop=True)
    train = load_official_split(root, "train")
    tel = train.telemetry.assign(src_seq=np.arange(len(train.telemetry)))
    ids = set(pts.tr_id)
    tel = tel[tel.tr_id.isin(ids)]
    veh = {tr: g.sort_values(["event_time", "src_seq"], kind="stable").reset_index(drop=True) for tr, g in tel.groupby("tr_id")}
    evs = {tr: v.event_time.to_numpy(dtype="datetime64[ns]").view("int64") for tr, v in veh.items()}
    raw_units = pd.read_csv(root / "train/traffic.csv", usecols=["tr_id", "unit_id"]).drop_duplicates()
    unit_of = dict(zip(raw_units.tr_id, raw_units.unit_id))
    plan = train.schedule_plan.copy()
    xy = np.array([parse_point_wkt(g) for g in plan.geom]); plan["lon"], plan["lat"] = xy[:, 0], xy[:, 1]
    plan_by = {tr: g for tr, g in plan.groupby("tr_id")}
    facts = load_train_schedule_facts(root)
    replay_dev = safe_current_deviation_seconds(pts[["sample_id", "tr_id", "T"]], facts)  # P1 = backend resolver (fixture-conformant)
    assert np.array_equal(replay_dev.to_numpy(), X0.cur_dev_s.to_numpy())
    matcher_plan = plan[~plan.tr_id.astype(str).str.startswith("9000")]

    rows, point_rows = [], []
    batches = {"BACKEND_REPLAY": {}, "BACKEND_CURRENT": {}}
    variants = ("CODE", "BACKEND_REPLAY", "BACKEND_CURRENT")
    for i, p in enumerate(pts.itertuples(index=False)):
        t = pd.Timestamp(p.T); t_ns = int(t.value)
        h = contract_history(veh[p.tr_id], evs[p.tr_id], t_ns)
        status, tgt = classify_target(plan_by[p.tr_id], t)
        target_matches = status == "ok" and int(tgt.tt_action_item_id) == int(p.target_stop_id)
        # trip-matcher audit (vehicles table empty -> global nearest, as in current seed)
        valid_h = h[gps_valid_mask(h)]
        lv = valid_h.iloc[-1] if len(valid_h) else None
        matched = None
        if lv is not None:
            win = matcher_plan[(matcher_plan.time_begin > t + pd.Timedelta(minutes=10)) & (matcher_plan.time_begin <= t + pd.Timedelta(minutes=15))]
            cand = matcher_plan[matcher_plan.tr_id.isin(set(win.tr_id))]
            d = haversine_m(cand.lon.to_numpy(), cand.lat.to_numpy(), lv.lon, lv.lat)
            ok = d <= 1000
            matched = int(cand.tr_id.to_numpy()[ok][np.argmin(d[ok])]) if ok.any() else None
        own = plan_by[p.tr_id]
        wown = own[(own.time_begin > t + pd.Timedelta(minutes=10)) & (own.time_begin <= t + pd.Timedelta(minutes=15))]
        own_ok = lv is not None and len(wown) and (haversine_m(own.lon.to_numpy(), own.lat.to_numpy(), lv.lon, lv.lat) <= 1000).any()
        labelled = plan_by[p.tr_id].set_index("tt_action_item_id").loc[p.target_stop_id]
        ref = X0.iloc[i]

        def request(deviation: float, target, *, backend: bool, speed0: bool, trunc: bool) -> dict:
            return dict(request_id=str(p.sample_id), prediction_time=iso_z(t, keep_fraction=False),
                        vehicle_context=dict(unit_id=str(unit_of[p.tr_id]), tr_id=str(p.tr_id), route_id=str(p.tr_id)),
                        schedule_context=dict(target_action_id=str(p.target_stop_id if target is None else target.tt_action_item_id),
                                              target_time_begin=iso_z((labelled if target is None else target).time_begin, keep_fraction=False),
                                              target_lat=float((labelled if target is None else target).lat),
                                              target_lon=float((labelled if target is None else target).lon),
                                              current_deviation_seconds=float(deviation),
                                              manual_fill=bool((labelled if target is None else target).manual_fill)),
                        telemetry=packets(h, backend=backend, speed_null_to_zero=speed0, truncate=trunc))

        rt = {}
        rt["CODE"] = features_from_request(request(replay_dev.iloc[i], None, backend=False, speed0=False, trunc=False), validate_schema=False)
        eligible = status == "ok"
        if eligible:
            k1, k2 = {}, {}
            rt["BACKEND_REPLAY"] = features_from_request(request(replay_dev.iloc[i], tgt, backend=True, speed0=True, trunc=True), validate_schema=True, keep=k1)
            rt["BACKEND_CURRENT"] = features_from_request(request(0.0, tgt, backend=True, speed0=True, trunc=True), validate_schema=True, keep=k2)
            batches["BACKEND_REPLAY"][i], batches["BACKEND_CURRENT"][i] = k1["batch"], k2["batch"]
            # counterfactuals for attribution (no schema: they deliberately keep missing speed / fractions)
            cf_nospeed = features_from_request(request(replay_dev.iloc[i], tgt, backend=True, speed0=False, trunc=True), validate_schema=False)
            cf_notrunc = features_from_request(request(replay_dev.iloc[i], tgt, backend=True, speed0=True, trunc=False), validate_schema=False)
            cf_neither = features_from_request(request(replay_dev.iloc[i], tgt, backend=True, speed0=False, trunc=False), validate_schema=False)
        for v in variants:
            for f in RUNTIME_SAFE_FEATURE_NAMES:
                r0 = float(ref[f])
                if v != "CODE" and not eligible:
                    rows.append(dict(sample_id=p.sample_id, variant=v, feature=f, offline=r0, runtime=np.nan, exact_match=False,
                                     category=f"INELIGIBLE_TARGET_{status.upper()}"))
                    continue
                r1 = float(rt[v][f]); same = (math.isnan(r0) and math.isnan(r1)) or r0 == r1
                cat = "EXACT"
                if not same:
                    if f == "cur_dev_s" and v == "BACKEND_CURRENT":
                        cat = "UNAVAILABLE_RUNTIME_STATE:current_deviation_no_fact_source"
                    elif v == "CODE":
                        cat = "UNEXPLAINED"
                    else:
                        eq = lambda s: (math.isnan(r0) and math.isnan(float(s[f]))) or r0 == float(s[f])
                        if eq(cf_nospeed) and not eq(cf_notrunc):
                            cat = "DATA_MAPPING:historical_missing_speed_sent_as_0"
                        elif eq(cf_notrunc) and not eq(cf_nospeed):
                            cat = "DATA_MAPPING:epoch_seconds_truncation_of_server_clock_packets"
                        elif eq(cf_neither):
                            cat = "DATA_MAPPING:missing_speed_as_0_and_seconds_truncation"
                        else:
                            cat = "UNEXPLAINED"
                rows.append(dict(sample_id=p.sample_id, variant=v, feature=f, offline=r0, runtime=r1, exact_match=bool(same), category=cat))
        point_rows.append(dict(sample_id=p.sample_id, tr_id=p.tr_id, target_status=status, target_matches_label=bool(target_matches),
                               matcher_trip_global_nearest=matched, matcher_correct_global=matched == p.tr_id, matcher_own_trip_within_1km=bool(own_ok),
                               # vehicles(unit_id -> current_tr_id) populated: preferred-trip query only, no global fallback
                               matcher_trip_with_vehicle_identity=p.tr_id if own_ok else None,
                               history_packets=len(h), history_has_missing_speed=bool(h.speed.isna().any()),
                               history_has_fractional_times=bool((h.event_time.dt.microsecond != 0).any())))
    P = pd.DataFrame(rows); PT = pd.DataFrame(point_rows)
    P.to_csv(out / "runtime_feature_parity.csv", index=False)
    PT.to_csv(out / "runtime_parity_points.csv", index=False)
    summary = {}
    for v in variants:
        s = P[P.variant == v]
        mism = s[~s.exact_match]
        summary[v] = dict(cells=len(s), exact_match_cells=int(s.exact_match.sum()), mismatched_cells=int((~s.exact_match).sum()),
                          ineligible_points=int(s[s.category.str.startswith("INELIGIBLE")].sample_id.nunique()),
                          affected_points=int(mism[~mism.category.str.startswith("INELIGIBLE")].sample_id.nunique()),
                          affected_features=sorted(mism[~mism.category.str.startswith("INELIGIBLE")].feature.unique().tolist()),
                          categories=mism.category.value_counts().to_dict(), unexplained_cells=int((s.category == "UNEXPLAINED").sum()))
    summary["targets"] = dict(PT.target_status.value_counts().to_dict(), ok_matches_label=int(PT.target_matches_label.sum()))
    summary["trip_matcher_audit"] = dict(global_nearest_correct=int(PT.matcher_correct_global.sum()), global_nearest_wrong_or_none=int((~PT.matcher_correct_global).sum()),
                                         global_nearest_none=int(PT.matcher_trip_global_nearest.isna().sum()),
                                         own_trip_reachable_within_1km=int(PT.matcher_own_trip_within_1km.sum()), n=len(PT),
                                         with_vehicle_identity_correct=int((PT.matcher_trip_with_vehicle_identity == PT.tr_id).sum()),
                                         with_vehicle_identity_none=int(PT.matcher_trip_with_vehicle_identity.isna().sum()),
                                         with_vehicle_identity_wrong=int((PT.matcher_trip_with_vehicle_identity.notna()
                                                                          & (PT.matcher_trip_with_vehicle_identity != PT.tr_id)).sum()))
    if args.artifact_dir is not None:
        from mostransport_ml.inference.predictor import ArtifactPredictor
        pred = ArtifactPredictor.load(args.artifact_dir)
        offline_pred = np.asarray(pred.predict(data.batch), float)
        summary["artifact"] = dict(model_version=pred.model_version(), feature_schema_version=pred.feature_schema_version(), bundle_sha256=pred.bundle_sha256)
        # Runtime predictions through the public predictor on the exact validated request batches.
        pred_rows = []
        for v, by_i in batches.items():
            d = np.array([float(pred.predict(b)[0]) - offline_pred[i] for i, b in by_i.items()])
            summary["artifact"][f"{v}_prediction_abs_diff"] = dict(n=len(d), exact=int((d == 0).sum()), median=float(np.median(np.abs(d))),
                                                                   p95=float(np.quantile(np.abs(d), .95)), max=float(np.abs(d).max()),
                                                                   mean_signed=float(d.mean()))
            pred_rows += [dict(sample_id=pts.sample_id.iloc[i], variant=v, offline_prediction=offline_pred[i], runtime_prediction=offline_pred[i] + di)
                          for i, di in zip(by_i.keys(), d)]
        pd.DataFrame(pred_rows).to_csv(out / "runtime_prediction_parity.csv", index=False)
    (out / "runtime_feature_parity_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
