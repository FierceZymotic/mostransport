// Pure-logic regression tests for the realtime dashboard state (Node built-in test runner,
// no extra dependency): `npm test` = `node --test tests/dashboard-state.test.ts`.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  applyTelemetry,
  displayReason,
  formatDelay,
  formatSpeed,
  latestPredictionByUnitId,
  mergeAlerts,
  normalizeSpeedKmh,
  reasonPresentation,
  riskLevelForDelay,
  riskOf,
  seedFromSummary,
  toVehicleView,
  type AlertRow,
  type DashboardState,
} from "../src/dashboard/state.ts";

const DEGRADED = "degraded:current_deviation_unavailable(no_fact_source)";

function alert(id: string, unitId: string, delaySeconds: number, generatedAt: string, routeId = "130238", reason = DEGRADED): AlertRow {
  return { id, unitId, routeId, status: "success", delaySeconds, reason, generatedAt };
}

describe("delay formatter", () => {
  it("keeps seconds instead of collapsing them to +0 мин", () => {
    assert.equal(formatDelay(10.9), "+11 сек");
    assert.equal(formatDelay(11), "+11 сек");
    assert.equal(formatDelay(135), "+2 мин 15 сек");
    assert.equal(formatDelay(120), "+2 мин");
    assert.equal(formatDelay(3600), "+1 ч");
    assert.equal(formatDelay(3661), "+1 ч 1 мин 1 сек");
  });

  it("preserves the sign of early predictions and shows zero compactly", () => {
    assert.equal(formatDelay(-11), "−11 сек");
    assert.equal(formatDelay(-135.4), "−2 мин 15 сек");
    assert.equal(formatDelay(0), "0 сек");
    assert.equal(formatDelay(-0.2), "0 сек");
  });
});

describe("latest prediction per unit from /alerts history", () => {
  it("selects the newest prediction per unit from deliberately unsorted rows", () => {
    const rows = [
      alert("b", "1003", 40, "2026-09-27T10:01:00.000Z"),
      alert("c", "1004", 300, "2026-09-27T10:00:30.000Z"),
      alert("a", "1003", 10, "2026-09-27T10:02:00.000Z"),
      alert("d", "1003", 999, "2026-09-27T09:59:00.000Z"),
    ];
    const latest = latestPredictionByUnitId(rows);
    assert.equal(latest["1003"].requestId, "a");
    assert.equal(latest["1003"].delaySeconds, 10);
    assert.equal(latest["1004"].requestId, "c"); // one unit never overwrites another
    assert.deepEqual(latestPredictionByUnitId([...rows].reverse()), latest);
  });

  it("uses the request id as a deterministic tie-breaker and never lets an older row replace a newer one", () => {
    const t = "2026-09-27T10:00:00.000Z";
    assert.equal(latestPredictionByUnitId([alert("x1", "7", 1, t), alert("x2", "7", 2, t)])["7"].requestId, "x2");
    const seeded = mergeAlerts({}, [alert("new", "7", 5, "2026-09-27T10:05:00.000Z")]);
    const afterOld = mergeAlerts(seeded, [alert("old", "7", 500, "2026-09-27T10:00:00.000Z")]);
    assert.equal(afterOld["7"].prediction?.requestId, "new");
  });

  it("attaches WS-only details (meta status, target time) to the matching request", () => {
    const row = alert("r1", "1003", 11, "2026-09-27T10:00:00.000Z");
    const first = mergeAlerts({}, [row]);
    const details = { r1: { targetTime: "2026-01-06T10:12:11Z", deviationStatus: "unavailable_no_fact_source" } };
    const second = mergeAlerts(first, [row], details);
    assert.equal(second["1003"].prediction?.deviationStatus, "unavailable_no_fact_source");
    assert.equal(second["1003"].prediction?.targetTime, "2026-01-06T10:12:11Z");
  });
});

describe("realtime ownership: telemetry and prediction are separate", () => {
  it("a prediction survives ten telemetry updates", () => {
    let state: DashboardState = mergeAlerts({}, [alert("p", "1003", 135, "2026-09-27T10:00:00.000Z")]);
    for (let i = 0; i < 10; i++) {
      state = applyTelemetry(state, { unitId: 1003, latitude: 55.7 + i / 1000, longitude: 37.6, locationValid: true, speed: 18 + i }, 1000 + i);
    }
    assert.equal(state["1003"].prediction?.requestId, "p");
    assert.equal(state["1003"].prediction?.delaySeconds, 135);
    assert.equal(state["1003"].telemetry?.speedKmh, 27);
    assert.equal(Object.keys(state).length, 1); // number 1003 and string "1003" are the same unit
  });

  it("a prediction does not reset telemetry position or speed", () => {
    let state = applyTelemetry({}, { unitId: "1004", latitude: 55.75, longitude: 37.61, locationValid: true, speed: 18 }, 5000);
    state = mergeAlerts(state, [alert("q", "1004", 11, "2026-09-27T10:00:00.000Z")]);
    assert.deepEqual(state["1004"].telemetry, { lat: 55.75, lon: 37.61, speedKmh: 18, receivedAt: 5000 });
  });

  it("an invalid fix updates speed and freshness but does not move the marker", () => {
    let state = applyTelemetry({}, { unitId: 5, latitude: 55.75, longitude: 37.61, locationValid: true, speed: 20 }, 1);
    state = applyTelemetry(state, { unitId: 5, latitude: 0, longitude: 0, locationValid: false, speed: 12 }, 2);
    assert.deepEqual(state["5"].telemetry, { lat: 55.75, lon: 37.61, speedKmh: 12, receivedAt: 2 });
  });

  it("the summary seed never overrides live telemetry", () => {
    const live = applyTelemetry({}, { unitId: 9, latitude: 55.8, longitude: 37.7, locationValid: true, speed: 30 }, 10);
    const seeded = seedFromSummary(live, [{ id: "9", lat: 55.1, lon: 37.1, speed: 3 }]);
    assert.equal(seeded["9"].telemetry?.lat, 55.8);
  });
});

describe("reason translation", () => {
  it("translates the degraded marker for the dispatcher and keeps the raw value", () => {
    assert.deepEqual(displayReason(DEGRADED), { kind: "degraded", text: "Недостаточно фактических данных по графику", raw: DEGRADED });
    assert.equal(displayReason("degraded:current_deviation_unavailable(other_detail)").kind, "degraded");
  });

  it("does not invent an explanation for unknown reasons", () => {
    assert.equal(displayReason("some_server_error").kind, "unknown");
    assert.notEqual(displayReason("some_server_error").text, "Недостаточно фактических данных по графику");
    assert.equal(displayReason(null).kind, "none");
  });

  it("the card never shows the raw degraded reason as its text: mapped text, amber warning, raw kept for the tooltip", () => {
    const state = mergeAlerts({}, [alert("d", "1003", 11, "2026-09-27T10:00:00.000Z")]);
    const view = toVehicleView(state["1003"]);
    const shown = reasonPresentation(view.reason!); // exactly what VehicleCard renders
    assert.equal(shown.text, "Недостаточно фактических данных по графику");
    assert.ok(!shown.text.includes("degraded:"));
    assert.equal(shown.tone, "warning");
    assert.equal(shown.tooltip, DEGRADED);
    assert.equal(state["1003"].prediction?.reason, DEGRADED); // raw backend reason preserved in state
    assert.equal(reasonPresentation(displayReason(null)).tone, "neutral");
  });

  it("a degraded prediction is still shown as a prediction", () => {
    const view = toVehicleView(mergeAlerts({}, [alert("d", "1", 11, "2026-09-27T10:00:00.000Z")])["1"]);
    assert.equal(view.delayText, "+11 сек");
    assert.equal(view.reason?.text, "Недостаточно фактических данных по графику");
  });
});

describe("canonical risk (organizer delay classes, -60 s / +120 s)", () => {
  it("maps late / on time / early to high / low / medium", () => {
    assert.equal(riskLevelForDelay(121), "high");
    assert.equal(riskLevelForDelay(120), "low");
    assert.equal(riskLevelForDelay(11), "low");
    assert.equal(riskLevelForDelay(-60), "low");
    assert.equal(riskLevelForDelay(-61), "medium");
  });

  it("gives marker, list and card the same level from one view, and no level without a prediction", () => {
    const state = mergeAlerts(applyTelemetry({}, { unitId: 3, latitude: 55.7, longitude: 37.6, locationValid: true, speed: 18 }, 1),
      [alert("h", "3", 300, "2026-09-27T10:00:00.000Z")]);
    const view = toVehicleView(state["3"]);
    assert.equal(view.risk, riskOf(state["3"].prediction).level); // marker class + list dot + card badge read view.risk
    assert.equal(view.risk, "high");
    assert.equal(view.riskLabel, "Высокий риск");
    assert.equal(view.status, "Прогноз задержки +5 мин");
    const noPrediction = toVehicleView(applyTelemetry({}, { unitId: 4, latitude: 55.7, longitude: 37.6, locationValid: true, speed: 18 }, 1)["4"]);
    assert.equal(noPrediction.risk, "unknown");
    assert.equal(noPrediction.status, "Прогноз: нет данных");
  });
});

describe("header, section and speed", () => {
  it("shows the real route id or an honest fallback, never LIVE", () => {
    const withRoute = toVehicleView(mergeAlerts({}, [alert("r", "1003", 11, "2026-09-27T10:00:00.000Z", "130238")])["1003"]);
    assert.equal(`${withRoute.id} · ${withRoute.route}`, "1003 · маршрут 130238");
    const unknownRoute = toVehicleView(mergeAlerts({}, [alert("u", "1003", 11, "2026-09-27T10:00:00.000Z", "unknown")])["1003"]);
    assert.equal(unknownRoute.route, "маршрут недоступен");
    const telemetryOnly = toVehicleView(applyTelemetry({}, { unitId: 1003, latitude: 55.7, longitude: 37.6, speed: 5 }, 1)["1003"]);
    assert.equal(telemetryOnly.route, "маршрут недоступен");
  });

  it("has no backend segment, so the section is unavailable", () => {
    const view = toVehicleView(applyTelemetry({}, { unitId: 1003, latitude: 55.7, longitude: 37.6, speed: 5 }, 1)["1003"]);
    assert.equal(view.sectionText, "данные недоступны");
    assert.ok(!view.sectionText.includes("Unit"));
  });

  it("displays km/h as delivered (NDTP speedAvg is km/h): 53 stays 53, not 191", () => {
    assert.equal(normalizeSpeedKmh(53), 53);
    assert.equal(formatSpeed(normalizeSpeedKmh(53)), "53 км/ч");
    assert.equal(formatSpeed(normalizeSpeedKmh(17.6)), "18 км/ч");
    assert.equal(normalizeSpeedKmh(-1), null);
    assert.equal(formatSpeed(null), "нет данных");
  });
});
