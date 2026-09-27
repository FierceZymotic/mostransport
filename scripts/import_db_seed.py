from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
DB_URL = "postgresql://postgres:postgres@localhost:5432/mostransport"


def ensure_schema(conn: psycopg.Connection) -> None:
    schema_sql = (ROOT / "database" / "mtr_DB_v2.sql").read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        cur.execute(schema_sql)
    conn.commit()


def parse_manual_fill(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower()
    if text.lower() in {"null", "none"}:
        return None
    try:
        parsed = json.loads(text)
        return json.dumps(parsed)
    except json.JSONDecodeError:
        return json.dumps(text)


def import_telemetry(conn: psycopg.Connection) -> int:
    csv_path = ROOT / "database" / "telemetry_seed.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    inserted = 0
    with conn.cursor() as cur:
        for row in df.to_dict(orient="records"):
            cur.execute(
                """
                INSERT INTO telemetry (
                    unit_id, tr_id, timestamp, longitude, latitude, location_valid,
                    speed, speed_max, course, track, altitude, nsat, pdop
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    str(row["unit_id"]),
                    None,
                    row["timestamp"],
                    float(row["longitude"]),
                    float(row["latitude"]),
                    bool(row["location_valid"]),
                    float(row["speed"]),
                    float(row["speed_max"]),
                    float(row["course"]),
                    float(row["track"]),
                    float(row["altitude"]),
                    int(row["nsat"]),
                    float(row["pdop"]),
                ),
            )
            inserted += cur.rowcount
    conn.commit()
    return inserted


def import_schedule_actions(conn: psycopg.Connection) -> int:
    csv_path = ROOT / "database" / "schedule_actions_import_fixed.csv"
    df = pd.read_csv(csv_path, parse_dates=["time_begin"])
    inserted = 0
    with conn.cursor() as cur:
        for row in df.to_dict(orient="records"):
            geom_wkt = str(row["geom"]).strip()
            cur.execute(
                """
                INSERT INTO schedule_actions (
                    tt_action_item_id, tr_id, time_begin, time_fact_begin, geom, manual_fill
                ) VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 4326), CAST(%s AS jsonb))
                ON CONFLICT (tt_action_item_id) DO NOTHING
                """,
                (
                    str(row["tt_action_item_id"]),
                    str(row["tr_id"]),
                    row["time_begin"],
                    None,
                    geom_wkt,
                    parse_manual_fill(row.get("manual_fill")),
                ),
            )
            inserted += cur.rowcount
    conn.commit()
    return inserted


def main() -> None:
    with psycopg.connect(DB_URL) as conn:
        ensure_schema(conn)
        telemetry_inserted = import_telemetry(conn)
        schedule_inserted = import_schedule_actions(conn)

        with conn.cursor() as cur:
            print("telemetry_rows=", cur.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0])
            print("schedule_rows=", cur.execute("SELECT COUNT(*) FROM schedule_actions").fetchone()[0])
            print("telemetry_imported=", telemetry_inserted)
            print("schedule_imported=", schedule_inserted)
            print(
                "sample_telemetry=",
                cur.execute(
                    "SELECT unit_id, timestamp, latitude, longitude FROM telemetry ORDER BY timestamp DESC LIMIT 3"
                ).fetchall(),
            )
            print(
                "sample_schedule=",
                cur.execute(
                    "SELECT tt_action_item_id, tr_id, ST_AsText(geom) FROM schedule_actions LIMIT 3"
                ).fetchall(),
            )


if __name__ == "__main__":
    main()
