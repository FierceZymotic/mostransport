#!/bin/sh
set -eu

CSV_PATH="/docker-entrypoint-initdb.d/data/schedule_actions_import_fixed.csv"

if [ -f "$CSV_PATH" ]; then
  echo "Importing schedule CSV into Postgres..."
  PGPASSWORD="$POSTGRES_PASSWORD" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v csv_path="$CSV_PATH" <<'SQL'
-- Organizer times are naive UTC: pin the session zone so TIMESTAMPTZ does not depend on it.
SET TIME ZONE 'UTC';
\copy schedule_actions (tt_action_item_id, tr_id, time_begin, geom, manual_fill)
FROM :'csv_path'
WITH (FORMAT csv, HEADER true, DELIMITER ',', NULL '');
SQL
fi
