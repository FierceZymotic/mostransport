-- PostgreSQL 14+
CREATE EXTENSION IF NOT EXISTS postgis;

-- telemetry
CREATE TABLE IF NOT EXISTS telemetry (
    id BIGSERIAL PRIMARY KEY,
    unit_id VARCHAR(64) NOT NULL,
    tr_id VARCHAR(64),
    timestamp TIMESTAMPTZ NOT NULL,
    
    longitude DOUBLE PRECISION NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    location_valid BOOLEAN NOT NULL DEFAULT FALSE,
    
    speed DOUBLE PRECISION,
    speed_max DOUBLE PRECISION,
    course DOUBLE PRECISION,
    track DOUBLE PRECISION,
    altitude DOUBLE PRECISION,
    
    nsat INTEGER,
    pdop DOUBLE PRECISION
);

-- Индексы для эффективной выборки истории по unit_id и временному интервалу
CREATE INDEX IF NOT EXISTS idx_telemetry_unit_time ON telemetry (unit_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry (timestamp);


-- реестр ТС и текущего рейса
CREATE TABLE IF NOT EXISTS vehicles (
    unit_id VARCHAR(64) PRIMARY KEY,
    current_tr_id VARCHAR(64)
);


-- расписание
CREATE TABLE IF NOT EXISTS schedule_actions (
    tt_action_item_id VARCHAR(64) PRIMARY KEY,
    tr_id VARCHAR(64) NOT NULL,
    time_begin TIMESTAMPTZ NOT NULL,
    geom geometry(Point, 4326) NOT NULL,
    manual_fill JSONB
);

CREATE INDEX IF NOT EXISTS idx_schedule_tr_time ON schedule_actions (tr_id, time_begin);


-- predictions (append-only)
CREATE TABLE IF NOT EXISTS predictions (
    request_id VARCHAR(64) PRIMARY KEY,
    
    unit_id VARCHAR(64) NOT NULL,
    tr_id VARCHAR(64),
    prediction_time TIMESTAMPTZ NOT NULL,
    target_action_id VARCHAR(64),
    target_time_begin TIMESTAMPTZ,
    
    status VARCHAR(32) NOT NULL,
    delay_seconds DOUBLE PRECISION,
    target_time TIMESTAMPTZ,
    reason TEXT,
    
    generated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(64),
    feature_schema_version VARCHAR(64)
);

CREATE INDEX IF NOT EXISTS idx_predictions_unit_time ON predictions (unit_id, prediction_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_target_action ON predictions (target_action_id);


-- крайнее состояние ТС
CREATE TABLE IF NOT EXISTS vehicle_last_state (
    unit_id VARCHAR(64) PRIMARY KEY,
    tr_id VARCHAR(64),
    timestamp TIMESTAMPTZ NOT NULL,
    longitude DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    location_valid BOOLEAN
);
