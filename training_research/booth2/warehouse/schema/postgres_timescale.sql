-- Booth2 plasma-spray warehouse: production schema for PostgreSQL 16 + TimescaleDB 2.x.
--
-- STATUS: ready to run, NOT applied anywhere. No database server was started for this work;
-- this file has not been executed against a real PostgreSQL instance. Review, then:
--     psql "$BOOTH2_PG_DSN" -v ON_ERROR_STOP=1 -f postgres_timescale.sql
--
-- The same logical tables exist, loaded with all history, in db/booth2.sqlite (see
-- schema/sqlite_schema.sql, generated). Names differ only where PostgreSQL needs them to
-- (sample_1hz -> history_sample_1hz, live_sample -> sample).
--
-- TIME. Two clocks, deliberately kept apart:
--   * Live data (OPC UA): SourceTimestamp / ServerTimestamp are UTC instants -> timestamptz.
--   * History (HMI CSV export, 2026-08-15 .. 2026-11-01, and the curated tables to 11-14):
--     the instrument's local wall clock with no recorded timezone -> `timestamp` (no tz),
--     plus `ts` = that wall clock counted as epoch seconds as if it were UTC (the source
--     project's anchor ids are 'A' || ts). Nothing here assigns a timezone to history.
--     Once the site confirms the HMI clock's zone, set asset.history_timezone and use
--     history_as_utc() below; until then do not join history to live data on time.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================ registry

CREATE TABLE asset (
    asset_id                smallint PRIMARY KEY,
    asset_key               text NOT NULL UNIQUE,            -- 'Booth2'
    name                    text NOT NULL,
    opcua_path              text NOT NULL,                   -- 'HA.Booth2.Plasma2'  (to be confirmed on site)
    opcua_namespace_index   smallint,                        -- 2                    (to be confirmed on site)
    opcua_namespace_uri     text,                            -- preferred over the index: indexes can change
    opcua_confirmed         boolean NOT NULL DEFAULT false,
    history_timezone        text,                            -- NULL until the HMI clock zone is confirmed
    note                    text
);

CREATE TABLE tag (
    tag_id                  smallint PRIMARY KEY,            -- same numbering as ingest/tags.json order
    asset_id                smallint NOT NULL REFERENCES asset,
    field_key               text NOT NULL,                   -- name in the historical CSV header and in every model
    name_cn                 text NOT NULL,
    unit                    text,
    tag_group               text,
    dtype                   text NOT NULL CHECK (dtype IN ('float', 'int', 'string')),
    kepware_tag             text NOT NULL,                   -- e.g. Press._Hopper_1 (history) <-> Press_Hopper_1 (Kepware)
    kepware_tag_confirmed   boolean NOT NULL DEFAULT false,
    opcua_node_id           text NOT NULL,                   -- expected 'ns=2;s=HA.Booth2.Plasma2.<kepware_tag>'
    node_id_confirmed       boolean NOT NULL DEFAULT false,  -- false = expectation, to be confirmed on site
    scan_rate_ms            integer NOT NULL DEFAULT 1000,   -- Kepware scan rate actually configured (confirm per tag)
    deadband_abs            double precision NOT NULL DEFAULT 0,
    max_age_s               integer NOT NULL DEFAULT 5,      -- older than this at an origin = missing for the model
    in_history              boolean NOT NULL,                -- present in the 28-column HMI export
    reserved                boolean NOT NULL DEFAULT false,  -- Gun_ID.DATA / Gun_ID.LEN / Gun_Hours / Ignitions
    model_v2_input          boolean NOT NULL DEFAULT false,
    pg_column               text NOT NULL,                   -- column name in the wide tables below
    note                    text,
    UNIQUE (asset_id, field_key)
);
-- Rows: load ingest/tags.json (32 tags: 28 measured + 4 reserved). The SQLite `tag` table holds the same rows.

-- ============================================================================ live raw samples

-- Narrow, one row per reported data change. Idempotent on (tag_id, source_time): the
-- collector inserts with ON CONFLICT DO NOTHING, so replays after a crash are harmless.
CREATE TABLE sample (
    tag_id          smallint NOT NULL REFERENCES tag,
    source_time     timestamptz NOT NULL,                    -- OPC UA SourceTimestamp (device / driver)
    server_time     timestamptz,                             -- OPC UA ServerTimestamp (Kepware)
    received_at     timestamptz NOT NULL DEFAULT now(),      -- collector clock
    value           double precision,                        -- NULL when bad, missing or not numeric; never zero-filled
    value_text      text,                                    -- string tags and unparseable values
    status_code     bigint NOT NULL DEFAULT 0,               -- raw OPC UA StatusCode
    good            boolean GENERATED ALWAYS AS ((status_code & 3221225472) = 0) STORED,  -- top two bits clear
    session_id      text NOT NULL,
    PRIMARY KEY (tag_id, source_time)
);
SELECT create_hypertable('sample', 'source_time', chunk_time_interval => INTERVAL '1 day');
CREATE INDEX sample_time_idx ON sample (source_time DESC);

-- Suggested policies (28 tags at 1 Hz is about 2.4 M rows/day before deadband):
--   ALTER TABLE sample SET (timescaledb.compress, timescaledb.compress_segmentby = 'tag_id',
--                           timescaledb.compress_orderby = 'source_time DESC');
--   SELECT add_compression_policy('sample', INTERVAL '7 days');     -- typically >10x smaller
--   SELECT add_retention_policy('sample', INTERVAL '3 years');      -- only if the plant's record rules allow;
--                                                                    -- raw data is the audit trail of every prediction
-- Late rows: compression makes a chunk expensive to modify; keep the 7-day window above the
-- longest expected store-and-forward outage.

-- OPC UA reports changes only. A heartbeat proves the session was alive, which is what
-- distinguishes "value unchanged" from "nothing collected".
CREATE TABLE collector_heartbeat (
    time            timestamptz NOT NULL,
    session_id      text NOT NULL,
    connected       boolean NOT NULL,
    monitored_items integer NOT NULL,
    notifications   integer NOT NULL,
    spool_rows      integer NOT NULL,
    PRIMARY KEY (session_id, time)
);
SELECT create_hypertable('collector_heartbeat', 'time', chunk_time_interval => INTERVAL '7 days');

CREATE TABLE ingest_event (                                  -- quality / session log
    event_id        bigint GENERATED ALWAYS AS IDENTITY,
    time            timestamptz NOT NULL,
    session_id      text,
    kind            text NOT NULL,   -- session_start / session_lost / subscribe_failed / sink_down / sink_up / bad_status / stale_tag / unparseable_value / empty_file
    tag_id          smallint REFERENCES tag,
    detail          text,
    PRIMARY KEY (event_id, time)
);
SELECT create_hypertable('ingest_event', 'time', chunk_time_interval => INTERVAL '30 days');

CREATE TABLE ingest_gap (                                    -- intervals with no usable data
    gap_id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    asset_id        smallint NOT NULL REFERENCES asset,
    origin          text NOT NULL CHECK (origin IN ('hmi_csv', 'opcua')),
    last_before     timestamptz,                             -- opcua gaps
    first_after     timestamptz,
    last_before_local  timestamp,                            -- hmi_csv gaps (wall clock)
    first_after_local  timestamp,
    missing_seconds bigint NOT NULL,
    cause           text                                     -- filled only when confirmed; never guessed
);

-- 10 s and 60 s continuous aggregates of good readings.
CREATE MATERIALIZED VIEW sample_10s WITH (timescaledb.continuous) AS
SELECT time_bucket(INTERVAL '10 seconds', source_time) AS bucket, tag_id,
       count(*) AS n, avg(value) AS mean, min(value) AS min, max(value) AS max,
       stddev_samp(value) AS std, last(value, source_time) AS last, max(source_time) AS last_time
FROM sample WHERE good AND value IS NOT NULL
GROUP BY 1, 2 WITH NO DATA;
CREATE MATERIALIZED VIEW sample_60s WITH (timescaledb.continuous) AS
SELECT time_bucket(INTERVAL '60 seconds', source_time) AS bucket, tag_id,
       count(*) AS n, avg(value) AS mean, min(value) AS min, max(value) AS max,
       stddev_samp(value) AS std, last(value, source_time) AS last, max(source_time) AS last_time
FROM sample WHERE good AND value IS NOT NULL
GROUP BY 1, 2 WITH NO DATA;
SELECT add_continuous_aggregate_policy('sample_10s', start_offset => INTERVAL '1 hour',
                                       end_offset => INTERVAL '20 seconds', schedule_interval => INTERVAL '10 seconds');
SELECT add_continuous_aggregate_policy('sample_60s', start_offset => INTERVAL '6 hours',
                                       end_offset => INTERVAL '2 minutes', schedule_interval => INTERVAL '1 minute');
-- CAUTION for model inputs: with change-only reporting, avg() over notifications is not the
-- time-weighted mean of a 1 Hz grid. The historical Voltage_mean60s / Voltage_std60s are the
-- plain mean and sample std (ddof = 1) of the last 60 one-second rows including the origin
-- second. The anchor builder must rebuild that 1 Hz grid (last value carried while the
-- heartbeat shows the session alive and the tag is within max_age_s) before it computes
-- them; these aggregates are for dashboards and trend queries.

-- ============================================================================ history (HMI export)

CREATE TABLE source_file (
    file_id         integer PRIMARY KEY,
    source_root     text NOT NULL,
    relative_path   text NOT NULL,
    priority        smallint NOT NULL,                       -- 0 = main tree; later roots only fill missing seconds
    bytes           bigint NOT NULL,
    sha256          text NOT NULL,
    header_ok       boolean NOT NULL,
    rows            integer NOT NULL,
    rows_loaded     integer NOT NULL,
    rows_duplicate_dropped integer NOT NULL,
    first_ts        bigint,
    last_ts         bigint,
    UNIQUE (source_root, relative_path)
);

-- Wide, as exported: one row per recorded second, 28 readings. 5,909,066 rows today.
CREATE TABLE history_sample_1hz (
    ts_local                    timestamp NOT NULL,          -- instrument wall clock, no timezone
    ts                          bigint NOT NULL,             -- wall clock as epoch seconds as if UTC
    argon                       double precision,
    nitrogen                    double precision,
    airjet                      double precision,
    silvent                     double precision,
    current                     double precision,
    power                       double precision,
    voltage                     double precision,
    cooling_water_flow          double precision,
    conductivity                double precision,
    cooling_water_inlet_temp    double precision,
    cooling_water_outlet_temp   double precision,
    feedrate_1                  double precision,
    disk_1                      double precision,
    carrier_gas_1               double precision,
    feed_factor_1               double precision,
    press_hopper_1              double precision,
    stirrer_1                   double precision,
    weight_hopper_1             double precision,
    feedrate_2                  double precision,
    disk_2                      double precision,
    carrier_gas_2               double precision,
    feed_factor_2               double precision,
    press_hopper_2              double precision,
    stirrer_2                   double precision,
    weight_hopper_2             double precision,
    station_1                   double precision,
    station_2                   double precision,
    o2_sensor                   double precision,
    file_id                     integer NOT NULL REFERENCES source_file,
    source_row                  integer NOT NULL,            -- 1-based line in the CSV (header = 1)
    PRIMARY KEY (ts_local)
);
SELECT create_hypertable('history_sample_1hz', 'ts_local', chunk_time_interval => INTERVAL '7 days');
-- Load: COPY from lake/raw_1hz/**/part-0.parquet (e.g. via DuckDB's postgres extension or
-- pyarrow -> COPY). Column names are tag.pg_column.
--   ALTER TABLE history_sample_1hz SET (timescaledb.compress, timescaledb.compress_orderby = 'ts_local DESC');
--   SELECT compress_chunk(c) FROM show_chunks('history_sample_1hz') c;   -- history is immutable

-- Wall clock -> instant, usable only after asset.history_timezone is confirmed on site.
CREATE FUNCTION history_as_utc(local_time timestamp, asset smallint DEFAULT 1) RETURNS timestamptz
LANGUAGE sql STABLE AS $$
    SELECT local_time AT TIME ZONE a.history_timezone FROM asset a WHERE a.asset_id = asset
$$;  -- returns NULL while the zone is unknown, by construction

-- ============================================================================ curated labels

CREATE TABLE electrode_cycle (
    cycle_id                text PRIMARY KEY,                -- L000 ..
    asset_id                smallint NOT NULL DEFAULT 1 REFERENCES asset,
    split                   text,                            -- v2 split: train / validation / test (NULL for new live cycles)
    source_split            text,                            -- v1 split, kept for traceability
    start_event             text,
    end_event               text,
    start_lower             timestamp,
    start_upper             timestamp,
    usable_start            timestamp NOT NULL,
    usable_end              timestamp,
    left_censored           boolean NOT NULL,
    right_censored          boolean NOT NULL,                -- natural voltage end of life not observed
    preventive_end          boolean NOT NULL,
    end_type                text,
    end_evidence            text,
    eol_endpoint            timestamp,                       -- T: first failed product of the final run starts
    source_eol_endpoint     timestamp,                       -- v1: second failed product ends (153-157 s later)
    eol_strict_observed     boolean,
    eol_sensor_observed     boolean,
    product_count           integer,
    eligible_products       integer,
    failure_products        integer,
    voltage_failed_products integer,
    feed_failed_products    integer,
    pressure_failed_products integer,
    observed_on_hours       double precision,
    recorded_old_electrode_hours double precision,
    unknown_clock_hours     double precision,
    reset_source            text,                            -- live: what confirmed the gun change (operator / Gun_ID / Gun_Hours); never inferred from voltage
    note                    text,
    end_note                text
);

CREATE TABLE maintenance_event (
    record_id               text PRIMARY KEY,                -- G001 .. / S001 ..
    asset_id                smallint NOT NULL DEFAULT 1 REFERENCES asset,
    record_date             date,
    action_type             text,                            -- natural_eol / preventive / low_voltage_then_replacement / ...
    confirmation            text,
    action_time_lower       timestamp,                       -- candidate range, not an exact removal second
    action_time_upper       timestamp,
    exact_change_time       timestamp,
    exact_time_confirmed    boolean NOT NULL DEFAULT false,
    old_serial              text,
    new_serial              text,
    working_hours           double precision,                -- register value for the removed electrode
    ignitions               integer,
    work_order              text,
    evidence_source         text,
    site_note               text,
    detail                  jsonb                            -- remaining source columns, verbatim
);

CREATE TABLE product (
    product_id              text PRIMARY KEY,                -- N000001 ..
    cycle_id                text REFERENCES electrode_cycle,
    split                   text,
    source_split            text,
    supervised_product_eligible boolean NOT NULL,
    station                 smallint,
    hopper                  smallint,
    start_time              timestamp NOT NULL,
    end_time                timestamp NOT NULL,
    label_available_at      timestamp,                       -- a label must never be used before this instant
    samples                 integer,
    coverage                double precision,
    voltage_mean            double precision,
    feedrate_mean           double precision,
    pressure_mean           double precision,
    voltage_fail            boolean,
    feedrate_fail           boolean,
    hopper_pressure_fail    boolean,
    failure                 boolean,                         -- NULL = excluded / unknown, never "normal"
    failure_type            text,
    label_valid             boolean,
    label_source            text,
    exclusion_reason        text,
    detail                  jsonb                            -- remaining source columns, verbatim
);
CREATE INDEX product_cycle_idx ON product (cycle_id, start_time);
CREATE INDEX product_start_idx ON product (start_time);

CREATE TABLE target_event (
    event_id                text PRIMARY KEY,                -- T001 ..
    cycle_id                text NOT NULL REFERENCES electrode_cycle,
    split                   text,
    event_time              timestamp NOT NULL,              -- first failed product of the run starts
    cause                   text NOT NULL,                   -- voltage / feedrate / pressure
    is_eol                  boolean NOT NULL,
    product_ids             text[] NOT NULL,
    run_length              smallint NOT NULL,
    second_completion       timestamp,
    last_completion         timestamp,
    deadline_20m            timestamp,
    plc_batch_confirmed     boolean NOT NULL DEFAULT false
);

-- ============================================================================ model inputs / targets

-- Processing origin: a 10-second point inside a confirmed product (>= 3 s after its start),
-- with the readings and causal 60 s voltage statistics the models were trained on.
CREATE TABLE anchor (
    origin_local            timestamp NOT NULL,              -- history: wall clock; live: fill origin_time too
    ts                      bigint NOT NULL,
    origin_time             timestamptz,                     -- live origins only
    anchor_id               text NOT NULL,                   -- 'A' || ts
    cycle_id                text REFERENCES electrode_cycle,
    split                   text,
    product_id              text REFERENCES product,
    seconds_since_product_start integer,
    voltage                 double precision,
    current                 double precision,
    power                   double precision,
    station_1               double precision,
    station_2               double precision,
    feedrate_1              double precision,
    feedrate_2              double precision,
    press_hopper_1          double precision,
    press_hopper_2          double precision,
    voltage_mean60s         double precision,                -- mean of the last 60 one-second rows incl. the origin second
    voltage_std60s          double precision,                -- sample std (ddof = 1) of the same rows
    features_28             jsonb,                           -- optional: 28 last values + 28 sixty-second means (anchor_feature_28)
    observation_horizon_seconds bigint,
    remaining_seconds       double precision,                -- eol_endpoint - origin; NULL when censored or after T
    has_raw_1hz             boolean,
    source                  text NOT NULL DEFAULT 'history' CHECK (source IN ('history', 'live')),
    PRIMARY KEY (ts)
);
CREATE INDEX anchor_cycle_idx ON anchor (cycle_id, ts);

CREATE TABLE target (
    ts                      bigint PRIMARY KEY REFERENCES anchor,
    y_eol_30m               smallint,                        -- 1 when 0 < remaining <= 1800 s; NULL = not decidable, never 0
    y_eol_45m               smallint,
    y_eol_60m               smallint,
    y_pre20_40m             smallint                         -- 1 when 1200 < remaining <= 2400 s
);

-- ============================================================================ models, predictions, alarms

CREATE TABLE model_registry (
    model_id                integer PRIMARY KEY,
    name                    text NOT NULL,
    family                  text NOT NULL,
    kind                    text NOT NULL,
    version                 text NOT NULL,
    head                    text,
    head_index              smallint,
    threshold               double precision,                -- 1.000001 = sentinel "alarm disabled"
    alarm_enabled           boolean NOT NULL,
    selected                boolean NOT NULL DEFAULT false,
    output_kind             text NOT NULL,                   -- uncalibrated_score / calibrated_probability / remaining_life_quantiles ...
    frozen_at_utc           timestamptz,
    source_repository       text,
    source_commit           text,
    artifact_path           text,
    artifact_sha256         text,
    alarm_policy            jsonb,
    spec                    jsonb,
    UNIQUE (name, version)
);

-- Append-only: a prediction, once stored, is never changed. A better model is a new model_id.
CREATE TABLE prediction (
    model_id                integer NOT NULL REFERENCES model_registry,
    ts                      bigint NOT NULL,                 -- origin (same clock as anchor.ts)
    origin_time             timestamptz,                     -- live origins only
    score                   double precision,                -- primary head
    outputs                 jsonb,                           -- other heads, remaining-life quantiles, per-risk scores
    input_coverage          real,                            -- share of the history window that was actually observed
    created_at              timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, ts)
);
CREATE FUNCTION prediction_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'prediction ledger is append-only: publish a new model version instead';
END $$;
CREATE TRIGGER prediction_no_change BEFORE UPDATE OR DELETE ON prediction
    FOR EACH ROW EXECUTE FUNCTION prediction_append_only();
-- Additionally: REVOKE UPDATE, DELETE, TRUNCATE ON prediction FROM <service role>;

CREATE TABLE alarm (
    alarm_id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id                integer NOT NULL REFERENCES model_registry,
    ts                      bigint NOT NULL,                 -- origin that emitted the alarm
    alarm_time              timestamptz,                     -- live only
    cycle_id                text REFERENCES electrode_cycle,
    score                   double precision NOT NULL,
    threshold               double precision NOT NULL,
    released_ts             bigint,                          -- set once, when the episode ends
    acknowledged_by         text,
    acknowledged_at         timestamptz,
    -- offline evaluation only (filled after the fact, never visible to the model):
    matched_event           text REFERENCES target_event,
    false_alarm             boolean,
    post_event              boolean,
    unknown                 boolean,
    UNIQUE (model_id, ts)
);

-- ============================================================================ roles (suggestion)
--   booth2_collector : INSERT on sample, collector_heartbeat, ingest_event
--   booth2_model     : SELECT on everything; INSERT on anchor, prediction, alarm
--   booth2_panel     : SELECT only
--   labels (product, target_event, electrode_cycle, target) are written by the offline labelling job only.
