"""SQLite form of the live-ingestion tables (same logical schema as schema/postgres_timescale.sql).

Used by load_history.py (the tables exist, empty, in db/booth2.sqlite) and by the collector's
SQLite sink / self-test. Live rows carry real UTC instants from OPC UA, unlike the historical
`sample_1hz` wall clock; the two are kept in separate tables until the site clock is confirmed.
"""

LIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_sample (
  tag_id            INTEGER NOT NULL,
  source_time_us    INTEGER NOT NULL,   -- OPC UA SourceTimestamp, UTC, microseconds since 1970
  server_time_us    INTEGER,            -- OPC UA ServerTimestamp (Kepware), UTC
  received_time_us  INTEGER NOT NULL,   -- collector clock when the notification arrived, UTC
  value             REAL,               -- NULL when the value is not numeric or the read was bad
  value_text        TEXT,               -- string tags (Gun_ID.DATA) and unparseable values
  status_code       INTEGER NOT NULL,   -- raw OPC UA StatusCode (0 = Good)
  session_id        TEXT NOT NULL,
  PRIMARY KEY (tag_id, source_time_us)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_live_sample_time ON live_sample (source_time_us);
CREATE TABLE IF NOT EXISTS collector_heartbeat (
  time_us           INTEGER NOT NULL,   -- collector clock, UTC
  session_id        TEXT NOT NULL,
  connected         INTEGER NOT NULL,   -- 1 while the OPC UA session answered the last check
  monitored_items   INTEGER NOT NULL,
  notifications     INTEGER NOT NULL,   -- data changes received since the previous heartbeat
  spool_rows        INTEGER NOT NULL,   -- rows waiting in the local store-and-forward buffer
  PRIMARY KEY (session_id, time_us)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS ingest_event (
  event_id          INTEGER PRIMARY KEY,
  time_us           INTEGER NOT NULL,
  session_id        TEXT,
  kind              TEXT NOT NULL,      -- session_start / session_lost / reconnect / subscribe_failed / sink_down / sink_up / bad_status / stale_tag
  tag_id            INTEGER,
  detail            TEXT
);
"""

APPEND_ONLY = """
CREATE TRIGGER IF NOT EXISTS prediction_no_update BEFORE UPDATE ON prediction
  BEGIN SELECT RAISE(ABORT, 'prediction ledger is append-only: publish a new model version instead'); END;
CREATE TRIGGER IF NOT EXISTS prediction_no_delete BEFORE DELETE ON prediction
  BEGIN SELECT RAISE(ABORT, 'prediction ledger is append-only'); END;
"""
