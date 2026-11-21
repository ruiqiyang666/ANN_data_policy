"""Local index of the plant history API (read-only towards the plant).

The Windows data server keeps every OPC UA notification of Booth2 Plasma2 and serves them
through a read-only HTTP API. This module keeps a rebuildable SQLite copy on the dashboard
host: the notifications as they were stored, a tag catalogue, per-minute and per-hour
aggregates, and the sync watermark. One writer (the sync), any number of readers (WAL).

Times are UTC milliseconds. `received_ms`, the moment the Linux collector received the
notification, is the time axis of every query, as in the source API. OPC UA reports a value
when it changes, so between two notifications the value is the earlier one: nothing here
interpolates, and silence of one tag is not a gap. A gap is a minute in which no tag at all
was heard.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from domain.real_contract import REAL_FIELDS

PREFIX = "HA.Booth2.Plasma2."
# Short names on site that differ from the dashboard's field keys (same table as api/field_live.py).
SITE_NAMES = {"AirJe": "AirJet", "Cooling_water_inlet_temp": "Cooling_water_inlet_temp.",
              "Cooling_water_outlet_temp": "Cooling_water_outlet_temp.", "Press_Hopper_1": "Press._Hopper_1",
              "Press_Hopper_2": "Press._Hopper_2"}
MAP_VERSION = "site-20261119-1"
SCHEMA_VERSION = 1  # layout of the index file (PRAGMA user_version)
RULES_VERSION = "1"  # buckets: low / high / sum / last over good numeric values; observed_count over every notification
MINUTE, HOUR = 60_000, 3_600_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
FIELDS = {f["key"]: f for f in REAL_FIELDS}

SCHEMA = """
CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY,
    equipment_key TEXT NOT NULL,
    path TEXT NOT NULL,
    node_id TEXT,
    name TEXT NOT NULL,
    canonical_key TEXT,
    display_name TEXT NOT NULL,
    unit TEXT,
    unit_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (unit_confirmed IN (0, 1)),
    map_version TEXT NOT NULL,
    UNIQUE (equipment_key, path)
);
CREATE TABLE IF NOT EXISTS raw_events (
    archive_key TEXT NOT NULL,
    history_id INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    source_epoch TEXT NOT NULL,
    source_seq INTEGER NOT NULL,
    tag_id INTEGER NOT NULL REFERENCES tags(tag_id),
    received_ms INTEGER NOT NULL,
    source_ms INTEGER,
    server_ms INTEGER,
    ingested_ms INTEGER,
    status_raw TEXT NOT NULL,
    quality TEXT NOT NULL CHECK (quality IN ('good', 'uncertain', 'bad', 'unknown')),
    value_kind TEXT NOT NULL,
    value_json TEXT NOT NULL,
    numeric_value REAL,
    PRIMARY KEY (archive_key, history_id),
    UNIQUE (source_id, source_epoch, source_seq)
);
CREATE INDEX IF NOT EXISTS idx_raw_tag_received ON raw_events (tag_id, received_ms, archive_key, history_id);
CREATE TABLE IF NOT EXISTS ingest_cursor (
    archive_key TEXT PRIMARY KEY,
    committed_history_id INTEGER NOT NULL CHECK (committed_history_id >= 0),
    last_success_ms INTEGER,
    last_source_max_id INTEGER
);
CREATE TABLE IF NOT EXISTS history_alias (
    archive_key TEXT NOT NULL,
    history_id INTEGER NOT NULL,
    canonical_archive_key TEXT NOT NULL,
    canonical_history_id INTEGER NOT NULL,
    PRIMARY KEY (archive_key, history_id),
    FOREIGN KEY (canonical_archive_key, canonical_history_id) REFERENCES raw_events(archive_key, history_id)
);
CREATE TABLE IF NOT EXISTS ingest_issue (
    archive_key TEXT NOT NULL,
    history_id INTEGER NOT NULL,
    reason TEXT NOT NULL,
    detail TEXT,
    original_json TEXT,
    detected_ms INTEGER NOT NULL,
    resolved_ms INTEGER,
    PRIMARY KEY (archive_key, history_id)
);
CREATE TABLE IF NOT EXISTS metric_bucket (
    tag_id INTEGER NOT NULL REFERENCES tags(tag_id),
    bucket_ms INTEGER NOT NULL,
    window_ms INTEGER NOT NULL,
    rules_version TEXT NOT NULL,
    observed_count INTEGER NOT NULL,
    valid_numeric_count INTEGER NOT NULL,
    bad_count INTEGER NOT NULL,
    low REAL,
    high REAL,
    sum REAL,
    last_value REAL,
    last_ms INTEGER,
    PRIMARY KEY (tag_id, window_ms, rules_version, bucket_ms)
);
CREATE INDEX IF NOT EXISTS idx_bucket_time ON metric_bucket (window_ms, bucket_ms, observed_count, bad_count);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT NOT NULL,
    source_epoch TEXT NOT NULL,
    events INTEGER NOT NULL,
    first_seq INTEGER NOT NULL,
    last_seq INTEGER NOT NULL,
    first_ms INTEGER NOT NULL,
    last_ms INTEGER NOT NULL,
    PRIMARY KEY (source_id, source_epoch)
);
CREATE TABLE IF NOT EXISTS sync_round (
    round_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_ms INTEGER NOT NULL,
    finished_ms INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    detail TEXT,
    pages INTEGER NOT NULL,
    fetched INTEGER NOT NULL,
    inserted INTEGER NOT NULL,
    idempotent INTEGER NOT NULL,
    aliased INTEGER NOT NULL,
    quarantined INTEGER NOT NULL,
    request_ms INTEGER NOT NULL,
    committed_history_id INTEGER NOT NULL,
    source_max_id INTEGER
);
"""


def now_ms():
    return time.time_ns() // 1_000_000


def parse_ms(text):
    """ISO 8601 with an offset -> UTC milliseconds. None stays None; a naive or malformed time raises."""
    if text is None:
        return None
    if not isinstance(text, str):
        raise ValueError("time is not a string")
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("time without an offset")
    return (stamp - EPOCH) // timedelta(milliseconds=1)


def iso_ms(ms):
    return None if ms is None else (EPOCH + timedelta(milliseconds=int(ms))).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def quality_of(status):
    """OPC UA status as reported -> good / uncertain / bad / unknown. The original text is kept beside it."""
    if isinstance(status, bool):
        return "unknown"
    code = status if isinstance(status, int) else None
    text = str(status)
    match = re.fullmatch(r"StatusCode\(value=(\d+)\)", text)
    if match:
        code = int(match.group(1))
    if code is not None:
        return ("good", "uncertain", "bad", "bad")[(code >> 30) & 3]
    for name in ("good", "uncertain", "bad"):
        if re.search(rf"(^|\W){name}", text.lower()):
            return name
    return "unknown"


def value_of(value):
    """JSON value -> (kind, number or None). Only finite int / float values are numbers."""
    if value is None:
        return "null", None
    if isinstance(value, bool):
        return "bool", None
    if isinstance(value, int):
        return "int", float(value)
    if isinstance(value, float):
        return "float", value if math.isfinite(value) else None
    if isinstance(value, str):
        return "str", None
    return "json", None


def describe_tag(path, name):
    """Catalogue entry of a site tag: the dashboard field it maps to, or an auxiliary reading."""
    short = path[len(PREFIX):] if path.startswith(PREFIX) else (name or path)
    key = SITE_NAMES.get(short, short)
    field = FIELDS.get(key)
    if field is None:
        return dict(canonical_key=None, display_name=short, unit=None)
    return dict(canonical_key=key, display_name=field["name"], unit=field["unit"])


def day_bounds(day, zone="Asia/Shanghai"):
    """A calendar day in the display time zone -> half-open [start, end) in UTC milliseconds."""
    tz = ZoneInfo(zone)
    start = datetime.fromisoformat(day).replace(tzinfo=tz)
    if start.hour or start.minute or start.second or start.microsecond:
        raise ValueError("a calendar day is expected")
    end = (start.replace(tzinfo=None) + timedelta(days=1)).replace(tzinfo=tz)
    return parse_ms(start.isoformat()), parse_ms(end.isoformat())


def local_ms(text, zone="Asia/Shanghai"):
    """A time typed in the page: with an offset as it is, without one in the display time zone."""
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=ZoneInfo(zone))
    return (stamp - EPOCH) // timedelta(milliseconds=1)


class Conflict(Exception):
    """A record contradicts what is already stored; ingestion stops before it."""

    def __init__(self, history_id, reason, detail):
        super().__init__(f"{reason} at history id {history_id}: {detail}")
        self.history_id, self.reason, self.detail = history_id, reason, detail


class HistoryIndex:
    def __init__(self, path, archive_key="plasma2-a", source_epoch="1", equipment_key="booth2-plasma2"):
        self.path = Path(path)
        self.archive_key, self.source_epoch, self.equipment_key = archive_key, source_epoch, equipment_key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect(write=True) as db:
            built = db.execute("PRAGMA user_version").fetchone()[0]
            if built not in (0, SCHEMA_VERSION) or (built == 0 and db.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]):
                # The index is a rebuildable copy: an old layout is set aside by hand, never converted silently.
                raise RuntimeError(f"History index {self.path} has layout {built}, this version writes layout {SCHEMA_VERSION}: move the file away to rebuild it")
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.execute("INSERT OR IGNORE INTO ingest_cursor (archive_key, committed_history_id) VALUES (?, 0)", (archive_key,))

    @contextmanager
    def connect(self, write=False):
        """A connection of its own for every caller; closed on exit. Readers cannot write."""
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            if write:
                db.execute("PRAGMA journal_mode = WAL")
                db.execute("PRAGMA synchronous = NORMAL")
            else:
                db.execute("PRAGMA query_only = ON")
            yield db
        finally:
            db.close()

    # ---------- writing (the sync only) ----------
    def cursor(self):
        with self.connect() as db:
            return int(db.execute("SELECT committed_history_id FROM ingest_cursor WHERE archive_key = ?", (self.archive_key,)).fetchone()[0])

    def _tag(self, db, cache, event):
        path = event["path"]
        if path not in cache:
            row = db.execute("SELECT tag_id FROM tags WHERE equipment_key = ? AND path = ?", (self.equipment_key, path)).fetchone()
            if row is None:
                info = describe_tag(path, event.get("name"))
                row = (db.execute(
                    "INSERT INTO tags (equipment_key, path, node_id, name, canonical_key, display_name, unit, map_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.equipment_key, path, event.get("node_id"), event.get("name") or path, info["canonical_key"], info["display_name"],
                     info["unit"], MAP_VERSION)).lastrowid,)
            cache[path] = int(row[0])
        return cache[path]

    @staticmethod
    def _parse(event):
        """The stored form of one history record; raises ValueError when it cannot be read."""
        if not isinstance(event.get("path"), str) or not event["path"]:
            raise ValueError("missing path")
        if not isinstance(event.get("source_id"), str) or isinstance(event.get("source_seq"), bool) or not isinstance(event.get("source_seq"), int):
            raise ValueError("missing source_id or source_seq")
        received = parse_ms(event.get("received_at"))
        if received is None:
            raise ValueError("missing received_at")
        kind, number = value_of(event.get("value"))
        status = event.get("status")
        return dict(source_id=event["source_id"], source_seq=event["source_seq"], received_ms=received,
                    source_ms=parse_ms(event.get("source_timestamp")), server_ms=parse_ms(event.get("server_timestamp")),
                    ingested_ms=parse_ms(event.get("ingested_at")), status_raw="" if status is None else str(status),
                    quality=quality_of(status), value_kind=kind,
                    value_json=json.dumps(None if kind == "float" and number is None else event.get("value"), ensure_ascii=False),
                    numeric_value=number)

    SAME = ("source_id", "source_seq", "received_ms", "source_ms", "status_raw", "value_json")

    def ingest_page(self, events, source_max_id=None):
        """Store one page of history records in one transaction and move the watermark.

        Returns counts and the last history id that is now accounted for. A record that
        contradicts a stored one stops the page before it (`conflict` in the result); a record
        that cannot be read is set aside, and the page continues. Both keep their original text
        in ingest_issue; a stored row keeps the parsed fields only (times to the millisecond),
        the data server remains the record. Feeding a page twice changes nothing.
        """
        result = dict(inserted=0, idempotent=0, aliased=0, quarantined=0, conflict=None)
        ids = [e.get("id") for e in events]
        if any(isinstance(i, bool) or not isinstance(i, int) or i <= 0 for i in ids) or any(b <= a for a, b in zip(ids, ids[1:])):
            raise ValueError("history ids must be positive and strictly increasing")
        with self.connect(write=True) as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                last = int(db.execute("SELECT committed_history_id FROM ingest_cursor WHERE archive_key = ?", (self.archive_key,)).fetchone()[0])
                tags, touched = {}, set()
                for event in events:
                    history_id = event["id"]
                    original = json.dumps(event, ensure_ascii=False, sort_keys=True)
                    try:
                        row = self._parse(event)
                    except (ValueError, TypeError, OverflowError) as cause:
                        db.execute("INSERT OR IGNORE INTO ingest_issue (archive_key, history_id, reason, detail, original_json, detected_ms) VALUES (?, ?, 'unreadable', ?, ?, ?)",
                                   (self.archive_key, history_id, str(cause), original, now_ms()))
                        result["quarantined"] += 1
                        last = max(last, history_id)
                        continue
                    tag_id = self._tag(db, tags, event)
                    try:
                        stored = db.execute("SELECT * FROM raw_events WHERE archive_key = ? AND history_id = ?", (self.archive_key, history_id)).fetchone()
                        if stored is not None:
                            if stored["tag_id"] != tag_id or any(stored[k] != row[k] for k in self.SAME):
                                raise Conflict(history_id, "history_id_reused", "the same history id now carries different content")
                            result["idempotent"] += 1
                        else:
                            twin = db.execute("SELECT * FROM raw_events WHERE source_id = ? AND source_epoch = ? AND source_seq = ?",
                                              (row["source_id"], self.source_epoch, row["source_seq"])).fetchone()
                            if twin is not None:
                                if twin["tag_id"] != tag_id or any(twin[k] != row[k] for k in self.SAME):
                                    raise Conflict(history_id, "source_sequence_reused", "a collector sequence number now carries different content")
                                db.execute("INSERT OR IGNORE INTO history_alias VALUES (?, ?, ?, ?)",
                                           (self.archive_key, history_id, twin["archive_key"], twin["history_id"]))
                                result["aliased"] += 1
                            else:
                                db.execute(
                                    "INSERT INTO raw_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    (self.archive_key, history_id, row["source_id"], self.source_epoch, row["source_seq"], tag_id, row["received_ms"],
                                     row["source_ms"], row["server_ms"], row["ingested_ms"], row["status_raw"], row["quality"], row["value_kind"],
                                     row["value_json"], row["numeric_value"]))
                                touched.add((tag_id, row["received_ms"] // HOUR * HOUR))
                                db.execute(
                                    """INSERT INTO sources VALUES (:id, :epoch, 1, :seq, :seq, :ms, :ms)
                                       ON CONFLICT (source_id, source_epoch) DO UPDATE SET events = events + 1, first_seq = MIN(first_seq, :seq),
                                           last_seq = MAX(last_seq, :seq), first_ms = MIN(first_ms, :ms), last_ms = MAX(last_ms, :ms)""",
                                    dict(id=row["source_id"], epoch=self.source_epoch, seq=row["source_seq"], ms=row["received_ms"]))
                                result["inserted"] += 1
                    except Conflict as conflict:
                        db.execute("INSERT OR IGNORE INTO ingest_issue (archive_key, history_id, reason, detail, original_json, detected_ms) VALUES (?, ?, ?, ?, ?, ?)",
                                   (self.archive_key, history_id, conflict.reason, conflict.detail, original, now_ms()))
                        result["conflict"] = dict(history_id=history_id, reason=conflict.reason, detail=conflict.detail)
                        break
                    last = max(last, history_id)
                for tag_id, hour in sorted(touched):
                    self._aggregate(db, tag_id, hour)
                db.execute("UPDATE ingest_cursor SET committed_history_id = ?, last_success_ms = ?, last_source_max_id = COALESCE(?, last_source_max_id) WHERE archive_key = ?",
                           (last, now_ms(), source_max_id, self.archive_key))
                db.execute("COMMIT")
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
        result["last_id"] = last
        return result

    @staticmethod
    def _aggregate(db, tag_id, hour):
        """Rebuild the minute buckets and the hour bucket of one tag-hour from the stored notifications.

        Recomputing from the notifications themselves keeps late-arriving old records exact.
        """
        ok = "quality = 'good' AND numeric_value IS NOT NULL"
        stats = db.execute(
            f"""SELECT received_ms / {MINUTE} * {MINUTE} AS bucket, COUNT(*) AS seen, SUM({ok}) AS valid, SUM(quality != 'good') AS bad,
                       MIN(CASE WHEN {ok} THEN numeric_value END) AS low, MAX(CASE WHEN {ok} THEN numeric_value END) AS high,
                       SUM(CASE WHEN {ok} THEN numeric_value END) AS total
                FROM raw_events WHERE tag_id = ? AND received_ms >= ? AND received_ms < ? GROUP BY bucket""", (tag_id, hour, hour + HOUR)).fetchall()
        latest = {r["bucket"]: (r["numeric_value"], r["received_ms"]) for r in db.execute(
            f"""SELECT bucket, numeric_value, received_ms FROM (
                    SELECT received_ms / {MINUTE} * {MINUTE} AS bucket, numeric_value, received_ms,
                           ROW_NUMBER() OVER (PARTITION BY received_ms / {MINUTE} ORDER BY received_ms DESC, archive_key DESC, history_id DESC) AS position
                    FROM raw_events WHERE tag_id = ? AND received_ms >= ? AND received_ms < ? AND {ok}) WHERE position = 1""",
            (tag_id, hour, hour + HOUR))}
        db.execute("DELETE FROM metric_bucket WHERE tag_id = ? AND rules_version = ? AND ((window_ms = ? AND bucket_ms >= ? AND bucket_ms < ?) OR (window_ms = ? AND bucket_ms = ?))",
                   (tag_id, RULES_VERSION, MINUTE, hour, hour + HOUR, HOUR, hour))
        rows = [(tag_id, r["bucket"], MINUTE, RULES_VERSION, r["seen"], r["valid"] or 0, r["bad"] or 0, r["low"], r["high"], r["total"],
                 *latest.get(r["bucket"], (None, None))) for r in stats]
        if rows:
            valid = [r for r in rows if r[5]]
            final = max(valid, key=lambda r: r[11]) if valid else None
            rows.append((tag_id, hour, HOUR, RULES_VERSION, sum(r[4] for r in rows), sum(r[5] for r in rows), sum(r[6] for r in rows),
                         min((r[7] for r in valid), default=None), max((r[8] for r in valid), default=None),
                         sum(r[9] for r in valid) if valid else None, final[10] if final else None, final[11] if final else None))
            db.executemany("INSERT INTO metric_bucket VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def record_round(self, **round_):
        with self.connect(write=True) as db:
            db.execute("""INSERT INTO sync_round (started_ms, finished_ms, outcome, detail, pages, fetched, inserted, idempotent, aliased, quarantined,
                                                  request_ms, committed_history_id, source_max_id)
                          VALUES (:started_ms, :finished_ms, :outcome, :detail, :pages, :fetched, :inserted, :idempotent, :aliased, :quarantined,
                                  :request_ms, :committed_history_id, :source_max_id)""", round_)
            db.execute("DELETE FROM sync_round WHERE round_id <= (SELECT MAX(round_id) FROM sync_round) - 2000")

    # ---------- reading ----------
    def summary(self):
        """Watermark, extent and totals of what is indexed."""
        with self.connect() as db:
            cursor = db.execute("SELECT * FROM ingest_cursor WHERE archive_key = ?", (self.archive_key,)).fetchone()
            span = db.execute("SELECT MIN(bucket_ms), MAX(last_ms), SUM(observed_count), SUM(bad_count) FROM metric_bucket WHERE window_ms = ? AND rules_version = ?",
                              (HOUR, RULES_VERSION)).fetchone()
            first, newest = db.execute("SELECT MIN(first_ms), MAX(last_ms) FROM sources").fetchone()  # kept current by every insert
            issues = db.execute("SELECT reason, COUNT(*) AS n FROM ingest_issue WHERE resolved_ms IS NULL GROUP BY reason").fetchall()
            aliases = db.execute("SELECT COUNT(*) FROM history_alias").fetchone()[0]
            tags = db.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
            rounds = [dict(r) for r in db.execute("SELECT * FROM sync_round ORDER BY round_id DESC LIMIT 5")]
        return dict(archive_key=self.archive_key, source_epoch=self.source_epoch, equipment_key=self.equipment_key,
                    committed_history_id=int(cursor["committed_history_id"]), last_success_ms=cursor["last_success_ms"],
                    source_max_id=cursor["last_source_max_id"], first_ms=first, last_ms=newest, events=int(span[2] or 0),
                    not_good=int(span[3] or 0), tags=int(tags), aliases=int(aliases), open_issues={r["reason"]: r["n"] for r in issues},
                    rounds=rounds, map_version=MAP_VERSION, rules_version=RULES_VERSION)

    def tag_ids(self):
        with self.connect() as db:
            return {r[0] for r in db.execute("SELECT tag_id FROM tags")}

    def tags(self):
        """Tag catalogue with what was last heard of each tag."""
        with self.connect() as db:
            rows = db.execute(
                """SELECT t.*, s.events, s.not_good, s.first_ms, s.last_ms FROM tags t LEFT JOIN (
                       SELECT tag_id, SUM(observed_count) AS events, SUM(bad_count) AS not_good, MIN(bucket_ms) AS first_ms, MAX(bucket_ms) AS last_ms
                       FROM metric_bucket WHERE window_ms = ? AND rules_version = ? GROUP BY tag_id) s USING (tag_id)
                   ORDER BY t.canonical_key IS NULL, t.tag_id""", (HOUR, RULES_VERSION)).fetchall()
            out = []
            for r in rows:
                # the index order (tag, time, archive, id) read backwards: one seek, no sort
                last = db.execute("SELECT received_ms, quality, value_kind, value_json, status_raw FROM raw_events WHERE tag_id = ? ORDER BY received_ms DESC, archive_key DESC, history_id DESC LIMIT 1",
                                  (r["tag_id"],)).fetchone()
                first = db.execute("SELECT MIN(received_ms) FROM raw_events WHERE tag_id = ?", (r["tag_id"],)).fetchone()[0]
                group = FIELDS[r["canonical_key"]]["group"] if r["canonical_key"] else "辅助读数"
                out.append(dict(tag_id=r["tag_id"], path=r["path"], name=r["name"], node_id=r["node_id"], key=r["canonical_key"],
                                display_name=r["display_name"], unit=r["unit"], unit_confirmed=bool(r["unit_confirmed"]), group=group,
                                events=int(r["events"] or 0), not_good=int(r["not_good"] or 0), first_ms=first,
                                last_ms=last["received_ms"] if last else None, last_quality=last["quality"] if last else None,
                                last_kind=last["value_kind"] if last else None, last_value=json.loads(last["value_json"]) if last else None,
                                last_status=last["status_raw"] if last else None))
        return out

    def series(self, tag_ids, start_ms, end_ms, max_points=1200):
        """Per tag: the notifications themselves when they fit, else low / high / last per time bucket.

        `before` is the last notification ahead of the window: the value that still held at its left edge.
        """
        out = []
        with self.connect() as db:
            for tag_id in tag_ids:
                count = db.execute("SELECT COUNT(*) FROM raw_events WHERE tag_id = ? AND received_ms >= ? AND received_ms < ?", (tag_id, start_ms, end_ms)).fetchone()[0]
                before = db.execute("SELECT received_ms, numeric_value, quality, value_json FROM raw_events WHERE tag_id = ? AND received_ms < ? ORDER BY received_ms DESC, archive_key DESC, history_id DESC LIMIT 1",
                                    (tag_id, start_ms)).fetchone()
                held = None if before is None else dict(t=before["received_ms"], v=before["numeric_value"], q=before["quality"], raw=json.loads(before["value_json"]))
                if count <= max_points:
                    rows = db.execute("SELECT received_ms, numeric_value, quality, value_kind, value_json FROM raw_events WHERE tag_id = ? AND received_ms >= ? AND received_ms < ? ORDER BY received_ms, archive_key, history_id",
                                      (tag_id, start_ms, end_ms)).fetchall()
                    points = [dict(t=r["received_ms"], v=r["numeric_value"], q=r["quality"], **({} if r["value_kind"] in ("float", "int") else {"raw": json.loads(r["value_json"])})) for r in rows]
                    out.append(dict(tag_id=tag_id, mode="raw", window_ms=None, events=count, before=held, points=points))
                    continue
                # the smallest round bucket that keeps the window within max_points: minutes up to half an hour, then hours
                span = end_ms - start_ms
                need = span / max_points
                size = next((m * MINUTE for m in (1, 2, 3, 5, 10, 15, 20, 30) if m * MINUTE >= need), None)
                window = MINUTE
                if size is None:
                    window = HOUR
                    size = next((h * HOUR for h in (1, 2, 3, 4, 6, 8, 12, 24) if h * HOUR >= need), math.ceil(need / (24 * HOUR)) * 24 * HOUR)
                # buckets are whole: the window is widened outwards to the bucket size and reported back
                first_bucket, end_bucket = start_ms // size * size, -(-end_ms // size) * size
                rows = db.execute(
                    """SELECT bucket_ms / :size * :size AS bucket, SUM(observed_count) AS seen, SUM(valid_numeric_count) AS valid, SUM(bad_count) AS bad,
                              MIN(low) AS low, MAX(high) AS high, SUM(sum) AS total
                       FROM metric_bucket WHERE tag_id = :tag AND window_ms = :window AND rules_version = :rules AND bucket_ms >= :start AND bucket_ms < :end
                       GROUP BY bucket ORDER BY bucket""",
                    dict(size=size, tag=tag_id, window=window, rules=RULES_VERSION, start=first_bucket, end=end_bucket)).fetchall()
                latest = {r["bucket"]: r["last_value"] for r in db.execute(
                    """SELECT bucket, last_value FROM (SELECT bucket_ms / :size * :size AS bucket, last_value,
                                  ROW_NUMBER() OVER (PARTITION BY bucket_ms / :size ORDER BY last_ms DESC) AS position
                           FROM metric_bucket WHERE tag_id = :tag AND window_ms = :window AND rules_version = :rules AND bucket_ms >= :start AND bucket_ms < :end
                             AND last_ms IS NOT NULL) WHERE position = 1""",
                    dict(size=size, tag=tag_id, window=window, rules=RULES_VERSION, start=first_bucket, end=end_bucket))}
                points = [dict(t=r["bucket"], n=r["seen"], bad=r["bad"], low=r["low"], high=r["high"], last=latest.get(r["bucket"])) for r in rows]
                out.append(dict(tag_id=tag_id, mode="bucket", window_ms=size, range_ms=[first_bucket, end_bucket], events=count, before=held, points=points))
        return out

    def events(self, tag_ids, start_ms, end_ms, quality=None, after=None, limit=200):
        """Notifications of the given tags in time order, with a keyset cursor (received_ms, archive_key, history_id)."""
        where = "tag_id = ? AND received_ms >= ? AND received_ms < ?" + (" AND (received_ms, archive_key, history_id) > (?, ?, ?)" if after else "")             + (" AND quality = ?" if quality else "")
        rows = []
        with self.connect() as db:
            for tag_id in tag_ids:
                args = [tag_id, start_ms, end_ms, *(after or ())] + ([quality] if quality else [])
                rows += db.execute(f"SELECT * FROM raw_events WHERE {where} ORDER BY received_ms, archive_key, history_id LIMIT ?", (*args, limit + 1)).fetchall()
        rows.sort(key=lambda r: (r["received_ms"], r["archive_key"], r["history_id"]))
        page = rows[:limit]
        cursor = (page[-1]["received_ms"], page[-1]["archive_key"], page[-1]["history_id"]) if len(rows) > limit else None
        return dict(rows=[dict(history_id=r["history_id"], archive_key=r["archive_key"], tag_id=r["tag_id"], received_ms=r["received_ms"],
                               source_ms=r["source_ms"], server_ms=r["server_ms"], ingested_ms=r["ingested_ms"], status=r["status_raw"],
                               quality=r["quality"], kind=r["value_kind"], value=json.loads(r["value_json"]), source_id=r["source_id"],
                               source_seq=r["source_seq"]) for r in page], next=cursor)

    def minutes(self, start_ms, end_ms):
        """Notifications heard per minute over all tags, for the minutes that have any."""
        with self.connect() as db:
            return [(r[0], r[1], r[2]) for r in db.execute(
                "SELECT bucket_ms, SUM(observed_count), SUM(bad_count) FROM metric_bucket WHERE window_ms = ? AND rules_version = ? AND bucket_ms >= ? AND bucket_ms < ? GROUP BY bucket_ms ORDER BY bucket_ms",
                (MINUTE, RULES_VERSION, start_ms // MINUTE * MINUTE, end_ms))]

    def hours(self, start_ms, end_ms):
        """Notifications heard per hour over all tags, for the hours that have any."""
        with self.connect() as db:
            return [dict(t=r[0], events=int(r[1]), not_good=int(r[2])) for r in db.execute(
                "SELECT bucket_ms, SUM(observed_count), SUM(bad_count) FROM metric_bucket WHERE window_ms = ? AND rules_version = ? AND bucket_ms >= ? AND bucket_ms < ? GROUP BY bucket_ms ORDER BY bucket_ms",
                (HOUR, RULES_VERSION, start_ms // HOUR * HOUR, end_ms))]

    def gaps(self, start_ms, end_ms, limit=50):
        """Runs of whole minutes inside the window in which no tag was heard, longest first."""
        heard = [m for m, _, _ in self.minutes(start_ms, end_ms)]
        runs = []
        for a, b in zip(heard, heard[1:]):
            if b - a > MINUTE:
                runs.append(dict(start_ms=a + MINUTE, end_ms=b, minutes=(b - a) // MINUTE - 1))
        runs.sort(key=lambda g: -g["minutes"])
        return dict(total=len(runs), silent_minutes=sum(g["minutes"] for g in runs), rows=sorted(runs[:limit], key=lambda g: g["start_ms"]))

    def days(self, zone="Asia/Shanghai"):
        """Notifications per calendar day of the display time zone."""
        tz = ZoneInfo(zone)
        with self.connect() as db:
            rows = db.execute("SELECT bucket_ms, SUM(observed_count), SUM(bad_count) FROM metric_bucket WHERE window_ms = ? AND rules_version = ? GROUP BY bucket_ms ORDER BY bucket_ms",
                              (HOUR, RULES_VERSION)).fetchall()
        out = {}
        for bucket, seen, bad in rows:
            day = (EPOCH + timedelta(milliseconds=bucket)).astimezone(tz).date().isoformat()
            entry = out.setdefault(day, dict(day=day, events=0, not_good=0, hours=0))
            entry["events"] += seen
            entry["not_good"] += bad
            entry["hours"] += 1
        return list(out.values())

    def sources(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM sources ORDER BY first_ms")]

    def tag_counts(self, start_ms, end_ms):
        """Per tag: notifications and not-good notifications in the minutes of the window."""
        with self.connect() as db:
            return {r[0]: dict(events=int(r[1]), not_good=int(r[2])) for r in db.execute(
                "SELECT tag_id, SUM(observed_count), SUM(bad_count) FROM metric_bucket WHERE window_ms = ? AND rules_version = ? AND bucket_ms >= ? AND bucket_ms < ? GROUP BY tag_id",
                (MINUTE, RULES_VERSION, start_ms // MINUTE * MINUTE, end_ms))}

    def delays(self, sample=5000):
        """Of the newest stored records: how long after reception they were stored on the data server,
        and how far reception lies after the OPC UA source time (a negative value points at a clock)."""
        with self.connect() as db:
            rows = db.execute("SELECT ingested_ms - received_ms, received_ms - source_ms FROM raw_events WHERE archive_key = ? ORDER BY history_id DESC LIMIT ?",
                              (self.archive_key, sample)).fetchall()

        def spread(values):
            values = sorted(v for v in values if v is not None)
            if not values:
                return None
            pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
            return dict(n=len(values), min=values[0], p50=pick(.5), p95=pick(.95), max=values[-1], negative=sum(v < 0 for v in values))
        return dict(store_after_receive_ms=spread(r[0] for r in rows), receive_after_source_ms=spread(r[1] for r in rows))

    def issues(self, limit=50):
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT archive_key, history_id, reason, detail, detected_ms, resolved_ms FROM ingest_issue ORDER BY detected_ms DESC LIMIT ?", (limit,))]

    def count_events(self, tag_ids, start_ms, end_ms):
        with self.connect() as db:
            return sum(db.execute("SELECT COUNT(*) FROM raw_events WHERE tag_id = ? AND received_ms >= ? AND received_ms < ?", (t, start_ms, end_ms)).fetchone()[0]
                       for t in tag_ids)
