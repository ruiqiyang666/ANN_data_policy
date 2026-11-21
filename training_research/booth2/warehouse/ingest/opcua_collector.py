"""Kepware OPC UA -> samples table collector (asyncua subscription client).

STATUS: written and self-tested only against an in-process asyncua server on localhost
(`--self-test`). It has NOT been deployed and has NOT been run against Kepware or any
plant endpoint; the endpoint, namespace, security policy and node ids in the config are
expectations that must be confirmed on site. The PostgreSQL sink has not been run against
a real database here (no server was started); the SQLite sink is the tested path.

Design
  subscription -> in-memory queue -> local spool (SQLite, WAL) -> sink (PostgreSQL or SQLite)
  * Every notification is written to the spool first, so a database outage, a restart of
    the collector or a slow sink never loses data: the forwarder drains the spool in order
    and only deletes rows the sink has acknowledged (store-and-forward).
  * Rows are idempotent on (tag_id, source_timestamp): replays after a crash are harmless.
  * A value that is bad or missing is stored as NULL with its status code. Nothing is
    forward-filled or zero-filled here; that is the feature builder's decision.
  * OPC UA only reports changes. A heartbeat row per interval records that the session
    was alive, which is what lets a reader tell "unchanged" from "not collected".

    python opcua_collector.py --config collector.example.json
    python opcua_collector.py --self-test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from live_schema import LIVE_SCHEMA  # noqa: E402

log = logging.getLogger("booth2.collector")

SPOOL_SCHEMA = """
CREATE TABLE IF NOT EXISTS spool (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,          -- sample / heartbeat / event
  payload TEXT NOT NULL
);
"""


def now_us():
    return time.time_ns() // 1000


def to_us(stamp):
    """datetime (OPC UA timestamps are UTC, usually naive) -> microseconds since 1970, or None."""
    if stamp is None:
        return None
    import datetime as dt

    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return int(stamp.timestamp() * 1_000_000)


def load_tags(path, asset_override=None):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    asset = dict(registry["asset"], **(asset_override or {}))
    tags = []
    for tag_id, tag in enumerate(registry["tags"], start=1):  # same numbering as load_history.tag_table
        kepware = tag.get("kepware_tag", tag["key"])
        tags.append(dict(tag_id=tag_id, key=tag["key"], dtype=tag["dtype"], reserved=bool(tag.get("reserved")),
                         node_id=tag.get("node_id") or registry["node_id_template"].format(
                             ns=asset["opcua_namespace_index"], path=asset["opcua_path"], kepware_tag=kepware)))
    return tags, registry["defaults"]


# ----------------------------------------------------------------------------- spool


class Spool:
    """Durable local buffer. One writer (the event loop); rows leave only after the sink took them."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, isolation_level=None)
        self.con.executescript("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;" + SPOOL_SCHEMA)

    def put(self, rows):
        self.con.execute("BEGIN")
        self.con.executemany("INSERT INTO spool (kind, payload) VALUES (?, ?)",
                             [(kind, json.dumps(payload, ensure_ascii=False)) for kind, payload in rows])
        self.con.execute("COMMIT")

    def take(self, limit):
        rows = self.con.execute("SELECT seq, kind, payload FROM spool ORDER BY seq LIMIT ?", (limit,)).fetchall()
        return [(seq, kind, json.loads(payload)) for seq, kind, payload in rows]

    def done(self, through):
        self.con.execute("DELETE FROM spool WHERE seq <= ?", (through,))

    def size(self):
        return self.con.execute("SELECT COUNT(*) FROM spool").fetchone()[0]

    def close(self):
        self.con.close()


# ----------------------------------------------------------------------------- sinks

SAMPLE_COLUMNS = ["tag_id", "source_time_us", "server_time_us", "received_time_us", "value", "value_text", "status_code", "session_id"]
HEARTBEAT_COLUMNS = ["time_us", "session_id", "connected", "monitored_items", "notifications", "spool_rows"]
EVENT_COLUMNS = ["time_us", "session_id", "kind", "tag_id", "detail"]


class SqliteSink:
    """The zero-install materialisation (live_sample / collector_heartbeat / ingest_event)."""

    def __init__(self, path):
        self.path = str(path)

    def write(self, batch):
        con = sqlite3.connect(self.path, timeout=5)
        try:
            con.executescript(LIVE_SCHEMA)
            for kind, columns, table, conflict in (("sample", SAMPLE_COLUMNS, "live_sample", "OR IGNORE"),
                                                   ("heartbeat", HEARTBEAT_COLUMNS, "collector_heartbeat", "OR IGNORE"),
                                                   ("event", EVENT_COLUMNS, "ingest_event", "")):
                rows = [[p.get(c) for c in columns] for k, p in batch if k == kind]
                if rows:
                    con.executemany(f"INSERT {conflict} INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})", rows)
            con.commit()
        finally:
            con.close()


class PostgresSink:
    """schema/postgres_timescale.sql. Needs `pip install psycopg[binary]`; untested against a real server here."""

    def __init__(self, dsn):
        import psycopg  # noqa: F401  (fail at start-up, not at the first batch)

        self.dsn = dsn
        self.con = None

    def write(self, batch):
        import psycopg

        try:
            if self.con is None or self.con.closed:
                self.con = psycopg.connect(self.dsn, connect_timeout=5)
            micro = "to_timestamp(%s::double precision / 1000000)"
            with self.con.cursor() as cur:
                rows = [p for k, p in batch if k == "sample"]
                if rows:
                    cur.executemany(
                        "INSERT INTO sample (tag_id, source_time, server_time, received_at, value, value_text, status_code, session_id) "
                        f"VALUES (%s, {micro}, {micro}, {micro}, %s, %s, %s, %s) ON CONFLICT (tag_id, source_time) DO NOTHING",
                        [[p[c] for c in SAMPLE_COLUMNS] for p in rows])
                rows = [p for k, p in batch if k == "heartbeat"]
                if rows:
                    cur.executemany(
                        "INSERT INTO collector_heartbeat (time, session_id, connected, monitored_items, notifications, spool_rows) "
                        f"VALUES ({micro}, %s, %s::int::boolean, %s, %s, %s) ON CONFLICT DO NOTHING",
                        [[p[c] for c in HEARTBEAT_COLUMNS] for p in rows])
                rows = [p for k, p in batch if k == "event"]
                if rows:
                    cur.executemany(
                        f"INSERT INTO ingest_event (time, session_id, kind, tag_id, detail) VALUES ({micro}, %s, %s, %s, %s)",
                        [[p[c] for c in EVENT_COLUMNS] for p in rows])
            self.con.commit()
        except Exception:
            if self.con is not None:
                try:
                    self.con.close()
                finally:
                    self.con = None
            raise


def make_sink(config):
    sink = config["sink"]
    if sink["kind"] == "sqlite":
        return SqliteSink(sink["path"])
    if sink["kind"] == "postgres":
        dsn = os.environ.get(sink.get("dsn_env", "BOOTH2_PG_DSN")) or sink.get("dsn")
        if not dsn:
            raise SystemExit("PostgreSQL DSN missing: set the environment variable named by sink.dsn_env")
        return PostgresSink(dsn)
    raise SystemExit(f"Unknown sink kind: {sink['kind']}")


# ----------------------------------------------------------------------------- collector


class Handler:
    """asyncua subscription callbacks. They must not block: they only enqueue."""

    def __init__(self, collector):
        self.collector = collector

    def datachange_notification(self, node, val, data):
        value = data.monitored_item.Value
        tag = self.collector.by_node.get(node.nodeid.to_string())
        if tag is None:
            return
        number, text = None, None
        if isinstance(val, bool):
            number = float(val)
        elif isinstance(val, (int, float)):
            number = float(val) if val == val and abs(val) != float("inf") else None
        elif val is not None:
            text = str(val)
        status = int(value.StatusCode.value) if value.StatusCode is not None else 0
        if status & 0xC0000000:  # Bad / Uncertain: keep the status, never keep the number as a reading
            text = text if text is not None else (None if number is None else repr(number))
            number = None
        received = now_us()
        self.collector.notifications += 1
        self.collector.pending.append(("sample", dict(
            tag_id=tag["tag_id"], source_time_us=to_us(value.SourceTimestamp) or received,
            server_time_us=to_us(value.ServerTimestamp), received_time_us=received, value=number,
            value_text=text, status_code=status, session_id=self.collector.session_id)))

    def status_change_notification(self, status):
        self.collector.event("subscription_status", detail=str(status))


class Collector:
    def __init__(self, config, tags, sink):
        self.config, self.tags, self.sink = config, tags, sink
        self.by_node = {}
        self.pending = []
        self.notifications = 0
        self.session_id = uuid.uuid4().hex
        self.spool = Spool(config["spool_path"])
        self.sink_ok = True
        self.connected = False
        self.monitored = 0
        self.stop = asyncio.Event()

    def event(self, kind, tag_id=None, detail=None):
        log.info("%s %s %s", kind, tag_id or "", detail or "")
        self.pending.append(("event", dict(time_us=now_us(), session_id=self.session_id, kind=kind, tag_id=tag_id, detail=detail)))

    async def flusher(self):
        """Queue -> spool. Cheap, local and synchronous on purpose: this is the durability point."""
        while not self.stop.is_set():
            await asyncio.sleep(self.config.get("flush_seconds", 0.5))
            if self.pending:
                rows, self.pending = self.pending, []
                self.spool.put(rows)

    async def forwarder(self):
        """Spool -> sink, in order. On failure the rows simply stay; retry with back-off."""
        first, longest = self.config.get("retry_min_seconds", 1.0), self.config.get("retry_max_seconds", 60.0)
        delay = first
        while not self.stop.is_set():
            batch = self.spool.take(self.config.get("batch_rows", 2000))
            if not batch:
                await asyncio.sleep(0.5)
                continue
            try:
                await asyncio.to_thread(self.sink.write, [(kind, payload) for _, kind, payload in batch])
            except Exception as exc:  # noqa: BLE001 - any sink failure means "keep the rows"
                if self.sink_ok:
                    self.sink_ok = False
                    self.event("sink_down", detail=f"{type(exc).__name__}: {exc}"[:300])
                await asyncio.sleep(delay)
                delay = min(delay * 2, longest)
                continue
            self.spool.done(batch[-1][0])
            delay = first
            if not self.sink_ok:
                self.sink_ok = True
                self.event("sink_up")

    async def heartbeat(self):
        while not self.stop.is_set():
            await asyncio.sleep(self.config.get("heartbeat_seconds", 10))
            count, self.notifications = self.notifications, 0
            self.pending.append(("heartbeat", dict(time_us=now_us(), session_id=self.session_id, connected=int(self.connected),
                                                    monitored_items=self.monitored, notifications=count, spool_rows=self.spool.size())))

    async def session(self):
        """One OPC UA session; returns when it is lost so the caller can reconnect."""
        from asyncua import Client

        opc = self.config["opcua"]
        client = Client(opc["endpoint"], timeout=opc.get("timeout_seconds", 10))
        client.application_uri = opc.get("application_uri", "urn:booth2:collector")
        if opc.get("security_string"):  # e.g. "Basic256Sha256,SignAndEncrypt,cert.der,key.pem"
            await client.set_security_string(opc["security_string"])
        if opc.get("username"):
            client.set_user(opc["username"])
            client.set_password(os.environ.get(opc.get("password_env", "BOOTH2_OPCUA_PASSWORD"), ""))
        async with client:
            self.session_id = uuid.uuid4().hex
            wanted = [t for t in self.tags if not (t["reserved"] and not opc.get("subscribe_reserved", False))]
            nodes = [client.get_node(t["node_id"]) for t in wanted]
            self.by_node = {node.nodeid.to_string(): tag for node, tag in zip(nodes, wanted)}
            subscription = await client.create_subscription(opc.get("publishing_interval_ms", 1000), Handler(self))
            results = await subscription.subscribe_data_change(
                nodes, queuesize=opc.get("queue_size", 10), sampling_interval=opc.get("sampling_interval_ms", 1000))
            results = results if isinstance(results, list) else [results]
            self.monitored = 0
            for tag, result in zip(wanted, results):
                if isinstance(result, int):
                    self.monitored += 1
                else:
                    self.event("subscribe_failed", tag_id=tag["tag_id"], detail=f"{tag['node_id']}: {result}")
            self.connected = True
            self.event("session_start", detail=f"{opc['endpoint']} monitored={self.monitored}/{len(wanted)}")
            try:
                while not self.stop.is_set():
                    await asyncio.sleep(opc.get("check_seconds", 5))
                    await client.check_connection()
            finally:
                self.connected = False

    async def run(self):
        tasks = [asyncio.create_task(c) for c in (self.flusher(), self.forwarder(), self.heartbeat())]
        delay = 1.0
        try:
            while not self.stop.is_set():
                try:
                    await self.session()
                    delay = 1.0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - lost session: record and reconnect
                    self.event("session_lost", detail=f"{type(exc).__name__}: {exc}"[:300])
                    try:
                        await asyncio.wait_for(self.stop.wait(), delay)
                    except asyncio.TimeoutError:
                        pass
                    delay = min(delay * 2, 60.0)
        finally:
            self.stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.pending:
                self.spool.put(self.pending)
                self.pending = []
            self.spool.close()


# ----------------------------------------------------------------------------- self-test


async def self_test():
    """In-process asyncua server on localhost, three tags, a simulated sink outage. No plant, no Kepware."""
    import tempfile

    from asyncua import Server, ua

    work = Path(tempfile.mkdtemp(prefix="booth2-collector-"))
    server = Server()
    await server.init()
    port = 48417
    server.set_endpoint(f"opc.tcp://127.0.0.1:{port}/booth2-selftest/")
    index = await server.register_namespace("urn:booth2:selftest")
    folder = await server.nodes.objects.add_object(index, "Plasma2")
    names = ["Voltage", "Current", "Press_Hopper_1", "Gun_ID.DATA"]
    variables = {}
    for name in names:
        initial = "AF4-TEST" if name == "Gun_ID.DATA" else 0.0
        variables[name] = await folder.add_variable(ua.NodeId(f"HA.Booth2.Plasma2.{name}", index), name, initial)
    tags = [dict(tag_id=i + 1, key=name, dtype="float", reserved=False, node_id=f"ns={index};s=HA.Booth2.Plasma2.{name}")
            for i, name in enumerate(names)]
    tags.append(dict(tag_id=99, key="Missing_Tag", dtype="float", reserved=False, node_id=f"ns={index};s=HA.Booth2.Plasma2.Missing_Tag"))
    target = work / "sink.sqlite"

    class FlakySink(SqliteSink):
        down = False

        def write(self, batch):
            if self.down:
                raise ConnectionError("simulated database outage")
            super().write(batch)

    sink = FlakySink(target)
    config = dict(spool_path=str(work / "spool.sqlite"), flush_seconds=0.1, heartbeat_seconds=0.5, batch_rows=500,
                  retry_min_seconds=0.2, retry_max_seconds=0.4,
                  opcua=dict(endpoint=f"opc.tcp://127.0.0.1:{port}/booth2-selftest/", publishing_interval_ms=100,
                             sampling_interval_ms=50, queue_size=50, check_seconds=0.5))
    collector = Collector(config, tags, sink)
    async with server:
        runner = asyncio.create_task(collector.run())
        await asyncio.sleep(1.5)
        for step in range(20):  # sink up
            await variables["Voltage"].write_value(55.0 - step * 0.1)
            await variables["Current"].write_value(549.0)
            await asyncio.sleep(0.12)
        sink.down = True
        await asyncio.sleep(0.3)
        for step in range(20):  # sink down: rows must wait in the spool
            await variables["Voltage"].write_value(52.0 - step * 0.05)
            await asyncio.sleep(0.12)
        # A bad-quality reading: the number must not be stored as a value.
        await variables["Press_Hopper_1"].write_value(ua.DataValue(ua.Variant(123.0, ua.VariantType.Double), ua.StatusCode(ua.StatusCodes.BadSensorFailure)))
        await asyncio.sleep(0.6)
        waiting = collector.spool.size()
        con = sqlite3.connect(target)
        before = con.execute("SELECT COUNT(*) FROM live_sample").fetchone()[0]
        con.close()
        sink.down = False
        await asyncio.sleep(3.5)
        collector.stop.set()
        await asyncio.wait_for(runner, 15)
    con = sqlite3.connect(target)
    total, distinct = con.execute("SELECT COUNT(*), COUNT(DISTINCT tag_id || ':' || source_time_us) FROM live_sample").fetchone()
    voltage = [r[0] for r in con.execute("SELECT value FROM live_sample WHERE tag_id = 1 ORDER BY source_time_us")]
    bad = con.execute("SELECT value, value_text, status_code FROM live_sample WHERE status_code != 0").fetchall()
    text = con.execute("SELECT value, value_text FROM live_sample WHERE tag_id = 4").fetchall()
    kinds = dict(con.execute("SELECT kind, COUNT(*) FROM ingest_event GROUP BY 1").fetchall())
    beats = con.execute("SELECT COUNT(*), MAX(spool_rows) FROM collector_heartbeat").fetchone()
    con.close()
    spool = sqlite3.connect(work / "spool.sqlite")
    # Heartbeats queued during shutdown legitimately stay for the next start; samples must all be delivered.
    left = spool.execute("SELECT COUNT(*) FROM spool WHERE kind = 'sample'").fetchone()[0]
    spool.close()
    checks = {
        "rows reached the sink while it was up": before > 0,
        "rows waited in the spool during the outage": waiting > 0,
        "every sample delivered after recovery (none left in the spool)": left == 0,
        "no duplicate (tag_id, source_time)": total == distinct,
        "all 40 written voltage values present, in order": [round(v, 2) for v in voltage][-40:] == [round(55.0 - s * 0.1, 2) for s in range(20)] + [round(52.0 - s * 0.05, 2) for s in range(20)],
        "bad-status reading stored without a value": len(bad) >= 1 and all(r[0] is None and r[2] & 0x80000000 for r in bad),
        "string tag stored as text": len(text) >= 1 and text[0] == (None, "AF4-TEST"),
        "unknown node reported, not fatal": kinds.get("subscribe_failed", 0) == 1,
        "outage and recovery logged": kinds.get("sink_down", 0) >= 1 and kinds.get("sink_up", 0) >= 1,
        "heartbeats written": beats[0] >= 3,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"samples={total} waiting_during_outage={waiting} events={kinds} heartbeats={beats[0]} work={work}")
    return 0 if all(checks.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.self_test:
        logging.getLogger("asyncua").setLevel(logging.ERROR)
        return asyncio.run(self_test())
    if args.config is None:
        parser.error("--config or --self-test is required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tags, _ = load_tags(args.config.parent / config.get("tags_file", "tags.json"), config.get("asset"))
    collector = Collector(config, tags, make_sink(config))
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
