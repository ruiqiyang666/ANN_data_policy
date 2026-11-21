"""A stand-in for the plant history API (the Windows data server, port 8780), for tests and preview.

It reproduces the behaviour that was verified on site on 2026-11-19: key check, `/api/stats`,
`/api/tags`, `/api/latest` and `/api/history` with ascending-id pagination across tags
(`after_id`, `limit` up to 5000, `next_after_id`, `has_more`), exact `tag` match on short name,
path or node id, and inclusive `start` / `end` on `received_at`. It is not the plant: nothing
here proves how the real server behaves under load or after a rebuild.

Events come either from `patterned_events` (plain synthetic values for tests) or from
`warehouse_events`, which replays recorded Booth2 seconds of the offline warehouse as
change-driven notifications. Five site tags have no recorded counterpart (FeedRate, Gun_Hours,
Gun_ID.DATA, Gun_ID.LEN, Ingitions); the replay derives stand-ins for them from the recorded
readings by the rules written at `EXTRA`.

usage (preview):
  python checks/history_mock.py --warehouse <booth2.sqlite> --start "2026-10-28 02:00:00" --hours 8 \
      --port 18793 --key-file <file with the test key> [--live]
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PREFIX = "HA.Booth2.Plasma2."
GOOD, BAD = "StatusCode(value=0)", "StatusCode(value=2147483648)"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
# Dashboard field key -> short name on site, where they differ (the table in api/field_live.py, inverted).
SITE = {"AirJet": "AirJe", "Cooling_water_inlet_temp.": "Cooling_water_inlet_temp", "Cooling_water_outlet_temp.": "Cooling_water_outlet_temp",
        "Press._Hopper_1": "Press_Hopper_1", "Press._Hopper_2": "Press_Hopper_2"}
EXTRA = ("FeedRate",      # Feedrate_1 + Feedrate_2 of the same second
         "Gun_Hours",     # hours with Current > 100 A since the start of the replay
         "Ingitions",     # times Current rose above 100 A since the start of the replay (the site spells it this way)
         "Gun_ID.DATA",   # a constant string
         "Gun_ID.LEN")    # its length, an integer


def iso(ms):
    return (EPOCH + timedelta(milliseconds=int(ms))).isoformat(timespec="milliseconds")


def parse(text):
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("offset required")
    return (stamp - EPOCH) // timedelta(milliseconds=1)


def event(seq, name, received_ms, value, status=GOOD, source_id="linux-collector"):
    """One record in the shape of the plant API, without the Windows id (assigned on append)."""
    return dict(source_id=source_id, source_seq=seq, node_id=f"ns=2;s={PREFIX}{name}", path=PREFIX + name, name=name,
                received_at=iso(received_ms), source_timestamp=iso(received_ms - 150), server_timestamp=iso(received_ms - 140),
                status=status, value=value)


def patterned_events(tags=("Voltage", "Current", "Press_Hopper_1", "Gun_ID.DATA", "Gun_ID.LEN"), start_ms=1_795_000_000_000, seconds=600,
                     first_seq=1):
    """Plain synthetic notifications: every value is a simple function of its second. For tests only."""
    out, seq = [], first_seq - 1
    for second in range(seconds):
        for k, name in enumerate(tags):
            if name == "Gun_ID.DATA":
                if second:
                    continue
                value = "TEST-GUN-01"
            elif name == "Gun_ID.LEN":
                if second:
                    continue
                value = 11
            elif second % (k + 1):
                continue  # tag k reports every (k + 1) seconds
            else:
                value = 0.0 if (name == "Press_Hopper_1" and second % 60 < 6) else round(50 + k + (second % 17) * 0.1, 3)
            seq += 1
            out.append(event(seq, name, start_ms + second * 1000 + 37 * k, value))
    return out


def warehouse_events(db_path, start, hours, anchor_ms):
    """Recorded Booth2 seconds as change-driven notifications, re-timed so the first second lands on `anchor_ms`.

    A tag reports when its value differs from its previous reported value (and once at the start), as an
    OPC UA subscription without a deadband would. A second missing in the recording produces nothing.
    """
    first = int((datetime.fromisoformat(start).replace(tzinfo=timezone.utc) - EPOCH).total_seconds())
    with sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True) as db:
        columns = [r[1] for r in db.execute("PRAGMA table_info(sample_1hz)")][1:29]
        quoted = ", ".join(f'"{c}"' for c in columns)
        rows = db.execute(f"SELECT ts, {quoted} FROM sample_1hz WHERE ts >= ? AND ts < ? ORDER BY ts", (first, first + int(hours * 3600))).fetchall()
    names = [SITE.get(c, c) for c in columns]
    at = {c: i for i, c in enumerate(columns)}
    last, out, seq = {}, [], 0
    powered, seconds_on, ignitions = False, 0, 0

    def report(name, ms, value):
        nonlocal seq
        if value is None or last.get(name) == value:
            return
        last[name] = value
        seq += 1
        out.append(event(seq, name, ms, value))

    for row in rows:
        ms = anchor_ms + (row[0] - first) * 1000
        values = row[1:]
        for k, name in enumerate(names):
            report(name, ms + 3 * k, values[k])
        current, f1, f2 = values[at["Current"]], values[at["Feedrate_1"]], values[at["Feedrate_2"]]
        on = current is not None and current > 100
        ignitions += on and not powered
        seconds_on += on
        powered = on
        if f1 is not None and f2 is not None:
            report("FeedRate", ms + 90, round(f1 + f2, 3))
        report("Gun_Hours", ms + 93, round(seconds_on / 3600, 2))
        report("Ingitions", ms + 96, float(ignitions))
        report("Gun_ID.DATA", ms + 99, "MOCK-GUN-0001")
        report("Gun_ID.LEN", ms + 102, 13)
    return out


class MockHistory:
    """The stored records and the switches that make the stand-in misbehave on purpose."""

    def __init__(self, key, events=(), now=None, first_id=9, hole_every=97):
        self.key = key
        self.now = now  # None: everything is visible; a callable: records with ingested time <= now() are visible
        self.lock = threading.Lock()
        self.rows, self.ids, self.ingested = [], [], []
        self.next_id, self.hole_every = first_id, hole_every
        self.faults = dict(unauthorized=False, forbidden=False, server_error=False, delay_seconds=0.0, stats_latest=None)
        self.requests = []
        self.append(events)

    def append(self, events, ingest_delay_ms=800, ingested_ms=None):
        """Store records as the data server would: ids ascend in arrival order, with occasional holes."""
        added = []
        with self.lock:
            for e in events:
                if self.hole_every and self.next_id % self.hole_every == 0:
                    self.next_id += 1
                try:
                    received = parse(e["received_at"])
                except (KeyError, TypeError, ValueError, AttributeError):
                    received = int(time.time() * 1000)  # an unreadable record is still stored by the server
                at = ingested_ms if ingested_ms is not None else received + ingest_delay_ms
                stored = dict(id=self.next_id, **e, ingested_at=iso(at))
                self.rows.append(stored)
                self.ids.append(stored["id"])
                self.ingested.append(at)
                added.append(stored)
                self.next_id += 1
        return added

    def rebuild(self, events):
        """A rebuilt database: the ids start again."""
        with self.lock:
            self.rows, self.ids, self.ingested, self.next_id = [], [], [], 1
        self.append(events)

    def visible(self):
        """The stored records, or in live mode those the server has stored by now (arrival order = time order there)."""
        if self.now is None:
            return self.rows
        return self.rows[:bisect.bisect_right(self.ingested, self.now())]

    # ---------- the API ----------
    def stats(self):
        rows = self.visible()
        sources = {}
        for r in rows:
            sources[r["source_id"]] = dict(source_id=r["source_id"], last_seq=r["source_seq"], last_ingested_at=r["ingested_at"])
        newest = max((r["id"] for r in rows), default=0)
        return dict(total_samples=len(rows), latest_event_id=newest if self.faults["stats_latest"] is None else self.faults["stats_latest"],
                    last_received_at=max((r["received_at"] for r in rows if isinstance(r.get("received_at"), str)), default=None),
                    tag_count=len({r["path"] for r in rows}), sources=list(sources.values()), generated_at=iso(time.time() * 1000))

    def tags(self):
        out = {}
        for r in self.visible():
            entry = out.setdefault(r["path"], dict(name=r["name"], path=r["path"], node_id=r["node_id"], sample_count=0, latest_received_at=None))
            entry["sample_count"] += 1
            entry["latest_received_at"] = r["received_at"]
        return dict(tags=list(out.values()), count=len(out), generated_at=iso(time.time() * 1000))

    def latest(self):
        out = {}
        for r in self.visible():
            out[r["path"]] = r
        return dict(tags=list(out.values()), count=len(out), generated_at=iso(time.time() * 1000))

    def history(self, after_id=0, limit=200, tag=None, start=None, end=None):
        limit = max(1, min(5000, int(limit)))
        lo = parse(start) if start else None
        hi = parse(end) if end else None
        picked = []
        rows = self.visible()
        for r in rows[bisect.bisect_right(self.ids, after_id, 0, len(rows)):]:
            if tag and tag not in (r["name"], r["path"], r["node_id"]):
                continue
            if lo is not None or hi is not None:
                at = parse(r["received_at"])
                if (lo is not None and at < lo) or (hi is not None and at > hi):
                    continue
            picked.append(r)
            if len(picked) > limit:
                break
        page = picked[:limit]
        return dict(events=page, count=len(page), next_after_id=page[-1]["id"] if page else after_id, has_more=len(picked) > limit,
                    order="id_ascending")


def handler_for(mock: MockHistory):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def reply(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except ConnectionError:
                pass  # the caller gave up waiting

        def do_GET(self):
            url = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            mock.requests.append((self.command, url.path, dict(query)))
            if url.path == "/health":
                return self.reply(200, dict(status="ok"))
            if mock.faults["delay_seconds"]:
                time.sleep(mock.faults["delay_seconds"])
            if mock.faults["server_error"]:
                return self.reply(500, dict(detail="mock server error"))
            given = self.headers.get("X-API-Key") or (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
            if mock.faults["unauthorized"] or given != mock.key:
                return self.reply(401, dict(detail="invalid token"))
            if mock.faults["forbidden"]:
                return self.reply(403, dict(detail="source address not allowed"))
            try:
                if url.path == "/api/stats":
                    return self.reply(200, mock.stats())
                if url.path == "/api/tags":
                    return self.reply(200, mock.tags())
                if url.path == "/api/latest":
                    return self.reply(200, mock.latest())
                if url.path == "/api/history":
                    return self.reply(200, mock.history(int(query.get("after_id", 0)), int(query.get("limit", 200)), query.get("tag"),
                                                        query.get("start"), query.get("end")))
            except ValueError as cause:
                return self.reply(422, dict(detail=str(cause)))
            return self.reply(404, dict(detail="not found"))

        def do_POST(self):  # the stand-in is read-only; a write must be visible in a test
            mock.requests.append((self.command, urlsplit(self.path).path, {}))
            self.reply(405, dict(detail="read-only"))

        do_PUT = do_DELETE = do_PATCH = do_POST
    return Handler


class MockServer:
    """The stand-in on a local port, in a background thread."""

    def __init__(self, mock, port=0):
        self.mock = mock
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_for(mock))
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Stand-in for the plant history API (preview and tests only)")
    parser.add_argument("--warehouse", type=Path, required=True, help="booth2.sqlite of the offline warehouse (read-only)")
    parser.add_argument("--start", default="2026-10-28 02:00:00", help="first recorded second to replay (instrument clock)")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--port", type=int, default=18793)
    parser.add_argument("--key-file", type=Path, required=True, help="file holding the key this stand-in accepts")
    parser.add_argument("--live", action="store_true", help="re-time the replay to end 20 minutes from now and release it in real time")
    args = parser.parse_args()
    now_ms = int(time.time() * 1000)
    span = int(args.hours * 3_600_000)
    anchor = now_ms - span + 20 * 60_000 if args.live else now_ms - span
    events = warehouse_events(args.warehouse, args.start, args.hours, anchor)
    mock = MockHistory(args.key_file.read_text(encoding="utf-8").strip(), events, now=(lambda: int(time.time() * 1000)) if args.live else None)
    server = MockServer(mock, args.port)
    print(f"mock history API on {server.url}: {len(events)} notifications of {len({e['path'] for e in events})} tags", flush=True)
    with server:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
