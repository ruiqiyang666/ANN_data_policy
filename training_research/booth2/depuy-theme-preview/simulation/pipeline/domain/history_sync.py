"""Read-only client of the plant history API and the incremental sync into the local index.

Only GET requests are made, to the three documented endpoints. The access key is read from the
environment (or a file named there), sent as X-API-Key, and never logged or returned.
A sync round reads `/api/stats`, then pages `/api/history` by ascending history id from the
committed watermark up to the id that was the newest when the round began, storing each page in
one transaction (domain/history_index.py). Rounds are idempotent: a crash or a repeated page
leaves the same rows and the same watermark.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from domain.history_index import HistoryIndex, now_ms

LOG = logging.getLogger(__name__)
MAX_BODY = 48 * 1024 * 1024
MAX_PAGE = 5000  # the API's own upper limit
PAGES_PER_ROUND = 200


class Unreachable(Exception):
    """The API did not answer in time or answered with a server error; worth retrying."""


class Refused(Exception):
    """401 / 403: the key or the allow-list is wrong; retrying quickly cannot help."""


class Unexpected(Exception):
    """The API answered with something the sync cannot use safely."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class HistoryClient:
    def __init__(self, base_url, api_key, timeout=15):
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment or parts.username:
            raise ValueError("The history API address must be a plain http(s) origin")
        self.base = base_url.rstrip("/")
        self.key, self.timeout = api_key, timeout
        # No proxy and no redirects: the request goes to the configured host or nowhere.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def get(self, path, **params):
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        request = urllib.request.Request(f"{self.base}{path}{'?' + query if query else ''}", method="GET",
                                         headers={"X-API-Key": self.key, "Accept": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read(MAX_BODY + 1)
        except urllib.error.HTTPError as cause:
            if cause.code in (401, 403):
                raise Refused(f"HTTP {cause.code}") from None
            if cause.code >= 500 or cause.code in (408, 429):
                raise Unreachable(f"HTTP {cause.code}") from None
            raise Unexpected(f"HTTP {cause.code} for {path}") from None
        except (urllib.error.URLError, OSError, TimeoutError) as cause:
            raise Unreachable(type(cause).__name__) from None
        if len(body) > MAX_BODY:
            raise Unexpected("response larger than the limit")
        try:
            document = json.loads(body)
        except ValueError:
            raise Unexpected(f"{path} did not return JSON") from None
        if not isinstance(document, dict):
            raise Unexpected(f"{path} did not return an object")
        return document

    def stats(self):
        document = self.get("/api/stats")
        newest = document.get("latest_event_id")
        if isinstance(newest, bool) or not isinstance(newest, int) or newest < 0:
            raise Unexpected("/api/stats has no latest_event_id")
        return document

    def history(self, after_id, limit):
        document = self.get("/api/history", after_id=after_id, limit=limit)
        events = document.get("events")
        if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
            raise Unexpected("/api/history has no events list")
        if document.get("order", "id_ascending") != "id_ascending":
            raise Unexpected("/api/history is not ordered by ascending id")
        return events, bool(document.get("has_more"))


class HistorySync:
    """One background thread that keeps the local index up to the plant history."""

    def __init__(self, index: HistoryIndex, client: HistoryClient, interval=5.0, page=2000):
        self.index, self.client = index, client
        self.interval, self.page = max(1.0, float(interval)), max(1, min(MAX_PAGE, int(page)))
        self.state = dict(status="starting", detail="", since_ms=now_ms(), source_max_id=None, last_round_ms=None,
                          last_success_ms=None, source=None)
        self.recorded = None
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.run, name="phase2-history-sync", daemon=True)

    def start(self):
        self.worker.start()

    def close(self):
        self.stop.set()
        self.worker.join(timeout=5)

    def set(self, status, detail="", **more):
        if status != self.state["status"]:
            self.state = dict(self.state, since_ms=now_ms())
        self.state = dict(self.state, status=status, detail=detail, last_round_ms=now_ms(), **more)

    def round(self):
        """One sync round. Returns (outcome, detail): caught_up, behind, conflict or regressed."""
        started, spent = now_ms(), 0
        totals = dict(pages=0, fetched=0, inserted=0, idempotent=0, aliased=0, quarantined=0)
        cursor = self.index.cursor()
        outcome, detail, upper = "failed", "", None
        try:
            t = time.monotonic()
            stats = self.client.stats()
            spent += int((time.monotonic() - t) * 1000)
            upper = stats["latest_event_id"]
            self.state = dict(self.state, source_max_id=upper, source=dict(
                total_samples=stats.get("total_samples"), tag_count=stats.get("tag_count"), last_received_at=stats.get("last_received_at")))
            outcome = "caught_up"
            if upper > cursor:
                self.set("behind")  # shown while a backlog is being worked off
            if upper < cursor:
                # The history database was rebuilt: its ids start again. Mixing generations would corrupt the index.
                outcome, detail = "regressed", f"source newest id {upper} is below the committed id {cursor}"
            while outcome == "caught_up" and cursor < upper and not self.stop.is_set():
                if totals["pages"] >= PAGES_PER_ROUND:
                    outcome = "behind"  # a long backlog is worked off over several rounds
                    break
                t = time.monotonic()
                events, more = self.client.history(cursor, self.page)
                spent += int((time.monotonic() - t) * 1000)
                ids = [e.get("id") for e in events]
                if any(isinstance(i, bool) or not isinstance(i, int) for i in ids) or any(b <= a for a, b in zip([cursor] + ids, ids)):
                    raise Unexpected("history ids did not advance past the cursor")
                kept = [e for e in events if e["id"] <= upper]  # newer records wait for the next round
                totals["pages"] += 1
                totals["fetched"] += len(kept)
                if kept:
                    result = self.index.ingest_page(kept, source_max_id=upper)
                    for key in ("inserted", "idempotent", "aliased", "quarantined"):
                        totals[key] += result[key]
                    cursor = result["last_id"]
                    if result["conflict"]:
                        outcome, detail = "conflict", f"{result['conflict']['reason']} at history id {result['conflict']['history_id']}"
                        break
                # Ids may have holes: the source saying "nothing more" ends the round, not reaching a particular id.
                if not events or len(kept) < len(events) or not more:
                    break
        except Refused as cause:
            outcome, detail = "unauthorized", str(cause)
            raise
        except Unreachable as cause:
            outcome, detail = "unreachable", str(cause)
            raise
        except Unexpected as cause:
            outcome, detail = "error", str(cause)
            raise
        finally:
            # Kept short: a round is recorded when it stored something or when the outcome changed.
            if totals["fetched"] or outcome != self.recorded:
                self.recorded = outcome
                self.index.record_round(started_ms=started, finished_ms=now_ms(), outcome=outcome, detail=detail, request_ms=spent,
                                        committed_history_id=cursor, source_max_id=upper, **totals)
        return outcome, detail

    def run(self):
        delay = self.interval
        while not self.stop.is_set():
            try:
                outcome, detail = self.round()
                self.set(outcome, detail, **({"last_success_ms": now_ms()} if outcome in ("caught_up", "behind") else {}))
                # A conflict or a rebuilt source needs a person; keep looking slowly so a fix is picked up.
                delay = 0.2 if outcome == "behind" else self.interval if outcome == "caught_up" else 60.0
            except Refused as cause:
                self.set("unauthorized", str(cause))
                delay = 60.0
            except Unreachable as cause:
                self.set("unreachable", str(cause))
                delay = min(60.0, max(self.interval, delay * 2))
            except Unexpected as cause:
                self.set("error", str(cause))
                delay = 60.0
            except Exception:  # the sync must never take the dashboard down
                LOG.exception("History sync round failed")
                self.set("error", "internal error, see the server log")
                delay = 60.0
            self.stop.wait(delay)


def read_key(env=os.environ):
    """The read-only key: from the file named by PHASE2_HISTORY_API_KEY_FILE, else PHASE2_HISTORY_API_KEY."""
    path = env.get("PHASE2_HISTORY_API_KEY_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return (env.get("PHASE2_HISTORY_API_KEY") or "").strip()


def from_environment(index_root, env=os.environ):
    """(index, sync) when the plant history API is configured, else None. Raises on a half-made configuration."""
    url = (env.get("PHASE2_HISTORY_API_URL") or "").strip()
    if not url:
        return None
    key = read_key(env)
    if not key:
        raise RuntimeError("PHASE2_HISTORY_API_URL is set but no read-only key was provided")
    index = HistoryIndex(Path(index_root) / "history" / "plasma2.sqlite3",
                         archive_key=env.get("PHASE2_HISTORY_ARCHIVE_KEY", "plasma2-a"),
                         source_epoch=env.get("PHASE2_HISTORY_SOURCE_EPOCH", "1"))
    sync = HistorySync(index, HistoryClient(url, key), interval=float(env.get("PHASE2_HISTORY_SYNC_SECONDS", "5")),
                       page=int(env.get("PHASE2_HISTORY_PAGE", "2000")))
    return index, sync
