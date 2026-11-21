"""Headless history collection, consistent snapshots and gated candidate jobs.

Standard library only. This worker never publishes a model or modifies the dashboard.
See docs/continuous-learning.md for the dataset and recipe contracts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "simulation/pipeline"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def exclusive(path):
    """OS lock releases after a crash; PID files alone cannot provide this guarantee."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def snapshot(database, destination):
    """SQLite backup includes committed WAL transactions; metadata is read from the copy."""
    database, destination = Path(database).resolve(), Path(destination).resolve()
    if not database.is_file():
        raise ValueError("history database does not exist")
    destination.mkdir(parents=True, exist_ok=True)
    with exclusive(destination / ".snapshot.lock"):
        with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=destination) as temporary:
            stage = Path(temporary)
            copy = stage / "history.sqlite3"
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(copy)) as target:
                    source.backup(target, pages=2048)
                    target.execute("PRAGMA journal_mode=DELETE")
                    if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("snapshot integrity check failed")
                    events, first, last = target.execute("SELECT COUNT(*), MIN(received_ms), MAX(received_ms) FROM raw_events").fetchone()
                    cursors = target.execute("SELECT archive_key, committed_history_id FROM ingest_cursor ORDER BY archive_key").fetchall()
                    issues = target.execute("SELECT COUNT(*) FROM ingest_issue WHERE resolved_ms IS NULL").fetchone()[0]
                    schema = target.execute("PRAGMA user_version").fetchone()[0]
            digest = sha256(copy)
            identity = "history-" + digest
            manifest = dict(schema_version=1, snapshot_id=identity, created_at=utc(), database_sha256=digest,
                            events=events, first_received_ms=first, last_received_ms=last,
                            cursors=dict(cursors), unresolved_issues=issues, history_schema_version=schema,
                            purpose="raw_history_snapshot", training_ready=False)
            final = destination / identity
            if final.exists():
                if sha256(final / "history.sqlite3") != digest:
                    raise ValueError("existing snapshot was modified")
                return read_json(final / "manifest.json")
            atomic_json(stage / "manifest.json", manifest)
            os.chmod(copy, 0o444)
            os.chmod(stage / "manifest.json", 0o444)
            # Rename both files together; TemporaryDirectory then removes the empty staging directory.
            ready = stage / "ready"
            ready.mkdir()
            os.replace(copy, ready / copy.name)
            os.replace(stage / "manifest.json", ready / "manifest.json")
            os.replace(ready, final)
            return manifest


def assess(manifest_path, previous_cycles=(), min_cycles=3, min_events=2):
    """Fail closed until labels, feature equivalence and provenance are provided."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        return dict(status="awaiting_feature_build", reason="No versioned training dataset", eligible=False)
    m = read_json(path)
    if m.get("schema_version") != 1:
        raise ValueError("unsupported dataset manifest")
    if m.get("status") != "ready":
        return dict(status=m.get("status", "awaiting_labels"), eligible=False)
    required = ("dataset_id", "snapshot_id", "feature_version", "label_version", "created_at", "cycles", "artifacts")
    if any(not m.get(k) for k in required):
        raise ValueError("dataset provenance is incomplete")
    created = datetime.fromisoformat(m["created_at"])
    if created.tzinfo is None:
        raise ValueError("dataset time requires a timezone")
    if m.get("feature_equivalence_review", {}).get("passed") is not True:
        return dict(status="awaiting_feature_validation", eligible=False)
    if m.get("quality_review", {}).get("passed") is not True:
        return dict(status="awaiting_quality_review", eligible=False)
    for review in ("feature_equivalence_review", "quality_review"):
        if not m[review].get("reviewer") or m[review].get("artifact") not in m["artifacts"]:
            raise ValueError("review needs a reviewer and hashed evidence artifact")
    for name, digest in m["artifacts"].items():
        artifact = (path.parent / name).resolve()
        if not artifact.is_relative_to(path.parent) or not artifact.is_file() or sha256(artifact) != digest:
            raise ValueError("missing, changed or out-of-directory dataset artifact")
    ids, events = set(), set()
    for c in m["cycles"]:
        if c.get("cycle_id") in ids or not c.get("cycle_id"):
            raise ValueError("cycle IDs must be nonempty and unique")
        if c.get("label_status") != "confirmed" or not c.get("reviewer"):
            return dict(status="awaiting_labels", eligible=False)
        confirmed = datetime.fromisoformat(c["confirmed_at"])
        if confirmed.tzinfo is None or confirmed > created:
            raise ValueError("labels must have been available when the dataset was created")
        if c.get("endpoint") not in ("eol", "right_censored"):
            raise ValueError("ambiguous failure cause must be resolved before admission")
        if c.get("closed") is not True:
            return dict(status="awaiting_mature_cycles", eligible=False)
        ids.add(c["cycle_id"])
        if c["endpoint"] == "eol":
            events.add(c["cycle_id"])
    new = ids - set(previous_cycles)
    ready = len(new) >= min_cycles and len(new & events) >= min_events
    return dict(status="ready_for_candidate" if ready else "awaiting_mature_cycles", eligible=ready,
                dataset_id=m["dataset_id"], new_cycles=sorted(new), new_eol_cycles=sorted(new & events),
                cycle_ids=sorted(ids), manifest_sha256=sha256(path))


def train_once(manifest_path, recipe_path, root):
    """Run a locally reviewed argv recipe once per exact dataset + recipe. Never shell=True."""
    root = Path(root).resolve()
    manifest_path, recipe_path = Path(manifest_path).resolve(), Path(recipe_path).resolve()
    with exclusive(root / ".training.lock"):
        admitted_path = root / "last_trained.json"
        previous = read_json(admitted_path)["cycle_ids"] if admitted_path.exists() else []
        gate = assess(manifest_path, previous)
        atomic_json(root / "gate.json", dict(gate, checked_at=utc()))
        if not gate["eligible"]:
            return gate
        recipe = read_json(recipe_path)
        if recipe.get("schema_version") != 1 or recipe.get("purpose") != "candidate_training":
            raise ValueError("only candidate_training recipes are supported")
        command = recipe.get("argv")
        if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
            raise ValueError("recipe argv must be a nonempty string list")
        identity = hashlib.sha256((gate["manifest_sha256"] + sha256(recipe_path)).encode()).hexdigest()
        run = root / "jobs" / identity
        if (run / "job.json").exists():
            return read_json(run / "job.json")  # failed/interrupted work needs an explicit new recipe revision
        run.mkdir(parents=True)
        output = run / "output"
        output.mkdir()
        argv = [a.replace("{dataset_dir}", str(manifest_path.parent)).replace("{output_dir}", str(output)) for a in command]
        job = dict(job_id=identity, status="running", started_at=utc(), dataset=gate,
                   recipe_sha256=sha256(recipe_path), argv=argv, automatic_promotion=False)
        atomic_json(run / "dataset.json", read_json(manifest_path))
        atomic_json(run / "recipe.json", recipe)
        atomic_json(run / "job.json", job)
        try:
            with (run / "train.log").open("wb") as log:
                result = subprocess.run(argv, cwd=manifest_path.parent, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=int(recipe.get("timeout_seconds", 21600)), check=False)
            job["exit_code"] = result.returncode
            job["status"] = "awaiting_evaluation_review" if result.returncode == 0 else "failed"
        except (OSError, subprocess.TimeoutExpired) as error:
            job.update(status="failed", error=type(error).__name__)
        # Freeze the output inventory, including evaluation failures. Successful process != model approval.
        job["artifacts"] = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
        job["finished_at"] = utc()
        if sha256(manifest_path) != gate["manifest_sha256"]:
            job.update(status="failed", error="dataset_manifest_changed_during_training")
        else:
            # Rehash actual inputs too; a stable manifest does not imply stable data files.
            try:
                assess(manifest_path, ())
            except ValueError:
                job.update(status="failed", error="dataset_artifact_changed_during_training")
        atomic_json(run / "job.json", job)
        if job["status"] == "awaiting_evaluation_review":
            atomic_json(admitted_path, dict(cycle_ids=gate["cycle_ids"], job_id=identity, finished_at=job["finished_at"]))
        return job


def collect(index_root, once=False):
    from domain.history_sync import from_environment
    with exclusive(Path(index_root) / ".collector.lock"):
        configured = from_environment(index_root)
        if configured is None:
            raise ValueError("PHASE2_HISTORY_API_URL is not configured; no collection was started")
        index, sync = configured
        if once:
            outcome, detail = sync.round()
            state = dict(status=outcome, detail=detail, committed_id=index.cursor(), checked_at=utc())
            atomic_json(Path(index_root) / "collector-status.json", state)
            if outcome not in ("caught_up", "behind"):
                raise RuntimeError(outcome)
            return state
        stopped = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stopped.set())
        sync.start()
        try:
            while not stopped.wait(10):
                atomic_json(Path(index_root) / "collector-status.json", dict(sync.state, committed_id=index.cursor(), checked_at=utc()))
        finally:
            sync.close()
        return dict(status="stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect"); p.add_argument("--index-root", type=Path, required=True); p.add_argument("--once", action="store_true")
    p = sub.add_parser("snapshot"); p.add_argument("--database", type=Path, required=True); p.add_argument("--destination", type=Path, required=True)
    p = sub.add_parser("assess"); p.add_argument("--manifest", type=Path, required=True)
    p = sub.add_parser("train-once"); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--recipe", type=Path, required=True); p.add_argument("--state-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(args.index_root, args.once)
    elif args.command == "snapshot":
        result = snapshot(args.database, args.destination)
    elif args.command == "assess":
        result = assess(args.manifest)
    else:
        result = train_once(args.manifest, args.recipe, args.state_root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result.get("status") == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
