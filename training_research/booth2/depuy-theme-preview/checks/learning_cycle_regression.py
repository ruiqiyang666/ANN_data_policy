"""Regression of WAL snapshots, dataset admission and candidate-only execution."""
import json
import os
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "simulation/pipeline"), str(REPO / "checks")]
import learning_cycle as L
import history_mock as M
from domain.history_index import HistoryIndex


class Learning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Snapshots intentionally use read-only files, also on Windows.
        def cleanup():
            for p in self.root.rglob("*"):
                if p.is_file():
                    p.chmod(0o600)
            self.tmp.cleanup()
        self.addCleanup(cleanup)

    def dataset(self):
        (self.root / "features.bin").write_bytes(b"fixture features")
        (self.root / "review.json").write_text('{"passed":true}')
        m = dict(schema_version=1, dataset_id="fixture-dataset", snapshot_id="history-fixture", status="ready",
                 feature_version="fixture-1", label_version="fixture-1", created_at="2026-11-20T00:00:00+00:00",
                 feature_equivalence_review=dict(passed=True, reviewer="test", artifact="review.json"),
                 quality_review=dict(passed=True, reviewer="test", artifact="review.json"),
                 artifacts={k: L.sha256(self.root / k) for k in ("features.bin", "review.json")},
                 cycles=[dict(cycle_id=f"C{i}", endpoint="eol" if i < 2 else "right_censored", closed=True,
                              label_status="confirmed", reviewer="test", confirmed_at="2026-11-19T00:00:00+00:00") for i in range(3)])
        path = self.root / "dataset.json"
        L.atomic_json(path, m)
        return path, m

    def recipe(self, code="from pathlib import Path; import sys; Path(sys.argv[1], 'result.json').write_text('{}')"):
        path = self.root / "recipe.json"
        L.atomic_json(path, dict(schema_version=1, purpose="candidate_training", argv=[sys.executable, "-c", code, "{output_dir}"]))
        return path

    def test_backup_contains_wal_and_is_independent(self):
        index = HistoryIndex(self.root / "index.sqlite3")
        rows = M.MockHistory("fixture-key", M.patterned_events(seconds=30, start_ms=1_795_000_000_000)).rows
        with closing(sqlite3.connect(index.path)) as kept_open:
            kept_open.execute("PRAGMA journal_mode=WAL")
            index.ingest_page(rows[:20])
            manifest = L.snapshot(index.path, self.root / "snapshots")
            path = self.root / "snapshots" / manifest["snapshot_id"] / "history.sqlite3"
            before = L.sha256(path)
            index.ingest_page(rows[20:])
            self.assertEqual(before, L.sha256(path))
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as copy:
                self.assertEqual(copy.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0], 20)
                self.assertEqual(copy.execute("SELECT committed_history_id FROM ingest_cursor").fetchone()[0], rows[19]["id"])
            self.assertEqual(manifest["events"], 20)
            self.assertFalse(manifest["training_ready"])
        self.assertEqual(L.snapshot(index.path, self.root / "snapshots")["events"], len(rows))

    def test_missing_labels_and_missing_features_do_not_train(self):
        self.assertEqual(L.assess(self.root / "absent.json")["status"], "awaiting_feature_build")
        path, m = self.dataset()
        m["cycles"][1]["label_status"] = "predicted"
        L.atomic_json(path, m)
        result = L.train_once(path, self.recipe(), self.root / "jobs")
        self.assertEqual(result["status"], "awaiting_labels")
        self.assertFalse((self.root / "jobs/jobs").exists())

    def test_enough_new_mature_cycles_required(self):
        path, m = self.dataset()
        self.assertTrue(L.assess(path)["eligible"])
        self.assertFalse(L.assess(path, ["C0"])["eligible"])
        m["cycles"][0]["endpoint"] = "right_censored"
        L.atomic_json(path, m)
        self.assertFalse(L.assess(path)["eligible"])

    def test_changed_and_escaped_inputs_rejected(self):
        path, m = self.dataset()
        (self.root / "features.bin").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            L.assess(path)
        m["artifacts"] = {"../escape": "not-a-hash"}
        L.atomic_json(path, m)
        with self.assertRaises(ValueError):
            L.assess(path)

    def test_late_labels_and_duplicate_cycles_rejected(self):
        path, m = self.dataset()
        m["cycles"][0]["confirmed_at"] = "2026-11-21T00:00:00+00:00"
        L.atomic_json(path, m)
        with self.assertRaises(ValueError):
            L.assess(path)
        m["cycles"][0]["confirmed_at"] = "2026-11-19T00:00:00+00:00"
        m["cycles"][1]["cycle_id"] = "C0"
        L.atomic_json(path, m)
        with self.assertRaises(ValueError):
            L.assess(path)

    def test_success_waits_for_review_and_does_not_repeat(self):
        path, _ = self.dataset()
        recipe = self.recipe()
        state = self.root / "state"
        result = L.train_once(path, recipe, state)
        self.assertEqual(result["status"], "awaiting_evaluation_review")
        self.assertFalse(result["automatic_promotion"])
        self.assertIn("result.json", result["artifacts"])
        self.assertEqual(L.train_once(path, recipe, state)["status"], "awaiting_mature_cycles")
        self.assertEqual(len(list((state / "jobs").iterdir())), 1)
        self.assertFalse((state / "active.json").exists())

    def test_failed_training_preserves_baseline_and_does_not_loop(self):
        path, _ = self.dataset()
        recipe = self.recipe("raise SystemExit(3)")
        state = self.root / "state"
        first = L.train_once(path, recipe, state)
        self.assertEqual(first["status"], "failed")
        self.assertFalse((state / "last_trained.json").exists())
        self.assertEqual(first, L.train_once(path, recipe, state))

    def test_mutation_during_training_fails(self):
        path, _ = self.dataset()
        recipe = self.recipe("from pathlib import Path; Path('features.bin').write_text('changed')")
        result = L.train_once(path, recipe, self.root / "state")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "dataset_artifact_changed_during_training")

    def test_exclusive_job_lock(self):
        with L.exclusive(self.root / ".lock"):
            with self.assertRaises(OSError):
                with L.exclusive(self.root / ".lock"):
                    self.fail("second writer acquired the lock")

    def test_headless_collect_and_restart(self):
        mock = M.MockHistory("fixture-key", M.patterned_events(seconds=30, start_ms=1_795_000_000_000))
        with M.MockServer(mock) as server:
            with patch.dict(os.environ, {"PHASE2_HISTORY_API_URL": server.url, "PHASE2_HISTORY_API_KEY": "fixture-key",
                                         "PHASE2_HISTORY_API_KEY_FILE": ""}):
                a = L.collect(self.root / "index", once=True)
                b = L.collect(self.root / "index", once=True)
        self.assertEqual(a["status"], "caught_up")
        self.assertEqual(a["committed_id"], b["committed_id"])
        self.assertEqual(HistoryIndex(self.root / "index/history/plasma2.sqlite3").summary()["events"], len(mock.rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
