"""Regression for the electrode end-of-life GRU shown beside the real 28-input replay.

usage: python -B checks/electrode_regression.py REAL28_BUNDLE ELECTRODE_BUNDLE

Covers: bundle integrity, score parity with the frozen ledger, the frozen alarm
moments, that the 28-input outputs are unchanged by the second model, the model
switch over HTTP, and that a service without the bundle refuses the model.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "simulation/pipeline"))
import torch  # noqa: E402,F401 -- native libraries before NumPy
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from api.application import create_app  # noqa: E402
from inference.electrode import ElectrodeBundle, encode, gru_forward  # noqa: E402
from inference.real28 import Real28Bundle, Real28Service  # noqa: E402

REAL28, ELECTRODE = Path(sys.argv[1]), Path(sys.argv[2])
HEADERS = {"X-Phase2-Request": "1"}


class ElectrodeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = ElectrodeBundle(ELECTRODE)
        cls.real = Real28Bundle(REAL28)

    def test_integrity_and_frozen_alarms(self):
        b = self.bundle
        self.assertTrue(b.verify())
        self.assertEqual(int(b.emitted.sum()), b.manifest["verification"]["alarms_reproduced"])
        first = b.times[b.emitted & (b.cycles == b.cycle_ids.index("L008"))]
        self.assertEqual(str(np.datetime64(int(first[0]), "s")), "2026-10-28T07:42:50")

    def test_scores_equal_the_ledger(self):
        b = self.bundle
        index = list(range(0, len(b.times), 97))
        heads, _ = gru_forward(b.weights, encode(b.times, b.values, b.cycles, index, b.pre))
        self.assertLess(float(np.abs(heads[:, b.head_index] - b.ledger[index]).max()), 1e-4)

    def test_inputs_are_the_recorded_readings(self):
        b, r = self.bundle, self.real
        at = np.searchsorted(r.raw_t, b.times)
        ok = at < len(r.raw_t)
        ok[ok] &= r.raw_t[at[ok]] == b.times[ok]
        self.assertGreater(int(ok.sum()), 8000)
        voltage = np.asarray(r.raw[at[ok], r.field_keys.index("Voltage")])
        self.assertTrue(np.allclose(voltage, b.values[ok, 0], atol=1e-3))

    def test_no_origin_means_no_score(self):
        b = self.bundle
        gap = int(np.argmax(np.diff(b.times))) + 1
        result = b.outputs(int(b.times[gap]) - 120)
        self.assertEqual(result["status"], "idle")
        self.assertNotIn("score", result)

    def test_second_model_leaves_the_first_unchanged(self):
        plain = Real28Service(self.real, default_segment="normal_20261029").create()
        both = Real28Service(self.real, default_segment="normal_20261029", electrode=self.bundle).create()
        for key in ("timestamp", "inputs", "features", "status"):
            self.assertEqual(plain[key], both[key])
        skip = {"latency_ms"}
        self.assertEqual({k: v for k, v in plain["outputs"].items() if k not in skip},
                         {k: v for k, v in both["outputs"].items() if k not in skip})
        self.assertEqual(plain["neural"]["heads"], both["neural"]["heads"])
        self.assertIsNone(plain["electrode"])
        self.assertEqual(both["model_id"], "active")


class ElectrodeHttp(unittest.TestCase):
    def app(self, electrode):
        folder = tempfile.mkdtemp()
        return TestClient(create_app(Path(folder), real_data=Path(folder), index_output=Path(folder),
                                     real28=REAL28, suite=True, electrode=electrode))

    def test_switch_and_alarm(self):
        client = self.app(ELECTRODE)
        self.assertEqual(client.get("/api/simulation/config").json()["electrode"]["threshold"], 0.9500000000000001)
        s = client.post("/api/simulation/sessions", json={"model_id": "electrode"}, headers=HEADERS).json()
        self.assertEqual((s["model_id"], s["instrument"]["source"]["name"]), ("electrode", "electrode_eol_20261028"))
        self.assertEqual(s["neural"]["topology"], "recurrent_last_step")
        self.assertIsNotNone(s["outputs"]["rul_hours"])  # other panels keep reading the 28-input outputs
        seq = s["seq"]
        for _ in range(13):
            s = client.post(f"/api/simulation/sessions/{s['session_id']}/advance",
                            json={"expected_seq": seq, "steps": 60, "model_id": "electrode"}, headers=HEADERS).json()
            seq = s["seq"]
        self.assertTrue(s["electrode"]["alarm_active"])
        self.assertEqual(s["electrode"]["alarm_at"], "2026-10-28T07:42:50")
        record = client.get("/api/real/experiments").json()
        self.assertEqual(len(record["electrode"]["candidates"]), 12)
        self.assertEqual(record["real28"]["features"], 292)

    def test_refused_without_the_bundle(self):
        client = self.app(None)
        self.assertIsNone(client.get("/api/simulation/config").json()["electrode"])
        response = client.post("/api/simulation/sessions", json={"model_id": "electrode"}, headers=HEADERS)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(client.get("/api/real/experiments").json()["electrode"], None)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
