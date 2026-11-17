"""Independent stream representation and saved-model adapter checks."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from inference import CausalBuffer, StreamingPredictor, predict_model, verify_frozen_files, voltage_summary_indices
from modeling import FIELDS, TARGETS, RiskNet, encode, fit_preprocessor, margin_score, predict_net, summaries
from test_causality import sensors


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.frame = sensors()
        cls.pre = fit_preprocessor(cls.frame[cls.frame.split.eq('train')])
        cls.x = encode(cls.frame, cls.pre)

    def test_incremental_window_equals_batch_for_every_origin(self):
        buffer = CausalBuffer(self.pre)
        for i, row in enumerate(self.frame.to_dict('records')):
            actual = buffer.push(row)
            np.testing.assert_array_equal(actual[0], self.x[i])

    def test_buffer_remains_bounded_and_rejects_out_of_order(self):
        buffer = CausalBuffer(self.pre)
        row = self.frame.iloc[0].to_dict()
        start = pd.Timestamp(row['timestamp'])
        for minute in range(140):
            row['timestamp'] = start + pd.Timedelta(minutes=minute)
            buffer.push(row)
        self.assertLessEqual(len(buffer.records), 62)
        with self.assertRaisesRegex(ValueError, 'increasing'):
            buffer.push(row)

    def test_unknown_future_metadata_ignored(self):
        a, b = CausalBuffer(self.pre), CausalBuffer(self.pre)
        for row in self.frame.to_dict('records')[:20]:
            changed = dict(row, remaining_seconds=-100, failure=1, voltage_mean=-999,
                           observation_horizon_seconds=0, future_maintenance='2099-01-01')
            np.testing.assert_array_equal(a.push(row), b.push(changed))

    def test_saved_neural_adapter_matches_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for kind in ['tcn', 'gru']:
                spec = dict(name=kind, kind=kind, hidden=8)
                out = root / 'artifacts/models' / kind
                out.mkdir(parents=True)
                net = RiskNet(kind, 8).eval()
                torch.save(dict(state_dict=net.state_dict()), out / 'best.pt')
                expected = predict_net(net, self.x[:10])
                actual = predict_model(spec, self.x[:10], self.frame.iloc[:10], root)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_saved_sklearn_heads_and_voltage_selection(self):
        z = summaries(self.x)
        indices = voltage_summary_indices(z.shape[1])
        self.assertEqual(len(indices), 58)
        y = (np.arange(len(z)) % 2).astype(int)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / 'artifacts/models/logistic_voltage'
            out.mkdir(parents=True)
            model = LogisticRegression(max_iter=500).fit(z[:, indices], y)
            joblib.dump([model] * 4, out / 'model.joblib')
            spec = dict(name='logistic_voltage', kind='logistic_voltage', summary_indices=indices)
            actual = predict_model(spec, self.x[:10], self.frame.iloc[:10], root)
            expected = model.predict_proba(z[:10, indices])[:, 1]
            self.assertEqual(actual.shape, (10, 4))
            for j in range(4):
                np.testing.assert_allclose(actual[:, j], expected)

    def test_streaming_selected_margin_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            art = root / 'artifacts'
            art.mkdir()
            pp = art / 'preprocessor.json'
            pp.write_text(json.dumps(self.pre), encoding='utf-8')
            config = dict(selected_model='margin', model_version='synthetic-test',
                          model_specs=[dict(name='margin', kind='margin')],
                          policies=dict(margin=dict(head='engineering_margin', head_index=0, threshold=.5)),
                          preprocessor_sha256=hashlib.sha256(pp.read_bytes()).hexdigest())
            (art / 'frozen_config.json').write_text(json.dumps(config), encoding='utf-8')
            predictor = StreamingPredictor(root)
            expected = margin_score(self.frame)
            for i, row in enumerate(self.frame.to_dict('records')):
                result = predictor.push(row)
                self.assertEqual(result['score'], float(expected[i]))
            pp.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'preprocessor'):
                verify_frozen_files(root, config)


if __name__ == '__main__':
    unittest.main(verbosity=2)
