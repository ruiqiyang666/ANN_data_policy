"""Independent synthetic checks; no held-out prediction is inspected here."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from modeling import FIELDS, VALUE_FIELDS, RiskNet, encode, fit_preprocessor, sequence_indices


def sensors():
    n = 240
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(dict(timestamp=pd.date_range('2026-01-01', periods=n, freq='10s'),
                              cycle_id=['L000'] * 120 + ['L001'] * 120,
                              split=['train'] * 120 + ['validation'] * 120))
    for j, col in enumerate(FIELDS):
        frame[col] = j + rng.normal(size=n)
    frame['Voltage'] = 55 + rng.normal(size=n)
    frame['Feedrate_1'] = np.where(np.arange(n) % 2 == 0, 10., 0.)
    frame['Feedrate_2'] = np.where(np.arange(n) % 2 == 1, 10., 0.)
    frame['Voltage_mean60s'] = frame.Voltage.rolling(6, min_periods=1).mean()
    frame['Voltage_std60s'] = frame.Voltage.rolling(6, min_periods=2).std().fillna(0)
    # A real observation gap is represented by missing anchors, not by zeros.
    frame = frame.drop(index=range(70, 90)).reset_index(drop=True)
    return frame


class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.frame = sensors()
        cls.pre = fit_preprocessor(cls.frame[cls.frame.split.eq('train')])

    def test_whole_history_agrees_with_prefix_replay(self):
        full = encode(self.frame, self.pre)
        for at in [0, 1, 28, 69, 70, 99, 100, 130, 219]:
            incremental = encode(self.frame.iloc[:at + 1], self.pre)
            np.testing.assert_array_equal(full[at], incremental[-1])

    def test_future_sensor_corruption_cannot_change_past_features(self):
        original = encode(self.frame, self.pre)
        corrupted = self.frame.copy()
        at = 65
        columns = FIELDS + ['Voltage_mean60s', 'Voltage_std60s']
        corrupted.loc[at:, columns] = 1e9
        corrupted.loc[at + 3:, 'Voltage'] = np.nan
        changed = encode(corrupted, self.pre)
        np.testing.assert_array_equal(original[:at], changed[:at])
        self.assertFalse(np.array_equal(original[at:], changed[at:]))

    def test_cycle_boundary_cannot_reuse_old_electrode_history(self):
        indices, _ = sequence_indices(self.frame)
        cycles = self.frame.cycle_id.to_numpy()
        for i, row in enumerate(indices):
            used = row[row >= 0]
            self.assertTrue(np.all(cycles[used] == cycles[i]))
            self.assertTrue(np.all(used <= i))
        first_new = np.flatnonzero(cycles == 'L001')[0]
        self.assertTrue(np.all(indices[first_new, :-1] == -1))

    def test_gap_not_filled_from_arbitrarily_old_samples(self):
        indices, ages = sequence_indices(self.frame)
        now = self.frame.timestamp.to_numpy(dtype='datetime64[s]').astype(np.int64)
        offsets = np.arange(60, -1, -1) * 60
        for i, row in enumerate(indices):
            valid = row >= 0
            delta = now[i] - offsets[valid] - now[row[valid]]
            self.assertTrue(np.all((delta >= 0) & (delta <= 60)))
        self.assertTrue(np.any(indices[70] < 0))

    def test_future_labels_and_completed_product_means_are_not_inputs(self):
        original = encode(self.frame, self.pre)
        changed = self.frame.copy()
        for name in ['failure', 'voltage_mean', 'gun_serial_from_record', 'remaining_seconds',
                     'maintenance_time_lower_seconds', 'label_available_at', 'previous_failure']:
            changed[name] = np.arange(len(changed)) * 12345
        np.testing.assert_array_equal(original, encode(changed, self.pre))
        self.assertTrue(set(VALUE_FIELDS).isdisjoint({'failure', 'voltage_mean', 'cycle_id', 'remaining_seconds'}))

    def test_validation_sensor_values_do_not_change_training_preprocessor(self):
        frame = self.frame.copy()
        expected = fit_preprocessor(frame[frame.split.eq('train')])
        columns = FIELDS + ['Voltage_mean60s', 'Voltage_std60s']
        frame.loc[frame.split.eq('validation'), columns] = -1e10
        actual = fit_preprocessor(frame[frame.split.eq('train')])
        self.assertEqual(actual, expected)
        self.assertEqual(actual['fit_split'], 'train')
        self.assertTrue(np.isfinite(actual['median']).all())
        self.assertTrue(np.isfinite(actual['scale']).all())

    def test_tcn_earlier_hidden_states_do_not_see_future_grid_points(self):
        torch.manual_seed(12)
        net = RiskNet('tcn', 8).eval()
        original = torch.randn(2, 61, 2 * len(VALUE_FIELDS) + 1)
        changed = original.clone()
        changed[:, 31:] = 999
        with torch.inference_mode():
            a = net.body(net.input(original.transpose(1, 2)))
            b = net.body(net.input(changed.transpose(1, 2)))
        torch.testing.assert_close(a[:, :, :31], b[:, :, :31], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
