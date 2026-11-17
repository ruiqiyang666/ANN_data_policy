"""Behavioral checks for alarm deadlines, episode state, and censoring."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation import AlarmMachine, build_alarms, evaluate, select_threshold


BASE = pd.Timestamp('2026-11-10 12:00:00')


def frame(seconds, horizon=20000, cycles=None):
    n = len(seconds)
    return pd.DataFrame(dict(timestamp=[BASE + pd.Timedelta(seconds=s) for s in seconds],
                             cycle_id=cycles or ['L012'] * n,
                             observation_horizon_seconds=np.broadcast_to(horizon, (n,)),
                             remaining_seconds=np.nan, split='validation'))


def events(seconds=3600, cycle='L012'):
    return pd.DataFrame([dict(event_id='T012', time=BASE + pd.Timedelta(seconds=seconds),
                              cycle_id=cycle, split='validation', is_eol=1)])


class AlarmEvaluationTests(unittest.TestCase):
    def test_exact_twenty_counts_and_one_second_later_does_not(self):
        for second, hit in [(2400, True), (2401, False)]:
            metrics, rows, _ = evaluate(frame([second - 10, second]), [.9, .9], .5, events(), 1)
            self.assertEqual(rows[0]['hit20'], hit)
            self.assertEqual(rows[0]['late'], not hit)
            self.assertEqual(metrics['missed'], 0)
            self.assertAlmostEqual(rows[0]['lead_minutes'], (3600 - second) / 60)

    def test_first_emitted_alarm_cannot_be_replaced_by_closer_one(self):
        data = frame([590, 600, 1200, 2390, 2400])
        metrics, rows, alarms = evaluate(data, [.9, .9, .1, .9, .9], .5, events(), 1)
        self.assertEqual(len(alarms), 2)
        self.assertEqual(rows[0]['first_alarm'], str(BASE + pd.Timedelta(seconds=600)))
        self.assertEqual(rows[0]['lead_minutes'], 50)
        self.assertEqual(rows[0]['excess_minutes'], 30)
        self.assertEqual(rows[0]['matched_alarm_count'], 2)
        self.assertEqual(metrics['hit20'], 1)

    def test_continuous_early_episode_not_credited_again_inside_window(self):
        data = frame(np.arange(0, 4201, 60))
        metrics, rows, alarms = evaluate(data, np.ones(len(data)), .5, events(7200), 2)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(metrics['false_alarms'], 1)
        self.assertTrue(rows[0]['missed'])
        self.assertEqual(rows[0]['earlier_alarm_count'], 1)
        self.assertEqual(rows[0]['all_episode_first'], str(BASE + pd.Timedelta(seconds=60)))

    def test_confirmation_gap_and_cycle_reset(self):
        data = frame([0, 91, 101, 200, 210], cycles=['L001', 'L001', 'L001', 'L002', 'L002'])
        alarms = build_alarms(data, [.9] * 5, .5)
        self.assertEqual([a['index'] for a in alarms], [2, 4])

    def test_below_threshold_and_missing_reset_confirmation(self):
        data = frame([0, 10, 20, 30, 40, 50])
        alarms = build_alarms(data, [.9, np.nan, .9, .1, .9, .9], .5)
        self.assertEqual([a['index'] for a in alarms], [5])

    def test_episode_release_and_cooldown_are_both_required(self):
        data = frame([0, 10, 610, 620, 1800, 1810])
        alarms = build_alarms(data, [.9] * len(data), .5)
        self.assertEqual([a['index'] for a in alarms], [1, 5])

    def test_future_barrier_prevents_event_credit_and_negative_claim(self):
        data = frame([2390, 2400], horizon=[1200, 1200])
        metrics, rows, alarms = evaluate(data, [.9, .9], .5, events(), 2)
        self.assertTrue(rows[0]['missed'])
        self.assertEqual(metrics['unknown_alarms'], 1)
        self.assertEqual(metrics['false_alarms'], 0)
        self.assertIsNone(alarms[0]['matched_event'])

    def test_post_terminal_alarm_is_separate_and_never_a_hit(self):
        metrics, rows, alarms = evaluate(frame([3600, 3610]), [.9, .9], .5, events(), 1)
        self.assertTrue(rows[0]['missed'])
        self.assertEqual(metrics['post_event_alarms'], 1)
        self.assertEqual(metrics['false_alarms'], 0)
        self.assertFalse(alarms[0]['unknown'])

    def test_no_events_still_counts_normal_and_censored_alarms(self):
        data = frame([0, 10, 2000, 2010], horizon=[4000, 4000, 500, 500])
        metrics, rows, alarms = evaluate(data, [.9] * len(data), .5, events().iloc[:0], 2)
        self.assertEqual(rows, [])
        self.assertEqual(metrics['false_alarms'], 1)
        self.assertEqual(metrics['unknown_alarms'], 1)
        self.assertEqual(metrics['false_alarms_per_hour'], .5)
        self.assertIsNone(metrics['event_success_rate'])

    def test_validation_only_threshold_selection(self):
        data = frame([2390, 2400])
        result = select_threshold(data, [.8, .8], events(), 2, grid=[.5, .9])
        self.assertEqual(result['threshold'], .5)
        data['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'validation'):
            select_threshold(data, [.8, .8], events(), 2)

    def test_incremental_machine_matches_batch_emissions(self):
        sec = [0, 10, 610, 620, 1800, 1810, 3000]
        p = [.7, .9, .9, .9, .9, .9, .1]
        machine = AlarmMachine(.5)
        emitted = [i for i, (at, value) in enumerate(zip(sec, p)) if machine.step(at, value, 'L012')]
        alarms = build_alarms(frame(sec), p, .5)
        self.assertEqual(emitted, [a['index'] for a in alarms])

    def test_zero_hours_is_not_a_zero_false_alarm_rate(self):
        metrics, _, _ = evaluate(frame([0, 10]), [.9, .9], .5, events().iloc[:0], 0)
        self.assertIsNone(metrics['false_alarms_per_hour'])

    def test_pandas_time_unit_does_not_change_calendar_lead(self):
        for unit in ['s', 'ms', 'us', 'ns']:
            data = frame([2390, 2400])
            data['timestamp'] = data.timestamp.astype('datetime64[' + unit + ']')
            ev = events()
            ev['time'] = ev.time.astype('datetime64[' + unit + ']')
            _, rows, _ = evaluate(data, [.9, .9], .5, ev, 1)
            self.assertEqual(rows[0]['lead_minutes'], 20.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
