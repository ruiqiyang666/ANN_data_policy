"""Causal alarm replay and event scoring for the frozen 20-minute protocol.

Alarm generation reads timestamps, cycle boundaries, and scores only.  Event
times and future observation horizons are used exclusively after emission for
scoring.  The score is not a prediction of remaining minutes.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlarmPolicy:
    confirm_origins: int = 2
    max_confirm_gap_seconds: int = 90
    release_seconds: int = 600
    cooldown_seconds: int = 1800


POLICY = AlarmPolicy()


class AlarmMachine:
    """An incremental state machine with no knowledge of outcomes or targets."""

    def __init__(self, threshold, policy=POLICY):
        self.threshold = float(threshold)
        self.policy = policy
        self.cycle = None
        self.last_origin = None
        self.last_signal = None
        self.last_alarm = None
        self.consecutive = 0
        self.active = False

    def step(self, at, probability, cycle_id):
        """Accept an epoch-second timestamp; return True only when an alarm emits."""
        at = float(at)
        if cycle_id != self.cycle:
            self.cycle = cycle_id
            self.last_origin = self.last_signal = self.last_alarm = None
            self.consecutive = 0
            self.active = False
        if self.last_origin is not None:
            if at <= self.last_origin:
                raise ValueError('Origins must be strictly chronological within a cycle')
            if at - self.last_origin > self.policy.max_confirm_gap_seconds:
                self.consecutive = 0
        self.last_origin = at
        if self.active and at - self.last_signal >= self.policy.release_seconds:
            self.active = False
        above = bool(np.isfinite(probability) and probability >= self.threshold)
        if not above:
            self.consecutive = 0
            return False
        if self.last_signal is None or at - self.last_signal > self.policy.max_confirm_gap_seconds:
            self.consecutive = 0
        self.consecutive += 1
        self.last_signal = at
        cooled = self.last_alarm is None or at - self.last_alarm >= self.policy.cooldown_seconds
        if self.consecutive >= self.policy.confirm_origins and not self.active and cooled:
            self.last_alarm = at
            self.active = True
            return True
        return False


def _time_seconds(values):
    # pandas preserves s/ms/us resolution for some inputs; asi8 alone does not
    # imply nanoseconds. Normalize units before the calendar-time arithmetic.
    return pd.DatetimeIndex(pd.to_datetime(values)).as_unit('ns').asi8.astype(np.float64) / 1e9


def build_alarms(frame, probabilities, threshold, policy=POLICY):
    """Replay every origin, retaining only actual first emissions of episodes."""
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 1 or len(probability) != len(frame):
        raise ValueError('probabilities must contain exactly one score per frame row')
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    timestamps = pd.to_datetime(frame.timestamp)
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError('frame timestamps must be present and globally chronological')
    if frame.cycle_id.isna().any():
        raise ValueError('cycle_id must be present for all origins')
    now = _time_seconds(timestamps)
    cycles = frame.cycle_id.astype(str).to_numpy()
    machine = AlarmMachine(threshold, policy)
    alarms = []
    seen = set()
    current = None
    for i, (at, value, cycle) in enumerate(zip(now, probability, cycles)):
        if cycle != current:
            if cycle in seen:
                raise ValueError('A cycle may not be split into disconnected groups')
            seen.add(cycle)
            current = cycle
        if machine.step(at, value, cycle):
            alarms.append(dict(index=int(i), time=str(timestamps.iloc[i]),
                               cycle_id=cycle, probability=float(value),
                               anchor_id=str(frame.anchor_id.iloc[i]) if 'anchor_id' in frame else None))
    return alarms


def evaluate(frame, probabilities, threshold, events, operating_hours, window_minutes=60):
    """Return summary, one row per event, and every emitted alarm.

    A credited alarm must be strictly before T, at most ``window_minutes``
    before T, and precede no observation barrier before T.  The first such
    emitted alarm is scored; later, more convenient alarms never replace it.
    Alerts after an observed terminal event are reported separately.
    """
    hours = float(operating_hours)
    if not np.isfinite(hours) or hours < 0:
        raise ValueError('operating_hours must be finite and non-negative')
    window_seconds = float(window_minutes) * 60
    if window_seconds < 1200:
        raise ValueError('The matching window must cover the required 20-minute lead')
    q = events.copy()
    if 'is_eol' in q:
        q = q[q.is_eol.eq(1)].copy()
    if len(q):
        q['time'] = pd.to_datetime(q.time)
        q = q.sort_values('time').reset_index(drop=True)
        if q.time.isna().any() or q.event_id.duplicated().any():
            raise ValueError('Event times must be present and event_id unique')
    else:
        q = pd.DataFrame(columns=['event_id', 'cycle_id', 'time'])
    alarms = build_alarms(frame, probabilities, threshold)
    now = _time_seconds(frame.timestamp)
    horizons = frame.observation_horizon_seconds.to_numpy(dtype=float)
    if np.any(~np.isfinite(horizons)) or np.any(horizons < 0):
        raise ValueError('observation horizons must be finite and non-negative')
    lookup = {}
    for cycle, group in q.groupby('cycle_id', sort=False):
        lookup[str(cycle)] = (group, _time_seconds(group.time))
    for alarm in alarms:
        i = alarm['index']
        group, event_times = lookup.get(alarm['cycle_id'], (q.iloc[:0], np.array([])))
        delta = event_times - now[i]
        candidates = np.flatnonzero((delta > 0) & (delta <= window_seconds) & (delta < horizons[i]))
        matched = str(group.event_id.iloc[candidates[0]]) if len(candidates) else None
        post_event = bool(np.any(delta <= 0)) and matched is None
        # Strict > matches the target builder: a barrier at the window boundary
        # leaves the outcome censored, including a preventive replacement.
        false_alarm = matched is None and not post_event and horizons[i] > window_seconds
        unknown = matched is None and not post_event and not false_alarm
        alarm.update(matched_event=matched, event_id=matched,
                     false_alarm=bool(false_alarm), unknown=bool(unknown), post_event=bool(post_event))
    rows = []
    frame_cycles = frame.cycle_id.astype(str).to_numpy()
    for event in q.itertuples():
        event_id = str(event.event_id)
        event_time = pd.Timestamp(event.time)
        at = event_time.value / 1e9
        matching = [a for a in alarms if a['matched_event'] == event_id]
        first = matching[0] if matching else None
        lead = (at - now[first['index']]) / 60 if first else None
        hit = lead is not None and lead >= 20
        late = lead is not None and lead < 20
        in_cycle = frame_cycles == str(event.cycle_id)
        delta = at - now
        opportunities = in_cycle & (delta > 0) & (delta <= window_seconds) & (delta < horizons)
        deadline_opportunities = opportunities & (delta >= 1200)
        prior_alarms = [a for a in alarms if a['cycle_id'] == str(event.cycle_id) and now[a['index']] < at]
        earlier = [a for a in prior_alarms if delta[a['index']] > window_seconds]
        rows.append(dict(
            event_id=event_id, cycle_id=str(event.cycle_id), time=str(event_time),
            deadline=str(event_time - pd.Timedelta(minutes=20)),
            first_alarm=first['time'] if first else None,
            lead_minutes=float(lead) if lead is not None else None,
            hit20=bool(hit), excess_minutes=float(lead - 20) if hit else None,
            late=bool(late), missed=first is None,
            matched_alarm_count=len(matching),
            all_episode_first=prior_alarms[0]['time'] if prior_alarms else None,
            earlier_alarm_count=len(earlier),
            maximum_observable_lead_minutes=float(delta[opportunities].max() / 60) if opportunities.any() else None,
            latest_deadline_origin=str(frame.timestamp.iloc[np.flatnonzero(deadline_opportunities)[-1]]) if deadline_opportunities.any() else None,
            latest_deadline_lead_minutes=float(delta[deadline_opportunities].min() / 60) if deadline_opportunities.any() else None,
            observable_origin_count=int(opportunities.sum()),
            deadline_origin_count=int(deadline_opportunities.sum()),
        ))
    total = len(rows)
    hit_count = sum(r['hit20'] for r in rows)
    excess = [r['excess_minutes'] for r in rows if r['hit20']]
    false_count = sum(a['false_alarm'] for a in alarms)
    metrics = dict(
        total_events=total, hit20=hit_count, event_success_rate=hit_count / total if total else None,
        late=sum(r['late'] for r in rows), missed=sum(r['missed'] for r in rows),
        unmet20=total - hit_count, false_alarms=false_count,
        false_alarms_per_hour=false_count / hours if hours > 0 else None,
        unknown_alarms=sum(a['unknown'] for a in alarms),
        post_event_alarms=sum(a['post_event'] for a in alarms),
        matched_alarms=sum(a['matched_event'] is not None for a in alarms),
        alarms=len(alarms), operating_hours=hours, threshold=float(threshold),
        matching_window_minutes=float(window_minutes),
        mean_excess_minutes=float(np.mean(excess)) if excess else None,
        median_excess_minutes=float(np.median(excess)) if excess else None,
        max_excess_minutes=float(np.max(excess)) if excess else None,
        excess_coverage_events=len(excess),
        note='excess统计仅覆盖提前至少20分钟达标事件；不是剩余寿命MAE。',
    )
    return metrics, rows, alarms


def selection_key(metrics, false_budget=0.1):
    """Predeclared lexicographic selection on validation outcomes only."""
    rate = metrics['false_alarms_per_hour']
    feasible = rate is not None and rate <= false_budget
    excess = metrics['mean_excess_minutes']
    return (bool(feasible), int(metrics['hit20']), -int(metrics['false_alarms']),
            -float(excess) if excess is not None else 0.,
            -int(metrics['late']), float(metrics['threshold']))


def select_threshold(frame, probabilities, events, operating_hours, grid=None, false_budget=0.1):
    """Select only from the fixed grid on the validation split.

    If a split column is supplied, any non-validation row is rejected. The
    default grid contains nineteen thresholds plus the no-alarm alternative.
    """
    if 'split' in frame and not frame.split.eq('validation').all():
        raise ValueError('Threshold selection may use validation origins only')
    if 'split' in events and not events.split.eq('validation').all():
        raise ValueError('Threshold selection may use validation events only')
    if grid is None:
        grid = np.r_[np.arange(.05, 1, .05), 1.000001]
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
        raise ValueError('Threshold grid must be nonempty and finite')
    comparisons = []
    for threshold in np.unique(grid):
        metrics, _, _ = evaluate(frame, probabilities, float(threshold), events, operating_hours)
        comparisons.append(dict(threshold=float(threshold), metrics=metrics))
    selected = max(comparisons, key=lambda r: selection_key(r['metrics'], false_budget))
    return dict(threshold=selected['threshold'], metrics=selected['metrics'],
                selection_key=list(selection_key(selected['metrics'], false_budget)),
                false_budget_per_hour=float(false_budget), grid_results=comparisons)
