"""Rebuild V2 targets from the 2026-11-17 time audit, without reading V1 targets.

Input sensor features and product segmentation remain immutable. Dates in outputs
are ISO local device wall-clock timestamps, not elapsed spraying time.
"""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "training_data"
PROJECT = Path("C:/Users/杨睿奇/Desktop/大三第一学期/JJ")
AUDIT = PROJECT / "11_17数据处理/预测事件时间核对.csv"
CLOCK = PROJECT / "11_16数据处理/本地核查材料/传感器计时与转台状态.npz"
FMT = "%d %m %Y %H:%M:%S"
FIELDS = ["Voltage", "Current", "Power", "Station_1", "Station_2",
          "Feedrate_1", "Feedrate_2", "Press._Hopper_1", "Press._Hopper_2"]


def seconds(values):
    return np.asarray(values, dtype="datetime64[s]").astype(np.int64)


def split_for(cycle_id):
    i = int(cycle_id[1:])
    return "train" if i <= 4 else "validation" if i <= 7 else "test"


def parse_dates(frame, columns):
    for col in columns:
        frame[col] = pd.to_datetime(frame[col], format=FMT, errors="raise")
    return frame


def rebuilt_cycles(source, time_audit):
    c = pd.read_csv(source / "electrode_cycles.csv")
    parse_dates(c, ["start_lower", "start_upper", "usable_start", "usable_end", "eol_endpoint"])
    c["source_split"] = c["split"]
    c["source_eol_endpoint"] = c["eol_endpoint"]
    c["source_right_censored"] = c["right_censored"]
    c["split"] = c.cycle_id.map(split_for)
    c["eol_endpoint"] = pd.NaT
    c["eol_strict_observed"] = 0
    c["eol_sensor_observed"] = 0
    c["right_censored"] = 1
    terminal = time_audit[time_audit["可作电极末期时间标签"].eq(1)]
    assert len(terminal) == 10 and terminal["周期"].is_unique
    for _, row in terminal.iterrows():
        m = c.cycle_id.eq(row["周期"])
        assert m.sum() == 1
        c.loc[m, "eol_endpoint"] = pd.to_datetime(row["连续第一件开始_新终点"], format=FMT)
        c.loc[m, "eol_strict_observed"] = 1
        c.loc[m, "right_censored"] = 0
        c.loc[m, "end_note"] = row["说明"]
    c.loc[c.cycle_id.eq("L012"), "end_type"] = "natural_eol"
    c.loc[c.cycle_id.eq("L012"), "end_evidence"] = "site_confirmed_2026_11_17_and_sensor"
    c.loc[c.cycle_id.eq("L013"), "left_censored"] = 0
    c.loc[c.cycle_id.eq("L013"), "note"] = (
        "11/17现场确认11/10约13点多更换；安装操作仅限定在13:34:43—14:03:01，"
        "自14:03:01之后开始本周期可用观察，非精确拆装秒。")
    c["censoring_semantics"] = "right_censored refers to unobserved natural voltage EOL, including preventive/damage replacements"
    assert c.groupby("split").eol_strict_observed.sum().to_dict() == {"test": 3, "train": 5, "validation": 2}
    assert c.groupby("split").size().to_dict() == {"test": 6, "train": 5, "validation": 3}
    return c


def rebuilt_events(products, cycles, time_audit):
    """Each >=2 consecutive failed-product run yields its FIRST product START."""
    rows = []
    causes = [("voltage", "voltage_fail"), ("feedrate", "feedrate_fail"), ("pressure", "hopper_pressure_fail")]
    lookup = {}
    for _, r in time_audit[time_audit["连续失败件数"].ge(2)].iterrows():
        cause = "pressure" if r["失败原因"] == "hopper_pressure" else r["失败原因"]
        key = (r["周期"], cause, pd.to_datetime(r["连续第一件开始_新终点"], format=FMT))
        lookup[key] = r
    for cycle in cycles.itertuples():
        q = products[products.cycle_id.eq(cycle.cycle_id)].sort_values("start")
        for cause, col in causes:
            runs, run = [], []
            for product in q.itertuples():
                good = product.supervised_product_eligible == 1 and getattr(product, col) == 1
                adjacent = not run or (product.start - run[-1].end).total_seconds() <= 1800
                if good and adjacent:
                    run.append(product)
                else:
                    if len(run) >= 2:
                        runs.append(run)
                    run = [product] if good else []
            if len(run) >= 2:
                runs.append(run)
            for run in runs:
                t = run[0].start
                matched = lookup.get((cycle.cycle_id, cause, t))
                is_eol = int(cause == "voltage" and pd.notna(cycle.eol_endpoint) and t == cycle.eol_endpoint)
                if matched is not None:
                    assert len(run) == int(matched["连续失败件数"])
                    assert is_eol == int(matched["可作电极末期时间标签"])
                rows.append(dict(event_id=matched["事件编号"] if matched is not None else f"RUN_{cause}_{run[0].product_id}",
                    cycle_id=cycle.cycle_id, split=cycle.split, time=t, cause=cause, is_eol=is_eol,
                    product_ids=";".join(r.product_id for r in run), run_length=len(run),
                    second_completion=run[1].end, last_completion=run[-1].end,
                    deadline_20m=t-pd.Timedelta(minutes=20), plc_batch_confirmed=0))
    events = pd.DataFrame(rows).sort_values(["time", "cause"]).reset_index(drop=True)
    assert len(events) == len(lookup) == 12, "Rebuilt failure runs differ from the latest reviewed event table"
    assert events.is_eol.sum() == 10
    for cycle in cycles[cycles.eol_strict_observed.eq(1)].itertuples():
        event = events[events.cycle_id.eq(cycle.cycle_id) & events.is_eol.eq(1)].iloc[0]
        run = products[products.product_id.isin(event.product_ids.split(";"))].sort_values("start")
        assert len(run) >= 2 and run.voltage_fail.eq(1).all()
        assert run.iloc[0].start == cycle.eol_endpoint
    return events


def observation_horizons(anchors, products, cycles, clock_path):
    """Future observation extent is supervision metadata, never a model input.

    An observation gap >2 seconds, unknown clock row, ineligible product or
    maintenance boundary terminates proof of an event-free future interval.
    Isolated missing seconds retain the source processing tolerance; they are
    not interpolated and do not split a reviewed complete-product label.
    """
    z = np.load(clock_path)
    ts, known = z["timestamp_seconds"], z["known"]
    assert np.all(np.diff(ts) > 0)
    missing_start = ts[:-1][np.diff(ts) > 2] + 1
    barriers_global = np.unique(np.r_[missing_start, ts[~known], ts[-1]+1])
    now = seconds(anchors.timestamp)
    horizon = np.zeros(len(anchors), dtype=np.int64)
    for cycle in cycles.itertuples():
        ix = np.flatnonzero(anchors.cycle_id.eq(cycle.cycle_id).to_numpy())
        if not len(ix):
            continue
        end = int(seconds([cycle.usable_end])[0])
        bad = seconds(products.loc[products.cycle_id.eq(cycle.cycle_id) & products.supervised_product_eligible.eq(0), "start"])
        barriers = np.unique(np.r_[barriers_global, bad, end])
        pos = np.searchsorted(barriers, now[ix], side="right")
        stop = barriers[np.minimum(pos, len(barriers)-1)]
        horizon[ix] = np.maximum(0, np.minimum(stop, end) - now[ix])
    return horizon, dict(clock_rows=len(ts), unknown_clock_rows=int((~known).sum()),
                        missing_intervals=int(len(missing_start)),
                        isolated_missing_seconds=int((np.diff(ts) == 2).sum()),
                        barrier_gap_rule="diff(timestamp_seconds)>2; same tolerance as source training_data.py; no interpolation")


def window_target(delta, horizon, minutes):
    window = minutes * 60
    observed = np.isfinite(delta) & (delta > 0) & (delta < horizon)
    return np.where(observed & (delta <= window), 1, np.where(horizon > window, 0, np.nan))


def construct(source=SOURCE, audit_path=AUDIT, clock_path=CLOCK):
    time_audit = pd.read_csv(audit_path)
    cycles = rebuilt_cycles(source, time_audit)
    products = pd.read_csv(source / "products.csv", low_memory=False)
    parse_dates(products, ["start", "end", "label_available_at"])
    products["source_split"] = products["split"]
    products["split"] = products.cycle_id.map(split_for)
    events = rebuilt_events(products, cycles, time_audit)
    features = pd.read_csv(source / "features_10s.csv.gz")
    features["timestamp"] = pd.to_datetime(features.timestamp, format=FMT)
    anchors = features[["anchor_id", "timestamp", "cycle_id"]].copy()
    anchors["split"] = anchors.cycle_id.map(split_for)
    for field in FIELDS:
        anchors[field] = features[field + "_last"]
    anchors["Voltage_mean60s"] = features["Voltage_mean60s"]
    anchors["Voltage_std60s"] = features["Voltage_std60s"]
    # Product lookup only validates benchmark opportunity and a causal 3 s start
    # confirmation latency; no full-product statistic/label enters model values.
    now = seconds(anchors.timestamp)
    keep = np.zeros(len(anchors), dtype=bool)
    for cycle, ix in anchors.groupby("cycle_id", sort=False).indices.items():
        q = products[products.cycle_id.eq(cycle)].sort_values("start")
        starts, ends = seconds(q.start), seconds(q.end)
        j = np.searchsorted(starts, now[ix], side="right") - 1
        safe = np.maximum(j, 0)
        keep[ix] = (j >= 0) & (now[ix] <= ends[safe]) & (now[ix] - starts[safe] >= 3)
        keep[ix] &= q.supervised_product_eligible.to_numpy()[safe].astype(bool)
    anchors = anchors.loc[keep].reset_index(drop=True)
    horizon, clock_info = observation_horizons(anchors, products, cycles, clock_path)
    anchors["observation_horizon_seconds"] = horizon
    endpoints = cycles.set_index("cycle_id").eol_endpoint
    terminal = pd.to_datetime(anchors.cycle_id.map(endpoints))
    remaining = (terminal - anchors.timestamp).dt.total_seconds()
    anchors["remaining_seconds"] = remaining.where(remaining > 0)
    targets = anchors[["anchor_id", "timestamp", "cycle_id", "split"]].copy()
    delta = anchors.remaining_seconds.to_numpy(float)
    after_terminal = terminal.notna() & anchors.timestamp.ge(terminal)
    for minutes in [30, 45, 60]:
        targets[f"y_eol_{minutes}m"] = window_target(delta, horizon, minutes)
    observed = np.isfinite(delta) & (delta > 0) & (delta < horizon)
    targets["y_pre20_40m"] = np.where(observed & (delta >= 1200) & (delta <= 2400), 1,
        np.where((observed & (delta < 1200)) | (horizon > 2400), 0, np.nan))
    target_cols = [c for c in targets if c.startswith("y_")]
    targets.loc[after_terminal, target_cols] = np.nan
    assert anchors.anchor_id.is_unique and anchors.timestamp.is_monotonic_increasing
    assert anchors.anchor_id.equals(targets.anchor_id)
    assert anchors.groupby("cycle_id").split.nunique().max() == 1
    assert len(anchors) == 81235
    assert targets.loc[after_terminal, target_cols].isna().all().all()
    assert not targets.loc[anchors.cycle_id.isin(["L007", "L010", "L011", "L013"]), target_cols].eq(1).any().any()
    summary = []
    for split in ["train", "validation", "test"]:
        p = products[products.split.eq(split)]
        c = cycles[cycles.split.eq(split)]
        eligible = p[p.supervised_product_eligible.eq(1)]
        summary.append(dict(split=split, cycles=len(c), products=len(p), eligible_products=len(eligible),
            production_hours=float(eligible.spray_samples.sum()/3600),
            failed_products=int(eligible.failure.eq(1).sum()), voltage_failed_products=int(eligible.voltage_fail.eq(1).sum()),
            feed_failed_products=int(eligible.feedrate_fail.eq(1).sum()), pressure_failed_products=int(eligible.hopper_pressure_fail.eq(1).sum()),
            eol_events=int(c.eol_strict_observed.sum()), censored_cycles=int(c.right_censored.sum()),
            feature_anchors=int(anchors.split.eq(split).sum())))
    counts = {s: {col: {"eligible": int(targets.loc[targets.split.eq(s), col].notna().sum()),
                         "positive": int(targets.loc[targets.split.eq(s), col].eq(1).sum())}
                  for col in target_cols} for s in ["train", "validation", "test"]}
    opportunities = []
    for event in events[events.is_eol.eq(1)].itertuples():
        a = anchors[anchors.cycle_id.eq(event.cycle_id) & anchors.timestamp.le(event.deadline_20m)].timestamp.max()
        opportunities.append(dict(event_id=event.event_id, cycle_id=event.cycle_id, time=event.time.isoformat(),
            deadline_20m=event.deadline_20m.isoformat(), last_existing_opportunity=a.isoformat(),
            opportunity_lead_minutes=(event.time-a).total_seconds()/60))
    metadata = dict(created_at=datetime.now().isoformat(), source_target_files_read=[],
        endpoint_policy="First product START of final consecutive voltage-failure run; 2026-11-17 audit",
        input_columns=FIELDS+["Voltage_mean60s", "Voltage_std60s"],
        never_input=["anchor_id", "timestamp", "cycle_id", "split", "observation_horizon_seconds", "remaining_seconds", "failure", "full_product_means", "future_maintenance", "Gun_ID", "Gun_Hours", "Ignitions"],
        source_feature_anchors=len(features), retained_anchors=len(anchors), start_confirmation_seconds=3,
        target_counts=counts, clock_audit=clock_info, event_opportunities=opportunities,
        split_policy="L000-L004 train; L005-L007 validation; L008-L013 test",
        historical_holdout_exposure="10/28 and 11/1 used by V1 validation; 11/10 also inspected previously; this is expanded historical holdout, not first-unseen prospective data",
        censoring="Natural EOL unobserved for L007,L010,L011,L013. Fully observed future windows may be negatives; insufficient future follow-up remains NaN. Post-EOL anchors remain available for replay but core targets are NaN.",
        replay_limit="Benchmark opportunities inherit offline complete-product/process eligibility and 3s start confirmation. They are not a full raw asynchronous PLC replay. No synthetic standby origins are added.",
        notes=["The original 60s trailing voltage statistics reset at maintenance/gaps/excluded intervals; no full-product voltage mean is used.",
               "Product failures preserve original site overrides, uncertain mask, full rotation/reversal segmentation and thresholds.",
               "No pressure or feed EOLs are relabeled as electrode failures; pressure run is a separate diagnostic event.",
               "Calendar timestamps are device local wall-clock; Working Hours is not a countdown."])
    return anchors, targets, products, cycles, events, pd.DataFrame(summary), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--time-audit", type=Path, default=AUDIT)
    parser.add_argument("--clock", type=Path, default=CLOCK)
    args = parser.parse_args()
    output, artifacts = ROOT/"data", ROOT/"artifacts"
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    anchors, targets, products, cycles, events, summary, metadata = construct(args.source, args.time_audit, args.clock)
    for name, frame in [("anchors.csv.gz", anchors), ("targets.csv.gz", targets), ("products.csv.gz", products),
                        ("cycles.csv", cycles), ("events.csv", events), ("split_summary.csv", summary)]:
        frame.to_csv(output/name, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    source_files = [args.source/"features_10s.csv.gz", args.source/"products.csv", args.source/"electrode_cycles.csv",
                    args.time_audit, args.clock]
    metadata["source_files"] = []
    for path in source_files:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        metadata["source_files"].append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest))
    metadata["output_files"] = {}
    for path in sorted(output.glob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                metadata["output_files"][path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    (artifacts/"data_audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"target_counts": metadata["target_counts"], "data_path": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
