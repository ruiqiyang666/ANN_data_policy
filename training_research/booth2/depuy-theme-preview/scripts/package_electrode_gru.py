"""Export the frozen Booth2 electrode-warning GRU (ANN_data_policy model_v2) as a panel bundle.

    python -I scripts/package_electrode_gru.py --source <ANN_data_policy checkout> --output <new folder>

The source checkout is read as data only: no module of it is imported and the
checkpoint is opened with weights_only. The panel's own NumPy implementation is
then run over every prepared origin and must reproduce the frozen score ledger
and the frozen alarm list before anything is written.
"""

import torch  # noqa: I001 -- native libraries must initialise before NumPy/pandas on Windows

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation" / "pipeline"))
from inference.electrode import (  # noqa: E402
    FIELDS, TARGETS, encode, gru_forward, raw_values, replay_alarms, sha256,
)

SELECTED = "gru_16"
SEGMENT = dict(name="electrode_eol_20261028", label="10月28日 07:30–08:32 电极末期",
               start="2026-10-28 07:30:00", stop="2026-10-28 08:32:00")
LIMITS = [
    "实验模型：历史测试 3 次电极末期事件中 2 次提前至少 20 分钟，不能作为自动换枪依据。",
    "评分是未校准的风险分数，不是概率，也不是剩余分钟数。",
    "加工原点沿用离线确认的产品资格，不等同于现场异步数据全链路。",
    "回放时段属于该模型的历史测试集，此前已被查看，不是新增盲测。",
]


def epoch(series):
    return pd.DatetimeIndex(pd.to_datetime(series)).as_unit("s").asi8


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, kind=float):
    return None if value in ("", None) or (isinstance(value, float) and not np.isfinite(value)) else kind(value)


def clean(value):
    """JSON-safe copy: NaN / inf become null, NumPy scalars become Python numbers."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v2, art = args.source / "model_v2_GRU", args.source / "model_v2_GRU" / "artifacts"
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty folder")
    args.output.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((art / "frozen_config.json").read_text(encoding="utf-8"))
    if frozen["selected_model"] != SELECTED or not frozen.get("selection_final"):
        raise SystemExit("The frozen selection is not the expected GRU-16")
    for name in ("models/gru_16/best.pt", "preprocessor.json"):
        expected = frozen["fitted_file_sha256"].get(name.replace("/", "\\"), frozen["fitted_file_sha256"].get(name))
        if name == "preprocessor.json":
            expected = frozen["preprocessor_sha256"]
        if expected is None or sha256(art / name) != expected:
            raise SystemExit(f"{name} differs from the frozen record")
    policy = frozen["policies"][SELECTED]
    pre = json.loads((art / "preprocessor.json").read_text(encoding="utf-8"))

    state = torch.load(art / "models/gru_16/best.pt", map_location="cpu", weights_only=True)["state_dict"]
    weights = {
        "weight_ih": state["body.weight_ih_l0"].numpy(), "weight_hh": state["body.weight_hh_l0"].numpy(),
        "bias_ih": state["body.bias_ih_l0"].numpy(), "bias_hh": state["body.bias_hh_l0"].numpy(),
        "head_weight": state["head.weight"].numpy(), "head_bias": state["head.bias"].numpy(),
    }
    weights = {k: np.ascontiguousarray(v, dtype="float32") for k, v in weights.items()}

    anchors = pd.read_csv(v2 / "data/anchors.csv.gz")
    ledger = pd.read_csv(art / "predictions.csv.gz")
    if not anchors.anchor_id.equals(ledger.anchor_id):
        raise SystemExit("Origins and score ledger are not aligned")
    cycle_ids = list(dict.fromkeys(anchors.cycle_id))
    times = epoch(anchors.timestamp)
    cycles = anchors.cycle_id.map({c: i for i, c in enumerate(cycle_ids)}).to_numpy("int16")
    values = anchors[FIELDS + ["Voltage_mean60s", "Voltage_std60s"]].to_numpy("float32")
    scores = ledger[SELECTED].to_numpy("float64")

    # 1. This implementation must equal the frozen ledger on every origin.
    values13 = raw_values(values)
    mine = np.empty((len(times), len(TARGETS)))
    for begin in range(0, len(times), 2048):
        index = list(range(begin, min(begin + 2048, len(times))))
        mine[index] = gru_forward(weights, encode(times, values13, cycles, index, pre))[0]
    difference = float(np.abs(mine[:, policy["head_index"]] - scores).max())
    if difference > 1e-4:
        raise SystemExit(f"Score mismatch against the frozen ledger: {difference}")

    # 2. The alarm replay must emit exactly the frozen alarms.
    emitted, _, _, _ = replay_alarms(times, scores, cycles, policy["threshold"], frozen["alarm_policy"])
    recorded = [r for r in rows(art / "alarm_results.csv") if r["model"] == SELECTED]
    if sorted(epoch([r["time"] for r in recorded]).tolist()) != times[emitted].tolist():
        raise SystemExit("Alarm replay differs from the frozen alarm list")

    np.savez(args.output / "weights.npz", **weights)
    np.savez_compressed(args.output / "anchors.npz", t=times, cycle=cycles, values=values,
                        ledger_score=scores.astype("float64"))
    (args.output / "preprocessor.json").write_bytes((art / "preprocessor.json").read_bytes())

    # ---- workbench summary -------------------------------------------------
    evaluation = json.loads((art / "evaluation.json").read_text(encoding="utf-8"))
    audit = json.loads((art / "data_audit.json").read_text(encoding="utf-8"))
    plan = json.loads((art / "experiment_plan.json").read_text(encoding="utf-8"))
    products = pd.read_csv(args.source / "training_data/products.csv", low_memory=False,
                           usecols=["failure", "supervised_product_eligible", "voltage_fail",
                                    "feedrate_fail", "hopper_pressure_fail"])
    keep = ("total_events", "hit20", "late", "missed", "false_alarms", "post_event_alarms", "alarms",
            "operating_hours", "threshold", "window_average_precision", "false_alarms_per_hour",
            "mean_excess_minutes")
    candidates = []
    for spec in frozen["model_specs"]:
        name = spec["name"]
        candidates.append(dict(
            name=name, kind=spec["kind"], hidden=spec.get("hidden"), C=spec.get("C"),
            max_depth=spec.get("max_depth"), parameters=spec.get("parameters"),
            best_epoch=spec.get("best_epoch"), epochs_run=spec.get("epochs_run"),
            fit_seconds=spec.get("fit_seconds"), head=frozen["policies"][name]["head"],
            threshold=frozen["policies"][name]["threshold"],
            alarm_enabled=frozen["policies"][name]["threshold"] <= 1,
            extension=name not in [s["name"] for s in plan["model_specs"]],
            selected=name == SELECTED,
            splits={split: {k: metrics.get(k) for k in keep}
                    for split, metrics in evaluation["models"][name].items()},
        ))
    curves = {c["name"]: json.loads((art / "models" / c["name"] / "history.json").read_text(encoding="utf-8"))
              for c in candidates if c["kind"] in ("gru", "tcn")}
    event_rows = [r for r in rows(art / "event_results.csv") if r["model"] == SELECTED]
    events = []
    for r in rows(v2 / "data/events.csv"):
        scored = next((e for e in event_rows if e["event_id"] == r["event_id"]), None)
        events.append(dict(
            id=r["event_id"], cycle=r["cycle_id"], split=r["split"], time=r["time"], cause=r["cause"],
            is_eol=r["is_eol"] == "1", run_length=int(r["run_length"]), deadline=r["deadline_20m"],
            first_alarm=(scored or {}).get("first_alarm") or None,
            lead_minutes=number((scored or {}).get("lead_minutes")),
            hit20=(scored or {}).get("hit20") == "True", late=(scored or {}).get("late") == "True",
            missed=(scored or {}).get("missed") == "True", scored=scored is not None,
            max_lead_minutes=number((scored or {}).get("maximum_observable_lead_minutes")),
        ))
    alarms = [dict(split=r["split"], time=r["time"], cycle=r["cycle_id"], score=float(r["probability"]),
                   event=r["event_id"] or None, false_alarm=r["false_alarm"] == "True",
                   post_event=r["post_event"] == "True") for r in recorded]
    cycle_rows = []
    for r in rows(v2 / "data/cycles.csv"):
        cycle_rows.append(dict(
            id=r["cycle_id"], split=r["split"], start=r["usable_start"], end=r["usable_end"],
            end_type=r["end_type"], eol=r["eol_endpoint"] or None, products=int(r["product_count"]),
            eligible=int(r["eligible_products"]), failures=int(r["failure_products"]),
            voltage_failures=int(r["voltage_failed_products"]), feed_failures=int(r["feed_failed_products"]),
            pressure_failures=int(r["pressure_failed_products"]), on_hours=number(r["observed_on_hours"]),
            origins=int((anchors.cycle_id == r["cycle_id"]).sum()),
        ))
    cases = []
    stamp = pd.to_datetime(anchors.timestamp)
    for event in events:
        if not (event["is_eol"] and event["scored"]):
            continue
        at = pd.Timestamp(event["time"])
        window = ((anchors.cycle_id == event["cycle"]) & (stamp >= at - pd.Timedelta(minutes=90))
                  & (stamp <= at + pd.Timedelta(minutes=8))).to_numpy()
        cases.append(dict(
            event=event["id"], cycle=event["cycle"], split=event["split"],
            t=[str(v) for v in anchors.timestamp[window]],
            score=np.round(scores[window], 4).tolist(),
            voltage=np.round(anchors.Voltage_mean60s[window].to_numpy("float64"), 3).tolist(),
            alarms=[a["time"] for a in alarms if a["cycle"] == event["cycle"]
                    and at - pd.Timedelta(minutes=90) <= pd.Timestamp(a["time"]) <= at + pd.Timedelta(minutes=8)],
        ))
    commit = subprocess.run(["git", "-C", str(args.source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    source = dict(repository="ruiqiyang666/ANN_data_policy", commit=commit, folder="model_v2_GRU",
                  frozen_at_utc=frozen["frozen_at_utc"], frozen_config_sha256=sha256(art / "frozen_config.json"))
    workbench = clean(dict(
        source=source, model_version=frozen["model_version"], selected=SELECTED,
        data=dict(
            range=[str(stamp.min()), str(stamp.max())],
            products=dict(candidates=int(len(products)), normal=int((products.failure == 0).sum()),
                          failed=int((products.failure == 1).sum()), excluded=int(products.failure.isna().sum()),
                          eligible=int((products.supervised_product_eligible == 1).sum()),
                          voltage_failed=int((products.voltage_fail == 1).sum()),
                          feed_failed=int((products.feedrate_fail == 1).sum()),
                          pressure_failed=int((products.hopper_pressure_fail == 1).sum())),
            rules=[dict(key="voltage", limit="< 51 V"), dict(key="feedrate", limit="< 9.1 或 > 11.1 g/min"),
                   dict(key="pressure", limit="< 100 或 > 250 mbar")],
            splits=[{k: number(v) if k != "split" else v for k, v in r.items()} for r in rows(v2 / "data/split_summary.csv")],
            cycles=cycle_rows, events=events,
            origins=dict(source=audit["source_feature_anchors"], retained=audit["retained_anchors"]),
            targets=audit["target_counts"], clock=audit["clock_audit"],
            inputs=FIELDS + ["Voltage_mean60s", "Voltage_std60s"], never_input=audit["never_input"],
        ),
        protocol=dict(history_minutes=plan["history_minutes"], max_epochs=plan["max_epochs"],
                      patience=plan["patience"], batch_size=plan["batch_size"],
                      learning_rate=plan["learning_rate"], weight_decay=plan["weight_decay"],
                      threshold_grid=plan["threshold_grid"],
                      false_budget_per_hour=plan["false_budget_per_production_hour"],
                      alarm_policy=frozen["alarm_policy"], seed=plan["seed"]),
        candidates=candidates, curves=curves, alarms=alarms, cases=cases,
        verification=dict(origins=int(len(times)), max_abs_score_difference=difference,
                          alarms_reproduced=int(emitted.sum())),
    ))
    (args.output / "workbench.json").write_text(json.dumps(workbench, ensure_ascii=False), encoding="utf-8")

    selected = next(c for c in candidates if c["selected"])
    segment = dict(SEGMENT, start=int(epoch([SEGMENT["start"]])[0]), stop=int(epoch([SEGMENT["stop"]])[0]))
    files = {name: sha256(args.output / name) for name in ("weights.npz", "anchors.npz", "preprocessor.json", "workbench.json")}
    manifest = clean(dict(
        version="booth2-v2-gru16-" + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12],
        model_version=frozen["model_version"], source=source, files=files,
        threshold=policy["threshold"], head_index=policy["head_index"], alarm_policy=frozen["alarm_policy"],
        cycle_ids=cycle_ids, showcase_segment=segment, limits=LIMITS,
        evaluation=dict(splits=selected["splits"], events=[e for e in events if e["scored"]]),
        verification=workbench["verification"],
    ))
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(version=manifest["version"], **workbench["verification"],
                          cases=len(cases), candidates=len(candidates)), ensure_ascii=False))


if __name__ == "__main__":
    main()
