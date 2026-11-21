"""Compile the read-only electrode-life research report shown in the training workbench.

Nothing is trained or re-scored here. The script only gathers numbers that the v3-v6
research runs already wrote (bundle workbench files, report.json, held-out predictions)
into one compact JSON file served as a static asset, so the workbench can show the
research candidates without mounting any of them as a live model.

    python scripts/export_life_research.py --root <.tmp folder holding model-v3..v6> \
        --out simulation/pipeline/static/research/life-research.json
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

POINTS = 90  # same picks as the v4 trajectory export; every second one is kept
SHORT_HOURS = 6.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def r(value, digits=3):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def life(row, last2h, last1h, coverage):
    return dict(mae=r(row["mae"]), last2h=r(row[last2h]), last1h=r(row[last1h]), bias2h=r(row.get("bias2h", row.get("bias_2h", np.nan))),
                coverage=r(row[coverage]), width=r(row["width"], 2))


def alarms(row):
    hours = float(row["operating_hours"])
    return dict(events=int(row["total_events"]), hit=int(row["hit20"]), late=int(row["late"]), missed=int(row["missed"]),
                false=int(row["false_alarms"]), hours=r(hours, 1), false_per_100h=r(100 * row["false_alarms"] / hours, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    paths = dict(
        v3=root / "model-v3-20261118/model_v3/bundle/workbench.json",
        v4=root / "model-v4-20261119/model_v4/bundle/workbench.json",
        v5=root / "model-v5-20261120/run1/report.json",
        v6=root / "model-v6-20261120/run1/report.json",
        v3_predictions=root / "model-v5-20261120/data/v3_predictions.csv.gz",
        v5_predictions=root / "model-v5-20261120/run1/predictions.npz",
        v6_predictions=root / "model-v6-20261120/run1/predictions.npz",
        frame=root / "model-v5-20261120/data/frame.parquet",
    )
    v3, v4, v5, v6 = (load(paths[k]) for k in ("v3", "v4", "v5", "v6"))
    frame = pd.read_parquet(paths["frame"], columns=["anchor_id", "cycle_id", "age_h", "rul_h", "event", "life_h"])

    models = []
    # v3: one hazard-layer model, metrics as stored with the bundle
    row = v3["evaluation"]["life"]["v3"]
    models.append(dict(id="v3:hazard", gen="v3", key="hazard", loco=life(row, "mae_last2h", "mae_last1h", "cover"), forward=None,
                       alarms=dict(loco=alarms(v3["evaluation"]["events"]), forward=None),
                       cycles={c["cycle"]: r(c["life_mae"]) for c in v3["evaluation"]["cycles"] if c["observed"]}))
    for key in ("life_tree", "life_gru", "life_fusion"):
        ev = v4["evaluation"][key]
        models.append(dict(id=f"v4:{key}", gen="v4", key=key, loco=life(ev["life"]["model"], "mae_2h", "mae_1h", "cover"), forward=None,
                           alarms=dict(loco=alarms(ev["events"]), forward=None),
                           cycles={c["cycle"]: r(c["life_mae"]) for c in ev["cycles"] if c["observed"]}))
    for gen, report in (("v5", v5), ("v6", v6)):
        for key, entry in report["comparison"]["loco"].items():
            forward = report["comparison"]["forward"][key]
            models.append(dict(
                id=f"{gen}:{key}", gen=gen, key=key, loco=life(entry["life"], "last2h", "last1h", "coverage"),
                forward=life(forward["life"], "last2h", "last1h", "coverage"),
                alarms=dict(loco=alarms(entry["alarms"]), forward=alarms(forward["alarms"])) if "alarms" in entry else None,
                cycles={c["cycle"]: r(c["mae"]) for c in entry["cycles"] if c["observed"]}))
    base = v3["evaluation"]["life"]["constant_quantiles"]
    baseline = dict(mae=r(base["mae"]), last2h=r(base["mae_last2h"]), coverage=r(base["cover"]), width=r(base["width"], 2))

    # held-out (leave-one-cycle-out) trajectories on the ten observed cycles
    quantiles = {}
    p3 = pd.read_csv(paths["v3_predictions"], usecols=["anchor_id", "q10", "q50", "q90"]).set_index("anchor_id").reindex(frame.anchor_id)
    if p3.q50.isna().any():
        raise SystemExit("v3 predictions do not cover every origin")
    quantiles["v3:hazard"] = p3[["q10", "q50", "q90"]].to_numpy()
    for gen in ("v5", "v6"):
        with np.load(paths[f"{gen}_predictions"]) as z:
            for name in z.files:
                if name.endswith("_quant"):
                    if len(z[name]) != len(frame):
                        raise SystemExit(f"{gen} predictions are not aligned with the origin table")
                    quantiles[f"{gen}:{name[:-6]}"] = z[name]
    # the stored table is re-derived from the arrays before anything is exported
    seen = frame.rul_h.notna() & frame.event.eq(1)
    for m in models:
        if m["gen"] in ("v5", "v6"):
            error = pd.Series(np.abs(quantiles[m["id"]][:, 1] - frame.rul_h.to_numpy()))[seen.to_numpy()]
            mae = float(error.groupby(frame.cycle_id[seen].to_numpy()).mean().mean())
            if abs(mae - m["loco"]["mae"]) > 2e-3:
                raise SystemExit(f"{m['id']}: recomputed MAE {mae:.4f} differs from the report {m['loco']['mae']}")
    cycles, lines = [], {key: {} for key in quantiles}
    lines.update({f"v4:{key}": {} for key in ("life_tree", "life_gru", "life_fusion")})
    v4_traj = {t["cycle"]: t for t in v4["trajectories"]}
    for cycle, q in frame.groupby("cycle_id", sort=False):
        observed = bool(q.event.iloc[0])
        hours = float(q.age_h.max())
        cycles.append(dict(id=cycle, observed=observed, hours=r(hours, 2), short=bool(observed and hours < SHORT_HOURS), origins=int(len(q))))
        q = q[q.rul_h.notna()]
        if not observed or not len(q):
            continue
        pick = q.index[np.unique(np.linspace(0, len(q) - 1, POINTS).round().astype(int))]
        keep = np.r_[np.arange(0, len(pick) - 1, 2), len(pick) - 1]
        actual = [round(float(v), 3) for v in frame.rul_h[pick]]
        cycles[-1]["actual"] = [round(actual[i], 2) for i in keep]
        cycles[-1]["life_hours"] = r(q.life_h.iloc[0], 2)
        for key, values in quantiles.items():
            lines[key][cycle] = [[round(float(v), 2) for v in values[pick[keep], j]] for j in range(3)]
        t = v4_traj.get(cycle)
        if t is not None and len(t["actual"]) == len(actual) and np.allclose(t["actual"], actual, atol=2e-3):
            for key, block in t["models"].items():
                lines[f"v4:{key}"][cycle] = [[round(float(block[name][i]), 2) for i in keep] for name in ("q10", "q50", "q90")]
    for m in models:
        m["trajectory"] = bool(lines.get(m["id"]))

    warning = [dict(id=key, loco=alarms(v6["warning"]["loco"][key]["totals"]), forward=alarms(v6["warning"]["forward"][key]["totals"]))
               for key in v6["warning"]["loco"]]
    folds = {gen: [dict(tag=run["tag"], held_out=run["held_out"], selected=run["selected"]) for run in report["runs"]]
             for gen, report in (("v5", v5), ("v6", v6))}
    record = dict(
        kind="life_research", generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset=dict(origins=int(len(frame)), cycles=len(cycles), observed=sum(c["observed"] for c in cycles), origin_seconds=10),
        protocol=dict(loco_folds=sum(run["tag"].startswith("loco") for run in v6["runs"]), forward_blocks=v6["protocol"]["forward_blocks"],
                      nominal_interval=0.8, false_budget_per_hour=v6["protocol"]["alarm_false_budget_per_hour"],
                      versions=dict(v5=v5["protocol"]["version"], v6=v6["protocol"]["version"])),
        cycles=cycles, models=models, baseline=baseline, folds=folds, warning=warning,
        # docs/model-v6-evaluation.md: exploratory per-cycle bootstrap of v6 nested minus v5 nested
        bootstrap=dict(loco=dict(difference=-0.561, interval=[-1.70, 0.95]), forward=dict(difference=0.502, interval=[-1.01, 2.93])),
        trajectories={key: value for key, value in lines.items() if value},
        sources={key: dict(file=str(path.relative_to(root)).replace("\\", "/"), sha256=sha(path)) for key, path in paths.items()},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"{args.out}: {args.out.stat().st_size / 1024:.0f} KB, {len(models)} models, {len(record['trajectories'])} with trajectories")


if __name__ == "__main__":
    main()
