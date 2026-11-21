"""Package the held-out predictions of the v5 / v6 research candidates for the realtime replay.

Nothing is trained or re-scored. The v5 and v6 runs already wrote, for every processing origin,
the prediction of the fold model that never saw that origin's electrode cycle (leave one cycle
out). This script cuts those ledgers down to the recorded mornings the realtime page replays and
adds what the page needs beside the curve: the alarm state of the frozen alarm policy, the inputs
of each model with their distance from the training median, and the reported evaluation.

The realtime page therefore shows recorded held-out predictions of research candidates. It does
not run their weights (the v6 weights are joblib files for trusted local research only), and none
of these models is a released model.

Checks before anything is written:
  * the origins of the v5 frame are the origins of the v4 bundle, one for one;
  * the v5 ledger equals the predictions stored fold by fold; the reported errors are copied from the reports;
  * v5 alarm thresholds, re-selected per fold with the frozen evaluation code on the fold's inner
    predictions, reproduce the reported alarm totals; the v6 warning head reproduces its totals.

    python scripts/package_life_ledger.py --root <.tmp folder holding model-v4..v6> --out <bundle folder>
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation" / "pipeline"))
from inference.electrode import replay_alarms  # noqa: E402

H = [1 / 3, 2 / 3, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24.0]
ONE_HOUR = 2
SMALL = ["margin51_30m", "d_peak_30m", "log_vsd_30m", "v_slope_30m", "vsd_ratio", "age_h"]
# id, generation, source name, feature set, family, zh name, en name, zh short, en short, zh description, en description
MODELS = [
    ("v5_aft_linear6", "v5", "aft_linear6", "small", "linear", "线性 AFT · 6 特征", "Linear AFT · 6 features", "线性 AFT v5", "Linear AFT v5",
     "加速失效时间线性模型：6 项特征 → 剩余寿命分布", "Accelerated-failure-time linear model: 6 features → remaining-life distribution"),
    ("v5_aft_xgb6", "v5", "aft_xgb6", "small", "trees", "XGBoost AFT · 6 特征", "XGBoost AFT · 6 features", "XGB AFT 6 v5", "XGB AFT 6 v5",
     "提升树加速失效时间模型：6 项特征", "Boosted-tree accelerated-failure-time model: 6 features"),
    ("v5_aft_xgb23", "v5", "aft_xgb23", "end", "trees", "XGBoost AFT · 23 特征", "XGBoost AFT · 23 features", "XGB AFT 23 v5", "XGB AFT 23 v5",
     "提升树加速失效时间模型：23 项特征", "Boosted-tree accelerated-failure-time model: 23 features"),
    ("v5_pch_linear6", "v5", "pch_linear6", "small", "linear", "分段常数风险 · 线性 6 特征", "Piecewise hazard · linear, 6 features", "分段风险线性 v5", "PCH linear v5",
     "分段常数风险的线性模型：6 项特征", "Piecewise-constant hazard, linear in 6 features"),
    ("v5_pch_mlp6", "v5", "pch_mlp6", "small", "network", "分段常数风险网络 · 6 特征", "Piecewise hazard network · 6 features", "分段风险网络 6 v5", "PCH net 6 v5",
     "小型网络输出各时间段风险：6 项特征", "Small network giving the hazard per time bin: 6 features"),
    ("v5_pch_mlp23", "v5", "pch_mlp23", "end", "network", "分段常数风险网络 · 23 特征", "Piecewise hazard network · 23 features", "分段风险网络 23 v5", "PCH net 23 v5",
     "小型网络输出各时间段风险：23 项特征", "Small network giving the hazard per time bin: 23 features"),
    ("v6_ipcw_hgb6", "v6", "ipcw_hgb6", "small", "trees", "IPCW 分位数树 · 6 特征", "IPCW quantile trees · 6 features", "分位数树 6 v6", "Quantile trees 6 v6",
     "删失加权的分位数提升树：6 项特征", "Censoring-weighted quantile boosted trees: 6 features"),
    ("v6_ipcw_hgb32", "v6", "ipcw_hgb32", "phys", "trees", "IPCW 分位数树 · 32 特征", "IPCW quantile trees · 32 features", "分位数树 32 v6", "Quantile trees 32 v6",
     "删失加权的分位数提升树：32 项特征", "Censoring-weighted quantile boosted trees: 32 features"),
    ("v6_cqrnn6", "v6", "cqrnn6", "small", "network", "删失分位数网络 · 6 特征", "Censored quantile network · 6 features", "CQRNN 6 v6", "CQRNN 6 v6",
     "删失分位数回归网络：6 项特征 → 19 个分位数", "Censored quantile regression network: 6 features → 19 quantiles"),
    ("v6_cqrnn32", "v6", "cqrnn32", "phys", "network", "删失分位数网络 · 32 特征", "Censored quantile network · 32 features", "CQRNN 32 v6", "CQRNN 32 v6",
     "删失分位数回归网络：32 项特征 → 19 个分位数", "Censored quantile regression network: 32 features → 19 quantiles"),
    ("v6_mix6", "v6", "mix6", "small", "network", "对数正态混合网络 · 6 特征", "Lognormal mixture network · 6 features", "混合网络 6 v6", "Mixture net 6 v6",
     "两分量对数正态混合的小型网络：6 项特征", "Small network with a two-component lognormal mixture: 6 features"),
    ("v6_mix32", "v6", "mix32", "phys", "network", "对数正态混合网络 · 32 特征", "Lognormal mixture network · 32 features", "混合网络 32 v6", "Mixture net 32 v6",
     "两分量对数正态混合的小型网络：32 项特征", "Small network with a two-component lognormal mixture: 32 features"),
    ("v6_rsf32", "v6", "rsf32", "phys", "trees", "随机生存森林 · 32 特征", "Random survival forest · 32 features", "生存森林 v6", "Survival forest v6",
     "随机生存森林：32 项特征", "Random survival forest: 32 features"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_evaluation(path):
    spec = importlib.util.spec_from_file_location("frozen_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.root, args.out
    v4 = root / "model-v4-20261119/model_v4/bundle"
    v5, v6 = root / "model-v5-20261120/run1", root / "model-v6-20261120/run1"
    data = root / "model-v5-20261120/data"
    manifest4 = load(v4 / "manifest.json")
    policy, cycle_ids = dict(manifest4["alarm_policy"]), list(manifest4["cycle_ids"])
    assert [float(h) for h in manifest4["horizons_hours"]] == [float(h) for h in H]

    frame = pd.read_parquet(data / "frame.parquet")
    with np.load(v4 / "origins.npz") as stored:
        times, cycles = np.asarray(stored["t"], dtype="int64"), np.asarray(stored["cycle"], dtype="int16")
    stamps = pd.DatetimeIndex(frame.timestamp).as_unit("s").asi8
    assert np.array_equal(stamps, times), "the v5 frame and the v4 bundle must list the same origins"
    assert np.array_equal(frame.cycle_id.map({c: i for i, c in enumerate(cycle_ids)}).to_numpy(), cycles)
    if "log_vsd_30m" not in frame:
        frame["log_vsd_30m"] = np.log(np.clip(frame.vsd_mean_30m.to_numpy(dtype="float64"), 0.02, None))
    sets = dict(small=SMALL, end=list(load(v5 / "final/aft_xgb23.json")["features"]), phys=list(load(v6 / "final/mix32.json")["features"]))
    assert sets["small"] == load(v5 / "final/aft_linear6.json")["features"] == load(v6 / "final/cqrnn6.json")["features"]
    assert len(sets["end"]) == 23 and len(sets["phys"]) == 32

    report5, report6 = load(v5 / "report.json"), load(v6 / "report.json")
    ev = frozen_evaluation(data / "evaluation.py")
    events = pd.read_csv(data / "events.csv", parse_dates=["time"])
    products = pd.read_csv(data / "products.csv", low_memory=False)
    remaining = frame.remaining_seconds.to_numpy(dtype="float64") / 3600 if "remaining_seconds" in frame else None

    def hours(names):
        q = products[products.cycle_id.isin(names) & products.supervised_product_eligible.eq(1)]
        return float(q.spray_samples.sum() / 3600)

    # The replayed mornings, each with the origins that can be current during it.
    segments = [dict(s) for s in manifest4["showcase_segments"]]
    keep = np.zeros(len(times), dtype=bool)
    for s in segments:
        keep |= (times >= s["start"] - 120) & (times <= s["stop"])
    at = np.flatnonzero(keep)
    arrays = dict(t=times[at], cycle=cycles[at], age_h=frame.age_h.to_numpy(dtype="float32")[at])

    # Inputs per feature set: value, and distance from the median of the cycles the fold model was fitted on.
    for key, names in sets.items():
        x = frame[names].to_numpy(dtype="float64")
        z = np.zeros_like(x)
        for k in np.unique(cycles[at]):
            train = x[cycles != k]
            median = np.nanmedian(train, axis=0)
            spread = np.nanpercentile(train, 75, axis=0) - np.nanpercentile(train, 25, axis=0)
            spread = np.where(spread > 1e-9, spread, 1.0)
            rows = cycles == k
            z[rows] = np.clip((np.where(np.isfinite(x[rows]), x[rows], median) - median) / spread, -8, 8)
        arrays[f"x_{key}"], arrays[f"z_{key}"] = x[at].astype("float32"), z[at].astype("float32")

    folds5 = {r["held_out"][0]: r for r in report5["runs"] if r["tag"].startswith("loco")}
    folds6 = {r["held_out"][0]: r for r in report6["runs"] if r["tag"].startswith("loco")}
    assert sorted(folds5) == sorted(folds6) == sorted(cycle_ids)

    # v6: one separate warning head, selected per fold; its scores and thresholds are in the run.
    warn6 = report6["warning"]["loco"]
    score6, threshold6, picked6 = np.zeros(len(times)), np.zeros(len(times)), {}
    for index, cycle in enumerate(cycle_ids):
        entry = warn6["nested_selected"]["folds"][index]
        heads = [name for name in ("warn_logit6", "warn_hgb32") if warn6[name]["folds"][index] == entry]
        assert heads, f"the selected v6 warning head of {cycle} is not identifiable"
        with np.load(v6 / "folds" / f"loco-{cycle}" / "predictions.npz", allow_pickle=False) as a:
            test = a["test_index"]
            assert set(cycles[test]) == {index}
            score6[test], threshold6[test], picked6[cycle] = a[f"{heads[0]}_score"], float(entry["threshold"]), heads[0]

    def replay(score, threshold):
        """The frozen alarm policy over every origin, cycle by cycle (thresholds differ per fold)."""
        emitted, active = np.zeros(len(times), dtype=bool), np.zeros(len(times), dtype=bool)
        confirmations, last = np.zeros(len(times), dtype="int16"), np.full(len(times), -1, dtype="int64")
        for k in range(len(cycle_ids)):
            rows = np.flatnonzero(cycles == k)
            emitted[rows], active[rows], confirmations[rows], last[rows] = replay_alarms(
                times[rows], score[rows], cycles[rows], float(threshold[rows[0]]), policy)
        return emitted, active, confirmations, last

    state6 = replay(score6, threshold6)
    totals6 = warn6["nested_selected"]["totals"]
    emitted_reported6 = sum(int(f["test"]["alarms"]) for f in warn6["nested_selected"]["folds"])
    assert int(state6[0].sum()) == emitted_reported6, (int(state6[0].sum()), emitted_reported6)

    with np.load(v5 / "predictions.npz", allow_pickle=False) as p5, np.load(v6 / "predictions.npz", allow_pickle=False) as p6:
        ledgers = dict(v5=p5, v6=p6)
        arrays["base"] = p6["km_landmark_curve"][at].astype("float32")  # the historical base rate: Kaplan-Meier by landmark
        specs, checks = [], {}
        for model_id, generation, name, feature_set, family, zh, en, short, short_en, describe, describe_en in MODELS:
            ledger = ledgers[generation]
            curve, quant = np.asarray(ledger[f"{name}_curve"], dtype="float64"), np.asarray(ledger[f"{name}_quant"], dtype="float64")
            reported = (report5 if generation == "v5" else report6)["comparison"]["loco"][name]
            # Cycle-macro MAE of the median against the observed remaining hours, as the reports define it.
            rows = {row["cycle"]: row for row in reported["cycles"]}
            life = reported["life"]
            if generation == "v5":
                score, threshold = curve[:, ONE_HOUR].copy(), np.zeros(len(times))
                totals = dict(total_events=0, hit20=0, late=0, missed=0, false_alarms=0, operating_hours=0.0)
                for cycle, run in folds5.items():
                    with np.load(v5 / "folds" / f"loco-{cycle}" / "predictions.npz", allow_pickle=False) as a:
                        train, test = a["train_index"], a["test_index"]
                        chosen = ev.select_threshold(
                            frame.loc[train].assign(split="validation"), a[f"{name}_inner_curve"][:, ONE_HOUR],
                            events[events.cycle_id.isin(run["training"])].assign(split="validation"), hours(run["training"]),
                            grid=np.array(report5["protocol"]["alarm_threshold_grid"]), false_budget=0.05)
                        threshold[test] = float(chosen["threshold"])
                        metrics, _, _ = ev.evaluate(frame.loc[test], a[f"{name}_curve"][:, ONE_HOUR], float(chosen["threshold"]),
                                                    events[events.cycle_id.isin(run["held_out"])], hours(run["held_out"]))
                        assert np.allclose(a[f"{name}_curve"], curve[test], atol=1e-6)
                    for key in totals:
                        totals[key] += float(metrics[key])
                for key in ("total_events", "hit20", "late", "missed", "false_alarms"):
                    assert int(totals[key]) == int(reported["alarms"][key]), (name, key, totals[key], reported["alarms"][key])
                state = replay(score, threshold)
                warning = dict(source="own_curve_1h", note="预警评分为模型自身的 1 小时内到寿概率，阈值按折在内层留出预测上选取")
                events_summary = {key: (float(totals[key]) if key == "operating_hours" else int(totals[key])) for key in totals}
            else:
                score, threshold, state = score6, threshold6, state6
                warning = dict(source="v6_warning_head", heads=picked6, note="预警来自 v6 独立预警头（各折内层选型），与寿命模型分开训练")
                events_summary = {key: (float(totals6[key]) if key == "operating_hours" else int(totals6[key]))
                                  for key in ("total_events", "hit20", "late", "missed", "false_alarms", "operating_hours")}
            emitted, active, confirmations, last = state
            arrays.update({
                f"{model_id}_curve": curve[at].astype("float32"), f"{model_id}_quant": quant[at].astype("float32"),
                f"{model_id}_warn": score[at].astype("float32"), f"{model_id}_threshold": threshold[at].astype("float32"),
                f"{model_id}_emitted": emitted[at], f"{model_id}_active": active[at],
                f"{model_id}_confirmations": confirmations[at], f"{model_id}_last_alarm": last[at],
            })
            checks[model_id] = dict(cycles=len(rows), mae=life["mae"], alarms_emitted=int(emitted.sum()))
            specs.append(dict(
                id=model_id, generation=generation, source_name=name, feature_set=feature_set, family=family,
                name=f"电极寿命 · {zh}", name_en=f"Electrode life · {en}", short=short, short_en=short_en,
                describe=f"{describe} · 留出预测回放", describe_en=f"{describe_en} · held-out predictions",
                features=sets[feature_set], warning=warning,
                evaluation=dict(
                    protocol="leave one electrode cycle out, 14 folds; research run, not a new blind test",
                    events=events_summary,
                    life=dict(mae=float(life["mae"]), mae_last2h=float(life["last2h"]), mae_last1h=float(life["last1h"]),
                              coverage=float(life["coverage"]), width=float(life["width"])))))

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "ledger.npz", **arrays)
    sources = {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in (
        v4 / "manifest.json", v4 / "origins.npz", v5 / "report.json", v5 / "predictions.npz", v6 / "report.json",
        v6 / "predictions.npz", data / "frame.parquet", data / "evaluation.py")}
    body = dict(
        kind="life_ledger_v1", created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status="research candidates; recorded held-out predictions; not released, not scored live",
        source="booth2 v5 / v6 research runs (leave one cycle out)", sensors=int(manifest4["sensors"]),
        horizons_hours=[float(h) for h in H], cap_hours=40.0, quantiles=[0.1, 0.5, 0.9],
        cycle_ids=cycle_ids, alarm_policy=policy, products_per_processing_hour=float(manifest4["products_per_processing_hour"]),
        showcase_segments=segments, origins=int(len(at)), origins_total=int(len(times)), models=specs,
        checks=dict(origins_equal_v4=True, life_errors_copied_from_reports=True, v5_alarm_totals_reproduced=True,
                    v6_warning_alarms_reproduced=emitted_reported6, per_model=checks),
        limits=["留出预测回放：页面读取研究运行已写出的逐原点留出预测，不在页面上运行模型权重",
                "研究候选，未发布；14 个周期均已在历次研究中查看，不是新增盲测",
                "十次观察到的到寿只能支持较宽的范围；分位区间未校准"],
        sources=sources, files={"ledger.npz": sha(out / "ledger.npz")})
    body["version"] = "booth2-research-ledger-" + hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    (out / "manifest.json").write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(version=body["version"], models=len(specs), origins=len(at), bytes=(out / "ledger.npz").stat().st_size,
                          checks=checks), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
