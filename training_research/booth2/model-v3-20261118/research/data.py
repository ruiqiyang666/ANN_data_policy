"""Booth2 electrode research frame: causal per-origin features, risk and life targets.

Everything is computed inside one electrode cycle and only from origins at or
before the current one. Life is measured in *processing time* (origins x 10 s),
because calendar time contains nights and stoppages.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("ANN_DATA_REPO", str(ROOT.parents[2])))
CACHE = ROOT / "research" / "cache"
STEP_H = 10 / 3600  # one processing origin
SENSORS28 = ["Argon", "Nitrogen", "AirJet", "Silvent", "Current", "Power", "Voltage", "Cooling_water_flow",
             "Conductivity", "Cooling_water_inlet_temp.", "Cooling_water_outlet_temp.", "Feedrate_1", "Disk_1",
             "Carrier_gas_1", "Feed_Factor_1", "Press._Hopper_1", "Stirrer_1", "Weight_Hopper_1", "Feedrate_2",
             "Disk_2", "Carrier_gas_2", "Feed_Factor_2", "Press._Hopper_2", "Stirrer_2", "Weight_Hopper_2",
             "Station_1", "Station_2", "O2_Sensor"]
W = {"5m": 30, "30m": 180, "3h": 1080}  # processing-time windows in origins


def roll(s, n, fn, minp=None):
    r = s.rolling(n, min_periods=minp or max(3, n // 6))
    return getattr(r, fn)() if isinstance(fn, str) else fn(r)


def slope(s, n):
    """Least-squares slope per processing hour over the last n origins."""
    x = np.arange(len(s), dtype="float64")
    sx, sy = pd.Series(x, index=s.index), s
    m = max(6, n // 6)
    cov = sx.rolling(n, min_periods=m).cov(sy)
    var = sx.rolling(n, min_periods=m).var()
    return cov / var / STEP_H


def cycle_features(q):
    """q: one cycle's origins in time order, with the 28 `_last` / `_mean60s` columns."""
    out = pd.DataFrame(index=q.index)
    n = len(q)
    t = q.timestamp.astype("int64").to_numpy() // 10**9
    gap = np.r_[0, np.diff(t)]
    out["age_h"] = np.arange(n) * STEP_H
    out["gap_s"] = np.minimum(gap, 86400)
    restart = gap > 600
    out["restarts"] = np.cumsum(restart)
    last_restart = np.maximum.accumulate(np.where(restart | (np.arange(n) == 0), np.arange(n), 0))
    out["since_restart_h"] = (np.arange(n) - last_restart) * STEP_H
    out["idle_h_cum"] = np.cumsum(np.where(restart, gap, 0)) / 3600
    v, sd = q["Voltage_mean60s"], q["Voltage_std60s"]
    out["v60"], out["vsd60"], out["v_last"] = v, sd, q["Voltage_last"]
    for name, k in W.items():
        out[f"v_med_{name}"] = roll(v, k, "median")
        out[f"v_q10_{name}"] = roll(v, k, lambda r: r.quantile(.1))
        out[f"v_std_{name}"] = roll(v, k, "std")
        out[f"vsd_mean_{name}"] = roll(sd, k, "mean")
        out[f"v_low53_{name}"] = roll((v < 53).astype(float), k, "mean")
        out[f"v_low52_{name}"] = roll((v < 52).astype(float), k, "mean")
    out["v_min_30m"] = roll(q["Voltage_last"], 180, "min")
    out["v_slope_30m"], out["v_slope_3h"], out["v_slope_10h"] = slope(v, 180), slope(v, 1080), slope(v, 3600)
    out["vsd_slope_3h"] = slope(sd, 1080)
    # cycle-to-date references (available once the cycle has run for a while)
    base = v.expanding(min_periods=180).median()
    peak = out["v_med_3h"].expanding(min_periods=1).max()
    out["v_base"], out["v_peak3h"] = base, peak
    out["v_first1h"] = v.iloc[:360].median() if n >= 360 else np.nan
    out.loc[out.index[:min(360, n)], "v_first1h"] = np.nan
    out["d_5m_3h"] = out["v_med_3h"] - out["v_med_5m"]
    out["d_30m_base"] = base - out["v_med_30m"]
    out["d_peak_30m"] = peak - out["v_med_30m"]
    out["d_peak_3h"] = peak - out["v_med_3h"]
    out["d_first_30m"] = out["v_first1h"] - out["v_med_30m"]
    out["vsd_ratio"] = out["vsd_mean_30m"] / sd.expanding(min_periods=180).median()
    out["margin51_30m"] = out["v_med_30m"] - 51
    out["margin51_q10"] = out["v_q10_30m"] - 51
    # thermal and electrical context from the other sensors
    flow = q["Cooling_water_flow_mean60s"]
    dT = q["Cooling_water_outlet_temp._mean60s"] - q["Cooling_water_inlet_temp._mean60s"]
    power = q["Power_mean60s"]
    loss = flow * dT * 4.186 / 60  # kW carried away by the cooling water (flow in l/min)
    out["cool_dT"], out["cool_loss_kw"], out["cool_flow"] = dT, loss, flow
    out["therm_eff"] = 1 - loss / power.where(power > 5)
    out["power"], out["current"] = power, q["Current_mean60s"]
    out["resistance"] = v / q["Current_mean60s"].where(q["Current_mean60s"] > 100) * 1000
    out["argon"], out["nitrogen"] = q["Argon_mean60s"], q["Nitrogen_mean60s"]
    out["inlet_T"] = q["Cooling_water_inlet_temp._mean60s"]
    out["o2"] = q["O2_Sensor_mean60s"]
    f1, f2 = q["Feedrate_1_last"] > 0, q["Feedrate_2_last"] > 0
    single = f1 ^ f2
    out["feed_act"] = np.where(single, np.where(f1, q["Feedrate_1_mean60s"], q["Feedrate_2_mean60s"]), np.nan)
    out["press_act"] = np.where(single, np.where(f1, q["Press._Hopper_1_mean60s"], q["Press._Hopper_2_mean60s"]), np.nan)
    out["carrier_act"] = np.where(single, np.where(f1, q["Carrier_gas_1_mean60s"], q["Carrier_gas_2_mean60s"]), np.nan)
    out["hopper"] = np.where(single, np.where(f1, 1.0, 2.0), np.nan)
    for col in ["therm_eff", "cool_loss_kw", "cool_dT", "power", "feed_act", "press_act", "argon", "nitrogen"]:
        s = out[col]
        out[f"{col}_med_30m"] = roll(s, 180, "median")
        out[f"{col}_med_3h"] = roll(s, 1080, "median")
        out[f"{col}_d_base"] = out[f"{col}_med_30m"] - s.expanding(min_periods=180).median()
    out["therm_eff_slope_3h"] = slope(out["therm_eff"], 1080)
    out["power_std_30m"] = roll(power, 180, "std")
    return out


def build(force=False):
    path = CACHE / "frame.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(REPO / "model_v2_GRU/data/anchors.csv.gz", parse_dates=["timestamp"])
    y = pd.read_csv(REPO / "model_v2_GRU/data/targets.csv.gz")
    f = pd.read_csv(REPO / "training_data/features_10s.csv.gz").set_index("anchor_id")
    cyc = pd.read_csv(REPO / "model_v2_GRU/data/cycles.csv", parse_dates=["usable_start", "usable_end", "eol_endpoint"])
    prod = pd.read_csv(REPO / "model_v2_GRU/data/products.csv.gz", parse_dates=["start", "end"], low_memory=False)
    assert a.anchor_id.equals(y.anchor_id)
    g = f.loc[a.anchor_id].reset_index(drop=True)
    assert np.allclose(g["Voltage_last"], a["Voltage"]) and np.allclose(g["Voltage_mean60s"], a["Voltage_mean60s"])
    raw = pd.concat([a[["anchor_id", "timestamp", "cycle_id", "split", "observation_horizon_seconds", "remaining_seconds"]],
                     g[[c for c in g.columns if c.endswith(("_last", "_mean60s", "_std60s"))]]], axis=1)
    parts = []
    for cid, q in raw.groupby("cycle_id", sort=False):
        c = cyc[cyc.cycle_id.eq(cid)].iloc[0]
        feats = cycle_features(q)
        eol = c.eol_endpoint
        observed = pd.notna(eol)
        end = eol if observed else c.usable_end
        before = (q.timestamp < end).to_numpy()
        total = int(before.sum())
        left = np.where(before, total - np.arange(len(q)), np.nan)  # origins from here to the end, inclusive
        feats["rul_h"] = left * STEP_H
        feats["event"] = int(observed)
        feats["life_h"] = total * STEP_H
        p = prod[prod.cycle_id.eq(cid) & prod.supervised_product_eligible.eq(1) & (prod.start < end)]
        starts = np.sort(p.start.astype("int64").to_numpy() // 10**9)
        now = q.timestamp.astype("int64").to_numpy() // 10**9
        feats["rul_pieces"] = np.where(before, len(starts) - np.searchsorted(starts, now, side="right"), np.nan)
        feats["pieces_done"] = np.searchsorted(starts, now, side="right")
        feats["left_censored"] = int(c.left_censored)
        parts.append(feats)
    feats = pd.concat(parts).loc[raw.index]
    frame = pd.concat([raw[["anchor_id", "timestamp", "cycle_id", "split", "observation_horizon_seconds", "remaining_seconds"]],
                       y[["y_eol_30m", "y_eol_45m", "y_eol_60m", "y_pre20_40m"]], feats], axis=1)
    frame.to_parquet(path)
    raw.to_parquet(CACHE / "raw28.parquet")
    return frame


META = ["anchor_id", "timestamp", "cycle_id", "split", "observation_horizon_seconds", "remaining_seconds",
        "y_eol_30m", "y_eol_45m", "y_eol_60m", "y_pre20_40m", "rul_h", "event", "life_h", "rul_pieces", "left_censored"]
AGE = ["age_h", "pieces_done", "restarts", "idle_h_cum"]


def feature_names(frame, age=True):
    cols = [c for c in frame.columns if c not in META]
    return cols if age else [c for c in cols if c not in AGE]


if __name__ == "__main__":
    fr = build(force=True)
    print(fr.shape, len(feature_names(fr)))
    print(fr.groupby("cycle_id").agg(n=("age_h", "size"), life=("life_h", "first"), ev=("event", "first"),
                                     pieces=("rul_pieces", "max"), eff=("therm_eff", "median"), loss=("cool_loss_kw", "median"),
                                     dT=("cool_dT", "median"), power=("power", "median")).round(3).to_string())
    print(fr[feature_names(fr)].isna().mean().sort_values().tail(12).round(3).to_string())
