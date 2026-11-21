"""Round 2: neural survival networks, Cox, degradation process and control charts, leave-one-cycle-out.

Training rows are one origin per processing minute plus every origin of the last three hours, the latter weighted 1/6,
so the base rates stay those of the data. Raw fold outputs are saved to out/bench_<tag>.npz.
usage: python bench2.py [tag] [groups]     groups: comma list of lin,mlp,gru,tcn,cox,wiener,chart (default all)
"""
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import v4lib as L

SEQ_FEAT = ["v60", "vsd60", "vsd_mean_30m", "d_peak_30m", "d_30m_base", "margin51_30m", "cool_loss_kw_d_base", "therm_eff_d_base",
            "power_d_base", "v_slope_30m", "vsd_ratio", "since_restart_h"]
RAW = ["v_c", "log_vsd", "vlast_dev", "therm_eff_c", "cool_dT_c", "power_c", "since_restart_h", "restart"]


def train_rows(frame, cycles):
    q = frame[frame.cycle_id.isin(cycles) & frame.rul_h.notna()]
    grid = np.arange(len(q)) % 6 == 0
    near = (q.event.eq(1) & q.rul_h.le(3)).to_numpy()
    keep = grid | near
    weight = np.where(near, 1 / 6, 1.0)[keep]  # near-end origins are all kept, so each counts a sixth
    return q[keep], weight.astype("float32")


class Scaler:
    def __init__(self, x):
        self.med = np.nanmedian(x, 0)
        q = np.nanpercentile(x, [25, 75], 0)
        self.scale = np.maximum(q[1] - q[0], 1e-3 + .05 * np.abs(self.med))

    def __call__(self, x):
        return np.clip((np.where(np.isfinite(x), x, self.med) - self.med) / self.scale, -8, 8).astype("float32")


def raw_channels(frame):
    """Baseline-normalised per-origin channels for the sequence networks (causal: cycle-to-date medians)."""
    out = pd.DataFrame(index=frame.index)
    out["v_c"] = frame.v60 - frame.v_base
    out["log_vsd"] = np.log(frame.vsd60.clip(lower=.02))
    out["vlast_dev"] = frame.v_last - frame.v60
    for col in ("therm_eff", "cool_dT", "power"):
        out[f"{col}_c"] = frame[col] - (frame[f"{col}_med_30m"] - frame[f"{col}_d_base"])
    out["since_restart_h"] = frame.since_restart_h.clip(upper=4)
    out["restart"] = (frame.gap_s > 600).astype(float)
    return out


def sequences(values, cycle, steps, stride):
    """Per origin: the last `steps` points `stride` origins apart inside its cycle, zero-padded, plus a presence channel."""
    out = np.zeros((len(values), steps, values.shape[1] + 1), "float32")
    start = 0
    for c in pd.unique(cycle):
        n = int((cycle == c).sum())
        pos = np.arange(n)[:, None] - np.arange(steps - 1, -1, -1)[None, :] * stride
        ok = pos >= 0
        out[start:start + n, :, :-1] = np.where(ok[:, :, None], values[start + np.maximum(pos, 0)], 0)
        out[start:start + n, :, -1] = ok
        start += n
    return out


def fit_net(kind, x, y, known, weight, hidden=16, epochs=24, seed=0, drop=.25, wd=2e-2, lr=3e-3):
    import torch
    from torch import nn
    torch.manual_seed(seed)

    class Block(nn.Module):
        def __init__(self, h, d):
            super().__init__(); self.pad = 2 * d
            self.conv = nn.Conv1d(h, h, 3, dilation=d, padding=self.pad); self.act = nn.GELU(); self.drop = nn.Dropout(drop)

        def forward(self, a):
            return a + self.drop(self.act(self.conv(a)[:, :, :-self.pad]))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(drop)
            if kind == "mlp":
                self.body = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Dropout(drop))
            elif kind == "gru":
                self.body = nn.GRU(x.shape[2], hidden, batch_first=True)
            else:
                self.inp = nn.Conv1d(x.shape[2], hidden, 1)
                self.body = nn.Sequential(*[Block(hidden, d) for d in (1, 2, 4, 8, 16)])
            self.head = nn.Linear(hidden, y.shape[1])

        def forward(self, a):
            if kind == "mlp":
                h = self.body(a)
            elif kind == "gru":
                h = self.drop(self.body(a)[0][:, -1])
            else:
                h = self.drop(self.body(self.inp(a.transpose(1, 2)))[:, :, -1])
            return self.head(h)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(x) // 256 + 1))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    wt = torch.from_numpy(known * weight[:, None])
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(xt), generator=g).split(256):
            loss = (nn.functional.binary_cross_entropy_with_logits(net(xt[ix]), yt[ix], reduction="none") * wt[ix]).sum() / wt[ix].sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net


def predict_net(net, x):
    import torch
    out = []
    with torch.inference_mode():
        for a in torch.from_numpy(x).split(4096):
            out.append(torch.sigmoid(net(a)).numpy().astype("float64"))
    return L.monotone(np.concatenate(out))


def wiener(frame, tr, te):
    """First hitting time of the 51 V line by a drifting Brownian motion on the 30-minute median voltage."""
    from scipy.stats import norm
    cyc = tr.groupby("cycle_id").agg(first=("v_first1h", "max"), life=("life_h", "first"), ev=("event", "first"))
    cyc = cyc[cyc.ev.eq(1) & cyc["first"].notna()]
    d0 = float(np.median((cyc["first"] - 51) / cyc.life))  # the average decline that would reach the line at the end
    inc = tr.groupby("cycle_id").v_med_30m.diff(360).dropna()  # change over one processing hour
    sigma2 = float(np.nanvar(inc))
    m = np.maximum(te.margin51_30m.to_numpy(float), .05)
    slope = np.nan_to_num(-te.v_slope_3h.to_numpy(float), nan=d0)
    d = np.maximum(.5 * slope + .5 * d0, .02)
    mu, lam = m / d, m ** 2 / sigma2
    m_ok = np.isfinite(mu)
    F = np.zeros((len(te), len(L.H)))
    for j, t in enumerate(L.H):
        a = np.sqrt(lam / t)
        F[:, j] = norm.cdf(a * (t / mu - 1)) + np.exp(np.minimum(2 * lam / mu, 50)) * norm.cdf(-a * (t / mu + 1))
    F[~m_ok] = np.nan
    return L.monotone(np.nan_to_num(F, nan=0.0))


def charts(te):
    """Classical control charts inside one cycle; scores are unbounded and get their own threshold grid."""
    sd = te.vsd60.to_numpy(float)
    base = te.vsd60.expanding(min_periods=180).median().fillna(.23).to_numpy()
    z = (sd - base) / .08
    drop = np.nan_to_num((te.v_base - te.v60).to_numpy(float), nan=0.0) / .5
    out = {}
    for name, series, k in (("cusum_vsd", z, .75), ("cusum_vdrop", drop, 1.0), ("cusum_both", .5 * z + .5 * drop, .75)):
        s, acc = np.zeros(len(te)), 0.0
        for i, v in enumerate(series):
            acc = max(0.0, min(acc + v - k, 200.0))
            s[i] = acc
        out[f"chart|{name}"] = s
    e, acc = np.zeros(len(te)), 0.0
    for i, v in enumerate(z):
        acc += .05 * (v - acc)
        e[i] = acc
    out["chart|ewma_vsd"] = e
    return out


def fold(frame, held, groups):
    import torch
    torch.set_num_threads(1)
    cycles = [c for c in frame.cycle_id.unique() if c != held]
    tr, w = train_rows(frame, cycles)
    te = frame[frame.cycle_id.eq(held)]
    rul, event = tr.rul_h.to_numpy(), tr.event.to_numpy().astype(bool)
    y, known = L.cdf_targets(rul, event)
    out = {}
    sc = Scaler(tr[L.END].to_numpy("float64"))
    ztr, zte = sc(tr[L.END].to_numpy("float64")), sc(te[L.END].to_numpy("float64"))
    if "lin" in groups:
        from sklearn.linear_model import LogisticRegression
        for c in (.01, .05, .3):
            cols = []
            for j in range(len(L.H)):
                m = known[:, j] > 0
                cols.append(LogisticRegression(C=c, max_iter=2000).fit(ztr[m], y[m, j], sample_weight=w[m]).predict_proba(zte)[:, 1])
            out[f"cdf_logit|END|C{c}"] = L.monotone(np.column_stack(cols))
    if "mlp" in groups:
        for h in (8, 32):
            out[f"cdf_mlp{h}|END"] = np.mean([predict_net(fit_net("mlp", ztr, y, known, w, hidden=h, epochs=30, seed=s, drop=.3), zte) for s in range(3)], 0)
    if {"gru", "tcn"} & set(groups):
        full_tr = frame[frame.cycle_id.isin(cycles)]
        pick = full_tr.index.get_indexer(tr.index)
        for tag, cols, steps, stride, values in (("feat", SEQ_FEAT, 37, 60, frame[SEQ_FEAT]), ("raw", RAW, 120, 6, raw_channels(frame))):
            s2 = Scaler(values.loc[tr.index].to_numpy("float64"))
            xs_tr = sequences(s2(values.loc[full_tr.index].to_numpy("float64")), full_tr.cycle_id.to_numpy(), steps, stride)[pick]
            xs_te = sequences(s2(values.loc[te.index].to_numpy("float64")), te.cycle_id.to_numpy(), steps, stride)
            for kind in ("gru", "tcn"):
                if kind in groups:
                    out[f"cdf_{kind}16|seq_{tag}"] = np.mean([predict_net(fit_net(kind, xs_tr, y, known, w, seed=s), xs_te) for s in range(3)], 0)
    if "cox" in groups:
        from lifelines import CoxPHFitter
        for pen in (.03, .3):
            d = pd.DataFrame(ztr.astype("float64"), columns=[f"f{i}" for i in range(ztr.shape[1])])
            d["T"], d["E"], d["W"] = np.maximum(rul, 1e-3), event.astype(int), w
            cph = CoxPHFitter(penalizer=pen).fit(d, "T", "E", weights_col="W", robust=False)
            s = cph.predict_survival_function(pd.DataFrame(zte.astype("float64"), columns=d.columns[:-3]), times=L.H).to_numpy().T
            out[f"cox|END|pen{pen}"] = L.monotone(1 - s)
    if "wiener" in groups:
        out["wiener_fht"] = wiener(frame, frame[frame.cycle_id.isin(cycles)], te)
    if "chart" in groups:
        out.update(charts(te))
    return held, te.index.to_numpy(), out


def main(tag="r2", groups="lin,mlp,gru,tcn,cox,wiener,chart"):
    frame = L.load()
    groups = groups.split(",")
    results = Parallel(n_jobs=14)(delayed(fold)(frame, c, groups) for c in frame.cycle_id.unique())
    store = {}
    for held, index, out in results:
        for name, value in out.items():
            store.setdefault(name, np.full((len(frame),) + value.shape[1:], np.nan))[index] = value
    np.savez_compressed(L.OUT / f"bench_{tag}.npz", **store)
    print("saved", sorted(store))


if __name__ == "__main__":
    main(*sys.argv[1:3])
