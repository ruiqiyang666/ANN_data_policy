"""model_v2's own GRU-16 / TCN-16 retrained leave-one-cycle-out.

Same inputs (`encode`), network (`RiskNet`), loss, optimiser, batch size and
one-origin-per-minute training stride as model_v2/src/train.py. There is no
validation split inside a fold, so the epoch count is fixed at the frozen run's
best epoch. Scored on the 30-minute head (the frozen alarm head) and the 60-minute head.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent))
import data as D
sys.path.insert(0, str(D.REPO / "model_v2_GRU" / "src"))
import modeling as M, evaluation as EV
warnings.filterwarnings("ignore")
EPOCHS = {"gru": int(sys.argv[1]) if len(sys.argv) > 1 else 10, "tcn": int(sys.argv[2]) if len(sys.argv) > 2 else 10}


def fold(held, kind, hidden, f, y, seed=20261117):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr = f.cycle_id.ne(held).to_numpy()
    pre = M.fit_preprocessor(f[tr])
    x = M.encode(f, pre)
    yy = y[M.TARGETS].to_numpy(np.float32)
    sample = ~pd.DataFrame({"cycle": f.cycle_id, "minute": f.timestamp.dt.floor("min")}).duplicated().to_numpy()
    ti = np.flatnonzero(tr & sample & np.isfinite(yy).any(1))
    net = M.RiskNet(kind, hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.003)
    valid = np.isfinite(yy[ti]); pos = np.nansum(yy[ti], axis=0); neg = valid.sum(0) - pos
    pw = torch.tensor(np.minimum(neg / np.maximum(pos, 1), 60), dtype=torch.float32)
    xt, yt, mt = torch.from_numpy(x[ti]), torch.from_numpy(np.nan_to_num(yy[ti])), torch.from_numpy(valid.astype(np.float32))
    for _ in range(EPOCHS[kind]):
        net.train(); order = rng.permutation(len(ti))
        for at in range(0, len(ti), 256):
            b = order[at:at + 256]; opt.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b]), yt[b], pos_weight=pw, reduction="none") * mt[b]).sum() / mt[b].sum().clamp_min(1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
    te = np.flatnonzero(~tr)
    return kind, hidden, te, M.predict_net(net, x[te])


def main():
    f, y, e, p = M.load_frames()
    cycles = list(f.cycle_id.unique())
    out = Parallel(n_jobs=14)(delayed(fold)(c, k, h, f, y) for k, h in (("gru", 16), ("gru", 8), ("tcn", 16)) for c in cycles)
    hours = float(p[p.supervised_product_eligible.eq(1)].spray_samples.sum() / 3600)
    store = {}
    for kind, hidden, te, pv in out:
        store.setdefault(f"{kind}_{hidden}", np.full((len(f), 4), np.nan))[te] = pv
    grid = np.r_[np.arange(.05, 1, .05), .97, .98, .99, .995]
    for name, s in store.items():
        for j, head in ((0, "30m"), (2, "60m"), (3, "pre20_40")):
            rows = []
            for th in grid:
                m, _, _ = EV.evaluate(f, s[:, j], float(th), e, hours)
                rows.append(dict(model=f"{name}|{head}", th=round(float(th), 3), hit=m["hit20"], late=m["late"], missed=m["missed"],
                                 fa=m["false_alarms"], fa_h=round(m["false_alarms_per_hour"], 3)))
            c = pd.DataFrame(rows)
            print(pd.concat([c[c.fa_h <= .1].sort_values(["hit", "fa"], ascending=[False, True]).head(2),
                             c[c.fa_h <= .05].sort_values(["hit", "fa"], ascending=[False, True]).head(1)]).to_string(index=False, header=False))
    np.savez_compressed(D.ROOT / "research/out/gru_loco.npz", **store)
    m, rows, _ = EV.evaluate(f, store["gru_16"][:, 0], .95, e, hours)
    print("gru_16 30m head at the frozen 0.95:", {k: m[k] for k in ("hit20", "late", "missed", "false_alarms")})
    print(pd.DataFrame(rows)[["event_id", "cycle_id", "lead_minutes", "hit20", "late", "missed"]].to_string())


if __name__ == "__main__":
    main()
