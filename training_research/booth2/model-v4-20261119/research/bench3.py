"""Round 3: which network to carry forward. Leave-one-cycle-out, properly weighted rows, 2 seeds.

Each network has 12 horizon outputs plus one warning output (model_v2's y_eol_60m, calendar minutes).
usage: python bench3.py [tag]
"""
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import bench2 as B
import nets
import v4lib as L

SEEDS = (0, 1)
VARIANTS = {
    "mlp16": dict(kind="mlp", hidden=16, drop=.3),
    "gru16": dict(kind="gru", hidden=16, drop=.25),
    "hybrid_gru16": dict(kind="hybrid", hidden=16, drop=.25),
    "hybrid_gru8": dict(kind="hybrid", hidden=8, drop=.25),
    "hybrid_gru16_wd": dict(kind="hybrid", hidden=16, drop=.4, wd=1e-1),
}


def targets(rows):
    """12 horizon targets and the warning target; weight 0 where unknown."""
    y, known = L.cdf_targets(rows.rul_h.to_numpy(), rows.event.to_numpy().astype(bool))
    warn = rows.y_eol_60m.to_numpy()
    return (np.column_stack([y, np.nan_to_num(warn)]).astype("float32"),
            np.column_stack([known, np.isfinite(warn)]).astype("float32"))


def fold(frame, held):
    import torch
    torch.set_num_threads(1)
    cycles = [c for c in frame.cycle_id.unique() if c != held]
    tr, w = B.train_rows(frame, cycles)
    te = frame[frame.cycle_id.eq(held)]
    y, known = targets(tr)
    weight = known * w[:, None]
    rate = float((y[:, -1] * weight[:, -1]).sum() / weight[:, -1].sum())
    weight[:, -1] *= np.where(y[:, -1] > 0, min((1 - rate) / max(rate, 1e-4), 20.0), 1.0)  # the warning is rare: weigh its positives up
    sc = B.Scaler(tr[L.END].to_numpy("float64"))
    ftr, fte = sc(tr[L.END].to_numpy("float64")), sc(te[L.END].to_numpy("float64"))
    raw = B.raw_channels(frame)
    full = frame[frame.cycle_id.isin(cycles)]
    s2 = B.Scaler(raw.loc[tr.index].to_numpy("float64"))
    str_ = B.sequences(s2(raw.loc[full.index].to_numpy("float64")), full.cycle_id.to_numpy(), 120, 6)[full.index.get_indexer(tr.index)]
    ste = B.sequences(s2(raw.loc[te.index].to_numpy("float64")), te.cycle_id.to_numpy(), 120, 6)
    out = {}
    for name, cfg in VARIANTS.items():
        cfg = dict(cfg)
        kind = cfg.pop("kind")
        logits = np.mean([nets.predict(nets.fit(kind, ftr, None if kind == "mlp" else str_, y, weight, seed=s, **cfg), fte, None if kind == "mlp" else ste)
                          for s in SEEDS], 0)
        out[f"cdf_{name}"] = L.monotone(L.expit(logits[:, :-1]))
        out[f"warn_{name}"] = L.expit(logits[:, -1])
    return held, te.index.to_numpy(), out


def main(tag="r3"):
    frame = L.load()
    results = Parallel(n_jobs=14)(delayed(fold)(frame, c) for c in frame.cycle_id.unique())
    store = {}
    for held, index, out in results:
        for name, value in out.items():
            store.setdefault(name, np.full((len(frame),) + value.shape[1:], np.nan))[index] = value
    np.savez_compressed(L.OUT / f"bench_{tag}.npz", **store)
    print("saved", sorted(store))


if __name__ == "__main__":
    main(*sys.argv[1:2])
