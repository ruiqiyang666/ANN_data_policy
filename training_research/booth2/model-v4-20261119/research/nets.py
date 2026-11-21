"""Survival networks for Booth2 v4: training in PyTorch, identical forward pass in NumPy.

kinds
  mlp     scaled features -> Linear(h) -> ReLU -> heads
  gru     sequence -> GRU(h) -> last state -> heads
  hybrid  [GRU(h) last state, scaled features] -> heads   (a linear path from the features plus a sequence state)
heads: one logit per horizon (P(end within h)) and optionally one warning logit.
"""
import numpy as np


def build(kind, n_feat, n_seq, hidden, n_out, drop):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.drop = nn.Dropout(drop)
            width = 0
            if kind in ("gru", "hybrid"):
                self.gru = nn.GRU(n_seq, hidden, batch_first=True)
                width += hidden
            if kind == "mlp":
                self.fc = nn.Linear(n_feat, hidden)
                width += hidden
            if kind == "hybrid":
                width += n_feat
            self.head = nn.Linear(width, n_out)

        def forward(self, feat, seq):
            parts = []
            if self.kind in ("gru", "hybrid"):
                parts.append(self.drop(self.gru(seq)[0][:, -1]))
            if self.kind == "mlp":
                parts.append(self.drop(torch.relu(self.fc(feat))))
            if self.kind == "hybrid":
                parts.append(feat)
            return self.head(torch.cat(parts, dim=1))
    return Net()


def fit(kind, feat, seq, y, weight, hidden=16, epochs=24, seed=0, drop=.25, wd=2e-2, lr=3e-3, batch=256):
    """y, weight: N x outputs (weight 0 where the answer is unknown). Fixed epochs: there is no data to stop early on."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    net = build(kind, feat.shape[1], seq.shape[2] if seq is not None else 0, hidden, y.shape[1], drop)
    decay = [p for n, p in net.named_parameters() if "bias" not in n]
    plain = [p for n, p in net.named_parameters() if "bias" in n]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=wd), dict(params=plain, weight_decay=0)], lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(feat) // batch + 1))
    ft, yt, wt = torch.from_numpy(feat), torch.from_numpy(y), torch.from_numpy(weight)
    st = torch.from_numpy(seq) if seq is not None else None
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(ft), generator=g).split(batch):
            out = net(ft[ix], st[ix] if st is not None else None)
            loss = (nn.functional.binary_cross_entropy_with_logits(out, yt[ix], reduction="none") * wt[ix]).sum() / wt[ix].sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net


def predict(net, feat, seq, batch=4096):
    """Logits, N x outputs."""
    import torch
    out = []
    with torch.inference_mode():
        for a in range(0, len(feat), batch):
            out.append(net(torch.from_numpy(feat[a:a + batch]), torch.from_numpy(seq[a:a + batch]) if seq is not None else None).numpy())
    return np.concatenate(out).astype("float64")


def export(net):
    """State dict as float64 arrays, with the kind recorded."""
    out = {k: v.detach().numpy().astype("float64") for k, v in net.state_dict().items()}
    out["kind"] = np.array(net.kind)
    return out


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def gru_states(w, seq):
    """Last hidden state of a one-layer GRU, PyTorch gate order (reset, update, new). seq: N x T x C."""
    wi, wh, bi, bh = w["gru.weight_ih_l0"], w["gru.weight_hh_l0"], w["gru.bias_ih_l0"], w["gru.bias_hh_l0"]
    hsize = wh.shape[1]
    h = np.zeros((seq.shape[0], hsize))
    for t in range(seq.shape[1]):
        gi = seq[:, t].astype("float64") @ wi.T + bi
        gh = h @ wh.T + bh
        r = _sigmoid(gi[:, :hsize] + gh[:, :hsize])
        z = _sigmoid(gi[:, hsize:2 * hsize] + gh[:, hsize:2 * hsize])
        n = np.tanh(gi[:, 2 * hsize:] + r * gh[:, 2 * hsize:])
        h = (1 - z) * n + z * h
    return h


def forward(w, feat, seq, parts=False):
    """NumPy forward pass of an exported network: logits, N x outputs (and the hidden parts when asked)."""
    kind = str(w["kind"])
    hidden = []
    if kind in ("gru", "hybrid"):
        hidden.append(gru_states(w, seq))
    if kind == "mlp":
        hidden.append(np.maximum(feat.astype("float64") @ w["fc.weight"].T + w["fc.bias"], 0))
    if kind == "hybrid":
        hidden.append(feat.astype("float64"))
    x = np.concatenate(hidden, axis=1)
    out = x @ w["head.weight"].T + w["head.bias"]
    return (out, x) if parts else out
