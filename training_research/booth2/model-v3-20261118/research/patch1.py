from pathlib import Path
import sys
def sub(path, pairs):
    p = Path(path); s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_bytes(s.encode("utf-8"))
W = Path(sys.argv[1])
sub(W / "simulation/pipeline/inference/life.py", [
("""def lift(hazard, p_warn):
    \"\"\"Raise the first-hour hazards so P(end within one hour) is at least the warning's.\"\"\"
""", """FAST_ALPHA = 0.05  # the fast path follows the hazard layer with a time constant of about three processing minutes


def lift(hazard, p_warn):
    \"\"\"Raise the first-hour hazards so P(end within one hour) is at least the warning's.\"\"\"
"""),
("""def survival(hazard):
    return np.concatenate([np.ones((len(hazard), 1)), np.cumprod(1 - hazard, axis=1)], axis=1)


def quantiles(hazard, qs=QUANTILES):
    \"\"\"Residual-life quantiles; survival is taken as linear inside a bin.\"\"\"
    s = survival(hazard)
    rows""", """def survival(hazard):
    return np.concatenate([np.ones((len(hazard), 1)), np.cumprod(1 - hazard, axis=1)], axis=1)


def fuse(logit, p_warn, alpha):
    \"\"\"Survival curve per origin (N x 14, at the bin edges) from hazard logits and the warning.

    Two causal readings of the same hazard layer are combined by taking the lower
    survival at every edge: a slow one (heavily smoothed, for the hours ahead) and
    a fast one (lightly smoothed and lifted by the warning, for the last hour).
    The lower envelope of two survival curves is itself a survival curve.
    \"\"\"
    slow = survival(sigmoid(ema(logit, alpha)))
    fast = survival(lift(sigmoid(ema(logit, FAST_ALPHA)), p_warn))
    return np.minimum(slow, fast)


def quantiles(s, qs=QUANTILES):
    \"\"\"Residual-life quantiles of survival curves; survival is taken as linear inside a bin.\"\"\"
    rows"""),
("""def horizon_risk(hazard, hours=HORIZONS):
    \"\"\"P(end of life within each horizon); the horizons are bin edges.\"\"\"
    s = survival(hazard)
    return""", """def horizon_risk(s, hours=HORIZONS):
    \"\"\"P(end of life within each horizon); the horizons are bin edges.\"\"\"
    return"""),
("""    raw = sigmoid(logit)
    smooth = sigmoid(ema(logit, alpha))
    p_warn = np.interp(warn, model["calibration_x"], model["calibration_y"])
    fused = lift(smooth, p_warn)
    q = quantiles(fused)
    widen = float(model["widen"])
    q[:, 0], q[:, 2] = q[:, 0] * np.exp(-widen), np.minimum(q[:, 2] * np.exp(widen), EDGES[-1])
    return dict(warn=warn, hazard=raw, hazard_smooth=smooth, hazard_fused=fused, p_warn=p_warn,
                quantiles=q, horizons=horizon_risk(fused), z=z)""", """    p_warn = np.interp(warn, model["calibration_x"], model["calibration_y"])
    s = fuse(logit, p_warn, alpha)
    q = quantiles(s)
    widen = float(model["widen"])
    q[:, 0], q[:, 2] = q[:, 0] * np.exp(-widen), np.minimum(q[:, 2] * np.exp(widen), EDGES[-1])
    fused = 1 - s[:, 1:] / np.maximum(s[:, :-1], 1e-12)  # the hazards of the combined curve
    return dict(warn=warn, hazard=sigmoid(logit), hazard_fused=fused, p_warn=p_warn,
                quantiles=q, horizons=horizon_risk(s), z=z)"""),
("""        Weights and activations are exact. The output activations are the fused
        hazards, so the first-hour units also reflect the tree warning.""", """        Weights and inputs are exact. The output activations are the hazards of the
        combined survival curve, so they also carry the smoothing and the tree warning."""),
])
sub(Path(sys.argv[2]) / "research/train_v3.py", [
("""    smooth = L.sigmoid(L.ema(logit, ALPHA))
    return L.quantiles(L.lift(smooth, np.interp(warn, model["calibration_x"], model["calibration_y"])))""",
 """    return L.quantiles(L.fuse(logit, np.interp(warn, model["calibration_x"], model["calibration_y"]), ALPHA))"""),
])
print("patched")
