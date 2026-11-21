"""Reload v5 research artifacts without pickle or PyTorch. XGBoost is loaded only for AFT trees."""
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri

H = np.array([1/3, 2/3, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24.])
EDGES = np.array([0, .25, .5, 1, 2, 4, 8, 12, 24, 40.])


def predict(artifact_path, features):
    path = Path(artifact_path)
    model = json.loads(path.read_text(encoding="utf-8"))
    frame = features.copy()
    if "log_vsd_30m" in model["features"]:
        frame["log_vsd_30m"] = np.log(frame.vsd_mean_30m.clip(lower=.02))
    x = frame[model["features"]].to_numpy(float)
    med, spread = np.array(model["median"]), np.array(model["spread"])
    z = np.clip((np.where(np.isfinite(x), x, med)-med)/spread, -8, 8)
    family = model["family"]
    if family.startswith("aft"):
        if family == "aft_linear6":
            w = np.array(model["weights"])
            mu, sigma = w[0]+z@w[1:-1], np.exp(w[-1])
        else:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(path.with_suffix(".ubj"))
            mu, sigma = booster.predict(xgb.DMatrix(z), output_margin=True), model["sigma"]
        curve = ndtr((np.log(H)[None,:]-mu[:,None])/sigma)
        quant = np.exp(np.clip(mu[:,None]+sigma*ndtri([.1,.5,.9]), -15, 15))
    else:
        seed_rates = []
        for raw in model["states"]:
            state = {k:np.array(v) for k,v in raw.items()}
            if model["hidden"]:
                a = np.tanh(z @ state["0.weight"].T + state["0.bias"])
                logits = a @ state["2.weight"].T + state["2.bias"]
            else:
                logits = z @ state["slope.weight"].T + state["base"]
            seed_rates.append(np.logaddexp(0, logits)+1e-5)
        rate = np.exp(np.mean(np.log(seed_rates), axis=0))
        exposure = np.maximum(np.minimum(H[:,None]-EDGES[:-1], np.diff(EDGES)), 0)
        exposure[:,-1] = np.maximum(H-EDGES[-2], 0)
        curve = -np.expm1(-(rate @ exposure.T))
        cumulative = np.cumsum(rate*np.diff(EDGES), axis=1)
        quant = np.empty((len(z),3))
        for j,p in enumerate((.1,.5,.9)):
            mass = -np.log1p(-p)
            for i in range(len(z)):
                k = min(np.searchsorted(cumulative[i], mass, side="left"), len(EDGES)-2)
                before = cumulative[i,k-1] if k else 0.
                quant[i,j] = EDGES[k]+(mass-before)/rate[i,k]
    if not np.isfinite(curve).all() or not np.isfinite(quant).all():
        raise ValueError("model returned non-finite predictions")
    return curve, quant
