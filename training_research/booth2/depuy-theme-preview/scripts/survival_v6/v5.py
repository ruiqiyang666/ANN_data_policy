"""Standalone censor-aware benchmark. All fits and selection exclude held-out cycles.

python train.py --data DATA --out OUT --jobs 8
DATA contains frame.parquet, events.csv, products.csv, evaluation.py and baseline JSON/CSV.
The snapshot, protocol and runtime versions are recorded with every run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr, ndtri
from joblib import Parallel, delayed

H = np.array([1/3, 2/3, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24.])
EDGES = np.array([0, .25, .5, 1, 2, 4, 8, 12, 24, 40.])
SMALL = ["margin51_30m", "d_peak_30m", "log_vsd_30m", "v_slope_30m", "vsd_ratio", "age_h"]
END = ["vsd_mean_5m", "vsd_mean_30m", "vsd_mean_3h", "vsd60", "vsd_ratio", "v_std_5m", "v_std_30m", "d_peak_30m", "d_peak_3h",
       "d_5m_3h", "d_30m_base", "v_slope_30m", "v_slope_3h", "v_low53_30m", "margin51_30m", "margin51_q10", "v_min_30m",
       "cool_loss_kw_d_base", "cool_dT_d_base", "therm_eff_d_base", "power_d_base", "power_std_30m", "therm_eff_slope_3h"]
NAMES = ["aft_linear6", "aft_xgb6", "aft_xgb23", "pch_linear6", "pch_mlp6", "pch_mlp23"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, data):
    Path(path).write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def train_rows(frame, cycles):
    parts = []
    for _, q in frame[frame.cycle_id.isin(cycles) & frame.rul_h.notna()].groupby("cycle_id", sort=False):
        r = q.iloc[::6].copy()
        r["weight"] = 1 / len(r)
        parts.append(r)
    q = pd.concat(parts)
    return q, q.weight.to_numpy() * len(q) / len(parts)


def scaler(x):
    med = np.nanmedian(x, axis=0)
    med = np.nan_to_num(med)
    p = np.nanpercentile(x, [25, 75], axis=0)
    spread = np.maximum(np.nan_to_num(p[1]-p[0]), .05*np.abs(med)+1e-3)
    return med, spread


def scaled(x, med, spread):
    return np.clip((np.where(np.isfinite(x), x, med)-med)/spread, -8, 8)


def aft_linear_fit(x, duration, event, w):
    a = np.column_stack([np.ones(len(x)), x])
    y = np.log(np.maximum(duration, 1e-4))
    w = w / w.sum()
    def loss(v):
        mu, sigma = a @ v[:-1], np.exp(v[-1])
        z = (y-mu)/sigma
        log_surv = log_ndtr(-z)
        mills = np.exp(np.minimum(-.5*z*z-.5*np.log(2*np.pi)-log_surv, 30))
        nll = np.where(event, .5*z*z+np.log(sigma)+y+.5*np.log(2*np.pi), -log_surv)
        dmu = np.where(event, -z/sigma, -mills/sigma)
        dsigma = np.where(event, 1-z*z, -z*mills)
        penalty = .025 * np.sum(v[1:-1]**2)
        grad = np.r_[a.T @ (w*dmu), np.sum(w*dsigma)]
        grad[1:-1] += .05*v[1:-1]
        return float(w @ nll + penalty), grad
    v = np.r_[np.log(np.median(duration[event])), np.zeros(x.shape[1]), 0.]
    result = minimize(loss, v, jac=True, method="L-BFGS-B", bounds=[(-15, 15)]*(len(v)-1)+[(-1.5, 1.4)], options={"maxiter":300})
    if not result.success and np.linalg.norm(result.jac) > .05:
        raise RuntimeError(f"AFT optimizer failed: {result.message}")
    return result.x, dict(success=bool(result.success), loss=float(result.fun), iterations=int(result.nit))


def aft_outputs(mu, sigma):
    cdf = ndtr((np.log(H)[None, :]-mu[:, None])/sigma)
    q = np.exp(np.clip(mu[:, None]+sigma*ndtri(np.array([.1,.5,.9])), -15, 15))
    return cdf, q


def pch_outputs(rate):
    def cumulative(t):
        exposure = np.maximum(np.minimum(t[..., None]-EDGES[:-1], np.diff(EDGES)), 0)
        exposure[..., -1] = np.maximum(t-EDGES[-2], 0)
        return exposure
    cdf = -np.expm1(-(rate @ cumulative(H).T))
    mass = rate*np.diff(EDGES)
    prev = np.column_stack([np.zeros(len(rate)), np.cumsum(mass, axis=1)[:, :-1]])
    q = np.empty((len(rate), 3))
    for j, p in enumerate((.1,.5,.9)):
        target = -np.log1p(-p)
        k = np.minimum((np.cumsum(mass, axis=1) < target).sum(1), len(EDGES)-2)
        q[:,j] = EDGES[k]+(target-prev[np.arange(len(rate)), k])/rate[np.arange(len(rate)), k]
    return cdf, q


def pch_fit(x, duration, event, w, hidden, seed):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if hidden:
        net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, len(EDGES)-1))
    else:
        # A common covariate effect, one baseline per time bin: proportional hazards.
        class PH(nn.Module):
            def __init__(self):
                super().__init__()
                self.slope = nn.Linear(x.shape[1], 1, bias=False)
                self.base = nn.Parameter(torch.full((len(EDGES)-1,), -2.3))
            def forward(self, a):
                return self.slope(a)+self.base
        net = PH()
    xt = torch.as_tensor(x, dtype=torch.float32)
    expo = np.maximum(np.minimum(duration[:,None]-EDGES[:-1], np.diff(EDGES)), 0)
    expo[:, -1] = np.maximum(duration-EDGES[-2], 0)
    expo = torch.as_tensor(expo, dtype=torch.float32)
    bins = torch.as_tensor(np.clip(np.searchsorted(EDGES, duration, side="right")-1, 0, len(EDGES)-2))
    et = torch.as_tensor(event, dtype=torch.float32)
    wt = torch.as_tensor(w/w.sum(), dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=.015, weight_decay=.04)
    losses = []
    for epoch in range(220):
        rate = nn.functional.softplus(net(xt)) + 1e-5
        nll = (rate*expo).sum(1)-et*torch.log(rate[torch.arange(len(xt)),bins])
        loss = (wt*nll).sum()
        # Smooth adjacent baseline bins, not successive predictions in time.
        smooth = .015*(torch.diff(torch.log(rate),dim=1)**2).mean()
        opt.zero_grad(); (loss+smooth).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        opt.step()
        if epoch % 20 == 0 or epoch == 219:
            losses.append([epoch, float(loss.detach())])
    net.eval()
    return net, losses


def fit_predict(frame, train_cycles, test_indices, name, model_file=None):
    tr, w = train_rows(frame, train_cycles)
    te = frame.loc[test_indices]
    cols = SMALL if name.endswith("6") else END
    x, test = tr[cols].to_numpy(float), te[cols].to_numpy(float)
    med, spread = scaler(x)
    z, zt = scaled(x,med,spread), scaled(test,med,spread)
    duration = tr.rul_h.to_numpy(float)
    event = tr.event.to_numpy(bool)
    artifact = dict(features=cols, median=med, spread=spread, family=name)
    if name == "aft_linear6":
        weights, fit = aft_linear_fit(z,duration,event,w)
        artifact.update(weights=weights, fit=fit)
        curve, quant = aft_outputs(np.column_stack([np.ones(len(zt)),zt]) @ weights[:-1], np.exp(weights[-1]))
    elif name.startswith("aft_xgb"):
        import xgboost as xgb
        d = xgb.DMatrix(z, weight=w)
        d.set_float_info("label_lower_bound", duration)
        d.set_float_info("label_upper_bound", np.where(event,duration,np.inf))
        sigma = 1.0
        params = dict(objective="survival:aft",aft_loss_distribution="normal",aft_loss_distribution_scale=sigma,
                      eval_metric="aft-nloglik",max_depth=2,eta=.035,min_child_weight=40,reg_lambda=8,
                      subsample=1,colsample_bytree=1,nthread=1,seed=20261120,tree_method="hist")
        model = xgb.train(params,d,num_boost_round=180)
        mu = model.predict(xgb.DMatrix(zt),output_margin=True)
        curve, quant = aft_outputs(mu,sigma)
        artifact.update(sigma=sigma,params=params)
        if model_file:
            model.save_model(str(model_file.with_suffix(".ubj")))
    else:
        import torch
        hidden = 0 if name == "pch_linear6" else 8
        outputs, states, logs = [], [], []
        for seed in (17,43):
            net, history = pch_fit(z,duration,event,w,hidden,seed)
            with torch.inference_mode():
                rate = (torch.nn.functional.softplus(net(torch.as_tensor(zt,dtype=torch.float32)))+1e-5).numpy()
            outputs.append(rate)
            states.append({k:v.detach().numpy() for k,v in net.state_dict().items()})
            logs.append(history)
        # Geometric ensemble of the piecewise exponential rates, fixed in protocol implementation.
        rate = np.exp(np.mean(np.log(outputs),axis=0))
        curve, quant = pch_outputs(rate)
        artifact.update(hidden=hidden,states=states,loss=logs)
    if not np.isfinite(curve).all() or not np.isfinite(quant).all():
        raise RuntimeError(f"Non-finite output for {name}")
    if model_file:
        save_json(model_file,artifact)
    return curve.astype("float32"), quant.astype("float32")


def km_fit(t, event, weight):
    times, ix = np.unique(t, return_inverse=True)
    n = np.bincount(ix, weights=weight)
    d = np.bincount(ix, weights=weight*event)
    at_risk = n[::-1].cumsum()[::-1]
    survival = np.cumprod(1-np.minimum(d/at_risk,1))
    return times, survival


def km_value(km, t, left=False):
    times, surv = km
    ix = np.searchsorted(times,t,side="left" if left else "right")-1
    return np.where(ix>=0,surv[np.maximum(ix,0)],1.)


def metrics(frame, indices, curve, quant, train_cycles):
    q = frame.loc[indices]
    tr,w = train_rows(frame,train_cycles)
    km = km_fit(tr.rul_h.to_numpy(),1-tr.event.to_numpy(),w)
    results=[]
    for cid in q.cycle_id.unique():
        m = q.cycle_id.eq(cid).to_numpy() & q.rul_h.notna().to_numpy()
        t,e = q.rul_h.to_numpy()[m],q.event.to_numpy(bool)[m]
        p,z = curve[m],quant[m]
        row=dict(cycle=cid,observed=bool(e[0]),origins=len(t))
        if e[0]:
            error=np.abs(z[:,1]-t)
            row.update(mae=float(error.mean()),last2h=float(error[t<=2].mean()),last1h=float(error[t<=1].mean()),
                       bias2h=float((z[:,1]-t)[t<=2].mean()),coverage=float(((z[:,0]<=t)&(t<=z[:,2])).mean()),width=float((z[:,2]-z[:,0]).mean()))
        else:
            row["median_below_censor_bound"]=float((z[:,1]<t).mean())
        for h in (1,4,8):
            j=int(np.flatnonzero(H==h)[0]); failure=e&(t<=h); alive=t>h
            gf=km_value(km,t,left=True); gh=float(km_value(km,np.array(h)))
            weights=failure/np.maximum(gf,.1)+alive/max(gh,.1)
            row[f"brier{h}h"]=float(np.mean(weights*(failure.astype(float)-p[:,j])**2))
            row[f"ipcw_clipped{h}h"]=float(np.mean((failure&(gf<.1))|(alive&(gh<.1))))
        results.append(row)
    return results


def summarize(rows):
    table=pd.DataFrame(rows)
    return {k:float(table[k].mean()) for k in table if k not in ("cycle","observed","origins")}


def inner_groups(cycles):
    return [cycles[k::3] for k in range(3) if cycles[k::3]]


def one_fold(data_path, out_path, tag, train_cycles, held_cycles):
    started=time.time()
    frame=pd.read_parquet(Path(data_path)/"frame.parquet")
    frame["log_vsd_30m"]=np.log(frame.vsd_mean_30m.clip(lower=.02))
    out=Path(out_path)/"folds"/tag; out.mkdir(parents=True,exist_ok=True)
    ti=frame.index[frame.cycle_id.isin(train_cycles)].to_numpy()
    hi=frame.index[frame.cycle_id.isin(held_cycles)].to_numpy()
    result=dict(tag=tag,training=train_cycles,held_out=held_cycles,models={})
    arrays=dict(test_index=hi,train_index=ti)
    for name in NAMES:
        inner_curve=np.zeros((len(frame),len(H)),dtype="float32")
        inner_quant=np.zeros((len(frame),3),dtype="float32")
        inner_rows=[]
        for val in inner_groups(train_cycles):
            fit=[c for c in train_cycles if c not in val]
            vi=frame.index[frame.cycle_id.isin(val)].to_numpy()
            c,q=fit_predict(frame,fit,vi,name)
            inner_curve[vi],inner_quant[vi]=c,q
            inner_rows+=metrics(frame,vi,c,q,fit)
        inner=summarize(inner_rows)
        c,q=fit_predict(frame,train_cycles,hi,name,out/f"{name}.json")
        result["models"][name]=dict(inner=inner,selection_loss=inner["mae"]+.5*inner["last2h"],
                                     test=metrics(frame,hi,c,q,train_cycles))
        arrays[f"{name}_curve"],arrays[f"{name}_quant"]=c,q
        arrays[f"{name}_inner_curve"]=inner_curve[ti]
        print(f"{tag} {name}: fit complete",flush=True)
    chosen=min(NAMES,key=lambda n:result["models"][n]["selection_loss"])
    result.update(selected=chosen,seconds=time.time()-started)
    np.savez_compressed(out/"predictions.npz",**arrays)
    save_json(out/"result.json",result)
    return result


def load_ev(path):
    spec=importlib.util.spec_from_file_location("frozen_evaluation",path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def alarm_eval(data, frame, result, arrays, name, grid):
    ev=load_ev(Path(data)/"evaluation.py")
    events=pd.read_csv(Path(data)/"events.csv",parse_dates=["time"])
    products=pd.read_csv(Path(data)/"products.csv",low_memory=False)
    def hours(cycles):
        q=products[products.cycle_id.isin(cycles)&products.supervised_product_eligible.eq(1)]
        return float(q.spray_samples.sum()/3600)
    ti,hi=arrays["train_index"],arrays["test_index"]
    train=frame.loc[ti].copy(); train["split"]="validation"
    selected=ev.select_threshold(train,arrays[f"{name}_inner_curve"][:,2],events[events.cycle_id.isin(result["training"])].assign(split="validation"),
                                 hours(result["training"]),grid=grid,false_budget=.05)
    threshold=float(selected["threshold"])
    m,rows,alarms=ev.evaluate(frame.loc[hi],arrays[f"{name}_curve"][:,2],threshold,events[events.cycle_id.isin(result["held_out"])],hours(result["held_out"]))
    return dict(threshold=threshold,metrics=m,events=rows)


def collect(data,out,results,protocol):
    frame=pd.read_parquet(Path(data)/"frame.parquet")
    report=dict(protocol=protocol,runs=results,comparison={})
    ledger={}
    for mode in ("loco","forward"):
        folds=[r for r in results if r["tag"].startswith(mode)]
        comparison={}
        for name in NAMES+["nested_selected"]:
            rows=[];alarm_rows=[]
            for r in folds:
                n=r["selected"] if name=="nested_selected" else name
                rows+=r["models"][n]["test"]
                with np.load(out/"folds"/r["tag"]/"predictions.npz",allow_pickle=False) as a:
                    arrays={k:a[k] for k in a.files}
                alarm=alarm_eval(data,frame,r,arrays,n,np.array(protocol["alarm_threshold_grid"]))
                alarm_rows.append(alarm)
                if mode=="loco":
                    ledger.setdefault(f"{name}_curve",np.zeros((len(frame),len(H)),"float32"))[arrays["test_index"]]=arrays[f"{n}_curve"]
                    ledger.setdefault(f"{name}_quant",np.zeros((len(frame),3),"float32"))[arrays["test_index"]]=arrays[f"{n}_quant"]
            totals={k:sum(float(a["metrics"].get(k,0)) for a in alarm_rows) for k in ("total_events","hit20","late","missed","false_alarms","unknown_alarms","operating_hours")}
            totals["false_per_100h"]=100*totals["false_alarms"]/max(totals["operating_hours"],1e-8)
            comparison[name]=dict(life=summarize(rows),alarms=totals,cycles=rows,alarm_folds=alarm_rows)
        report["comparison"][mode]=comparison
    np.savez_compressed(out/"predictions.npz",**ledger)
    save_json(out/"report.json",report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--jobs",type=int,default=8);p.add_argument("--smoke",action="store_true");p.add_argument("--collect",action="store_true")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
    save_json(args.out/"protocol.json",protocol)
    import sklearn,scipy,torch,xgboost
    save_json(args.out/"provenance.json",dict(source_sha256=digest(__file__),protocol_sha256=digest(Path(__file__).with_name("protocol.json")),
        inputs={p.name:digest(p) for p in args.data.iterdir() if p.is_file()},versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,
        sklearn=sklearn.__version__,scipy=scipy.__version__,torch=torch.__version__,xgboost=xgboost.__version__)))
    frame=pd.read_parquet(args.data/"frame.parquet");cycles=sorted(frame.cycle_id.unique())
    if args.smoke:
        frame["log_vsd_30m"]=np.log(frame.vsd_mean_30m.clip(lower=.02))
        for name in NAMES:
            ix=frame.index[frame.cycle_id.eq(cycles[-1])][::53]
            c,q=fit_predict(frame,cycles[:4],ix,name,args.out/f"smoke-{name}.json")
            assert np.all(np.diff(c,axis=1)>=-1e-6) and np.all(np.diff(q,axis=1)>=-1e-6)
            print(name,len(ix),np.median(q[:,1]),flush=True)
        return
    tasks=[(f"loco-{c}",[v for v in cycles if v!=c],[c]) for c in cycles]
    tasks += [(f"forward-{i}",[c for c in cycles if c<held[0]],held) for i,held in enumerate(protocol["forward_blocks"])]
    if args.collect:
        results=[json.loads((args.out/"folds"/tag/"result.json").read_text()) for tag,_,_ in tasks]
    else:
        results=Parallel(n_jobs=args.jobs)(delayed(one_fold)(str(args.data),str(args.out),tag,tr,te) for tag,tr,te in tasks)
    report=collect(args.data,args.out,results,protocol)
    for mode,table in report["comparison"].items():
        print(mode,flush=True)
        for name,r in table.items():
            print(name,{k:round(v,3) for k,v in r["life"].items()},r["alarms"],flush=True)


if __name__=="__main__":
    main()
