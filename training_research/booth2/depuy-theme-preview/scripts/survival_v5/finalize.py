"""Fit all six research artifacts, reload independently and audit held-out predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train as T
import runtime as R


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True);p.add_argument("--run",type=Path,required=True)
    args=p.parse_args()
    frame=pd.read_parquet(args.data/"frame.parquet")
    frame["log_vsd_30m"]=np.log(frame.vsd_mean_30m.clip(lower=.02))
    cycles=sorted(frame.cycle_id.unique())
    report=json.loads((args.run/"report.json").read_text(encoding="utf-8"))
    with np.load(args.run/"predictions.npz",allow_pickle=False) as source:
        ledger={k:source[k] for k in source.files}
    seen=np.zeros(len(frame),int)
    for fold in report["runs"]:
        assert not set(fold["training"])&set(fold["held_out"])
        if fold["tag"].startswith("forward"):
            assert max(fold["training"])<min(fold["held_out"])
        if fold["tag"].startswith("loco"):
            seen[frame.cycle_id.isin(fold["held_out"])] += 1
        for val in T.inner_groups(fold["training"]):
            assert not set(val)&set(fold["held_out"])
    assert np.all(seen==1)
    parity={}; errors={}
    for name in T.NAMES:
        curve,quant=ledger[name+"_curve"],ledger[name+"_quant"]
        assert curve.shape==(len(frame),12) and quant.shape==(len(frame),3)
        assert np.isfinite(curve).all() and np.isfinite(quant).all()
        assert np.all((curve>=0)&(curve<=1)) and np.all(np.diff(curve,axis=1)>=-1e-6)
        assert np.all(quant>0) and np.all(np.diff(quant,axis=1)>=-1e-6)
        mae=[]; tail=[]
        for _,g in frame[frame.event.eq(1)&frame.rul_h.notna()].groupby("cycle_id"):
            e=np.abs(quant[g.index,1]-g.rul_h.to_numpy())
            mae.append(e.mean());tail.append(e[g.rul_h.to_numpy()<=2].mean())
        expected=report["comparison"]["loco"][name]["life"]
        assert abs(np.mean(mae)-expected["mae"])<1e-6
        assert abs(np.mean(tail)-expected["last2h"])<1e-6
        errors[name]=dict(mae=float(np.mean(mae)),last2h=float(np.mean(tail)))
        delta_c,delta_q=0.,0.
        for cid in cycles:
            ix=frame.index[frame.cycle_id.eq(cid)][::181]
            c,q=R.predict(args.run/"folds"/f"loco-{cid}"/f"{name}.json",frame.loc[ix])
            delta_c=max(delta_c,float(np.max(np.abs(c-curve[ix]))))
            delta_q=max(delta_q,float(np.max(np.abs(q-quant[ix]))))
        assert delta_c<2e-5 and delta_q<.002,(name,delta_c,delta_q)
        parity[name]=dict(cdf_max_abs=delta_c,quantile_hours_max_abs=delta_q)
        print("reloaded folds",name,parity[name],flush=True)
    final=args.run/"final";final.mkdir(exist_ok=True)
    ix=frame.index[::37]
    final_parity={}
    for name in T.NAMES:
        c,q=T.fit_predict(frame,cycles,ix,name,final/f"{name}.json")
        rc,rq=R.predict(final/f"{name}.json",frame.loc[ix])
        dc,dq=float(np.abs(c-rc).max()),float(np.abs(q-rq).max())
        assert dc<2e-5 and dq<.002,(name,dc,dq)
        final_parity[name]=dict(cdf_max_abs=dc,quantile_hours_max_abs=dq)
        print("final fit",name,final_parity[name],flush=True)
    # This selects a final research family using all-cycle CV, not a new unbiased test result.
    selected=min(T.NAMES,key=lambda n:report["comparison"]["loco"][n]["life"]["mae"]+.5*report["comparison"]["loco"][n]["life"]["last2h"])
    verification=dict(outer_folds=14,forward_blocks=3,rows=len(frame),observed_cycles=int(frame.groupby("cycle_id").event.max().sum()),
                      independent_recomputed_errors=errors,held_out_runtime_parity=parity,final_runtime_parity=final_parity)
    T.save_json(args.run/"verification.json",verification)
    manifest=dict(schema_version=1,status="research_only_not_approved",auto_publish=False,
                  research_selection=selected,evaluation="report.json; final weights are fitted on all cycles and are not evaluation predictions",
                  models=T.NAMES,features_source="versioned historical frame; live causal compiler not validated",units="remaining_processing_hours",
                  files={p.name:T.digest(p) for p in sorted(final.iterdir()) if p.is_file() and p.name!="manifest.json"})
    T.save_json(final/"manifest.json",manifest)
    print("verified",verification["rows"],"origins; selected research family",selected,flush=True)


if __name__=="__main__":
    main()
