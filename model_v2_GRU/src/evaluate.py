"""One frozen evaluation of all candidates; no fitting, tuning or reselection."""
import argparse,json,time,platform
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from modeling import ROOT,load_frames,encode,TARGETS
from inference import predict_model
from evaluation import evaluate
from train import digest,dump,operating_hours

def verify(cfg):
    for name,sha in cfg['data_sha256'].items():assert digest(ROOT/'data'/name)==sha,('Data changed',name)
    for name,sha in cfg['fitted_file_sha256'].items():assert digest(ROOT/'artifacts'/name)==sha,('Weights changed',name)
    for name,sha in cfg['source_sha256'].items():assert digest(ROOT/'src'/name)==sha,('Source changed',name)
    assert digest(ROOT/'artifacts/preprocessor.json')==cfg['preprocessor_sha256']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    art=ROOT/'artifacts';cfg=json.loads((art/'frozen_config.json').read_text('utf-8'))
    assert cfg.get('selection_final'),'Run finalize_freeze.py after all validation decisions'
    verify(cfg)
    if args.verify_only:
        result=json.loads((art/'evaluation.json').read_text('utf-8'))
        assert digest(art/'frozen_config.json')==result['frozen_config_sha256']
        for name,sha in result['result_sha256'].items():assert digest(art/name)==sha,('Result changed',name)
        print('PASS: frozen code, data, weights and saved results verified');return
    if (art/'evaluation.json').exists():raise RuntimeError('Already evaluated; use --verify-only. Do not reopen for tuning.')
    started=time.perf_counter();torch.set_num_threads(4)
    opened=dict(opened_utc=datetime.now(timezone.utc).isoformat(),frozen_config_sha256=digest(art/'frozen_config.json'),selected_model=cfg['selected_model'])
    if (art/'test_opened.json').exists():
        previous=json.loads((art/'test_opened.json').read_text('utf-8'))
        assert previous['frozen_config_sha256']==opened['frozen_config_sha256'],'Cannot resume with changed selection'
    else:dump(art/'test_opened.json',opened)
    f,y,e,p=load_frames();pre=json.loads((art/'preprocessor.json').read_text('utf-8'));x=encode(f,pre)
    predictions=f[['anchor_id','timestamp','cycle_id','split']].copy();model_results={};all_events=[];all_alarms=[]
    for spec in cfg['model_specs']:
        name=spec['name'];all_scores=predict_model(spec,x,f);policy=cfg['policies'][name];score=all_scores[:,policy['head_index']]
        predictions[name]=score;model_results[name]={}
        for split in ['train','validation','test']:
            mask=f.split.eq(split).to_numpy();sf=f[mask].reset_index(drop=True);se=e[e.split.eq(split)&e.is_eol.eq(1)]
            metrics,events,alarms=evaluate(sf,score[mask],policy['threshold'],se,operating_hours(p,split))
            target='y_eol_30m' if spec['kind']=='margin' else policy['head']
            truth=y.loc[mask,target].to_numpy(float);valid=np.isfinite(truth);truth=truth[valid];risk=score[mask][valid]
            metrics['window_average_precision']=float(average_precision_score(truth,risk)) if len(np.unique(truth))==2 else None
            metrics['score_target']=target;metrics['score_semantics']='uncalibrated risk score; not remaining minutes or a calibrated probability'
            model_results[name][split]=metrics
            all_events.extend([dict(model=name,split=split,**r) for r in events]);all_alarms.extend([dict(model=name,split=split,**r) for r in alarms])
            if split=='validation':
                old=cfg['validation_results'][name]['metrics']
                for key in ['hit20','late','missed','false_alarms']:assert metrics[key]==old[key],(name,key,'validation replay changed')
        print(name,json.dumps(model_results[name]['test'],ensure_ascii=False),flush=True)
    predictions.to_csv(art/'predictions.csv.gz',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_events).to_csv(art/'event_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_alarms,columns=['model','split','index','time','cycle_id','probability','anchor_id','matched_event','event_id','false_alarm','unknown','post_event']).to_csv(art/'alarm_results.csv',index=False,encoding='utf-8-sig')
    result=dict(selected_model=cfg['selected_model'],model_version=cfg['model_version'],models=model_results,
      evaluated_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
      frozen_config_sha256=digest(art/'frozen_config.json'),
      result_sha256={name:digest(art/name) for name in ['predictions.csv.gz','event_results.csv','alarm_results.csv']},
      secondary_causes={cause:{split:int((e.cause.eq(cause)&e.split.eq(split)).sum()) for split in ['train','validation','test']} for cause in ['voltage','feedrate','pressure']},
      limitations=['10 total natural voltage endpoints, only 2 validation events; validation selection is optimistic.',
      'Historical test partly examined in V1; not an independent prospective deployment test.',
      'No feedrate consecutive event; only one pressure consecutive event in train. No reliable cause-specific early-warning model claimed.',
      'No remaining-life MAE claimed: models emit risk scores and measured alarm lead times, not a constant 20min countdown.',
      'Replay opportunities inherit offline complete-product eligibility; not full asynchronous raw PLC replay.',
      'Matching window fixed at 60min. Earlier alarms remain visible as false or unknown; persistent old episode is not re-issued at T-20.'],
      execution=dict(host=platform.node(),device='cpu',server_execution=False,site_connected=False))
    dump(art/'evaluation.json',result)
    chosen=pd.DataFrame(all_events);chosen=chosen[chosen.model.eq(cfg['selected_model'])&chosen.split.eq('test')]
    print('SELECTED',cfg['selected_model']);print(chosen[['event_id','time','deadline','first_alarm','lead_minutes','hit20','late','missed']].to_string(index=False),flush=True)

if __name__=='__main__':main()
