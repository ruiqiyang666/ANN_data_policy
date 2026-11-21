"""Local synthetic artifact integration checks. Never connects to equipment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import synthetic_training as builder
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from api.application import create_app
from domain.contract import REG_TARGETS
from domain.features import feature_rows


def check(output):
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in builder.SOURCE.rglob('*') if p.is_file()}
    report = builder.build_artifacts(output)
    output = Path(output).resolve()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in source_hashes.items())
    checks = ['Original source hashes unchanged', 'Actual baseline and Patch model fitted']
    manifest = json.loads((output/'数据清单.json').read_text())
    groups = [set(report['splits'][s]['guns']) for s in ('train','calibration','test')]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert groups == [set(range(1,7)),{7,8},{9,10}]
    assert set(report['feature_state_counts']['train']) == set(builder.STATES)
    checks.append('Whole-cycle split 6/2/2; all ten training states sampled')
    frame,_ = builder.make_gun(9)
    indices=np.arange(59,len(frame)-300,60)
    features=feature_rows(frame,indices)
    changed=frame.copy();changed['gun_id']=999;changed['y_rul_hours']=99999;changed['workpiece_id']=777
    assert np.array_equal(features,feature_rows(changed,indices))
    checks.append('Target, cycle ID and piece ID excluded from original features')
    frame2,_=builder.make_gun(9)
    pd.testing.assert_frame_equal(frame,frame2)
    checks.append('Seeded data generation reproducible')
    try: builder.build_artifacts(output)
    except FileExistsError: checks.append('Existing artifacts refused without overwrite')
    else: raise AssertionError('Existing artifacts overwritten')
    app=create_app(output,real_data=builder.COPY_ROOT/'sample-data/csv')
    endpoints=[]
    with TestClient(app,base_url='http://127.0.0.1') as c:
        for path in ['/api/health','/api/overview','/workbench/api/training/realtime-ann/models','/workbench/api/training/realtime-ann/readiness','/api/series?gun=9','/api/gun-analysis?gun=9','/api/evaluation?gun=9','/api/simulation/config']:
            response=c.get(path);assert response.status_code==200,(path,response.text)
            endpoints.append(dict(path=path,status=response.status_code))
        for model,filename in [('baseline','评估预测.parquet'),('patch','neural/evaluation.parquet')]:
            evaluation=pd.read_parquet(output/filename)
            for target in REG_TARGETS:
                path=f'/api/model-report?model_id={model}&target={target[2:]}'
                response=c.get(path);assert response.status_code==200,(path,response.text)
                data=response.json(); expected=float(np.mean(np.abs(evaluation[target]-evaluation['pred_'+target[2:]])))
                assert np.isclose(data['mae'],expected)
                assert len(data['samples'])>0
                endpoints.append(dict(path=path,status=response.status_code,mae=data['mae'],rows=data['rows']))
        for model in ('baseline','active'):
            for path in [f'/api/snapshot?gun=9&index=1000&model_id={model}&with_trace=true',f'/api/detail?gun=9&index=1000&model_id={model}']:
                response=c.get(path);assert response.status_code==200,(path,response.text)
                data=response.json();assert data['outputs'] and np.isfinite(data['outputs']['rul_hours'])
                if model=='active': assert data['neural']['real_neuron_activations_available']
                endpoints.append(dict(path=path,status=response.status_code,outputs=True,neural=model=='active'))
            headers={'X-Phase2-Request':'1'}
            response=c.post('/api/simulation/sessions',json={'model_id':model},headers=headers)
            assert response.status_code==200,response.text
            first=response.json();assert first['prediction_ready'] and first['outputs']
            sid=first['session_id']
            response=c.post(f'/api/simulation/sessions/{sid}/advance',json={'model_id':model,'expected_seq':first['seq'],'steps':3},headers=headers)
            assert response.status_code==200,response.text
            second=response.json();assert second['seq']==first['seq']+3 and second['outputs']
            assert second['timestamp']!=first['timestamp']
            if model=='active': assert second['neural']['sample_id']!=first['neural']['sample_id']
            endpoints.append(dict(path='/api/simulation/sessions + advance',model=model,status=200,sequence_advances=True,outputs=True))
            assert c.delete(f'/api/simulation/sessions/{sid}',headers=headers).status_code==200
    checks.append('Native reports recompute matching MAE for all five outputs, both models')
    checks.append('Native snapshot, detail and generated live simulation return actual predictions and neural activations')
    result=dict(checks=checks,endpoints=endpoints,source='synthetic',artifact_directory=str(output),source_files_checked=len(source_hashes),
                limitations=builder.LIMITATIONS)
    destination=output.parent/(output.name+'.verification.json');builder.write_json(destination,result)
    print(json.dumps(dict(verification=str(destination),checks=len(checks),endpoint_checks=len(endpoints)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    check(parser.parse_args().output)
