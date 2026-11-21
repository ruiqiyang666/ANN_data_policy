"""Regression checks for genuine neural inference, split isolation and retained ingress validation."""
import torch
import json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi.testclient import TestClient
from neural_net import encode_frame
from neural_api import NeuralPredictor
from core import FIELDS,NUMERIC,load_model
from server import create_app

OUT=Path(r'D:\Learning\GC\IPP\Phase2\simulation\pipeline\results')

class NeuralChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.base=load_model(OUT);cls.nn=NeuralPredictor(OUT)
        m=json.loads((OUT/'数据清单.json').read_text(encoding='utf8'));g=next(g for g in m['guns'] if g['gun_id']==9)
        cls.frame=pq.ParquetFile(OUT/'模拟数据.parquet').read_row_group(g['row_group']).to_pandas()
        cls.client=TestClient(create_app(OUT))
    def test_whole_gun_splits(self):
        c=json.loads((OUT/'neural'/'cache.json').read_text(encoding='utf8'));sets=[set(g) for g in c['splits'].values()]
        for i,a in enumerate(sets):
            for b in sets[i+1:]:self.assertFalse(a&b)
        self.assertEqual(c['identity']['sampling'],1);self.assertEqual(c['identity']['window'],60)
        self.assertNotIn(153,set.union(*sets))
    def test_no_label_id_or_diagnostic_message_leak(self):
        f=self.frame.iloc[300:1001].copy();q,x=encode_frame(f)
        for c in f:
            if c.startswith('y_'):f[c]=999999
        f['gun_id']=999;f['seq']=np.arange(len(f))+900000
        f['machine_message']='fault-name-and-future-answer'
        # Diagnostic text is neutral, just as the unchanged current SIM_OK message.
        q2,x2=encode_frame(f)
        np.testing.assert_array_equal(q[...,:29],q2[...,:29]);np.testing.assert_array_equal(x[:,:87],x2[:,:87])
        self.assertEqual(q.shape,(1,60,30));self.assertEqual(x.shape,(1,88))
        self.assertTrue(np.all(q2[...,29]==0));self.assertEqual(x2[0,87],0)
    def test_real_head_and_complete_outputs(self):
        r=self.nn.predict(self.frame.iloc[300:1001],self.base,True);o=r['outputs'];p=r['neural']
        self.assertEqual(o['model_type'],'neural_network');self.assertEqual(len(o['input_parameters']),21)
        self.assertEqual(p['scope'],'dense_readout_head');self.assertEqual(p['input_parameter_count'],21)
        acts=p['activation_layers'];layers=[self.nn.net.hidden1,self.nn.net.hidden2,self.nn.net.output]
        for i,layer in enumerate(layers):
            before=np.array(acts[i]['values']);expected=np.array(p['graph']['connections'][i]['weights'])@before+layer.bias.detach().numpy()
            if i<2:expected=np.maximum(0,expected)
            np.testing.assert_allclose(expected,acts[i+1]['values'],atol=2e-5,rtol=2e-5)
        self.assertAlmostEqual(max(0,acts[-1]['values'][0]*35),o['rul_hours'],places=4)
        for key in ['rul_pieces','rul_interval_hours','pressure_60s','gun_health','tank_health','anomaly_type','anomaly_300s_probability','root_candidates','predicted_anomaly_eta_s']:self.assertIn(key,o)
        self.assertGreaterEqual(o['anomaly_300s_probability'],0);self.assertLessEqual(o['anomaly_300s_probability'],1)
    def test_fault_attribution_and_normal_empty(self):
        normal=self.nn.predict(self.frame.iloc[300:1001],self.base)['outputs'];self.assertEqual(normal['anomaly_type'],'正常');self.assertEqual(normal['root_candidates'],[])
        fault=self.nn.predict(self.frame.iloc[3700:4401],self.base)['outputs'];self.assertNotEqual(fault['anomaly_type'],'正常');self.assertTrue(fault['root_candidates'])
        self.assertTrue(all(r['parameter'] in NUMERIC and r['score']>0 for r in fault['root_candidates']))
    def test_api_baseline_and_neural_distinct(self):
        nn=self.client.get('/api/detail?gun=9&index=1000').json();base=self.client.get('/api/detail?gun=9&index=1000&model_id=baseline').json()
        self.assertIn('neural',nn);self.assertNotIn('neural',base);self.assertNotEqual(nn['outputs']['model_version'],base['outputs']['model_version'])
        self.assertEqual(self.client.get('/api/detail?model_id=bad').status_code,422)
    def test_ingress_still_requires_full_window_and_rejects_units(self):
        with TestClient(create_app(OUT)) as client:
            records=json.loads(self.frame.iloc[:60][[x['key'] for x in FIELDS]+['timestamp','gun_id','seq']].to_json(orient='records',date_format='iso'))
            for i,row in enumerate(records):
                r=client.post('/api/live',json=row,headers={'X-Phase2-Request':'1'}).json()
                if i<59:self.assertIsNone(r['outputs'])
            self.assertEqual(r['outputs']['model_type'],'neural_network')
            bad=records[-1].copy();bad['seq']+=1;bad['timestamp']=(pd.Timestamp(bad['timestamp'])+pd.Timedelta(seconds=1)).isoformat();bad['units']={'gun_current':'not-amps'}
            r=client.post('/api/live',json=bad,headers={'X-Phase2-Request':'1'}).json();self.assertEqual(r['status'],'invalid_input');self.assertIsNone(r['outputs'])
    def test_missing_window_and_replacement(self):
        self.assertIsNone(self.nn.predict(self.frame.iloc[:59],self.base)['outputs'])
        f=self.frame.iloc[:60].copy();f.loc[f.index[-1],'gun_id']=10
        self.assertEqual(self.nn.predict(f,self.base)['status'],'gap_or_replacement')

    def test_causal_lifetime_history_matches_stream(self):
        from lifecycle_neural import history_features,HistoryState
        frame=self.frame.iloc[:400];batch=history_features(frame);state=HistoryState()
        stream=np.array([state.step(row) for row in frame.to_dict('records')])
        np.testing.assert_allclose(batch,stream,rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(batch[:200],history_features(frame.iloc[:200]),rtol=1e-5,atol=1e-5)
        changed=frame.copy();changed['gun_id']=123;changed['y_rul_hours']=99999
        np.testing.assert_array_equal(batch,history_features(changed))

    def test_lifetime_report_matches_saved_test(self):
        report=self.client.get('/api/model-report?model_id=lifetime').json()
        summary=json.loads((OUT/'neural/lifetime_summary.json').read_text(encoding='utf8'))
        self.assertAlmostEqual(report['mae'],summary['test']['rul_hours']['mae'],places=5)
        self.assertLess(report['mae'],4.55)
        self.assertEqual(report['importance'],[])

    def test_tank_health_head_matches_report_and_inference(self):
        summary=json.loads((OUT/'neural/tank_summary.json').read_text(encoding='utf8'))
        report=self.client.get('/api/model-report?model_id=lifetime&target=tank_health').json()
        self.assertAlmostEqual(report['mae'],summary['test']['mae'],places=5)
        self.assertLess(report['mae'],4.1487)
        from lifecycle_neural import LifetimeNet
        from neural_net import normalize_inputs
        q,x=encode_frame(self.frame.iloc[300:1001]);self.nn.load();_,flat=normalize_inputs(q,x,self.nn.checkpoint['stats'])
        c=torch.load(OUT/'neural/tank_health.pt',map_location='cpu',weights_only=True);net=LifetimeNet(88,'ple',c['bins'],1);net.load_state_dict(c['state_dict']);net.eval()
        with torch.inference_mode():expected=float(net(torch.tensor(flat))[0,0].clamp(0,1)*100)
        actual=self.nn.predict(self.frame.iloc[300:1001],self.base)['outputs']['tank_health']
        self.assertAlmostEqual(actual,expected,places=4)

    def test_lifetime_trace_and_offline_prediction_agree(self):
        from lifecycle_neural import LifetimePredictor,history_features
        from neural_net import normalize_inputs
        frame=self.frame.iloc[:1001];q,x=encode_frame(frame)
        self.nn.load();_,flat=normalize_inputs(q,x,self.nn.checkpoint['stats'])
        values,c,graph,acts=LifetimePredictor(OUT/'neural').predict(flat,history_features(frame,[-1])[0])
        live=self.client.get('/api/detail?gun=9&index=1000').json()
        self.assertAlmostEqual(live['outputs']['rul_hours'],values[0],places=4)
        self.assertEqual(live['outputs']['history_context'],'完整观测周期')
        self.assertEqual(live['neural']['graph']['layers'][-1]['units'],3)
        self.assertIn('SiLU',live['neural']['note'])

    def test_explicit_start_stream_and_cold_start(self):
        rows=json.loads(self.frame.iloc[:61][[x['key'] for x in FIELDS]+['timestamp','gun_id','seq']].to_json(orient='records',date_format='iso'))
        with TestClient(create_app(OUT)) as client:
            for i,row in enumerate(rows[:60]):
                row={**row,'lifecycle_start':i==0};result=client.post('/api/live',json=row,headers={'X-Phase2-Request':'1'}).json()
            self.assertEqual(result['outputs']['history_context'],'完整观测周期')
            snapshot=client.get('/api/snapshot?gun=9&index=59').json()
            self.assertAlmostEqual(result['outputs']['rul_hours'],snapshot['outputs']['rul_hours'],places=4)
            gap=rows[60].copy();gap['seq']+=2;gap['timestamp']=(pd.Timestamp(gap['timestamp'])+pd.Timedelta(seconds=2)).isoformat()
            result=client.post('/api/live',json=gap,headers={'X-Phase2-Request':'1'}).json()
            self.assertEqual(result['status'],'warming_up');self.assertIsNone(result['outputs'])

if __name__=='__main__':unittest.main(verbosity=2)
