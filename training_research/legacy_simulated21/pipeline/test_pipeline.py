import sys
sys.dont_write_bytecode=True
import unittest, time, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from core import simulate, feature_rows, NUMERIC, STATES, ENERGIZED, split_for, dump, predict_window, load_model, LivePredictor

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame,cls.info,cls.plant=simulate(1,short=True)

    def test_all_channels_and_seconds(self):
        f=self.frame
        self.assertEqual(len(NUMERIC),19)
        self.assertTrue(np.isfinite(f[NUMERIC].to_numpy()).all())
        self.assertTrue((f[NUMERIC].to_numpy()>=0).all())
        self.assertTrue((f.timestamp.diff().dropna().dt.total_seconds()==1).all())
        self.assertTrue(f.operation_state.isin(STATES).all())

    def test_replacement_and_piece_rul(self):
        f=self.frame;valid=f.y_rul_valid
        self.assertEqual(f.gun_replaced.sum(),1)
        self.assertEqual(f.y_terminal.sum(),1)
        self.assertEqual(f.operation_state.iloc[-1],'maintenance')
        np.testing.assert_array_equal(f.loc[valid,'y_rul_pieces'].to_numpy(),(f.workpiece_complete.sum()-f.loc[valid,'completed_pieces']).to_numpy())
        self.assertTrue(f.loc[~valid,'y_rul_hours'].isna().all())

    def test_exchange_stops_powder(self):
        f=self.frame;v=f.loc[f.operation_state=='exchange','powder_feed_rate']
        self.assertGreater(len(v),0)
        self.assertLess(v.max(),.2)

    def test_energy_and_pressure_labels(self):
        f=self.frame
        electric=f.gun_current*f.gun_voltage/1000
        self.assertTrue((f.net_power<=electric+1e-10).all())
        np.testing.assert_allclose(f.y_pressure_60s.iloc[:-60],f.powder_tank_pressure.iloc[60:])
        self.assertTrue(f.y_pressure_60s.iloc[-60:].isna().all())

    def test_censored_labels(self):
        f,_,_=simulate(4,censored=True)
        self.assertTrue(f.y_rul_hours.isna().all())
        self.assertTrue(f.y_rul_pieces.isna().all())
        self.assertFalse(f.y_rul_valid.any())
        self.assertFalse(f.gun_replaced.any())

    def test_features_causal_no_labels(self):
        f=self.frame.copy();before=feature_rows(f,[100])
        f.loc[101:,NUMERIC]=90000
        for c in f:
            if c.startswith('y_'):f[c]=0
        f['machine_message']='SIM_WARN_FUTURE_FAILURE'
        # At row100 baseline message SIM_OK; diagnostic text produces the same safe flag.
        np.testing.assert_allclose(before,feature_rows(f,[100]))

    def test_offline_online_feature_same(self):
        f=self.frame
        for i in [59,100,200,500]:
            np.testing.assert_equal(feature_rows(f,[i]),feature_rows(f.iloc[:i+1],[i]))

    def test_plant_continuity(self):
        f,_,plant=simulate(2,len(self.frame),self.plant,short=True)
        self.assertGreaterEqual(plant['tank_load'],self.plant['tank_load'])
        self.assertLess(abs(f.powder_tank_pressure.iloc[0]-self.frame.powder_tank_pressure.iloc[-1]),.1)

    def test_repeatable(self):
        f,i,_=simulate(1,short=True)
        pd.testing.assert_frame_equal(f,self.frame)
        self.assertEqual(i,self.info)

    def test_splits(self):
        self.assertEqual([split_for(i) for i in range(1,11)],['train']*6+['calibration']*2+['test']*2)

    def test_stream_reject_and_reset(self):
        live=LivePredictor(None);records=self.frame.iloc[:30].to_dict('records')
        for r in records:result=live.push(r)
        self.assertEqual(result['status'],'warming_up')
        self.assertEqual(live.push(records[-1])['status'],'late_or_duplicate')
        nextrow=dict(records[-1],seq=31,timestamp=records[-1]['timestamp']+pd.Timedelta(seconds=2))
        self.assertEqual(live.push(nextrow)['needed'],59)
        self.assertEqual(live.heartbeat(nextrow['timestamp']+pd.Timedelta(seconds=4))['status'],'stale')
        self.assertEqual(len(live.buffer),0)

    def test_stream_units_and_retired(self):
        live=LivePredictor(None);r=self.frame.iloc[0].to_dict()
        r['units']={'gun_current':'mA'}
        self.assertEqual(live.push(r)['status'],'invalid_input')
        r=self.frame.iloc[1].to_dict();r['gun_id']=2
        self.assertEqual(live.push(r)['status'],'warming_up')
        r=self.frame.iloc[2].to_dict()
        self.assertEqual(live.push(r)['status'],'retired_gun')

def validate_dataset(out):
    out=Path(out);manifest=json.loads((out/'数据清单.json').read_text(encoding='utf-8'))
    pf=pq.ParquetFile(out/'模拟数据.parquet');previous=-1;rows=0;normal=0;anomaly=0
    mins=np.full(19,np.inf);maxs=np.full(19,-np.inf);counts={s:0 for s in STATES};checks=0;fault_counts=np.zeros(6,np.int64)
    sums=np.zeros(19);sum2=np.zeros(19);cross=np.zeros((19,19));started=time.time()
    for i,g in enumerate(manifest['guns']):
        f=pf.read_row_group(i).to_pandas();a=f[NUMERIC].to_numpy();n=len(f)
        assert f.seq.iloc[0]==previous+1 and np.all(np.diff(f.seq)==1)
        assert f.gun_id.nunique()==1 and int(f.gun_id.iloc[0])==g['gun_id']
        assert np.isfinite(a).all() and (a>=0).all()
        assert f.operation_state.isin(STATES).all() and f.machine_message.notna().all()
        assert f.timestamp.diff().dropna().dt.total_seconds().eq(1).all()
        assert not f.y_pressure_valid.iloc[-60:].any()
        np.testing.assert_allclose(f.y_pressure_60s.iloc[:-60],f.powder_tank_pressure.iloc[60:])
        valid=f.y_rul_valid
        if not g['censored']:
            assert f.gun_replaced.sum()==1 and f.y_terminal.sum()==1
            np.testing.assert_allclose(f.loc[valid,'y_rul_hours'],f.energized_hours.iloc[-1]-f.loc[valid,'energized_hours'])
            np.testing.assert_array_equal(f.loc[valid,'y_rul_pieces'].to_numpy(),(f.workpiece_complete.sum()-f.loc[valid,'completed_pieces']).to_numpy())
        else:assert f.y_rul_hours.isna().all() and not valid.any()
        exchange=f.operation_state=='exchange'
        assert f.loc[exchange,'powder_feed_rate'].max()<.2
        electric=f.gun_current*f.gun_voltage/1000
        assert (f.net_power<=electric+1e-10).all()
        mins=np.minimum(mins,a.min(0));maxs=np.maximum(maxs,a.max(0));sums+=a.sum(0);sum2+=(a*a).sum(0);cross+=a.T@a
        for s,c in f.operation_state.value_counts().items():counts[s]+=int(c)
        fault_counts+=np.bincount(f.y_anomaly,minlength=6)
        previous=int(f.seq.iloc[-1]);rows+=n;checks+=13
        if i%20==0:print(f'validate gun={i+1}/{len(manifest["guns"])}',flush=True)
    assert rows==manifest['rows']==pf.metadata.num_rows
    sd=np.sqrt(sum2/rows-(sums/rows)**2)
    corr=(cross/rows-np.outer(sums/rows,sums/rows))/np.outer(sd,sd)
    result={'rows_checked':rows,'row_groups':pf.num_row_groups,'checks':checks,'missing_process_values':0,'timestamp_gaps':0,
        'duplicate_rows':0,'seconds':time.time()-started,'parameter_stats':[{'key':k,'min':float(mins[j]),'max':float(maxs[j]),'mean':float(sums[j]/rows),'std':float(sd[j])} for j,k in enumerate(NUMERIC)],
        'correlation':corr.tolist(),'state_counts':counts,'fault_counts':fault_counts.tolist()}
    if (out/'模型.joblib').exists():
        bundle=load_model(out);f=pf.read_row_group(0).to_pandas();lat=[]
        terminal=int(np.flatnonzero(f.y_terminal)[0])
        for i in np.linspace(100,terminal-60,30,dtype=int):
            r=predict_window(bundle,f.iloc[max(0,i-700):i+1]);assert r['status']=='ok'
            lat.append(r['outputs']['latency_ms'])
        result['inference_p95_ms']=float(np.percentile(lat,95))
        checks_input={}
        for scenario in ['missing','negative','infinite','gap','duplicate','replacement','unknown_state','warmup']:
            sample=f.iloc[:70].copy()
            if scenario=='missing':sample.loc[69,NUMERIC[0]]=np.nan
            if scenario=='negative':sample.loc[69,NUMERIC[0]]=-1
            if scenario=='infinite':sample.loc[69,NUMERIC[0]]=np.inf
            if scenario=='gap':sample.loc[69,'seq']+=2
            if scenario=='duplicate':sample.loc[69,'seq']-=1
            if scenario=='replacement':sample.loc[69,'gun_id']=999
            if scenario=='unknown_state':sample.loc[69,'operation_state']='unknown'
            if scenario=='warmup':sample=sample.iloc[:20]
            r=predict_window(bundle,sample);assert r['outputs'] is None
            checks_input[scenario]=r['status']
        maintenance=predict_window(bundle,f.iloc[-701:])
        assert maintenance['status']=='maintenance' and maintenance['outputs']['rul_hours'] is None
        checks_input['maintenance']='suppressed_life_prediction'
        result['input_guard_tests']=checks_input
    dump(out/'验证结果.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['parameter_stats','correlation','state_counts']},ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':unittest.main()
