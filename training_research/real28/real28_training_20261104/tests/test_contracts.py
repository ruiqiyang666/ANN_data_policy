import sys
from pathlib import Path
import unittest
import numpy as np
import torch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from build_windows import hazard_targets,rolling_stats,lookup_ids,terminal_chain
from modeling import RealTCN,hazard_distribution,weighted_stratified_ids
from survival import lognormal_nll
from train_lifetime import bounds
from evaluate_run import alert_indices

class Contracts(unittest.TestCase):
    def test_count_interval_keeps_zero_remaining_pieces(self):
        samples=pd.DataFrame({'rul_lower_pieces':[0,3], 'rul_upper_pieces':[0,np.inf], 'rul_event':[True,False]})
        lo,hi,event=bounds(samples,'pieces')
        np.testing.assert_array_equal(lo,[0,3]);self.assertEqual(hi[0],1)
        self.assertTrue(np.isinf(hi[1]))
        self.assertTrue(np.isfinite(lognormal_nll(np.zeros(2),1,lo,hi,event)).all())
    def test_interval_likelihood_handles_both_tails(self):
        loss=lognormal_nll(np.array([-20.,20.]),.3,np.array([1.,1.]),np.array([1.01,1.01]),np.array([True,True]))
        self.assertTrue(np.isfinite(loss).all());self.assertTrue((loss>0).all())
    def test_alarm_merging_does_not_count_dense_windows_as_events(self):
        samples=pd.DataFrame({'timestamp':pd.to_datetime([0,10,20,30,40,310,320],unit='s'),
                              'event_in_300s':False,'observed_seconds':300})
        alarms=alert_indices(samples,np.array([.8,.8,.1,.8,.8,.8,.8]),.5)
        # Sustained scores form one alert; quick retrigger is suppressed; long gap resets.
        np.testing.assert_array_equal(alarms,[0,5])
    def test_unobserved_no_event_horizon_is_not_false_alarm(self):
        samples=pd.DataFrame({'timestamp':pd.to_datetime([0,10],unit='s'),
                              'event_in_300s':[False,True],'observed_seconds':[100,100]})
        alarms=alert_indices(samples,np.ones(2),.5)
        np.testing.assert_array_equal(alarms,[1])
    def test_censor_likelihood_rewards_survival_beyond_observation(self):
        a=lognormal_nll(np.log(np.array([1.,4.])),.5,np.array([2.,2.]),np.array([np.inf,np.inf]),np.array([False,False]))
        self.assertGreater(a[0],a[1]);self.assertTrue(np.isfinite(a).all())
    def test_zero_remaining_pieces_interval_has_finite_likelihood(self):
        loss=lognormal_nll(np.array([0.]),1,np.array([0.]),np.array([1.]),np.array([True]))
        self.assertAlmostEqual(float(loss[0]),np.log(2),places=7)
    def test_failure_onset_precedes_next_morning_replacement(self):
        products=pd.DataFrame({'product_id':['normal','evening_low','morning_low1','morning_low2'],
                               'label_valid':[1,1,1,1],'voltage_fail':[0,1,1,1]})
        tail=terminal_chain(products)
        self.assertEqual(tail[-1]['product_id'],'evening_low')
    def test_no_future_event_id_uses_empty_sentinel(self):
        self.assertEqual(lookup_ids(['E1','E2'],np.array([0,2])).tolist(),['E1',''])
        self.assertEqual(lookup_ids([],np.array([0])).tolist(),[''])
    def test_event_at_horizon_and_partial_followup(self):
        y,m,e,d,o=hazard_targets(np.zeros(6,dtype='int64'),np.array([30,31,300,301,240,0]),np.array([300,300,300,300,239,300]))
        self.assertEqual(np.argmax(y[0]),0);self.assertEqual(np.argmax(y[1]),1)
        self.assertEqual(np.argmax(y[2]),9);self.assertEqual(m[2].sum(),10)
        self.assertFalse(e[3]);self.assertFalse(e[4]);self.assertFalse(e[5])
        self.assertEqual(m[4].sum(),7)
    def test_causal_encoder_has_no_future_dependence(self):
        torch.manual_seed(42);model=RealTCN(3).eval()
        a=torch.randn(2,56,120);b=a.clone();b[:,:,80:]+=100
        with torch.no_grad():u=model.encode(a)[0];v=model.encode(b)[0]
        torch.testing.assert_close(u[:,:,:80],v[:,:,:80],atol=0,rtol=0)
    def test_hazard_distribution_and_eta(self):
        p,eta,m=hazard_distribution(torch.zeros(2,10))
        self.assertTrue(torch.allclose(p,torch.full((2,),1-2**-10)))
        self.assertTrue(((eta>=15)&(eta<=285)).all())
        self.assertTrue(torch.allclose(p,m.sum(1)))
    def test_rolling_summary_stops_at_block_and_time(self):
        t=np.arange(1000);x=np.ones((1000,28));mask=np.ones_like(x,dtype=bool)
        x[:500]=100;x[601:]=999
        block=np.r_[np.zeros(500,dtype=int),np.full(500,500)]
        stats,_=rolling_stats(t,x,mask,np.array([600]),block)
        np.testing.assert_allclose(stats[0,:11],1)
    def test_stratified_importance_preserves_prevalence(self):
        ids=np.arange(10000);event=ids<50;rng=np.random.default_rng(2)
        draw,w=weighted_stratified_ids(ids,event,1000,rng)
        self.assertAlmostEqual(float((w*event[draw]).sum()/w.sum()),.005,places=6)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
