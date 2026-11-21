"""Regression checks for ongoing episodes and persistent pre-product alarms."""
import sys
sys.dont_write_bytecode=True
import torch  # Windows DLL import order matches the training processes.
import numpy as np
import pandas as pd
from prepare_v3 import build_episodes
from metrics_v2 import alarms
from common_v2 import ROOT,dump

def main():
    # Overnight stop is not recovery; two observed normal results resolve it.
    t=pd.to_datetime(['2026-09-01 17:00','2026-09-01 17:03','2026-09-02 07:00','2026-09-02 07:03','2026-09-02 07:06','2026-09-02 07:09'])
    p=pd.DataFrame({'start':t,'label_available_at':t+pd.Timedelta(seconds=65),'label_valid':1,'voltage_fail':[1,1,1,0,0,1],'product_id':[f'p{i}' for i in range(6)]})
    e=build_episodes(p,'voltage');assert len(e)==2 and e.iloc[0].failed_products==3
    assert e.iloc[0].recovery_at==t[4]+pd.Timedelta(seconds=65)
    p.voltage_fail=0;assert len(build_episodes(p,'voltage'))==0
    # A single score spike does not alert; a sustained pair alerts after 10s.
    s=pd.DataFrame({'timestamp':pd.to_datetime(np.arange(6)*10,unit='s'),'event_in_300s':True,'observed_seconds':300,'event_id':'e1'})
    _,_,a=alarms(s,np.array([.9,.1,.9,.9,.9,.1]),.5);np.testing.assert_array_equal(a,[3])
    # A clock gap cannot bridge the persistence requirement.
    s.timestamp=pd.to_datetime([0,10,100,200,300,400],unit='s');_,_,a=alarms(s,np.ones(6),.5);np.testing.assert_array_equal(a,[1])
    dump({'passed':True,'checks':['no new failure solely from overnight stop','ordinary normal products create no event','two normal results resolve ongoing episode','isolated spike suppressed','persistent alert timing exact','clock gaps cannot create false persistence']},ROOT/'evaluation/episode_policy_regression.json')
    print('6 episode/alarm regression checks passed')
if __name__=='__main__':main()
