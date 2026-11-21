"""Phase2 synthetic, local-only workflow. Labels are never feature columns."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import json, math, time, hashlib, os
from pathlib import Path
from functools import lru_cache
from collections import deque
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / 'contract.json').read_text(encoding='utf-8'))
FIELDS = CONTRACT['fields']
NUMERIC = [f['key'] for f in FIELDS if f['type'] == 'number']
STATES = CONTRACT['states']
ENERGIZED = CONTRACT['energized_states']
FAULTS = ['normal', 'water_restriction', 'carrier_leak', 'feeder_jam', 'primary_gas_sag', 'tank_pressure_loss']
FAULT_NAMES = ['正常', '冷却水路受限', '载气泄漏', '送粉机构卡滞', '主气供给下降', '料斗压力损失']
CAUSES = ['cooling_water_flow', 'feeder_carrier_gas', 'feeder_disk_speed', 'primary_gas_flow', 'powder_tank_pressure']
REG_TARGETS = ['y_rul_hours', 'y_rul_pieces', 'y_gun_health', 'y_tank_health', 'y_pressure_60s']
VERSION = 'phase2-21ch-1hz-20261005'
SEED = 20261005

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def response(x, tau, initial=0.):
    a = np.exp(-1 / tau)
    return lfilter([1-a], [1, -a], np.asarray(x, dtype=float), zi=[a*initial])[0]

@lru_cache(maxsize=1)
def reference():
    raw = pd.read_csv(ROOT / 'reference.csv', skiprows=[1]).groupby('time')[['Current', 'Voltage', 'flow rate']].mean()
    regular = pd.DataFrame({c: np.interp(np.arange(3599), raw.index, raw[c]) for c in raw})
    return (regular - regular.rolling(61, center=True, min_periods=1).mean()).to_numpy()

def schedule(n, rng):
    states, piece = [], []
    t = 0; p = 0
    def add(s, duration):
        nonlocal t
        k = min(int(duration), n-t)
        states.extend([s]*k); piece.extend([p]*k); t += k
    while t < n:
        add('startup_purge', 10); add('ignition_settle', 20)
        for _ in range(int(rng.integers(20, 36))):
            p += 1
            add('feed_prepare', 3); add('spraying', rng.integers(20, 46))
            add('powder_clear', 7); add('exchange', rng.integers(5, 13))
            if t == n: break
        add('shutdown_purge', 20); add('cooling', 60); add('standby', rng.integers(120, 301))
    return np.array(states), np.asarray(piece, dtype=np.int32)

def simulate(gun, offset=0, plant=None, seed=SEED, censored=False, short=False):
    """One independent gun; non-gun plant history crosses replacements."""
    plant = dict(plant or {'tank_load': .05, 'tank_pressure': 2.1})
    health, timing, signal, fault_rng = [np.random.default_rng(x) for x in np.random.SeedSequence([seed, gun]).spawn(4)]
    category = int(health.choice(3, p=[.85, .12, .03]))
    low, high = [(26, 42), (10, 26), (2, 10)][category]
    nominal = health.triangular(*[(28.5, 34, 39.5), (11.5, 19, 24.5), (3, 6, 9)][category])
    if short: low, high, nominal = .08, .22, .14
    n = 3600 if censored else int(high*3600*1.55)+7200
    states, pieces = schedule(n, timing)
    arc = np.isin(states, ENERGIZED)
    # Moderate process faults: gradual precursor, active period, recovery. No terminal labels in messages/features.
    severity = np.zeros((n, 5)); anomaly = np.zeros(n, np.int8); events = []
    onset = 180 if short else int(fault_rng.integers(2400, 4800))
    while onset < n-600:
        kind = int(fault_rng.integers(0, 5)); duration = int(fault_rng.integers(150, 481))
        lead = int(fault_rng.integers(120, 301)); end = min(n, onset+duration)
        amplitude = fault_rng.uniform(.55, 1.)
        begin = max(0, onset-lead)
        severity[begin:onset, kind] = np.linspace(0, .38, onset-begin)*amplitude
        severity[onset:end, kind] = amplitude
        recover = min(n, end+120)
        severity[end:recover, kind] = np.linspace(amplitude, 0, recover-end)
        anomaly[onset:recover-45] = kind+1
        events.append({'onset': onset, 'end': recover-45, 'kind': kind+1, 'precursor_start': begin})
        onset = recover + int(fault_rng.integers(1800, 7200))
    hours = np.cumsum(arc)/3600
    load = health.lognormal(-.0128, .16, int(pieces.max())+1)[pieces]
    increment = arc*load*(1+.7*severity[:,0]+.2*severity[:,3]) / (nominal*3600)
    shock_rate = [.012, .065, .3][category]
    increment += (health.random(n) < shock_rate/3600)*arc*health.uniform(.002, .03 if category else .007, n)
    damage = np.cumsum(increment)
    damage = np.where(hours < low, np.minimum(damage, .985), damage)
    damage = np.maximum.accumulate(np.where(hours >= high, 1., np.minimum(damage, 1.)))
    terminal = None
    if not censored:
        terminal = int(np.flatnonzero(damage >= 1)[0])
        cut = terminal+1
        downtime = int(timing.integers(900,1801)) if not short else 120
        tail = np.array(['shutdown_purge']*20+['cooling']*60+['maintenance']*downtime)
        states = np.r_[states[:cut], tail]; pieces = np.r_[pieces[:cut], np.zeros(len(tail), np.int32)]
        damage = np.r_[damage[:cut], np.ones(len(tail))]
        severity = np.vstack([severity[:cut], np.zeros((len(tail), 5))])
        anomaly = np.r_[anomaly[:cut], np.zeros(len(tail), np.int8)]
    n = len(states); sec = np.arange(n); global_sec = offset+sec
    arc = np.isin(states, ENERGIZED); feed = np.isin(states, ['feed_prepare', 'spraying'])
    present = np.isin(states, ['feed_prepare','spraying','powder_clear'])
    purge = np.isin(states, ['startup_purge','shutdown_purge'])
    completed = (states == 'powder_clear') & np.r_[False, states[:-1] == 'spraying']
    # Truncated terminal pieces never count as completed.
    if terminal is not None: completed[terminal+1:] = False
    count = np.cumsum(completed).astype(np.int32)
    hours = np.cumsum(arc)/3600
    raw = reference()
    # Disjoint reference thirds by gun bucket; all remain one limited public source.
    block = (gun-1) % 10
    part = 0 if block < 6 else 1 if block < 8 else 2
    source = raw[part*1199:(part+1)*1199]
    starts = signal.integers(0, len(source)-30, (n+29)//30)
    residual = source[(starts[:,None]+np.arange(30)).ravel()][:n]
    wear = damage**health.uniform(1.7, 3.0) * health.uniform(.85, 1.2)
    recipe = 1 + .025*np.sin(2*np.pi*global_sec/(9*3600))
    tank_load = plant['tank_load'] + np.cumsum(feed)/(70*3600)
    service = np.zeros(n, bool)
    eligible = np.flatnonzero((tank_load >= .75) & (states == 'standby'))
    if len(eligible):
        k = int(eligible[0]); service[k] = True; tank_load[k:] -= tank_load[k]-.04
    tank_load = np.clip(tank_load, 0, 1)
    noise = lambda scale: response(signal.normal(0,scale,n), 2.)
    d = {}
    d['primary_gas_flow'] = response(np.maximum(0,24*recipe*(1-.30*severity[:,3])+residual[:,2])*arc + 5*purge, .6)
    d['secondary_gas_flow'] = response((6*recipe+noise(.015))*arc, 1.2)
    d['shroud_gas_flow'] = response((12+noise(.04))*arc, 1.5)
    d['cooling_gas_flow'] = response(np.where(arc|purge,80.,20.)+noise(.2), 2.,20.)
    d['air_jet'] = response(np.where(present,4.,1.)+noise(.015),.7,1.)
    d['silvent_air_knife'] = response(np.where(present,3.,.5)+noise(.012),1.,.5)
    d['feeder_carrier_gas'] = response((1.5+noise(.005))*(arc|purge)*(1-.42*severity[:,1]),1.)
    d['powder_tank_pressure'] = response(2.1-.15*feed-.12*tank_load-.24*severity[:,1]-.65*severity[:,4]+noise(.02),4.,plant['tank_pressure'])
    d['feeder_disk_speed'] = response((42*recipe+noise(.15))*feed*(1-.48*severity[:,2]),.8)
    d['feeder_stirrer_speed'] = response((30+noise(.1))*arc*(1-.12*severity[:,2]),2.)
    d['feeder_vibrator_pressure'] = response((1.38+noise(.015))*feed*(1-.14*severity[:,2]),.5)
    d['kerosene_back_pressure'] = response((7.6+.12*feed+.08*wear+noise(.025))*arc,1.5)
    d['cooling_water_flow'] = response(12*(1-.32*severity[:,0])+noise(.055),3.,12.)
    d['gun_current'] = response(np.maximum(0,520*recipe+signal.normal(0,1.2)+residual[:,0]*(1+.4*wear))*arc,.4)
    d['gun_voltage'] = response(np.maximum(0,56+signal.normal(0,.2)+1.8*wear+residual[:,1]*(1+.8*wear)+.20*(d['secondary_gas_flow']-6)+.7*severity[:,3])*arc,.3)
    d['power_supply_voltage'] = d['gun_voltage']+.003*d['gun_current']
    electric = d['gun_current']*d['gun_voltage']/1000
    # First-order heat removal. Water flow changes inferred temperature rise and wear stress.
    heat = response(.35*electric, 2.)
    d['net_power'] = np.maximum(0, electric-heat)
    d['cooling_water_conductivity'] = 2+.001*heat/(d['cooling_water_flow']/60*4.186)+.15*(1-np.exp(-global_sec/(120*3600)))+noise(.004)
    powder = 40*(d['feeder_disk_speed']/42)*np.clip(d['feeder_carrier_gas']/1.5,0,1)*np.clip(d['feeder_vibrator_pressure']/1.38,.8,1.05)*np.clip(d['powder_tank_pressure']/1.95,.5,1)*(1+signal.normal(0,.004,n))
    d['powder_feed_rate'] = response(np.r_[0.,powder[:-1]],.6)
    for key in NUMERIC: d[key] = np.maximum(0,d[key])
    change = np.r_[True, states[1:] != states[:-1]]
    message = np.where(change, np.char.add('SIM_STATE_', states), 'SIM_OK')
    # State messages are ingested/displayed; diagnostic messages are excluded from learning.
    frame = pd.DataFrame({k:d[k] for k in NUMERIC})
    frame['timestamp'] = pd.Timestamp('2026-01-01',tz='Asia/Shanghai') + pd.to_timedelta(global_sec,unit='s')
    frame['seq'] = global_sec.astype(np.int64); frame['gun_id'] = np.int32(gun)
    frame['operation_state'] = states; frame['machine_message'] = message
    frame['workpiece_id'] = pieces; frame['workpiece_complete'] = completed
    frame['energized_hours'] = hours; frame['completed_pieces'] = count
    frame['tank_service'] = service; frame['gun_replaced'] = False
    if terminal is not None: frame.loc[n-1,'gun_replaced'] = True
    valid_life = np.ones(n,bool) if censored else sec<=terminal
    frame['y_rul_valid'] = valid_life & (not censored)
    frame['y_rul_hours'] = np.where(valid_life, hours[-1]-hours, np.nan) if not censored else np.nan
    frame['y_rul_pieces'] = np.where(valid_life, count[-1]-count,np.nan) if not censored else np.nan
    frame['y_remaining_wall_s'] = np.where(valid_life, terminal-sec,np.nan) if not censored else np.nan
    frame['y_gun_health'] = 100*(1-damage)
    frame['y_tank_health'] = 100*np.clip(1-tank_load-.65*severity[:,4]-.1*severity[:,1],0,1)
    frame['y_anomaly'] = anomaly
    next_kind = anomaly.copy(); eta = np.where(anomaly>0,0.,np.nan)
    observed_events = []
    for event in events:
        onset = event['onset']
        if onset >= (terminal if terminal is not None else n): continue
        end = min(event['end'], terminal+1 if terminal is not None else n)
        begin = max(0,onset-300)
        next_kind[begin:onset] = event['kind']; eta[begin:onset] = np.arange(onset-begin,0,-1)
        observed_events.append(dict(event, end=end, onset_seq=offset+onset))
    frame['y_next_anomaly_300s'] = next_kind
    frame['y_anomaly_eta_s'] = eta
    frame['y_anomaly_valid'] = (sec < n-300) | (next_kind>0)
    frame['y_root_parameter'] = np.where(anomaly>0, np.asarray([-1]+[NUMERIC.index(x) for x in CAUSES],np.int16)[anomaly],-1)
    frame['y_pressure_60s'] = frame['powder_tank_pressure'].shift(-60)
    frame['y_pressure_valid'] = sec < n-60
    frame['y_terminal'] = False
    if terminal is not None: frame.loc[terminal,'y_terminal'] = True
    frame['is_censored'] = censored
    info = {'gun_id':gun,'rows':n,'start_seq':offset,'end_seq':offset+n-1,'life_hours':float(hours[-1]),
            'spray_hours':float((states=='spraying').sum()/3600),'wall_hours':n/3600,'pieces':int(count[-1]),
            'category':['normal_lifetime','early_degradation','gun_fault'][category], 'censored':censored,
            'terminal_seq':offset+terminal if terminal is not None else None,'events':observed_events,
            'tank_services':int(service.sum()),'reference_partition':part}
    return frame, info, {'tank_load':float(tank_load[-1]),'tank_pressure':float(d['powder_tank_pressure'][-1])}

def split_for(gun):
    """Deterministic entire-gun split; raw-reference pools are also partitioned."""
    b = (gun-1)%10
    return 'train' if b<6 else 'calibration' if b<8 else 'test'

def generate(out, target_gb=5., max_guns=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    final=out/'模拟数据.parquet'; temp=out/'模拟数据.parquet.partial'
    if final.exists(): raise FileExistsError('Dataset exists; use a different output or explicitly remove the intended file.')
    writer=None; guns=[]; offset=0; plant=None; started=time.time(); gun=1
    try:
        while True:
            frame, info, plant = simulate(gun,offset,plant)
            info['split']=split_for(gun)
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:
                schema=table.schema.with_metadata({b'phase2':VERSION.encode(),b'sample_seconds':b'1',b'process_channels':b'21'})
                writer=pq.ParquetWriter(temp,schema,compression='NONE',use_dictionary=['operation_state','machine_message'],write_statistics=True)
            writer.write_table(table,row_group_size=len(frame))
            info['row_group']=len(guns);guns.append(info);offset+=len(frame)
            size=temp.stat().st_size
            dump(out/'进度.json',{'stage':'generating','guns':gun,'rows':offset,'bytes':size,'target_bytes':target_gb*1e9,'elapsed_s':time.time()-started})
            print(f'generate gun={gun} rows={offset:,} disk={size/1e9:.3f} GB elapsed={time.time()-started:.1f}s',flush=True)
            if (size>=target_gb*1e9 and gun>=10) or (max_guns and gun>=max_guns): break
            gun+=1
        frame,info,plant=simulate(gun+1,offset,plant,censored=True)
        writer.write_table(pa.Table.from_pandas(frame,preserve_index=False),row_group_size=len(frame))
        info.update(row_group=len(guns),split='replay');guns.append(info);offset+=len(frame)
    finally:
        if writer: writer.close()
    os.replace(temp,final)
    manifest={'version':VERSION,'seed':SEED,'sample_seconds':1,'numeric_channels':19,'process_channels':21,
              'rows':offset,'bytes':final.stat().st_size,'compression':'NONE','target_bytes':target_gb*1e9,
              'complete_guns':len(guns)-1,'guns':guns,'fields':FIELDS,'states':STATES,'faults':FAULT_NAMES,
              'generation_seconds':time.time()-started,'format_note':'One row group per gun, full 1 Hz trajectories, no padding or duplicate copies.',
              'assumption':'Hypothetical composite equipment; timings, faults, damage and auxiliary channels are synthetic assumptions.'}
    dump(out/'数据清单.json',manifest)
    return manifest

def feature_rows(frame, indices=None):
    """Exactly causal 60 s, no target/ID/future/replacement/diagnostic-message input."""
    n=len(frame); idx=np.arange(59,n) if indices is None else np.asarray(indices)
    idx=idx[idx>=59]
    a=frame[NUMERIC].to_numpy(float)
    windows=a[idx[:,None]-np.arange(59,-1,-1)]
    st=frame['operation_state'].to_numpy()
    changed=np.r_[True,st[1:]!=st[:-1]]
    start=np.maximum.accumulate(np.where(changed,np.arange(n),0))
    # Operational message flag is safe; fault/error text never becomes a feature.
    msgs=frame['machine_message'].fillna('').astype(str)
    msg_state=msgs.str.startswith('SIM_STATE_').to_numpy()[idx].astype(float)
    feature=np.column_stack([a[idx],windows.mean(1),windows.std(1),(windows[:,-1]-windows[:,0])/59,
        *[(st[idx]==s).astype(float) for s in STATES], np.minimum(idx-start[idx],600),msg_state])
    return feature.astype(np.float32)

def sample_training(out, stride=60):
    out=Path(out); manifest=json.loads((out/'数据清单.json').read_text(encoding='utf-8')); pf=pq.ParquetFile(out/'模拟数据.parquet')
    parts={s:[] for s in ['train','calibration','test']}; stats=[]
    for g in manifest['guns']:
        if g['split']=='replay':continue
        f=pf.read_row_group(g['row_group']).to_pandas()
        idx=np.arange(59,len(f)-60,stride)
        X=feature_rows(f,idx)
        target=f.iloc[idx].copy();target['local_index']=idx
        parts[g['split']].append((X,target))
        if g['split']=='train':
            normal=f[(f.y_anomaly==0)&(f.y_gun_health>60)].iloc[::30]
            stats.append(normal[NUMERIC+['operation_state']])
    arrays={s:(np.concatenate([p[0] for p in chunks]),pd.concat([p[1] for p in chunks],ignore_index=True)) for s,chunks in parts.items()}
    normals=pd.concat(stats)
    mean=normals.groupby('operation_state')[NUMERIC].mean().reindex(STATES).fillna(0).to_numpy()
    std=normals.groupby('operation_state')[NUMERIC].std().reindex(STATES).fillna(1).to_numpy()
    floor=np.maximum(np.abs(mean)*.01,.01);std=np.maximum(std,floor)
    return manifest,arrays,mean,std

def risk_level(rul, sigma=False):
    if rul<=4:return 'red'
    if rul<=12:return 'orange'
    if rul<=24 or sigma:return 'yellow'
    return 'green'

def classifier_prob(model,X, count=6):
    prob=np.zeros((len(X),count));prob[:,model.classes_.astype(int)]=model.predict_proba(X)
    return prob

def parameter_contributions(bundle,X,prob):
    """Grouped parameter ablation: change a parameter's 4 features to the training state baseline.
    This explains model sensitivity, not physical causality. All 19 numeric parameters compete.
    """
    count=len(X); state=X[:,76:86].argmax(1); classes=prob.argmax(1)
    raw=np.zeros((count,19))
    for begin in range(0,count,512):
        stop=min(count,begin+512); source=X[begin:stop]; k=len(source)
        altered=np.repeat(source,19,axis=0); j=np.tile(np.arange(19),k); ix=np.arange(k*19)
        st=np.repeat(state[begin:stop],19)
        altered[ix,j]=bundle['normal_mean'][st,j]
        altered[ix,19+j]=bundle['normal_mean'][st,j]
        altered[ix,38+j]=bundle['normal_std'][st,j]
        altered[ix,57+j]=0
        p=classifier_prob(bundle['clf'],altered)
        cls=np.repeat(classes[begin:stop],19)
        raw[begin:stop]=np.maximum(0,(np.repeat(prob[np.arange(begin,stop),classes[begin:stop]],19)-p[ix,cls]).reshape(k,19))
    raw[classes==0]=0
    scores=np.divide(raw,raw.sum(1,keepdims=True),out=np.zeros_like(raw),where=raw.sum(1,keepdims=True)>0)
    return scores,raw

def train(out):
    out=Path(out); started=time.time()
    dump(out/'进度.json',{'stage':'features','started':started})
    manifest, parts, mean, std=sample_training(out)
    X,t=parts['train']; C,c=parts['calibration']; T,test=parts['test']
    good=t.y_rul_valid & t.y_pressure_valid
    # Normalization of target scales prevents piece RUL from dominating shared splits.
    scale=np.array([35.,2500.,100.,100.,2.1])
    reg=ExtraTreesRegressor(n_estimators=48,max_depth=18,min_samples_leaf=8,max_features=.8,n_jobs=4,random_state=SEED)
    print(f'train rows={len(X)} calibration={len(C)} test={len(T)}',flush=True)
    dump(out/'进度.json',{'stage':'training','train_rows':len(X),'calibration_rows':len(C),'test_rows':len(T)})
    reg.fit(X[good],t.loc[good,REG_TARGETS].to_numpy()/scale)
    clf=ExtraTreesClassifier(n_estimators=48,max_depth=18,min_samples_leaf=4,max_features=.8,n_jobs=4,random_state=SEED,class_weight='balanced')
    clf.fit(X,t.y_anomaly.to_numpy())
    forecast=ExtraTreesClassifier(n_estimators=48,max_depth=18,min_samples_leaf=4,max_features=.8,n_jobs=4,random_state=SEED+1,class_weight='balanced')
    v=t.y_anomaly_valid.to_numpy();forecast.fit(X[v],t.y_next_anomaly_300s.to_numpy()[v])
    eg=t.y_anomaly_eta_s.notna().to_numpy() & (t.y_anomaly.to_numpy()==0)
    eta=ExtraTreesRegressor(n_estimators=32,max_depth=12,min_samples_leaf=5,n_jobs=4,random_state=SEED)
    eta.fit(X[eg],t.y_anomaly_eta_s.to_numpy()[eg])
    cal=reg.predict(C)*scale; cg=c.y_rul_valid.to_numpy()
    # Finite-sample split conformal, nominal marginal 90%; report actual held-out coverage.
    residual=np.abs(cal[cg,:2]-c.loc[cg,['y_rul_hours','y_rul_pieces']].to_numpy())
    qlevel=min(1.,math.ceil((len(residual)+1)*.90)/len(residual))
    quantile=np.quantile(residual,qlevel,axis=0,method='higher')
    bundle={'version':VERSION,'reg':reg,'clf':clf,'forecast':forecast,'eta':eta,'scale':scale,'conformal':quantile,'normal_mean':mean,'normal_std':std,'fields':NUMERIC,'targets':REG_TARGETS}
    joblib.dump(bundle,out/'模型.joblib.tmp',compress=3)
    os.replace(out/'模型.joblib.tmp',out/'模型.joblib')
    dump(out/'进度.json',{'stage':'evaluating','train_rows':len(X)})
    pred=reg.predict(T)*scale; prob=classifier_prob(clf,T); future=classifier_prob(forecast,T)
    label=prob.argmax(1); predeta=np.clip(eta.predict(T),0,300)
    valid=test.y_rul_valid.to_numpy(); pv=test.y_pressure_valid.to_numpy()
    metrics={}
    for j,k in enumerate(REG_TARGETS):
        m=valid if j<2 else pv if j==4 else np.ones(len(T),bool)
        truth=test[k].to_numpy()[m]; y=pred[m,j]
        metrics[k]={'mae':float(mean_absolute_error(truth,y)),'r2':float(r2_score(truth,y)),'rows':int(m.sum())}
    coverage=np.mean(np.abs(pred[valid,:2]-test.loc[valid,['y_rul_hours','y_rul_pieces']].to_numpy())<=quantile,axis=0)
    actual=test.y_anomaly.to_numpy(); norm=actual==0
    anomaly_metrics={'accuracy':float(accuracy_score(actual,label)), 'macro_f1':float(f1_score(actual,label,average='macro',zero_division=0)),
        'false_positive_rate':float(np.mean(label[norm]!=0)), 'confusion':confusion_matrix(actual,label,labels=np.arange(6)).tolist()}
    attribution,raw_contribution=parameter_contributions(bundle,T,prob)
    roots=np.argsort(-attribution,axis=1)[:,:3]
    anomalous=actual>0
    truth_roots=test.y_root_parameter.to_numpy()
    root_hit=float(np.mean(np.any(roots[anomalous]==truth_roots[anomalous,None],axis=1)&(raw_contribution[anomalous].sum(1)>0)))
    # Event-level warning lead. Missing warnings count as missed; include short-life infeasibility.
    event_results=[]
    for g in manifest['guns']:
        if g['split']!='test':continue
        mask=(test.gun_id.to_numpy()==g['gun_id'])&valid
        local=np.flatnonzero(mask); alerts=local[pred[local,0]<=24]
        lead=(g['terminal_seq']-int(test.seq.iloc[alerts[0]]))/3600 if len(alerts) else None
        event_results.append({'gun_id':g['gun_id'],'warning_lead_wall_hours':lead,'available_wall_hours':(g['terminal_seq']-g['start_seq'])/3600,'reached_24h':lead is not None and lead>=24})
    anomaly_events=[]
    for g in manifest['guns']:
        if g['split']!='test':continue
        for ev in g['events']:
            ids=np.flatnonzero((test.gun_id.to_numpy()==g['gun_id'])&(test.seq.to_numpy()>=ev['onset_seq']-300)&(test.seq.to_numpy()<ev['onset_seq']))
            hits=ids[future[ids,ev['kind']]>=.5]
            anomaly_events.append({'gun_id':g['gun_id'],'kind':ev['kind'],'onset_seq':ev['onset_seq'],'lead_seconds':ev['onset_seq']-int(test.seq.iloc[hits[0]]) if len(hits) else None})
    warning_valid=test.y_anomaly_valid.to_numpy()
    forecast_metrics={'macro_f1':float(f1_score(test.y_next_anomaly_300s.to_numpy()[warning_valid],future.argmax(1)[warning_valid],average='macro',zero_division=0)),
        'event_count':len(anomaly_events),'warned_before_onset':sum(x['lead_seconds'] is not None for x in anomaly_events)}
    baselines={'rul_train_median_mae':float(mean_absolute_error(test.loc[valid,'y_rul_hours'],np.full(valid.sum(),t.loc[good,'y_rul_hours'].median()))),
               'pressure_persistence_mae':float(mean_absolute_error(test.loc[pv,'y_pressure_60s'],test.loc[pv,'powder_tank_pressure']))}
    report={'version':VERSION,'algorithm':'ExtraTrees baseline models, not a selected ANN architecture','training_stride_seconds':60,'sampling_seconds':1,'forecast_seconds':60,'anomaly_horizon_seconds':300,
        'splits':{s:{'rows':len(p[0]),'guns':[g['gun_id'] for g in manifest['guns'] if g['split']==s]} for s,p in parts.items()},
        'features':X.shape[1],'metrics':metrics,'anomaly':anomaly_metrics,'forecast':forecast_metrics,'root_top3_hit':root_hit,
        'conformal_nominal':.90,'conformal_half_width':quantile.tolist(),'conformal_test_coverage':coverage.tolist(),'baselines':baselines,
        'gun_warning_events':event_results,'anomaly_warning_events':anomaly_events,'seconds':time.time()-started,
        'root_method':'Grouped ablation of current/mean/std/slope against training-only same-state baseline; normalized positive change in selected anomaly probability; no causal proof.',
        'limitations':['Synthetic-only holdout; public residual source is limited.', 'Root contribution is model sensitivity, not causal proof.', '24h warning is impossible for lifetimes observed less than 24 wall hours.', 'No production or business benefit claim.']}
    report['acceptance']={
       'RUL MAE < 5h':metrics['y_rul_hours']['mae']<5,'RUL R2 > 0.85':metrics['y_rul_hours']['r2']>.85,
       '异常准确率 > 90%':anomaly_metrics['accuracy']>.9,'异常宏平均F1 > 0.88':anomaly_metrics['macro_f1']>.88,
       '异常误报率 < 15%':anomaly_metrics['false_positive_rate']<.15,'Top-3命中 > 85%':root_hit>.85,
       '全部测试枪预警 >= 24h':all(x['reached_24h'] for x in event_results)}
    result=test[['seq','timestamp','gun_id','operation_state','machine_message','y_rul_hours','y_rul_pieces','y_pressure_60s','y_gun_health','y_tank_health','y_anomaly','y_next_anomaly_300s','y_root_parameter','y_rul_valid']].copy()
    for j,k in enumerate(REG_TARGETS):result['pred_'+k[2:]]=np.maximum(0,pred[:,j])
    result['rul_low']=np.maximum(0,pred[:,0]-quantile[0]);result['rul_high']=pred[:,0]+quantile[0]
    result['pred_anomaly']=label;result['anomaly_probability']=1-prob[:,0];result['forecast_probability']=1-future[:,0]
    result['pred_next_anomaly']=future.argmax(1);result['pred_eta_seconds']=predeta
    for j in range(3):
        result[f'root_{j+1}']=[NUMERIC[k] if raw_contribution[i].sum()>0 else None for i,k in enumerate(roots[:,j])]
        result[f'root_{j+1}_contribution']=attribution[np.arange(len(T)),roots[:,j]]
    result.to_parquet(out/'评估预测.parquet',index=False)
    dump(out/'评估报告.json',report)
    write_report(out,report,manifest)
    dump(out/'进度.json',{'stage':'ready','rows':manifest['rows'],'bytes':manifest['bytes'],'complete_guns':manifest['complete_guns'],'training_seconds':report['seconds']})
    print(f'training complete in {report["seconds"]:.1f}s',flush=True)
    return report

def write_report(out,r,m):
    lines=['# Phase2 本地模拟与输出测试报告','',f'数据：{m["rows"]:,} 条，{m["bytes"]/1e9:.3f} GB，{m["complete_guns"]} 个完整寿命周期和 1 个截尾周期。',
        '','输入为既有21项，原始采样1秒。训练及离线评分每60秒取一个完整60秒窗口；在线回放每秒推理，不改变数据采样率。',
        '','采用 ExtraTrees 基线完成输出测试，未选择正式神经网络架构。按完整喷枪分训练/校准/测试，公开残差块也分池；合成测试不代表实际设备表现。',
        '','|输出|MAE|R²|','|---|---:|---:|']
    for k,v in r['metrics'].items():lines.append(f'|{k}|{v["mae"]:.4f}|{v["r2"]:.4f}|')
    lines+=['',f'RUL中位数基线MAE：{r["baselines"]["rul_train_median_mae"]:.4f} h；罐压当前值基线MAE：{r["baselines"]["pressure_persistence_mae"]:.4f} bar。',
        '',f'异常准确率 {r["anomaly"]["accuracy"]:.2%}；宏平均F1 {r["anomaly"]["macro_f1"]:.4f}；正常样本误报率 {r["anomaly"]["false_positive_rate"]:.2%}；原因Top-3命中 {r["root_top3_hit"]:.2%}。',
        '',f'标称90%寿命预测区间：测试集小时覆盖率 {r["conformal_test_coverage"][0]:.2%}，件数覆盖率 {r["conformal_test_coverage"][1]:.2%}。它是预测区间，并非参数估计置信区间。',
        '',f'异常事件提前检出：{r["forecast"]["warned_before_onset"]}/{r["forecast"]["event_count"]}；5分钟异常预测宏平均F1 {r["forecast"]["macro_f1"]:.4f}。',
        '','|执行方案目标|合成测试结果|','|---|---|']
    for k,v in r['acceptance'].items():lines.append(f'|{k}|{"达到" if v else "未达到"}|')
    lines+=['','24/12/4h采用剩余带电寿命阈值；实际预警提前量按失效前日历时间计算。短寿命枪不足24小时观测时不可能满足24小时预警。',
        '','Top-3从19项数值参数中计算：将某参数的当前值、均值、波动和斜率同时替换为训练集同工况基准，观察当前异常类别概率减少多少，再将正变化归一化排序。它解释模型判断，不证明物理因果。没有异常或没有正贡献时不强行给出原因。3σ以训练集相同工况基准计算，工况切换后10秒暂不启用。',
        '','运行速度与页面验收见验证结果.json。未测试真实MES/ERP/PLC和产线收益；停机减少50%/寿命增加10%不能用本数据证明。','']
    (Path(out)/'运行报告.md').write_text('\n'.join(lines),encoding='utf-8')

def load_model(out):
    # Only our locally produced trusted artifact; no model upload endpoint.
    return joblib.load(Path(out)/'模型.joblib')

def predict_window(bundle,frame):
    started=time.perf_counter()
    current=frame.iloc[-1]; n=len(frame)
    if n<60:return {'status':'warming_up','needed':60-n,'outputs':None}
    last=frame.iloc[-60:]
    if len(last.gun_id.unique())!=1 or not np.all(np.diff(last.seq.to_numpy())==1):return {'status':'gap_or_replacement','outputs':None}
    values=last[NUMERIC].to_numpy()
    if not np.isfinite(values).all() or (values<0).any() or not last.operation_state.isin(STATES).all():return {'status':'invalid_input','outputs':None}
    X=feature_rows(frame,[n-1]); p=(bundle['reg'].predict(X)*bundle['scale'])[0]
    prob=classifier_prob(bundle['clf'],X)[0]; future=classifier_prob(bundle['forecast'],X)[0]
    eta=float(np.clip(bundle['eta'].predict(X)[0],0,300))
    state=STATES.index(current.operation_state)
    z=np.abs((current[NUMERIC].to_numpy(float)-bundle['normal_mean'][state])/bundle['normal_std'][state])
    stable=last.operation_state.iloc[-10:].nunique()==1
    deviations=[{'parameter':NUMERIC[i],'sigma':round(float(z[i]),2)} for i in np.argsort(-z) if z[i]>3] if stable else []
    anomaly=int(prob.argmax()); confidence=float(1-prob[0]); next_kind=int(future.argmax())
    attribution,raw_contribution=parameter_contributions(bundle,X,prob[None,:]) if anomaly else (np.zeros((1,19)),np.zeros((1,19)))
    roots=np.argsort(-attribution[0])[:3]
    health=float(np.clip(p[2],0,100));tankhealth=float(np.clip(p[3],0,100))
    available=current.operation_state not in ['maintenance']
    rul=max(0,float(p[0]));pieces=max(0,int(round(p[1])))
    risk=risk_level(rul,bool(deviations)) if available else 'unavailable'
    reasons=[]
    if available and rul<=24: reasons.append(f'预测剩余带电寿命 {rul:.2f} h，进入 {4 if rul<=4 else 12 if rul<=12 else 24} h 阈值')
    if deviations:reasons.append('同工况参数偏离训练基准超过3σ')
    if anomaly:reasons.append(FAULT_NAMES[anomaly])
    output={'rul_hours':rul if available else None,'rul_pieces':pieces if available else None,
        'rul_interval_hours':[max(0,rul-bundle['conformal'][0]),rul+bundle['conformal'][0]] if available else None,
        'rul_interval_pieces':[max(0,int(pieces-bundle['conformal'][1])),int(pieces+bundle['conformal'][1])] if available else None,
        'interval_nominal':.9,'pressure_now':float(current.powder_tank_pressure),'pressure_60s':max(0,float(p[4])),
        'pressure_trend':'上升' if p[4]-current.powder_tank_pressure>.03 else '下降' if p[4]-current.powder_tank_pressure<-.03 else '平稳',
        'gun_health':health if available else None,'tank_health':tankhealth,'anomaly_type':FAULT_NAMES[anomaly],'anomaly_probability':confidence,
        'next_anomaly_type':FAULT_NAMES[next_kind],'anomaly_300s_probability':float(1-future[0]),
        'predicted_anomaly_eta_s':0. if anomaly else eta if 1-future[0]>=.5 else None,
        'warning_level':risk,'warning_reasons':reasons,'sigma_deviations':deviations,
        'root_candidates':[{'parameter':NUMERIC[i],'name':FIELDS[i]['name'],'score':float(attribution[0,i]),'probability_drop':float(raw_contribution[0,i]),'sigma':float(z[i])} for i in roots] if raw_contribution.sum()>0 else [],
        'root_status':'model_contribution' if raw_contribution.sum()>0 else 'no_positive_evidence',
        'model_version':bundle['version'],'input_parameters':{f['key']:(float(current[f['key']]) if f['type']=='number' else str(current[f['key']])) for f in FIELDS},
        'timestamp':current.timestamp.isoformat(),'gun_id':int(current.gun_id),'seq':int(current.seq)}
    output['latency_ms']=(time.perf_counter()-started)*1000
    return {'status':'maintenance' if not available else 'ok','outputs':output}

class LivePredictor:
    """Actual one-record ingestion, bounded memory, late/gap/unit protection."""
    def __init__(self,bundle):
        self.bundle=bundle;self.buffer=deque(maxlen=701);self.retired=set()
        self.last_time=None;self.last_seq=None;self.gun=None

    def heartbeat(self,now):
        now=pd.Timestamp(now)
        if now.tzinfo is None:raise ValueError('heartbeat timestamp must include timezone')
        if self.last_time is not None and (now-self.last_time).total_seconds()>3:
            self.buffer.clear()
            return {'status':'stale','outputs':None}
        return {'status':'alive' if self.last_time is not None else 'empty','outputs':None}

    def push(self,record):
        try:
            stamp=pd.Timestamp(record['timestamp']);seq=record['seq'];gun=record['gun_id']
            if stamp.tzinfo is None or pd.isna(stamp):raise ValueError('Timezone required')
            if isinstance(seq,bool) or not isinstance(seq,(int,np.integer)):raise ValueError('seq must be integer')
            if isinstance(gun,bool) or not isinstance(gun,(int,np.integer)):raise ValueError('gun_id must be integer')
        except (KeyError,ValueError,TypeError):
            self.buffer.clear();return {'status':'invalid_metadata','outputs':None}
        if self.last_time is not None and (stamp<=self.last_time or seq<=self.last_seq):return {'status':'late_or_duplicate','outputs':None}
        if gun in self.retired:return {'status':'retired_gun','outputs':None}
        if self.gun is not None and gun!=self.gun:
            self.retired.add(self.gun);self.buffer.clear()
        if self.last_time is not None and ((stamp-self.last_time).total_seconds()!=1 or seq!=self.last_seq+1):self.buffer.clear()
        self.gun=gun;self.last_time=stamp;self.last_seq=seq
        try:
            for f in FIELDS:
                value=record[f['key']]
                if f['type']=='number':
                    if isinstance(value,(bool,str)) or not np.isfinite(value) or value<0:raise ValueError('invalid numeric value')
                elif not isinstance(value,str):raise ValueError('invalid text value')
            if record['operation_state'] not in STATES:raise ValueError('unknown state')
            for k,u in record.get('units',{}).items():
                expected=next((f['unit'] for f in FIELDS if f['key']==k),None)
                if expected is None or u!=expected:raise ValueError('unit mismatch')
        except (KeyError,ValueError,TypeError):
            self.buffer.clear();return {'status':'invalid_input','outputs':None}
        clean={f['key']:record[f['key']] for f in FIELDS}
        clean.update(timestamp=stamp,seq=int(seq),gun_id=int(gun));self.buffer.append(clean)
        if len(self.buffer)<60:return {'status':'warming_up','needed':60-len(self.buffer),'outputs':None}
        return predict_window(self.bundle,pd.DataFrame(self.buffer))
