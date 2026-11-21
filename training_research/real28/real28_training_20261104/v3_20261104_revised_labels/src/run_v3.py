"""Run one GPU worker and one bounded CPU worker; record actual process exits."""
import sys,os,subprocess,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'logs';LOG.mkdir(exist_ok=True)
def launch(task,script,names,label):
    env=os.environ.copy();env.update(TRAIN_TASK=task,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    stream=(LOG/f'{label}.log').open('a',encoding='utf-8')
    p=subprocess.Popen([sys.executable,'-B',str(ROOT/'src'/script),'--models',*names],env=env,stdout=stream,stderr=subprocess.STDOUT)
    print(f'Started {label} PID={p.pid}',flush=True)
    return p,stream,label
def main():
    started=time.time();risk=['logistic','histgb','extra_trees','random_forest','tcn','gru','patch_transformer']
    life=['constant','lognormal_aft','weibull_aft','spline_aft','mlp_aft','logistic_hazard','boosted_hazard','forest_hazard']
    pending=[('life','train_life_v2.py',life,'life'),('feedrate','train_risk.py',['logistic','histgb','extra_trees'],'feedrate'),('hopper_pressure','train_risk.py',['logistic','histgb','extra_trees'],'hopper_pressure')]
    workers=[launch('voltage','train_risk.py',risk,'voltage'),launch(*pending.pop(0))];results=[]
    while workers:
        for item in workers[:]:
            p,stream,label=item;code=p.poll()
            if code is None:continue
            stream.close();workers.remove(item);results.append({'worker':label,'exit_code':code})
            print(f'Finished {label}: exit={code}',flush=True)
            if code!=0:
                for q,_,_ in workers:q.terminate()
                raise RuntimeError(f'{label} failed; inspect logs/{label}.log')
            if label!='voltage' and pending:workers.append(launch(*pending.pop(0)))
        count=len(list((ROOT/'models').rglob('fit.json')))+len([p for p in (ROOT/'models/life').rglob('*.joblib')])
        print(f'Training elapsed={(time.time()-started)/60:.1f} min; saved main fits={count}',flush=True)
        if workers:time.sleep(20)
    report={'training_processes':results,'main_fit_count':count,'elapsed_seconds':time.time()-started,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (ROOT/'training_processes.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report,flush=True)
if __name__=='__main__':main()
