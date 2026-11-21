"""Finish the ordered experiment once the initial GPU/CPU workers have exited."""
import sys,os,time,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(task,script,label):
    env=os.environ.copy();env.update(TRAIN_TASK=task,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    print('Running',label,flush=True)
    with (ROOT/'logs'/f'{label}.log').open('a',encoding='utf-8') as stream:
        p=subprocess.run([sys.executable,'-B',str(ROOT/'src'/script)],env=env,stdout=stream,stderr=subprocess.STDOUT)
    if p.returncode!=0:raise RuntimeError(f'{label} failed ({p.returncode}); inspect log')
    print('Finished',label,flush=True)
def main():
    deadline=time.monotonic()+3600
    while not (ROOT/'training_processes.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('Initial training did not complete within deadline')
        print('Waiting for initial GPU/CPU workers to exit',flush=True);time.sleep(20)
    initial=json.loads((ROOT/'training_processes.json').read_text('utf-8'));assert all(v['exit_code']==0 for v in initial['training_processes'])
    assert len(list((ROOT/'models/life/life').glob('stage_*/complete.json')))==6
    run('voltage','train_ssl.py','tcn_ssl')
    for task in ['voltage','feedrate','hopper_pressure','life']:run(task,'evaluate_v3.py',f'evaluate_{task}')
    run('voltage','probability_guard.py','conditional_probability_guard')
    run('voltage','verify_report_v3.py','verification_report')
    print('All training, frozen comparisons and final artifact verification completed.',flush=True)
if __name__=='__main__':main()
