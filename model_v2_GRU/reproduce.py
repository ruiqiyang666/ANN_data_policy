"""Reproduce into a new child run, preserving the delivered frozen result."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',default=datetime.now().strftime('%Y%m%dT%H%M%S'))
    ap.add_argument('--rebuild-data',action='store_true')
    args=ap.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        raise ValueError('Use an alphanumeric run name')
    run=ROOT/'runs'/args.name
    run.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/'src',run/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'tests',run/'tests',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'docs',run/'docs')
    shutil.copy2(ROOT/'README.md',run/'README.md')
    shutil.copy2(ROOT/'requirements.txt',run/'requirements.txt')
    (run/'artifacts').mkdir();(run/'data').mkdir()
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['OMP_NUM_THREADS']='4'
    commands=[]
    if args.rebuild_data:
        commands.append(['src/prepare.py','--source',str(ROOT.parent/'training_data')])
    else:
        for path in (ROOT/'data').iterdir():
            if path.is_file():
                # Keep named experiments independent of the delivered data.
                shutil.copy2(path,run/'data'/path.name)
        shutil.copy2(ROOT/'artifacts/data_audit.json',run/'artifacts/data_audit.json')
    commands.extend([['-m','unittest','discover','-s','tests'],['src/train.py'],['src/extend_validation.py'],
       ['src/finalize_freeze.py'],['src/evaluate.py'],['src/replay.py'],['src/report.py']])
    for command in commands:
        print('RUN',command,flush=True)
        log_name=Path(command[0]).stem+'.log' if command[0]!='-m' else 'tests.log'
        with (run/'artifacts'/log_name).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,*command],cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Command failed; see {run / "artifacts" / log_name}')
    print('Reproduction complete:',run)

if __name__=='__main__':main()
