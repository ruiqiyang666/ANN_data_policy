"""Train in a new experiment directory; never overwrite the delivered v1."""
from pathlib import Path
import argparse,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    output=args.output.resolve();data=ROOT.parent/'training_data'
    if output.is_relative_to(ROOT) or output.is_relative_to(data) or ROOT.is_relative_to(output) or data.is_relative_to(output):
        raise ValueError('Use a separate experiment directory outside model_v1 and training_data')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a new empty experiment directory')
    output.mkdir(parents=True,exist_ok=True);(output/'configs').mkdir();(output/'artifacts').mkdir()
    shutil.copy2(ROOT/'configs/train.json',output/'configs/train.json')
    jobs=[('prepare',[str(ROOT/'src/prepare.py'),'--data',str(data),'--output',str(output/'artifacts')]),
          ('train',[str(ROOT/'src/train.py'),'--project',str(output)]),
          ('evaluate',[str(ROOT/'src/evaluate.py'),'--project',str(output)])]
    for n,(name,command) in enumerate(jobs,1):
        log=output/(name+'.log');print(f'{n}/3 {name}: {log}',flush=True)
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-B','-u','-X','utf8',*command],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        if result.returncode:raise RuntimeError(f'{name} failed; read {log}')
    print('Finished. See artifacts/evaluation.json, event_results.csv and frozen_config.json in '+str(output),flush=True)

if __name__=='__main__':main()
