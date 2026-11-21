"""Continue the already-running GPU experiment and finalize its isolated artifacts."""
import sys
sys.dont_write_bytecode = True
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / '20261104_v1'


def wait_for(files):
    deadline = time.monotonic() + 3600
    while not all((MODEL / f).exists() for f in files):
        if time.monotonic() > deadline:
            raise RuntimeError('Training files still unavailable after one hour; inspect neural log')
        state = ROOT / 'execution_status.json'
        if state.exists() and time.time() - state.stat().st_mtime > 300:
            raise RuntimeError('No neural training heartbeat for 300 seconds; inspect process before continuing')
        print('Waiting for: ' + ', '.join(f for f in files if not (MODEL / f).exists()), flush=True)
        time.sleep(30)


def run(name, *args):
    print('Running ' + name, flush=True)
    subprocess.run([sys.executable, '-B', str(ROOT / 'src' / name), *args], check=True)


def main():
    wait_for(['selected_model.json'])
    run('train_lifetime.py')
    wait_for(['b1_ssl_tcn_seed43_metrics.json', 'b1_ssl_tcn_seed44_metrics.json'])
    run('evaluate_run.py')
    run('verify_run.py', '--final')
    run('write_report.py')
    print('Completed first run and verified report.', flush=True)


if __name__ == '__main__':
    main()
