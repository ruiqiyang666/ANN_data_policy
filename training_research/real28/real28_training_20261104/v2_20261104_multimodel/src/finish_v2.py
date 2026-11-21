"""Wait for authorized fits, then freeze, compare and verify; never launch duplicate fits."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import time,subprocess

ROOT=Path(__file__).resolve().parents[1]
risk=['logistic','histgb','extra_trees','random_forest','tcn','gru','patch_transformer','spline_logistic','rbf_logistic']
life=['constant','lognormal_aft','weibull_aft','spline_aft','mlp_aft','logistic_hazard','boosted_hazard','forest_hazard']
deadline=time.monotonic()+5400
while True:
    missing=[f'risk/{n}' for n in risk if not (ROOT/'models/risk'/n/'complete.json').exists()]+[f'life/{n}' for n in life if not (ROOT/'models/life'/n/'complete.json').exists()]
    if not missing:break
    if time.monotonic()>deadline:raise RuntimeError('Required models not completed; inspect logs: '+str(missing))
    print('Waiting for '+', '.join(missing),flush=True);time.sleep(30)
for name in ['evaluate_v2.py','verify_v2.py','report_v2.py']:
    print('Running '+name,flush=True);subprocess.run([sys.executable,'-B',str(ROOT/'src'/name)],check=True)
print('All second-round experiments completed and verified.',flush=True)
