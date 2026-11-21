"""Copy reusable implementations into a new run; never modify earlier runs."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
root=Path(__file__).resolve().parent
old=root.parent/'v2_20261104_multimodel'/'src'
src=root/'src';src.mkdir(parents=True,exist_ok=True)
for name in ['common_v2.py','risk_models.py','train_risk.py','metrics_v2.py','life_models.py','train_life_v2.py']:
    target=src/name
    if target.exists():raise RuntimeError('Refuse to overwrite '+str(target))
    text=(old/name).read_text('utf-8')
    if name=='common_v2.py':
        text=text.replace("DATA = ROOT / 'data'", "SHARED = ROOT / 'data'\nGRID = PARENT / 'v2_20261104_multimodel' / 'data'\nTASK = os.environ.get('TRAIN_TASK','voltage')\nDATA = SHARED / TASK")
        text=text.replace("MODELS = ROOT / 'models'", "MODELS = ROOT / 'models' / TASK").replace("EVAL = ROOT / 'evaluation'", "EVAL = ROOT / 'evaluation' / TASK")
        text=text.replace("ROOT/'execution_status.json'", "ROOT/f'execution_status_{TASK}.json'")
        text=text.replace("ROOT/'protocol.json'", "ROOT/'legacy_unused_protocol.json'")
        a=text.index('def protocol():')
        text=text[:a]+"def protocol(): return load(ROOT/'protocol.json')\n"
    if name=='risk_models.py':
        text=text.replace('DATA,TYPES,load','DATA,TYPES,load,SHARED,GRID')
        text=text.replace("DATA/'features.npy'", "SHARED/'features.npy'")
        for f in ['grid_x.npy','grid_seen.npy']:text=text.replace("DATA/'"+f+"'", "GRID/'"+f+"'")
    if name=='train_life_v2.py':
        text=text.replace('DATA,MODELS,EVAL,load', 'DATA,MODELS,EVAL,SHARED,load').replace("DATA/'features.npy'", "SHARED/'features.npy'")
        text=text.replace("'observed_operation_amp_hours']", "'observed_operation_amp_hours','history_Voltage_mean_7200s','history_Voltage_slope_per_hour_7200s','history_observed_powered_hours','matched_long_peak_drop','matched_long_voltage_slope','matched_long_voltage_std','matched_long_count','matched_long_margin_51','phase_rising','phase_falling','phase_near_peak']")
    if name=='train_risk.py':
        text=text.replace("args=parser.parse_args()", "args=parser.parse_args()")
        # The conditional type head now has a single known subtype; it does not choose another task.
    target.write_text(text,encoding='utf-8')
print('Created isolated implementation copies',src)
