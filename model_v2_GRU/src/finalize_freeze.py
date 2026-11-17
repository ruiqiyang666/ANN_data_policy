"""Seal final source and file hashes after validation, before first test access."""
from datetime import datetime,timezone
from modeling import ROOT
from train import dump,digest
import json

def main():
    a=ROOT/'artifacts'
    if (a/'test_opened.json').exists() or (a/'evaluation.json').exists():
        raise RuntimeError('Cannot reseal after test is opened')
    cfg=json.loads((a/'frozen_config.json').read_text('utf-8'))
    required=['inference.py','evaluate.py','replay.py']
    assert all((ROOT/'src'/s).exists() for s in required)
    cfg['source_sha256']={p.name:digest(p) for p in (ROOT/'src').glob('*.py') if p.name!='report.py'}
    cfg['data_sha256']={p.name:digest(p) for p in (ROOT/'data').iterdir() if p.is_file()}
    cfg['sealed_at_utc']=datetime.now(timezone.utc).isoformat()
    cfg['selection_final']=True
    dump(a/'frozen_config.json',cfg)
    print('SEALED',cfg['selected_model'],cfg['sealed_at_utc'])

if __name__=='__main__':main()
