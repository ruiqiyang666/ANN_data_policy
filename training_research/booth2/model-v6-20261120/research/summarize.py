"""Generate a review table from frozen v6/v5 results, without modifying any models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--v5',type=Path,required=True)
    args=p.parse_args()
    report=json.loads((args.run/'report.json').read_text(encoding='utf-8'))
    previous=json.loads((args.v5/'report.json').read_text(encoding='utf-8'))
    tables=[]
    for version,r in [('v5',previous),('v6',report)]:
        for mode,table in r['comparison'].items():
            for name,entry in table.items():
                row=dict(version=version,evaluation=mode,model=name)
                row.update(entry['life']);tables.append(row)
    df=pd.DataFrame(tables)
    df.to_csv(args.run/'comparison.csv',index=False)
    observed=[]
    for name,entry in report['comparison']['loco'].items():
        rows=[r for r in entry['cycles'] if r['observed']]
        without=[r for r in rows if r['cycle']!='L000']
        observed.append(dict(model=name,mae=np.mean([r['mae'] for r in rows]),mae_excluding_left_truncated_L000=np.mean([r['mae'] for r in without]),
                             short_L006=next(r['mae'] for r in rows if r['cycle']=='L006'),short_L012=next(r['mae'] for r in rows if r['cycle']=='L012')))
    pd.DataFrame(observed).to_csv(args.run/'cycle-sensitivity.csv',index=False)
    print(df[['version','evaluation','model','mae','last2h','bias2h','coverage','width']].round(3).to_string(index=False))
    print('WARNINGS',json.dumps({mode:{n:e['totals'] for n,e in table.items()} for mode,table in report['warning'].items()},indent=2))


if __name__=='__main__':main()
