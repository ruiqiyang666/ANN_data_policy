"""Verify every delivered research file without loading executable model pickles."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
errors = []
for row in manifest['files']:
    p = ROOT / row['path']
    if not p.is_file():
        errors.append(f'missing: {row["path"]}')
        continue
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    if p.stat().st_size != row['bytes'] or h.hexdigest() != row['sha256']:
        errors.append(f'hash mismatch (run git lfs pull if needed): {row["path"]}')
if errors:
    print('\n'.join(errors))
    sys.exit(1)
print(f'PASS: {len(manifest["files"])} files verified')
