"""Generate the v4 incremental code tree from the read-only v1/v2/v3 sources.

Every edit is an exact, asserted string replacement, so the difference from the
earlier rounds is explicit and a silent non-match is impossible. Earlier-round
directories are only read. Output: ./tree (mirrors the server work/real28).
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib, json, os, shutil, stat

HERE = Path(__file__).resolve().parent
OLD = HERE.parent
V1SRC, REF = OLD / 'src', OLD / 'reference'
V2SRC = OLD / 'v2_20261104_multimodel' / 'src'
V3SRC = OLD / 'v3_20261104_revised_labels' / 'src'
TREE = HERE / 'tree'
RUN = '20261112_v4'
NEW_START, NEW_END = '2026-10-27', '2026-11-02'
record = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def emit(source, target, edits=()):
    text = Path(source).read_text(encoding='utf-8')
    for old, new, count in edits:
        found = text.count(old)
        if found != count:
            raise SystemExit(f'{source.name}: expected {count} match(es), found {found}: {old[:70]!r}')
        text = text.replace(old, new)
    target = TREE / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8', newline='\n')
    record.append({'target': target.relative_to(TREE).as_posix(), 'source': source.relative_to(OLD).as_posix(),
                   'source_sha256': sha(source), 'edits': len(edits), 'target_sha256': sha(target)})


def main():
    if TREE.exists():
        def writable(function, path, _):  # generated copies may inherit a read-only bit
            os.chmod(path, stat.S_IWRITE); function(path)
        for child in TREE.iterdir():  # keep the directory itself: a shell may be standing in it
            if child.is_dir(): shutil.rmtree(child, onexc=writable)
            else: writable(os.unlink, child, None)
    # ---- stage 1: raw rows, products, events, replacement candidates (v1 scripts)
    emit(V1SRC / 'common.py', 'src/common.py', [
        ('RUN = "20261104_v1"', f'RUN = "{RUN}"', 1),
        ("PARTITIONS = ['train','selection','calibration','test']",
         "PARTITIONS = ['train','selection','calibration','test','new_blind']", 1)])
    emit(V1SRC / 'prepare_data.py', 'src/prepare_data.py', [
        ("FOLDERS=['20260815-20260911','20260912-20261009','20261010-20261016','20261019-20261023']",
         "FOLDERS=['20260815-20260911','20260912-20261009','20261010-20261016','20261019-20261023','20261027-20261101']", 1),
        # replacement_windows() filters stop windows by analysis_end_exclusive; the
        # 2026-10-24 value from config.json would silently drop every new-week candidate.
        ("cfg.update(analysis_start='2026-08-15 00:00:00',minimum_sampling_coverage=0.0,\n"
         "               version='real28-20261104-company1103-v1')",
         f"cfg.update(analysis_start='2026-08-15 00:00:00',analysis_end_exclusive='{NEW_END} 00:00:00',minimum_sampling_coverage=0.0,\n"
         "               version='real28-20261112-incremental-v4')", 1),
        ("legacy=pd.read_csv(SOURCE/'label_samples.csv')", "legacy=pd.read_csv(ROOT/'reference'/'label_samples.csv')", 1),
        ("    if len(t)!=5_402_169 or len(files)!=1505: raise AssertionError('Main source scope changed from audited delivery')",
         "    old_rows=int((t<pd.Timestamp('2026-10-24').value//10**9).sum())\n"
         "    # The audited 2026-10-23 delivery must be reproduced exactly inside the larger scope.\n"
         "    if old_rows!=5_402_169 or len(files)!=1635: raise AssertionError(f'Main source scope changed: {old_rows} rows through 20261023, {len(files)} files')", 1),
        ("'scope':'main device 2026-08-15 through 2026-10-23; Voltage Sample excluded'}",
         "'rows_through_20261023':old_rows,'new_rows_20261027_20261101':int(len(t)-old_rows),\n"
         "        'scope':'main device 2026-08-15 through 2026-11-01; Voltage Sample excluded'}", 1)])
    emit(V1SRC / 'build_windows.py', 'src/build_windows.py', [
        ("['2026-08-15','2026-10-04','2026-10-08','2026-10-10']", f"['2026-08-15','2026-10-04','2026-10-08','2026-10-10','{NEW_START}']", 1),
        ("['2026-09-30','2026-10-08','2026-10-10','2026-10-24']", f"['2026-09-30','2026-10-08','2026-10-10','2026-10-24','{NEW_END}']", 1),
        ('for k in range(4)', 'for k in range(5)', 2)])
    emit(V1SRC / 'lifetime_labels.py', 'src/lifetime_labels.py', [('for k in range(4)', 'for k in range(5)', 1)])
    for name in ['label_ann.py', 'revise_labels.py', 'config.json']:
        emit(REF / name, f'reference/{name}')
    # ---- stage 2: complete chronological 1 s grid and 277 causal features (v2 script)
    emit(V2SRC / 'common_v2.py', 'v2_grid/src/common_v2.py', [("V1 = PARENT / 'data' / '20261104_v1'", f"V1 = PARENT / 'data' / '{RUN}'", 1)])
    emit(V2SRC / 'prepare_v2.py', 'v2_grid/src/prepare_v2.py', [
        ("('2026-10-10','2026-10-24')]", f"('2026-10-10','2026-10-24'),('{NEW_START}','{NEW_END}')]", 1),
        ('for k in range(4)', 'for k in range(5)', 1)])
    # ---- stage 3: revised weak labels, 292 features, models (v3 scripts)
    folds = ("FOLDS = [\n"
             " {'name':'f1','train_end':'2026-09-01','validation_end':'2026-09-08'},\n"
             " {'name':'f2','train_end':'2026-09-08','validation_end':'2026-09-15'},\n"
             " {'name':'f3','train_end':'2026-09-15','validation_end':'2026-09-30'},\n")
    emit(V3SRC / 'common_v2.py', 'v4/src/common_v2.py', [
        ("V1 = PARENT / 'data' / '20261104_v1'", f"V1 = PARENT / 'data' / '{RUN}'", 1),
        ("GRID = PARENT / 'v2_20261104_multimodel' / 'data'", "GRID = PARENT / 'v2_grid' / 'data'\nOLD_V3 = PARENT / 'old_v3'", 1),
        (folds + "]", folds +
         " {'name':'f4','train_end':'2026-10-04','validation_end':'2026-10-10'},\n"
         " {'name':'f5','train_end':'2026-10-10','validation_end':'2026-10-24'},\n"
         f" {{'name':'f6','train_end':'{NEW_START}','validation_end':'{NEW_END}'}},\n]\n"
         "# f1-f3 are the v3 selection folds. f4/f5 cover October dates that earlier rounds\n"
         "# already inspected. f6 validates on the 2026-10-27..11-01 week no round had seen.\n"
         "BLIND_FOLD = 'f6'\nNEW_PARTITION = 4", 1)])
    start = V3SRC.joinpath('prepare_v3.py').read_text(encoding='utf-8')
    a = start.index("    protocol={'created_at'")
    b = start.index("    p=pd.read_parquet(V1/'products.parquet')")
    emit(V3SRC / 'prepare_v3.py', 'v4/src/prepare_v4.py', [
        (start[a:b], "    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'))  # declared before preparation\n"
                     "    prepared=json.loads((V1/'prepared.json').read_text('utf-8'))\n", 1),
        ('for k in range(4)', 'for k in range(5)', 1),
        ("'raw_rows':5402169,'main_csv':1505,", "'raw_rows':prepared['rows'],'main_csv':prepared['source_files'],'rows_through_20261023':prepared['rows_through_20261023'],", 1)])
    for name in ['risk_models.py', 'metrics_v2.py', 'life_models.py', 'train_stage_life.py', 'train_life_v2.py', 'probability_guard.py']:
        emit(V3SRC / name, f'v4/src/{name}')
    # ---- hand-written v4 files (protocol, training, evaluation) live in ./authored
    authored = HERE / 'authored'
    for path in sorted(p for p in authored.rglob('*') if p.is_file()):
        target = TREE / path.relative_to(authored)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
        record.append({'target': target.relative_to(TREE).as_posix(), 'source': 'v4_20261112_incremental/authored/' + path.relative_to(authored).as_posix(),
                       'source_sha256': sha(path), 'edits': 0, 'target_sha256': sha(target), 'authored_in_v4': True})
    (TREE / 'reference' / 'label_samples.csv').write_bytes((OLD.parent / 'Real-time Data' / 'label_samples.csv').read_bytes())
    (TREE / 'generated_from.json').write_text(json.dumps({'run': RUN, 'files': record}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(record)} files generated, {sum(r["edits"] for r in record)} asserted edits')
    for r in record:
        print(f'  {r["edits"]} edit(s)  {r["target"]}')


if __name__ == '__main__':
    main()
