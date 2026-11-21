"""Rebuild the Booth2 warehouse (Parquet lake + SQLite) from the read-only sources.

Sources (never modified):
  * the ANN_data_policy clone, read through `git show HEAD:<path>` so the bytes are
    the committed ones regardless of the checkout's newline settings;
  * the hourly 1 Hz instrument CSVs (`Real-time Data/<week folders>` and, optionally,
    extra folders that fill holes of the main tree). `Voltage Sample` is excluded.

Outputs under <out>/:  lake/raw_1hz/date=YYYY-MM-DD/part-0.parquet, lake/curated/*.parquet,
db/booth2.sqlite, schema/sqlite_schema.sql, manifest.json.

Time: every timestamp is the instrument's local wall clock as written in the CSVs.
No timezone is recorded, none is invented. `ts` columns are that wall clock counted as
seconds since 1970-01-01 00:00:00 *as if it were UTC* (the convention the source project
already uses for `anchor_id = 'A' + ts`). Do not convert them with a timezone.

Deterministic: fixed file order, no wall-clock values in any output.

    python load_history.py --repo ../../repo --raw "D:/.../Real-time Data" \
        --raw-extra "D:/.../20261009" --out ..
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from live_schema import APPEND_ONLY, LIVE_SCHEMA  # noqa: E402

REGISTRY = json.loads((HERE / "tags.json").read_text(encoding="utf-8"))
KEYS = [t["key"] for t in REGISTRY["tags"] if not t.get("reserved")]
MONTHS = dict(zip("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), range(1, 13)))
HOUR_FILE = re.compile(r"(\d{2})_([A-Za-z]{3})_(\d{4})H(\d{2})\.csv")
DMY = r"\d{2} \d{2} \d{4} \d{2}:\d{2}:\d{2}"
DMY_ONE = re.compile(rf"^{DMY}$")
DMY_LIST = re.compile(rf"^{DMY}([;,|] ?{DMY})*$")
DMY_PART = re.compile(r"(\d{2}) (\d{2}) (\d{4}) (\d{2}:\d{2}:\d{2})")
ISO_ONE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
SCHEMA_VERSION = "booth2-warehouse-1"
CLOCK = ("instrument-local wall clock as exported by the HMI; timezone not recorded and not assumed; "
         "ts = that wall clock counted as epoch seconds as if UTC")
POWERED = "Current > 100 A and Power > 5 kW (analysis setting of the source project, not a PLC state)"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epoch(series):
    """ISO text -> Int64 epoch seconds of the naive wall clock; missing stays missing."""
    stamps = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S")
    values = (stamps.astype("int64") // 10**9).astype("Int64")
    return values.mask(stamps.isna())


def iso(ts):
    return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------- raw 1 Hz


def file_order(name):
    m = HOUR_FILE.fullmatch(name)
    return f"{m[3]}-{MONTHS[m[2]]:02}-{m[1]}T{m[4]}" if m and m[2] in MONTHS else name


def parse_hour(job):
    """One hourly CSV -> (info, frame, unparseable cells). Runs in a worker process."""
    root, path = Path(job[0]), Path(job[1])
    data = path.read_bytes()
    info = dict(source_root=root.name, relative_path=path.relative_to(root).as_posix(),
                bytes=len(data), sha256=sha256_bytes(data), rows=0, header_ok=1,
                first_ts=None, last_ts=None)
    raw = pd.read_csv(io.BytesIO(data), sep=";", dtype=str, keep_default_na=False, skipinitialspace=True)
    if list(raw.columns) != ["Date-Time", "Time_s"] + KEYS:
        info["header_ok"] = 0
        return info, None, []
    info["rows"] = len(raw)
    if raw.empty:
        return info, None, []
    stamps = pd.to_datetime(raw["Date-Time"], format="%d %m %Y %H:%M:%S", errors="raise")
    ts = (stamps.astype("int64") // 10**9).to_numpy()
    if (np.diff(ts) <= 0).any():
        raise ValueError(f"Timestamps not strictly increasing inside {path}")
    if not (pd.to_numeric(raw["Time_s"]).to_numpy() == ts % 3600).all():
        raise ValueError(f"Time_s is not the second of the hour in {path}")
    values = raw[KEYS].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(values.to_numpy(dtype=float))
    cells = [dict(ts=int(ts[r]), field_key=KEYS[c], kind="unparseable_value", raw_text=raw.iat[r, c + 2],
                  source_row=int(r) + 2) for r, c in zip(*np.where(bad))]
    frame = values.where(~bad, np.nan).astype("float64")
    frame.insert(0, "ts", ts)
    frame["source_row"] = np.arange(len(frame), dtype=np.int32) + 2  # 1-based line in the CSV, header = 1
    info.update(first_ts=int(ts[0]), last_ts=int(ts[-1]))
    return info, frame, cells


def load_raw(roots, workers):
    """All hourly files in clock order. Later roots only fill seconds the earlier ones lack."""
    jobs = []
    for priority, root in enumerate(roots):
        paths = [p for p in root.rglob("*.csv") if "Voltage Sample" not in p.parts and HOUR_FILE.fullmatch(p.name)]
        jobs += [(priority, file_order(p.name), p.relative_to(root).as_posix(), str(root), str(p)) for p in paths]
    jobs.sort()
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(parse_hour, [j[3:] for j in jobs], chunksize=8))
    files, frames, quality = [], [], []
    for file_id, (job, (info, frame, cells)) in enumerate(zip(jobs, results), start=1):
        info = dict(file_id=file_id, priority=job[0], **info)
        files.append(info)
        if not info["header_ok"]:
            quality.append(dict(ts=None, field_key=None, kind="unexpected_header", raw_text=None, source_row=1, file_id=file_id))
        elif info["rows"] == 0:
            quality.append(dict(ts=None, field_key=None, kind="empty_file", raw_text=None, source_row=None, file_id=file_id))
        quality += [dict(c, file_id=file_id) for c in cells]
        if frame is not None:
            frame["file_id"] = np.int32(file_id)
            frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    del frames
    # Stable sort by (ts, file_id): file ids follow source priority, so the main tree wins a tie.
    samples = samples.sort_values(["ts", "file_id"], kind="stable", ignore_index=True)
    duplicate = samples.duplicated("ts", keep="first").to_numpy()
    if duplicate.any():
        first = samples[~duplicate].set_index("ts")
        again = samples[duplicate]
        same = np.isclose(first.loc[again.ts.to_numpy(), KEYS].to_numpy(), again[KEYS].to_numpy(), equal_nan=True).all(axis=1)
        if not same.all():
            raise ValueError(f"{int((~same).sum())} seconds are recorded twice with different readings")
        dropped = again.groupby("file_id").size()
        samples = samples[~duplicate].reset_index(drop=True)
    else:
        dropped = pd.Series(dtype="int64")
    kept = samples.groupby("file_id").size()
    for info in files:
        info["rows_loaded"] = int(kept.get(info["file_id"], 0))
        info["rows_duplicate_dropped"] = int(dropped.get(info["file_id"], 0))
    return files, samples, quality


def gap_table(ts):
    step = np.diff(ts)
    at = np.flatnonzero(step > 1)
    missing = step[at] - 1
    bucket = np.select([missing < 60, missing < 3600], ["lt_1min", "lt_1h"], "ge_1h")
    return pd.DataFrame(dict(gap_id=np.arange(1, len(at) + 1), last_ts_before=ts[at], first_ts_after=ts[at + 1],
                             last_before=iso(ts[at]), first_after=iso(ts[at + 1]),
                             missing_seconds=missing, bucket=bucket, origin="hmi_csv"))


def minute_table(samples):
    powered = (samples["Current"] > 100) & (samples["Power"] > 5)
    grouped = samples.assign(powered=powered).groupby(samples.ts // 60 * 60, sort=True)
    out = grouped[KEYS].mean()
    out.insert(0, "n_rows", grouped.size())
    out.insert(1, "n_powered", grouped["powered"].sum().astype("int64"))
    out["Voltage_min"], out["Voltage_max"] = grouped["Voltage"].min(), grouped["Voltage"].max()
    return out.rename_axis("ts").reset_index()


def day_table(samples):
    powered = (samples["Current"] > 100) & (samples["Power"] > 5)
    day = pd.to_datetime(samples.ts, unit="s").dt.strftime("%Y-%m-%d")
    grouped = samples.assign(powered=powered, date=day).groupby("date", sort=True)
    out = pd.DataFrame(dict(rows=grouped.size(), first_ts=grouped.ts.min(), last_ts=grouped.ts.max(),
                            powered_seconds=grouped.powered.sum().astype("int64"),
                            hour_files=grouped.file_id.nunique()))
    out["missing_seconds"] = 86400 - out["rows"]
    return out.reset_index()


def write_raw_lake(samples, folder):
    if folder.exists():
        shutil.rmtree(folder)
    day = pd.to_datetime(samples.ts, unit="s").dt.strftime("%Y-%m-%d").to_numpy()
    edges = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
    rows = {}
    for a, b in zip(edges[:-1], edges[1:]):
        part = samples.iloc[a:b]
        table = pa.table({"ts": pa.array(part.ts.to_numpy(), pa.int64()),
                          "ts_local": pa.array(part.ts.to_numpy().astype("datetime64[s]"), pa.timestamp("s")),
                          **{k: pa.array(part[k].to_numpy(), pa.float64(), from_pandas=True) for k in KEYS},
                          "file_id": pa.array(part.file_id.to_numpy(), pa.int32()),
                          "source_row": pa.array(part.source_row.to_numpy(), pa.int32())})
        target = folder / f"date={day[a]}"
        target.mkdir(parents=True)
        pq.write_table(table, target / "part-0.parquet", compression="zstd", row_group_size=3600)
        rows[f"date={day[a]}/part-0.parquet"] = b - a
    return rows


# ----------------------------------------------------------------------------- curated tables


class Repo:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.commit = self.git("rev-parse", "HEAD").decode().strip()
        listing = self.git("ls-tree", "-r", "-z", "HEAD").decode("utf-8")
        self.blobs = {line.split("\t", 1)[1]: line.split()[2] for line in listing.split("\0") if line}
        self.used = {}

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.folder), "-c", "core.quotepath=off", *args],
                              check=True, capture_output=True).stdout

    def read(self, path):
        data = self.git("show", f"HEAD:{path}")
        self.used[path] = dict(git_blob=self.blobs[path], bytes=len(data), sha256=sha256_bytes(data))
        return data

    def csv(self, path):
        data = self.read(path)
        return pd.read_csv(io.BytesIO(data), compression="gzip" if path.endswith(".gz") else None,
                           low_memory=False, encoding="utf-8")

    def json(self, path):
        return json.loads(self.read(path).decode("utf-8"))


def normalise(frame):
    """Rewrite 'DD MM YYYY HH:MM:SS' (single values and ;-lists) as ISO. Returns single-valued time columns."""
    single = []
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        values = frame[column].dropna()
        if values.empty or not values.map(lambda v: isinstance(v, str)).all():
            continue
        if values.map(lambda v: bool(DMY_LIST.match(v))).all():
            frame[column] = frame[column].map(lambda v: DMY_PART.sub(r"\3-\2-\1 \4", v), na_action="ignore")
            if values.map(lambda v: bool(DMY_ONE.match(v))).all():
                single.append(column)
        elif values.map(lambda v: bool(ISO_ONE.match(v))).all():
            single.append(column)
    return single


def curated(repo):
    """name -> (frame, primary key, indexes, single-valued time columns, description)."""
    t = {}

    def add(name, frame, key, indexes=(), note=""):
        t[name] = dict(frame=frame, key=key, indexes=list(indexes), times=normalise(frame), note=note)

    v2, v1 = "model_v2_GRU/data", "training_data"

    cycles = repo.csv(f"{v2}/cycles.csv")
    normalise(cycles)
    for column in ("usable_start", "usable_end", "eol_endpoint"):
        cycles.insert(cycles.columns.get_loc(column) + 1, f"{column}_ts", epoch(cycles[column]))
    add("electrode_cycle", cycles, "cycle_id",
        note="电极安装周期（v2，11/17 纠正后）。eol_endpoint = 最终连续失败序列第一件开始加工 T；source_* 为 v1 口径。")
    legacy = repo.csv(f"{v1}/electrode_cycles.csv").rename(columns={"split": "split_v1"})
    add("electrode_cycle_v1_legacy", legacy, "cycle_id",
        note="v1（11/16）周期表原样保留：eol_endpoint = 第二件失败产品结束；划分为 v1 划分。仅供追溯。")

    events = repo.csv(f"{v2}/events.csv")
    normalise(events)
    events.insert(events.columns.get_loc("time") + 1, "ts", epoch(events["time"]))
    add("target_event", events, "event_id", ["cycle_id"],
        note="连续失败事件（v2）：time = 第一件失败产品开始加工；is_eol=1 为电极末期事件。")
    add("product_failure_event_v1", repo.csv(f"{v1}/target_events.csv").rename(columns={"split": "split_v1"}),
        "event_id", ["cycle_id", "product_id"],
        note="v1 逐产品事件（first_* / adjacent_two_*）：time = 该产品结束（标签可得）时刻。")

    products = repo.csv(f"{v2}/products.csv.gz")
    v1_products = repo.csv(f"{v1}/products.csv")
    normalise(products), normalise(v1_products)
    shared = [c for c in v1_products.columns if c != "split"]
    if not products[shared].equals(v1_products[shared]) or not (products["source_split"] == v1_products["split"]).all():
        raise ValueError("v2 products differ from training_data/products.csv beyond the split column")
    products.insert(products.columns.get_loc("start") + 1, "start_ts", epoch(products["start"]))
    products.insert(products.columns.get_loc("end") + 1, "end_ts", epoch(products["end"]))
    add("product", products, "product_id", ["cycle_id", "start_ts", "failure"],
        note="全量产品（13,181）。split = v2 划分，source_split = v1 划分；其余列与 training_data/products.csv 相同。")

    add("maintenance_event", repo.csv(f"{v1}/maintenance_events.csv"), "record_id",
        note="换枪登记 / 现场确认记录；action_time_lower/upper 是候选范围，不是精确拆装秒。")
    labels = repo.csv("data_processing_11_16/label_samples.csv").rename(columns={"split": "split_v1"})
    add("label_sample", labels, "sample_id", ["cycle_id"], note="353 件关注产品（审阅子集，不能单独用于训练）。")
    add("split_summary", repo.csv(f"{v2}/split_summary.csv"), "split", note="v2 划分汇总。")
    add("split_summary_v1_legacy", repo.csv(f"{v1}/split_summary.csv"), "split", note="v1 划分汇总。")

    anchors = repo.csv(f"{v2}/anchors.csv.gz")
    normalise(anchors)
    anchors.insert(0, "ts", epoch(anchors["timestamp"]).astype("int64"))
    if not ("A" + anchors.ts.astype(str) == anchors.anchor_id).all():
        raise ValueError("anchor_id is not 'A' + epoch seconds of the wall clock")
    order = np.argsort(products.start_ts.to_numpy(), kind="stable")
    starts, ends = products.start_ts.to_numpy()[order], products.end_ts.to_numpy()[order]
    at = np.searchsorted(starts, anchors.ts.to_numpy(), side="right") - 1
    inside = (at >= 0) & (anchors.ts.to_numpy() <= ends[np.maximum(at, 0)])
    anchors["product_id"] = np.where(inside, products.product_id.to_numpy()[order][np.maximum(at, 0)], None)
    anchors["seconds_since_product_start"] = pd.array(np.where(inside, anchors.ts.to_numpy() - starts[np.maximum(at, 0)], -1), "Int64")
    anchors["seconds_since_product_start"] = anchors["seconds_since_product_start"].mask(~inside)
    add("anchor", anchors, "ts", ["cycle_id", "split", "product_id"],
        note="v2 加工原点（10 s，开工确认 ≥3 s）：9 个传感器当秒读数 + 过去 60 s 电压均值/标准差。product_id 为本库派生。")

    targets = repo.csv(f"{v2}/targets.csv.gz")
    if not (targets.anchor_id == anchors.anchor_id).all():
        raise ValueError("v2 targets are not aligned with v2 anchors")
    targets.insert(0, "ts", anchors.ts)
    add("target", targets.drop(columns=["timestamp", "cycle_id", "split"]), "ts",
        note="v2 监督目标：y=1 当 lo < remaining_seconds ≤ hi；观测范围不足以判定时为 NULL（不能当 0）。")

    features = repo.csv(f"{v1}/features_10s.csv.gz").rename(columns={"split": "split_v1"})
    normalise(features)
    features.insert(0, "ts", features.anchor_id.str[1:].astype("int64"))
    if not (epoch(features["timestamp"]) == features.ts).all():
        raise ValueError("training_data anchor ids disagree with their timestamps")
    features["in_v2"] = features.anchor_id.isin(set(anchors.anchor_id)).astype("int64")
    lead = ["ts", "anchor_id", "timestamp", "cycle_id", "split_v1", "in_v2", "history_points60s"]
    add("anchor_feature_28", features[lead + [c for c in features.columns if c not in lead]], "ts", ["cycle_id", "in_v2"],
        note="v1 原点特征（28 路当秒值 + 60 s 均值 + 电压 60 s 标准差）。in_v2=0 的原点在产品开始后 <3 s，被 v2 剔除。")
    old = repo.csv(f"{v1}/targets_10s.csv.gz").rename(columns={"split": "split_v1"})
    if not (old.anchor_id == features.anchor_id).all():
        raise ValueError("training_data targets are not aligned with their features")
    old.insert(0, "ts", features.ts)
    add("target_v1_legacy", old.drop(columns=["timestamp", "cycle_id"]), "ts",
        note="v1 目标原样保留。rul_strict_seconds 以 v1 终点（第二件失败结束）计，比 v2 remaining_seconds 多 153–157 s；L012 在 v1 为空。")

    frozen = repo.json("model_v2_GRU/artifacts/frozen_config.json")
    hashes = {k.replace("\\", "/"): v for k, v in frozen["fitted_file_sha256"].items()}
    rows = []
    for model_id, spec in enumerate(frozen["model_specs"], start=1):
        name = spec["name"]
        policy = frozen["policies"][name]
        artifact = next((k for k in (f"models/{name}/best.pt", f"models/{name}/model.joblib") if k in hashes), None)
        rows.append(dict(
            model_id=model_id, name=name, family="booth2-v2", kind=spec["kind"], version=frozen["model_version"],
            head=policy["head"], head_index=policy["head_index"], threshold=policy["threshold"],
            alarm_enabled=int(policy["threshold"] <= 1.0), selected=int(name == frozen["selected_model"]),
            output_kind="uncalibrated_score", frozen_at_utc=frozen["frozen_at_utc"],
            source_repository="ruiqiyang666/ANN_data_policy", source_commit=repo.commit,
            artifact_path=f"model_v2_GRU/artifacts/{artifact}" if artifact else None,
            artifact_sha256=hashes.get(artifact), alarm_policy=json.dumps(frozen["alarm_policy"], sort_keys=True),
            spec=json.dumps(spec, sort_keys=True, ensure_ascii=False)))
    registry = pd.DataFrame(rows)
    add("model_registry", registry, "model_id",
        note="模型登记。阈值 1.000001 是“不报警”哨兵（验证误报预算内无可用阈值），alarm_enabled=0。")
    ids = dict(zip(registry.name, registry.model_id))

    ledger = repo.csv("model_v2_GRU/artifacts/predictions.csv.gz")
    if not (ledger.anchor_id == anchors.anchor_id).all():
        raise ValueError("The v2 prediction ledger is not aligned with the v2 anchors")
    narrow = pd.concat([pd.DataFrame(dict(model_id=ids[name], ts=anchors.ts.to_numpy(), score=ledger[name].to_numpy()))
                        for name in registry.name], ignore_index=True)
    add("prediction", narrow, ["model_id", "ts"],
        note="追加式预测台账；当前内容为 v2 冻结台账（12 个候选 × 81,235 原点）的评分。")

    alarms = repo.csv("model_v2_GRU/artifacts/alarm_results.csv")
    normalise(alarms)
    alarms.insert(0, "alarm_id", np.arange(1, len(alarms) + 1))
    alarms.insert(1, "model_id", alarms.pop("model").map(ids))
    alarms.insert(2, "ts", epoch(alarms["time"]))
    add("alarm", alarms.rename(columns={"probability": "score", "index": "origin_index"}), "alarm_id", ["model_id", "ts"],
        note="v2 冻结评估发出的报警（gru_16 与 hgb_depth2）；score 是未校准评分，不是概率。")
    results = repo.csv("model_v2_GRU/artifacts/event_results.csv")
    normalise(results)
    results.insert(0, "model_id", results.pop("model").map(ids))
    add("event_result", results, ["model_id", "event_id"], note="v2 冻结评估的逐事件结果（12 候选 × 10 个 EOL 事件）。")
    return t


# ----------------------------------------------------------------------------- SQLite


def sql_type(series):
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    return "REAL" if pd.api.types.is_float_dtype(series) else "TEXT"


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def records(frame):
    """Row tuples with missing as None and NumPy scalars as Python values."""
    columns = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series):
            series = series.astype("Int64")
        values = series.astype(object).where(series.notna(), None).tolist()
        columns.append([v.item() if isinstance(v, np.generic) else v for v in values])
    return zip(*columns)


def create(con, name, frame, key, indexes=(), strict_key=True):
    keys = [key] if isinstance(key, str) else list(key)
    parts = [f"{quote(c)} {sql_type(frame[c])}" + (" NOT NULL" if c in keys else "") for c in frame.columns]
    integer_rowid = len(keys) == 1 and sql_type(frame[keys[0]]) == "INTEGER"
    if integer_rowid:
        parts[list(frame.columns).index(keys[0])] = f"{quote(keys[0])} INTEGER PRIMARY KEY"
    else:
        parts.append(f"PRIMARY KEY ({', '.join(quote(k) for k in keys)})")
    con.execute(f"CREATE TABLE {quote(name)} (\n  " + ",\n  ".join(parts) + "\n)" + ("" if integer_rowid else " WITHOUT ROWID"))
    marks = ", ".join("?" * len(frame.columns))
    con.executemany(f"INSERT INTO {quote(name)} VALUES ({marks})", records(frame))
    for column in indexes:
        con.execute(f"CREATE INDEX {quote(f'ix_{name}_{column}')} ON {quote(name)} ({quote(column)})")


VIEWS = """
CREATE VIEW v_sample_1hz AS
  SELECT datetime(ts, 'unixepoch') AS ts_local, s.* FROM sample_1hz s;
CREATE VIEW v_tag_opcua AS
  SELECT t.tag_id, t.field_key, t.kepware_tag, t.opcua_node_id, t.node_id_confirmed, t.unit,
         t.scan_rate_ms, t.deadband_abs, t.in_history, t.reserved, t.model_v2_input
  FROM tag t ORDER BY t.tag_id;
CREATE VIEW v_anchor_supervised AS
  SELECT a.*, t.y_eol_30m, t.y_eol_45m, t.y_eol_60m, t.y_pre20_40m
  FROM anchor a JOIN target t USING (ts);
CREATE VIEW v_eol_event AS
  SELECT e.event_id, e.cycle_id, c.split, e.time, e.ts, e.deadline_20m, e.run_length, e.product_ids,
         e.second_completion AS v1_endpoint, c.end_type, c.end_evidence
  FROM target_event e JOIN electrode_cycle c USING (cycle_id) WHERE e.is_eol = 1;
CREATE VIEW v_cycle_summary AS
  SELECT c.cycle_id, c.split, c.usable_start, c.usable_end, c.end_type, c.eol_endpoint,
         c.right_censored, c.preventive_end, c.left_censored, c.product_count, c.eligible_products,
         c.failure_products, c.voltage_failed_products, c.feed_failed_products, c.pressure_failed_products,
         c.observed_on_hours, c.recorded_old_electrode_hours, c.unknown_clock_hours,
         (SELECT COUNT(*) FROM anchor a WHERE a.cycle_id = c.cycle_id) AS anchors,
         (SELECT COUNT(*) FROM anchor a WHERE a.cycle_id = c.cycle_id AND a.has_raw_1hz = 1) AS anchors_with_raw_1hz,
         (SELECT COUNT(*) FROM anchor a WHERE a.cycle_id = c.cycle_id AND a.remaining_seconds IS NOT NULL) AS anchors_with_remaining,
         (SELECT COUNT(*) FROM target_event e WHERE e.cycle_id = c.cycle_id AND e.is_eol = 1) AS eol_events,
         (SELECT COUNT(*) FROM target_event e WHERE e.cycle_id = c.cycle_id AND e.is_eol = 0) AS other_run_events
  FROM electrode_cycle c ORDER BY c.cycle_id;
CREATE VIEW v_day_coverage AS
  WITH days AS (
    SELECT date FROM raw_day UNION SELECT substr(timestamp, 1, 10) FROM anchor UNION SELECT substr(start, 1, 10) FROM product),
  a AS (SELECT substr(timestamp, 1, 10) AS date, COUNT(*) AS anchors FROM anchor GROUP BY 1),
  p AS (SELECT substr(start, 1, 10) AS date, COUNT(*) AS products, SUM(failure = 1) AS failed_products,
               SUM(failure IS NULL) AS unlabelled_products FROM product GROUP BY 1)
  SELECT d.date, r.rows AS raw_rows, r.missing_seconds AS raw_missing_seconds, r.powered_seconds,
         COALESCE(a.anchors, 0) AS anchors, COALESCE(p.products, 0) AS products,
         COALESCE(p.failed_products, 0) AS failed_products, COALESCE(p.unlabelled_products, 0) AS unlabelled_products,
         CASE WHEN r.rows IS NULL THEN 'curated_only' WHEN a.anchors IS NULL AND p.products IS NULL THEN 'raw_only' ELSE 'both' END AS coverage
  FROM days d LEFT JOIN raw_day r USING (date) LEFT JOIN a USING (date) LEFT JOIN p USING (date) ORDER BY d.date;
CREATE VIEW v_prediction AS
  SELECT m.name AS model, m.version, p.ts, datetime(p.ts, 'unixepoch') AS ts_local, p.score, m.threshold,
         m.alarm_enabled, m.selected
  FROM prediction p JOIN model_registry m USING (model_id);
"""


def tag_table():
    asset, defaults = REGISTRY["asset"], REGISTRY["defaults"]
    rows = []
    for tag_id, tag in enumerate(REGISTRY["tags"], start=1):
        kepware = tag.get("kepware_tag", tag["key"])
        rows.append(dict(
            tag_id=tag_id, asset_id=asset["asset_id"], field_key=tag["key"], name_cn=tag["name"], unit=tag["unit"],
            tag_group=tag["group"], dtype=tag["dtype"], kepware_tag=kepware,
            kepware_tag_confirmed=int(tag.get("kepware_tag_confirmed", False)),
            opcua_node_id=REGISTRY["node_id_template"].format(ns=asset["opcua_namespace_index"], path=asset["opcua_path"], kepware_tag=kepware),
            node_id_confirmed=0, scan_rate_ms=defaults["scan_rate_ms"], deadband_abs=defaults["deadband_abs"],
            max_age_s=defaults["max_age_s"], in_history=int(not tag.get("reserved", False)),
            reserved=int(tag.get("reserved", False)), model_v2_input=int(tag.get("model_v2_input", False)),
            pg_column=re.sub(r"[^a-z0-9]+", "_", tag["key"].lower()).strip("_"), note=tag.get("note")))
    return pd.DataFrame(rows)


def build_sqlite(path, files, samples, quality, tables, meta):
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript("PRAGMA page_size = 8192; PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;")
    counts = {}

    def put(name, frame, key, indexes=()):
        create(con, name, frame, key, indexes)
        counts[name] = len(frame)

    put("meta", pd.DataFrame(sorted(meta.items()), columns=["key", "value"]), "key")
    asset = REGISTRY["asset"]
    put("asset", pd.DataFrame([dict(asset_id=asset["asset_id"], asset_key=asset["key"], name=asset["name"],
                                    opcua_path=asset["opcua_path"], opcua_namespace_index=asset["opcua_namespace_index"],
                                    opcua_confirmed=0, note=asset["note"])]), "asset_id")
    put("tag", tag_table(), "tag_id", ["field_key"])
    put("source_file", pd.DataFrame(files)[["file_id", "source_root", "relative_path", "priority", "bytes", "sha256", "header_ok",
                                           "rows", "rows_loaded", "rows_duplicate_dropped", "first_ts", "last_ts"]], "file_id")
    columns = ["ts"] + KEYS + ["file_id", "source_row"]
    con.execute("CREATE TABLE sample_1hz (\n  ts INTEGER PRIMARY KEY,\n  " + ",\n  ".join(f"{quote(k)} REAL" for k in KEYS)
                + ",\n  file_id INTEGER NOT NULL REFERENCES source_file(file_id),\n  source_row INTEGER NOT NULL\n)")
    marks = ", ".join("?" * len(columns))
    for a in range(0, len(samples), 200_000):
        con.executemany(f"INSERT INTO sample_1hz VALUES ({marks})", records(samples.iloc[a:a + 200_000][columns]))
    counts["sample_1hz"] = len(samples)
    put("sample_1min", minute_table(samples), "ts")
    put("raw_day", day_table(samples), "date")
    put("ingest_gap", gap_table(samples.ts.to_numpy()), "gap_id", ["last_ts_before"])
    events = pd.DataFrame(quality, columns=["ts", "field_key", "kind", "raw_text", "source_row", "file_id"])
    events.insert(0, "quality_id", np.arange(1, len(events) + 1))
    events["ts"] = events["ts"].astype("Int64")
    events["source_row"] = events["source_row"].astype("Int64")
    put("quality_event", events, "quality_id", ["ts"])
    for name, spec in tables.items():
        put(name, spec["frame"], spec["key"], spec["indexes"])
    con.executescript(LIVE_SCHEMA)
    con.executescript(APPEND_ONLY)
    counts.update(live_sample=0, collector_heartbeat=0, ingest_event=0)
    con.executescript(VIEWS)
    con.commit()
    con.execute("ANALYZE")
    con.commit()
    con.execute("VACUUM")
    con.close()
    return counts


def dump_schema(db, target, notes):
    con = sqlite3.connect(db)
    rows = con.execute("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY "
                       "CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END, rowid").fetchall()
    con.close()
    lines = ["-- Generated by ingest/load_history.py from db/booth2.sqlite. Do not edit.",
             f"-- Clock: {CLOCK}.", ""]
    for kind, name, sql in rows:
        if kind == "table" and name in notes:
            lines.append(f"-- {notes[name]}")
        lines += [sql.strip() + ";", ""]
    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- lake + manifest


def write_curated_lake(tables, folder):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    rows = {}
    for name, spec in tables.items():
        frame = spec["frame"].copy()
        for column in spec["times"]:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d %H:%M:%S").astype("datetime64[s]")
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), folder / f"{name}.parquet", compression="zstd")
        rows[f"{name}.parquet"] = len(frame)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True, help="ANN_data_policy clone")
    parser.add_argument("--raw", type=Path, required=True, help="Real-time Data folder (hourly 1 Hz CSVs)")
    parser.add_argument("--raw-extra", type=Path, action="append", default=[], help="extra CSV folder; only fills missing seconds")
    parser.add_argument("--out", type=Path, default=HERE.parent)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = args.out.resolve()

    print("raw 1 Hz ...", flush=True)
    files, samples, quality = load_raw([args.raw.resolve()] + [p.resolve() for p in args.raw_extra], args.workers)
    print(f"  {len(files)} files, {len(samples):,} seconds, {iso(samples.ts.iloc[[0, -1]].to_numpy()).tolist()}", flush=True)
    raw_rows = write_raw_lake(samples, out / "lake" / "raw_1hz")

    print("curated tables ...", flush=True)
    repo = Repo(args.repo)
    tables = curated(repo)
    anchors = tables["anchor"]["frame"]
    raw_index = pd.Index(samples.ts)
    present = raw_index.get_indexer(anchors.ts)
    anchors.insert(anchors.columns.get_loc("remaining_seconds") + 1, "has_raw_1hz", (present >= 0).astype("int64"))
    curated_rows = write_curated_lake(tables, out / "lake" / "curated")

    meta = dict(
        schema_version=SCHEMA_VERSION, clock=CLOCK, powered_rule=POWERED,
        source_repository="ruiqiyang666/ANN_data_policy", source_commit=repo.commit,
        raw_first=iso(int(samples.ts.iloc[0])), raw_last=iso(int(samples.ts.iloc[-1])),
        raw_seconds=str(len(samples)), raw_roots=json.dumps([args.raw.name] + [p.name for p in args.raw_extra], ensure_ascii=False),
        curated_first=anchors.timestamp.min(), curated_last=anchors.timestamp.max(),
        anchors_without_raw_1hz=str(int((present < 0).sum())),
        label_authority="model_v2_GRU/data (endpoints corrected 2026-11-17); training_data kept as *_v1_legacy / split_v1",
        missing_values="NULL; never zero-filled, never interpolated",
    )
    print("sqlite ...", flush=True)
    db = out / "db" / "booth2.sqlite"
    counts = build_sqlite(db, files, samples, quality, tables, meta)
    notes = {name: spec["note"] for name, spec in tables.items()}
    dump_schema(db, out / "schema" / "sqlite_schema.sql", notes)

    outputs = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        relative = path.relative_to(out).as_posix()
        if relative.split("/")[0] in ("scratch", "spool") or "__pycache__" in relative or relative == "manifest.json":
            continue
        entry = dict(bytes=path.stat().st_size, sha256=sha256_file(path))
        if relative.startswith("lake/raw_1hz/"):
            entry["rows"] = int(raw_rows[relative[len("lake/raw_1hz/"):]])
        elif relative.startswith("lake/curated/"):
            entry["rows"] = int(curated_rows[relative[len("lake/curated/"):]])
        outputs[relative] = entry
    raw_digest = sha256_bytes("\n".join(f"{f['source_root']}/{f['relative_path']} {f['bytes']} {f['sha256']}" for f in files).encode())
    manifest = dict(
        schema_version=SCHEMA_VERSION, clock=CLOCK,
        sources=dict(repository=dict(name="ruiqiyang666/ANN_data_policy", commit=repo.commit, files=repo.used),
                     raw=dict(roots=[args.raw.name] + [p.name for p in args.raw_extra], files=len(files),
                              bytes=int(sum(f["bytes"] for f in files)), listing_sha256=raw_digest,
                              rows=int(sum(f["rows"] for f in files)),
                              duplicate_rows_dropped=int(sum(f["rows_duplicate_dropped"] for f in files)))),
        coverage=dict(raw_first=meta["raw_first"], raw_last=meta["raw_last"], curated_first=meta["curated_first"],
                      curated_last=meta["curated_last"], anchors_without_raw_1hz=int((present < 0).sum())),
        sqlite_tables=counts, table_notes=notes, files=outputs,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(counts, indent=1))
    print(f"db {db.stat().st_size / 1e6:.0f} MB, lake {sum(v['bytes'] for k, v in outputs.items() if k.startswith('lake/')) / 1e6:.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
