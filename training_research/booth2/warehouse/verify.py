"""Re-check the Booth2 warehouse: file hashes, row counts and integrity invariants.

Standard library only, so it runs on the server host (python3 without numpy/pandas):

    python3 verify.py            # full check
    python3 verify.py --quick    # skip hashing the Parquet lake

Exit code 0 only if every check passes.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="hash only the database, schema and scripts")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

    # ---- files
    wrong, missing, hashed = [], [], 0
    for relative, entry in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size != entry["bytes"]:
            wrong.append(relative)
            continue
        if args.quick and relative.startswith("lake/"):
            continue
        hashed += 1
        if sha256(path) != entry["sha256"]:
            wrong.append(relative)
    check("every manifest file present", not missing, f"{len(manifest['files'])} listed, {len(missing)} missing")
    check("sizes and sha256 match the manifest", not wrong, f"{hashed} hashed" + (f", wrong: {wrong[:3]}" if wrong else ""))

    con = sqlite3.connect(f"file:{(ROOT / 'db' / 'booth2.sqlite').as_posix()}?mode=ro", uri=True)
    one = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731

    # ---- row counts
    counts = {name: one(f'SELECT COUNT(*) FROM "{name}"') for name in manifest["sqlite_tables"]}
    different = {k: (v, manifest["sqlite_tables"][k]) for k, v in counts.items() if v != manifest["sqlite_tables"][k]}
    check("SQLite row counts match the manifest", not different, f"{len(counts)} tables" + (f", {different}" if different else ""))
    lake_rows = sum(e["rows"] for k, e in manifest["files"].items() if k.startswith("lake/raw_1hz/"))
    check("raw lake rows = sample_1hz rows", lake_rows == counts["sample_1hz"], f"{lake_rows:,}")
    curated = {k[len("lake/curated/"):-len(".parquet")]: e["rows"] for k, e in manifest["files"].items() if k.startswith("lake/curated/")}
    check("curated lake rows = SQLite rows", all(counts[k] == v for k, v in curated.items()), f"{len(curated)} tables")
    check("SQLite integrity_check", one("PRAGMA integrity_check") == "ok")

    # ---- raw
    check("raw seconds unique and strictly increasing by construction (PK)",
          one("SELECT COUNT(*) FROM sample_1hz") == one("SELECT COUNT(DISTINCT ts) FROM sample_1hz"))
    check("source files account for every raw row",
          one("SELECT SUM(rows_loaded) FROM source_file") == counts["sample_1hz"]
          and one("SELECT SUM(rows) - SUM(rows_loaded) - SUM(rows_duplicate_dropped) FROM source_file") == 0)
    check("gaps + rows cover the raw span exactly",
          one("SELECT MAX(ts) - MIN(ts) + 1 FROM sample_1hz") == counts["sample_1hz"] + one("SELECT SUM(missing_seconds) FROM ingest_gap"))
    check("no NaN text and no invented zeros for unparseable cells",
          one("""SELECT COUNT(*) FROM quality_event q JOIN sample_1hz s USING (ts)
                 WHERE q.kind = 'unparseable_value' AND q.field_key = 'O2_Sensor' AND s."O2_Sensor" IS NOT NULL""") == 0)

    # ---- curated invariants
    check("anchor <-> target 1:1",
          counts["anchor"] == counts["target"] == one("SELECT COUNT(*) FROM anchor JOIN target USING (ts)"))
    check("v1 feature <-> v1 target 1:1",
          counts["anchor_feature_28"] == counts["target_v1_legacy"] == one("SELECT COUNT(*) FROM anchor_feature_28 JOIN target_v1_legacy USING (ts)"))
    check("every v2 anchor exists in the v1 28-feature table and is flagged in_v2",
          one("SELECT COUNT(*) FROM anchor a JOIN anchor_feature_28 f USING (ts) WHERE f.in_v2 = 1") == counts["anchor"]
          and one("SELECT SUM(in_v2) FROM anchor_feature_28") == counts["anchor"])
    check("anchor_id = 'A' || ts, origins on the 10 s grid",
          one("SELECT COUNT(*) FROM anchor WHERE anchor_id != 'A' || ts OR ts % 10 != 0") == 0)
    check("anchor timestamp text = ts", one("SELECT COUNT(*) FROM anchor WHERE strftime('%s', timestamp) + 0 != ts") == 0)
    check("anchor readings equal the raw 1 Hz row wherever raw exists",
          one("""SELECT COUNT(*) FROM anchor a JOIN sample_1hz s USING (ts)
                 WHERE a.Voltage != s.Voltage OR a.Current != s.Current OR a.Power != s.Power
                    OR a.Station_1 != s.Station_1 OR a.Station_2 != s.Station_2
                    OR a.Feedrate_1 != s.Feedrate_1 OR a.Feedrate_2 != s.Feedrate_2
                    OR a."Press._Hopper_1" != s."Press._Hopper_1" OR a."Press._Hopper_2" != s."Press._Hopper_2" """) == 0,
          f"{one('SELECT SUM(has_raw_1hz) FROM anchor'):,} of {counts['anchor']:,} anchors have raw")
    check("has_raw_1hz flag is exact",
          one("SELECT COUNT(*) FROM anchor a JOIN sample_1hz s USING (ts)") == one("SELECT SUM(has_raw_1hz) FROM anchor"))
    check("every anchor lies inside one eligible product of its cycle, >= 3 s after its start",
          one("""SELECT COUNT(*) FROM anchor a LEFT JOIN product p USING (product_id)
                 WHERE p.product_id IS NULL OR p.supervised_product_eligible != 1 OR p.cycle_id != a.cycle_id
                    OR a.ts < p.start_ts + 3 OR a.ts > p.end_ts""") == 0)
    check("products per cycle = cycles table counts (failure_products counts supervised-eligible failures)",
          one("""SELECT COUNT(*) FROM electrode_cycle c WHERE c.product_count != (SELECT COUNT(*) FROM product p WHERE p.cycle_id = c.cycle_id)
                    OR c.eligible_products != (SELECT COUNT(*) FROM product p WHERE p.cycle_id = c.cycle_id AND p.supervised_product_eligible = 1)
                    OR c.failure_products != (SELECT COUNT(*) FROM product p WHERE p.cycle_id = c.cycle_id AND p.failure = 1
                                                AND p.supervised_product_eligible = 1)""") == 0,
          f"{counts['product']:,} products, {counts['electrode_cycle']} cycles")
    check("product / anchor split = cycle split (v2)",
          one("SELECT COUNT(*) FROM product p JOIN electrode_cycle c USING (cycle_id) WHERE p.split != c.split") == 0
          and one("SELECT COUNT(*) FROM anchor a JOIN electrode_cycle c USING (cycle_id) WHERE a.split != c.split") == 0)
    check("each cycle with an EOL endpoint has exactly one EOL event at that instant, and vice versa",
          one("""SELECT COUNT(*) FROM electrode_cycle c WHERE (c.eol_endpoint IS NOT NULL) !=
                    ((SELECT COUNT(*) FROM target_event e WHERE e.cycle_id = c.cycle_id AND e.is_eol = 1 AND e.time = c.eol_endpoint) = 1)""") == 0
          and one("SELECT COUNT(*) FROM target_event WHERE is_eol = 1") == one("SELECT COUNT(*) FROM electrode_cycle WHERE eol_endpoint IS NOT NULL"),
          f"{one('SELECT COUNT(*) FROM target_event WHERE is_eol = 1')} EOL events")
    check("EOL instant = start of the first product of the run; v1 endpoint = second completion",
          one("""SELECT COUNT(*) FROM target_event e JOIN product p ON p.product_id = substr(e.product_ids, 1, 7)
                 WHERE p.start != e.time""") == 0
          and one("""SELECT COUNT(*) FROM target_event e JOIN electrode_cycle c USING (cycle_id)
                     WHERE e.is_eol = 1 AND e.second_completion != c.source_eol_endpoint""") == 0)
    check("remaining_seconds = eol_endpoint - origin, present exactly before a known endpoint",
          one("""SELECT COUNT(*) FROM anchor a JOIN electrode_cycle c USING (cycle_id)
                 WHERE (a.remaining_seconds IS NOT NULL) != (c.eol_endpoint_ts IS NOT NULL AND a.ts < c.eol_endpoint_ts)
                    OR (a.remaining_seconds IS NOT NULL AND a.remaining_seconds != c.eol_endpoint_ts - a.ts)""") == 0)
    check("targets follow their definition (y = 1 iff lo < remaining <= hi; NULL never used as 0)",
          one("""SELECT COUNT(*) FROM anchor a JOIN target t USING (ts) WHERE
                    (t.y_eol_30m = 1) != (a.remaining_seconds > 0 AND a.remaining_seconds <= 1800 AND t.y_eol_30m IS NOT NULL)
                 OR (t.y_eol_45m = 1) != (a.remaining_seconds > 0 AND a.remaining_seconds <= 2700 AND t.y_eol_45m IS NOT NULL)
                 OR (t.y_eol_60m = 1) != (a.remaining_seconds > 0 AND a.remaining_seconds <= 3600 AND t.y_eol_60m IS NOT NULL)
                 OR (t.y_pre20_40m = 1) != (a.remaining_seconds > 1200 AND a.remaining_seconds <= 2400 AND t.y_pre20_40m IS NOT NULL)""") == 0)
    check("v1 strict RUL = v2 remaining + (v1 endpoint - v2 endpoint) wherever both exist",
          one("""SELECT COUNT(*) FROM anchor a JOIN target_v1_legacy o USING (ts) JOIN electrode_cycle c USING (cycle_id)
                 WHERE a.remaining_seconds IS NOT NULL AND o.rul_strict_seconds IS NOT NULL
                   AND o.rul_strict_seconds - a.remaining_seconds != strftime('%s', c.source_eol_endpoint) - c.eol_endpoint_ts""") == 0)
    check("failure counts: 13,046 normal / 51 failed / 84 unlabelled",
          con.execute("SELECT SUM(failure = 0), SUM(failure = 1), SUM(failure IS NULL) FROM product").fetchone() == (13046, 51, 84))
    check("exactly one failed product is not supervised-eligible (site timing uncertain)",
          con.execute("SELECT COUNT(*), MIN(product_id), MIN(site_timing_uncertain) FROM product WHERE failure = 1 AND supervised_product_eligible = 0").fetchone() == (1, "N000499", 1))
    check("prediction ledger: one score per model and anchor, exactly one selected model",
          counts["prediction"] == counts["model_registry"] * counts["anchor"] and one("SELECT SUM(selected) FROM model_registry") == 1)
    check("every stored alarm refers to an anchor and its ledger score",
          one("""SELECT COUNT(*) FROM alarm a LEFT JOIN prediction p USING (model_id, ts)
                 WHERE p.ts IS NULL OR abs(p.score - a.score) > 1e-6 OR 'A' || a.ts != a.anchor_id""") == 0, f"{counts['alarm']} alarms")
    try:
        con.execute("UPDATE prediction SET score = 0 WHERE 0")  # read-only handle; the trigger is checked by name
    except sqlite3.OperationalError:
        pass
    check("append-only triggers present on the prediction ledger",
          one("SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'prediction'") == 2)
    check("tag registry: 28 measured + 4 reserved, hopper pressure mapped to Kepware names",
          con.execute("SELECT SUM(in_history), SUM(reserved) FROM tag").fetchone() == (28, 4)
          and one("""SELECT COUNT(*) FROM tag WHERE field_key LIKE 'Press._Hopper_%' AND kepware_tag LIKE 'Press\\_Hopper\\__' ESCAPE '\\'""") == 2)
    check("live tables exist and are empty (nothing has been collected)",
          counts["live_sample"] == 0 and counts["collector_heartbeat"] == 0 and counts["ingest_event"] == 0)
    con.close()

    span = manifest["coverage"]
    print(f"\nraw 1 Hz : {span['raw_first']} .. {span['raw_last']}  ({counts['sample_1hz']:,} s)")
    print(f"curated  : {span['curated_first']} .. {span['curated_last']}  ({span['anchors_without_raw_1hz']:,} anchors have no raw 1 Hz)")
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
