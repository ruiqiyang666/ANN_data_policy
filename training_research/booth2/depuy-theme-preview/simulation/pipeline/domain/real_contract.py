"""Instrument names and units; company limits are read from local annotations."""

import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

BASE_FIELDS = [
    ("Argon", "氩气流量", "NLPM", "气体"),
    ("Nitrogen", "氮气流量", "NLPM", "气体"),
    ("AirJet", "喷枪前端冷却气压", "bar", "气体"),
    ("Silvent", "转台侧冷却气压", "bar", "气体"),
    ("Current", "喷枪电流", "A", "电气"),
    ("Power", "喷枪功率", "kW", "电气"),
    ("Voltage", "喷枪电压", "V", "电气"),
    ("Cooling_water_flow", "冷却水流量", "L/min", "冷却"),
    ("Conductivity", "冷却水电导率", "μs（原标注，待确认）", "冷却"),
    ("Cooling_water_inlet_temp.", "冷却水入口温度", "°C", "冷却"),
    ("Cooling_water_outlet_temp.", "冷却水出口温度", "°C", "冷却"),
]
FEEDER_FIELDS = [
    ("Feedrate", "送料量", "g/min"),
    ("Disk", "送粉盘设定", "%"),
    ("Carrier_gas", "载气流量", "NLPM"),
    ("Feed_Factor", "送料系数", "—"),
    ("Press._Hopper", "料斗压力", "mbar"),
    ("Stirrer", "搅拌器设定", "%"),
    ("Weight_Hopper", "料斗重量", "kg"),
]
REAL_FIELDS = [dict(key=k, name=n, unit=u, group=g) for k, n, u, g in BASE_FIELDS]
for feeder in (1, 2):
    REAL_FIELDS.extend(
        dict(key=f"{key}_{feeder}", name=f"{name} {feeder}", unit=unit, group=f"料斗 {feeder}")
        for key, name, unit in FEEDER_FIELDS
    )
REAL_FIELDS.extend(
    [
        dict(key="Station_1", name="转台 1 转速", unit="rpm", group="环境 / 转台"),
        dict(key="Station_2", name="转台 2 转速", unit="rpm", group="环境 / 转台"),
        dict(key="O2_Sensor", name="环境氧气", unit="%", group="环境 / 转台"),
    ]
)
REAL_KEYS = [f["key"] for f in REAL_FIELDS]
# Reviewable state rules (see NOTES); the agent's answer check also accepts them.
ENERGIZED_CURRENT = 100  # A
ACTIVE_FEEDRATE = 1  # g/min


def load_annotation_notes(source):
    """Read the original field notes and their workbook cell addresses."""
    path = Path(source) / "标注.xlsx"
    if not path.is_file():
        return []
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", ns):
                strings.append("".join(t.text or "" for t in item.findall(".//s:t", ns)))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall("s:sheetData/s:row", ns)[:2]:
            cells = {}
            for cell in row.findall("s:c", ns):
                address = re.sub(r"\d", "", cell.attrib["r"])
                value = cell.find("s:v", ns)
                text = (
                    value.text
                    if value is not None
                    else "".join(t.text or "" for t in cell.findall(".//s:t", ns))
                )
                cells[address] = strings[int(text)] if cell.attrib.get("t") == "s" else text
            rows.append(cells)
    if len(rows) < 2:
        return []
    return [dict(field=key.strip(" ;\t\r\n"), note=rows[1].get(column, ""),
                 source=f"标注.xlsx · Sheet1!{column}2")
            for column, key in rows[0].items()]


def load_local_rules(source):
    """Accept only explicit limits from local notes, with no numerical defaults.

    Keep this shape stable: it participates in the existing index signature.
    """
    notes = {row["field"]: row["note"] for row in load_annotation_notes(source)}
    rules = []
    for field in REAL_FIELDS:
        key = field["key"]
        if key != "Voltage" and not key.startswith(
            ("Feedrate_", "Press._Hopper_", "Weight_Hopper_")
        ):
            continue
        note = notes.get(key, "")
        lower = re.search(r"(?:低于|报警值)\s*(\d+(?:\.\d+)?)", note)
        upper = re.search(r"高于\s*(\d+(?:\.\d+)?)", note)
        if lower is None:
            continue
        low = float(lower[1])
        high = float(upper[1]) if upper else None
        if high is not None and high <= low:
            continue
        scope = "energized" if key == "Voltage" else "feeder" + key[-1]
        label = (
            f"{field['name']} "
            + (f"低于 {low:g}" if high is None else f"超出 {low:g}–{high:g}")
            + f" {field['unit']}"
        )
        rules.append(dict(field=key, low=low, high=high, scope=scope, label=label))
    return rules


NOTES = [
    "时间按仪器原始时钟显示，原文件未注明时区；不自动进行 UTC 换算。",
    "带电筛选暂用 Current > 100 A；活动送料暂用带电且对应 Feedrate > 1 g/min，属于可复核的分析口径。",
    "越界统计针对瞬时样本。原工艺规则按单件喷涂平均值判定；瞬时越界不等于产品不合格或已确认故障。",
    "连续片段仅合并相邻 1 秒的同类记录；断档切断片段。缺少记录不等于采集丢包，也可能是停产。",
    "没有喷枪身份、更换和维修真值，本页不推断喷枪寿命、故障根因或产品实际质量。",
]
