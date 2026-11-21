from pathlib import Path
import sys


def sub(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (str(path), old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_bytes(s.encode("utf-8"))


S = Path(sys.argv[1]) / "simulation/pipeline/static/js"
sub(S / "components/real28_view.js", [
("""    const second = d.electrode && window.P2Electrode;
    if (second) window.P2Electrode.setup(d);
    const current = select.value === "electrode" && second ? "electrode" : "active";
    select.innerHTML = `<option value="active">${e(name)}</option>` +
      (second ? `<option value="electrode">${e(window.P2Electrode.option())}</option>` : "");""",
 """    // Further models of the same replay: [model id, its view], listed when the bundle is mounted.
    const extra = [["electrode", d.electrode && window.P2Electrode], ["life", d.life && window.P2Life]].filter(([, view]) => view);
    extra.forEach(([, view]) => view.setup(d));
    const current = extra.some(([id]) => id === select.value) ? select.value : "active";
    select.innerHTML = `<option value="active">${e(name)}</option>` +
      extra.map(([id, view]) => `<option value="${id}">${e(view.option())}</option>`).join("");"""),
("""  // The electrode model is a second reading of the same replay; see components/electrode_view.js.
  const second = (s) => (window.P2Electrode?.shows(s) ? window.P2Electrode : null);
  function draw(s, tab = "life") {
    if (second(s)) {
      if (language !== P2.lang()) { relabel(config); modelOption(config); }
      return second(s).draw(s, tab);
    }
    window.P2Electrode?.leave();""",
 """  // The electrode models are further readings of the same replay; see components/electrode_view.js
  // and components/life_view.js. At most one of them owns the panel at a time.
  const views = () => [window.P2Electrode, window.P2Life].filter(Boolean);
  const second = (s) => views().find((view) => view.shows(s)) || null;
  function draw(s, tab = "life") {
    const view = second(s);
    views().forEach((other) => { if (other !== view) other.leave(); });
    if (view) {
      if (language !== P2.lang()) { relabel(config); modelOption(config); }
      return view.draw(s, tab);
    }"""),
])
sub(S / "shared/live_session.js", [
("""      if (!["active", "baseline", "electrode"].includes(id) || id === modelId) return;""",
 """      if (!["active", "baseline", "electrode", "life"].includes(id) || id === modelId) return;"""),
("""      // The electrode model replays its own recorded morning, so entering or leaving it starts
      // a new session; the other models are two readings of the same sample.
      const replay = id === "electrode" || previous === "electrode";""",
 """      // The electrode models replay their own recorded morning, so entering or leaving one starts
      // a new session; the other models are two readings of the same sample.
      const own = ["electrode", "life"];
      const replay = own.includes(id) || own.includes(previous);"""),
])
sub(S / "components/neural_view.js", [
("""    "序列输入": "Sequence in", "GRU 隐状态": "GRU state", "末期风险": "End of life",
  };""", """    "序列输入": "Sequence in", "GRU 隐状态": "GRU state", "末期风险": "End of life", "分段风险": "Hazard by interval",
  };"""),
(""""序列输入": "输入", "GRU 隐状态": "隐状态", "末期风险": "输出",
  };""", """"序列输入": "输入", "GRU 隐状态": "隐状态", "末期风险": "输出", "分段风险": "分段风险",
  };"""),
("""  const LAYER_SHORT_EN = { "Hidden 1": "Layer 1", "Hidden 2": "Layer 2", "Raw out": "Out", Outputs: "Out" };""",
 """  const LAYER_SHORT_EN = { "Hidden 1": "Layer 1", "Hidden 2": "Layer 2", "Raw out": "Out", Outputs: "Out", "Hazard by interval": "Hazard" };"""),
])
print("frontend patched")
