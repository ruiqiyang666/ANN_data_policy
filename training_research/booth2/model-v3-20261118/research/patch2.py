from pathlib import Path
import sys
def sub(path, pairs):
    p = Path(path); s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (str(path), old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_bytes(s.encode("utf-8"))
R = Path(sys.argv[1]) / "simulation/pipeline"
sub(R / "inference/real28.py", [
("""    def __init__(self, bundle, default_segment=None, electrode=None):
        self.bundle = bundle
        # Optional second model scored on the same recorded seconds; see inference/electrode.py.
        self.electrode = electrode
        if electrode is not None:
            segment = electrode.segment
            if not (bundle.raw_t[0] <= segment["start"] and segment["stop"] <= bundle.raw_t[-1] + 1):
                raise ValueError("The electrode replay segment lies outside the recorded rows")
""", """    def __init__(self, bundle, default_segment=None, electrode=None, life=None):
        self.bundle = bundle
        # Optional further models scored on the same recorded seconds; see inference/electrode.py
        # and inference/life.py. Each replays its own recorded segment when selected.
        self.electrode = electrode
        self.life = life
        self.extra = {key: model for key, model in (("electrode", electrode), ("life", life)) if model is not None}
        for key, model in self.extra.items():
            segment = model.segment
            if not (bundle.raw_t[0] <= segment["start"] and segment["stop"] <= bundle.raw_t[-1] + 1):
                raise ValueError(f"The {key} replay segment lies outside the recorded rows")
"""),
("""            "electrode": self.electrode.summary() if self.electrode is not None else None,
            "display\"""", """            "electrode": self.electrode.summary() if self.electrode is not None else None,
            "life": self.life.summary() if self.life is not None else None,
            "display\""""),
("""            if model_id == "electrode":
                if self.electrode is None:
                    raise ValueError("电极末期预警模型未接入")
                # Its own recorded morning: the score rises, alarms, and the electrode then fails.
                segment = self.electrode.segment
            else:""", """            if model_id in EXTRA_MODELS:
                # Its own recorded morning: the score rises, alarms, and the electrode then fails.
                segment = self.extra_model(model_id).segment
            else:"""),
("""        electrode = model_id == "electrode" and self.electrode is not None
        if model_id == "electrode" and self.electrode is None:
            raise ValueError("电极末期预警模型未接入")
        return {""", """        extra = self.extra_model(model_id) if model_id in EXTRA_MODELS else None
        return {"""),
("""            "model_kind": "real28", "model_id": "electrode" if electrode else "active",""",
 """            "model_kind": "real28", "model_id": model_id if extra is not None else "active","""),
("""            "electrode": self.electrode.outputs(stamp) if self.electrode is not None else None,
            "inputs\"""", """            "electrode": self.electrode.outputs(stamp) if self.electrode is not None else None,
            "life": self.life.outputs(stamp) if self.life is not None else None,
            "inputs\""""),
("""            "neural": self.electrode.neural_payload(stamp, seq) if electrode else self.bundle.neural_payload(position, seq),""",
 """            "neural": extra.neural_payload(stamp, seq) if extra is not None else self.bundle.neural_payload(position, seq),"""),
("""    def _session(self, session_id):""", """    def extra_model(self, model_id):
        model = self.extra.get(model_id)
        if model is None:
            raise ValueError(f"{EXTRA_MODELS[model_id]}未接入")
        return model

    def _session(self, session_id):"""),
("""QUANTILES = (-1.2815515655, 0.0, 1.2815515655)  # 10 %, 50 %, 90 % of a lognormal
""", """QUANTILES = (-1.2815515655, 0.0, 1.2815515655)  # 10 %, 50 %, 90 % of a lognormal
EXTRA_MODELS = {"electrode": "电极末期预警模型", "life": "电极寿命与预警模型"}  # selectable beside the 28-input model
"""),
])
sub(R / "run.py", [
("""    args = parser.parse_args()""", """    parser.add_argument(
        "--life",
        type=Path,
        help="Read-only exported bundle of the electrode life-and-warning model (v3); shown beside --real28 on the same replay",
    )
    args = parser.parse_args()"""),
("""    import torch
""", """    if args.life is not None and args.real28 is None:
        parser.error("--life replays on the recorded rows of --real28 and needs it")
    if args.life is not None and not (args.life / "manifest.json").is_file():
        parser.error(f"--life has no manifest.json: {args.life}")
    import torch
"""),
("""real28=args.real28, suite=args.suite, electrode=args.electrode,""",
 """real28=args.real28, suite=args.suite, electrode=args.electrode, life=args.life,"""),
])
sub(R / "api/application.py", [
("""def create_app(output, real_data=None, *, demo=False, index_output=None, real28=None, suite=False, electrode=None):""",
 """def create_app(output, real_data=None, *, demo=False, index_output=None, real28=None, suite=False, electrode=None, life=None):"""),
("""        app.state.real28 = Real28Service(bundle, default_segment="normal_20261029" if suite else None, electrode=second)""",
 """        third = None
        if life is not None:
            from inference.life import LifeBundle

            third = LifeBundle(life)  # recomputes every origin and compares with the training ledger
            third.verify()
        app.state.real28 = Real28Service(
            bundle, default_segment="normal_20261029" if suite else None, electrode=second, life=third
        )"""),
("""        return HTMLResponse(document.replace("</body>", launch + "</body>", 1))""",
 """        if app.state.real28 is not None and app.state.real28.life is not None:
            document = document.replace(
                page_script, f'<script defer src="{asset("js/components/life_view.js")}"></script>\n    ' + page_script, 1
            )
        return HTMLResponse(document.replace("</body>", launch + "</body>", 1))"""),
])
sub(R / "api/simulation_routes.py", [
("""    model_id: Literal["active", "baseline", "electrode"] = "active"


class AdvanceRequest""", """    model_id: Literal["active", "baseline", "electrode", "life"] = "active"


class AdvanceRequest"""),
("""def current_session(session_id: str, model_id: Literal["active", "baseline", "electrode"] = "active"):""",
 """def current_session(session_id: str, model_id: Literal["active", "baseline", "electrode", "life"] = "active"):"""),
])
sub(R / "api/workbench_routes.py", [
("""            electrode = getattr(service, "electrode", None)
""", """            electrode = getattr(service, "electrode", None)
            life = getattr(service, "life", None)
"""),
("""                if electrode is not None else None,
            )""", """                if electrode is not None else None,
                life=dict(life.workbench(), version=life.version, summary=life.summary()) if life is not None else None,
            )"""),
])
print("backend patched")
