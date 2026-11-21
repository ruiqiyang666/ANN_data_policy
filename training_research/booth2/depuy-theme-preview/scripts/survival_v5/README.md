# v5 censored survival benchmark

Research only; the November 20 run did not justify replacing the active model. See `docs/model-v5-evaluation.md` and `docs/continuous-learning.md`.

Input snapshot contains `frame.parquet`, `events.csv`, `products.csv` and frozen `evaluation.py`. Additional baseline files are hashed in provenance. The frame is the versioned v4 historical feature cache, not a live feature builder. Do not feed raw notification rows directly to this trainer.

```bash
python scripts/survival_v5/train.py --data /path/to/snapshot --out /path/to/new-run --jobs 8
python scripts/survival_v5/finalize.py --data /path/to/snapshot --run /path/to/new-run
```

Use a new output directory. `--smoke` exercises all families; `--collect` regenerates the report from completed folds without refitting. `finalize.py` audits holdout predictions and fits/reloads all six full-data research artifacts. Its full-data outputs must never replace the held-out predictions used for evaluation.

Frozen training environment: Python 3.11.10, NumPy 2.1.2, pandas 3.0.3, scikit-learn 1.9.0, SciPy 1.17.1, PyTorch 2.5.1+cu124, XGBoost 3.2.0; joblib and Parquet engine included in the recorded Docker image. School image digest is in the report. No packages were installed into the dashboard runtime.

`runtime.predict(artifact_path, feature_frame)` reloads JSON/UBJ weights and returns the 12-horizon CDF and q10/q50/q90 in processing hours. It requires NumPy/SciPy/pandas and XGBoost only for the tree family. It supplies no dashboard adapter, live alignment, alarm debounce or label conversion. Those need independent equivalence and deployment validation.

The protocol remains frozen beside the source. Changes to loss, features, selection or thresholds require a new protocol and run directory; do not tune on the outer holdouts and then call the same result an unbiased test.
