# v6 research: loss alignment and heterogeneous degradation

This run follows v5 error analysis. All old historical cycles have already been inspected; neither LOCO nor forward blocking makes this a new blind test. `protocol.json` is frozen before full training. School experiments are isolated from the dashboard and company field machine.

Compared methods:

- Two Kaplan–Meier baselines: landmark remaining life and conditional lifetime given observed age. The age baseline excludes left-truncated L000 during baseline fitting; its test result remains visible.
- Inverse-censoring-weighted quantile boosted trees, 6/32 features. Five raw-hour quantiles and marginal training-only censoring KM, G floor 0.1. Conditional independent censoring is not established by this dataset; IPCW is an assumption-dependent benchmark.
- CQRNN-style monotone neural quantiles, 6/32 features. Portnoy weights follow the author's implementation, with weight 1 below the inferred censor quantile. Architecture, cycle weighting and monotone increments are project adaptations, not a claim to reproduce the paper's experiments. Probabilities between quantiles require interpolation assumptions.
- Two-component lognormal mixture networks, 6/32 features, exact observed/censored mixture likelihood. Inspired by DSM, independently implemented; not full DSM or a competing-risk model.
- Random survival forest with controlled depth and coarse training duration grid. Unidentified tail quantiles are recorded and bounded by a declared support+24h display convention; this convention is not a calibrated survival estimate.
- Separate logistic and boosted-tree wall-clock 60-minute warning heads, with inner-fold thresholds. Processing-time RUL probabilities are not used as wall-clock risk probabilities.

All scalers/censoring weights/selection use whole training cycles. Every candidate uses the same 14 LOCO folds and 3 forward blocks. Outer test outcomes are not used for architecture changes or threshold selection within this run. Outer prediction confidence should reflect the 10 independent failure cycles, not 81,235 correlated origins.

## Commands

```bash
python research/math_checks.py
python research/synthetic_checks.py
python research/train.py --data /data --out smoke2 --smoke
python research/train.py --data /data --out run1 --jobs 6
python research/verify.py --data /data --run run1 --v5 /v5
python research/summarize.py --run run1 --v5 /v5
```

`--collect` rebuilds the report from complete folds; a fold's `result.json` is written after all its models and warnings are saved. Resume only with unchanged sources, protocol, environment and data. Every run needs its own output folder. The school training image is `booth2-survival-v6:20261120`, built from the existing image plus pinned scikit-survival 0.28.0; exact installed versions are recorded in provenance.

Model `joblib` is for trusted local research artifacts only. Verify the adjacent metadata SHA-256 before loading; never accept user-uploaded joblib/pickle. This is not the dashboard deployment format. Full-data fitted weights must not replace held-out predictions for evaluation. Deployment requires a reviewed export/runtime with equivalent results, separately validated live causal inputs, and human approval.

## References read for this run

- Pearce et al., NeurIPS 2022, [Censored Quantile Regression Neural Networks](https://arxiv.org/abs/2205.13496), [author implementation](https://github.com/TeaPearce/Censored_Quantile_Regression_NN/blob/main/01_code/models.py). The below-censor quantile branch was checked against the implementation and a focused failing-then-passing mathematical assertion.
- Nagpal et al., [Deep Survival Machines](https://arxiv.org/abs/2003.01176), and [time-varying covariate extension](https://proceedings.mlr.press/v146/nagpal21a.html). Mixture components motivate testing multiple degradation modes; they do not prove the modes are identifiable here.
- [scikit-survival RandomSurvivalForest](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html), [evaluation methods](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html).
- [DeepQuantreg](https://www.sciencedirect.com/science/article/pii/S0167947321001572): IPCW-adjusted quantile loss. Our tree candidate uses this general weighting idea, not the neural implementation.
- [Deep Cox Mixtures](https://proceedings.mlr.press/v149/nagpal21a.html), [dependent-censoring survival models](https://proceedings.mlr.press/v216/gharari23a.html): reviewed as subsequent directions, not claimed as trained models in this run.

## Diagnostic evidence

`diagnostics/v5-cycle-errors.csv` retains all cycles, including short-lived L006/L012. `feature-prefix-audit.json` checks 12 truncated prefixes against the historical cache; it covers 7 selected base features and does not establish equivalence of the whole live pipeline. Timestamp/restart reconstruction matches all 14 cycles. `end-of-life-trajectories.png` illustrates abrupt versus progressive voltage changes; retrospective categorization is not an input to any candidate.

L002/L005 show voltage collapse only in the final products, while L006/L012 show earlier downward trends. This is evidence for heterogeneous patterns, not proof that the first group is fundamentally unpredictable. Future work should test other causal sensors and true maintenance/cause labels.

## Domain-specific and recent research follow-up

- Oerlikon Metco's published [WO2026027724A1](https://patents.google.com/patent/WO2026027724A1/en) describes electrode/nozzle prognostics using high-frequency voltage (20–100 kHz, examples at 50 kHz), spectral representations, DeepSAD and a mixture of local experts. This is a directly relevant technical disclosure, not independent evidence of performance on Booth2. Existing 1 Hz records cannot reconstruct the omitted high-frequency waveform. An instrumentation feasibility study and actual comparable waveform data are prerequisites; no hardware was changed here.
- [APS ensemble modelling with electrode ageing](https://pmc.ncbi.nlm.nih.gov/articles/PMC9821370/) examines process/particle characteristics under ageing. Its target and controlled settings differ from this dataset's end-of-life task, so its reported performance is not a transferable RUL benchmark.
- [SurvPFN, 2026](https://arxiv.org/abs/2606.04564) proposes prior-data fitted survival prediction with censor-aware distributional loss. Author repository checked at commit `4f26fe31efd811f546a6a3a12acd28293828ca7e`; README now links an official `samuelboehm/SurvPFN` checkpoint and limits features to 2–10. It requires a separate Python >=3.12 / PyTorch >=2.9 environment plus a pinned TFM-Playground dependency. This run has not downloaded or evaluated the checkpoint; the v6 Python 3.11 / PyTorch 2.5 image remains fixed. Synthetic task priors, dependent maintenance censoring and correlated per-cycle origins need testing before relying on it.
- A 2025 [contrastive mixer study for sparse run-to-failure plasma arc cutting data](https://www.sciencedirect.com/science/article/pii/S0007850625001301) appeared in search results; the publisher full text returned 403. Only the abstract-level description was inspected, so implementation and claimed effect are not treated as verified evidence here.

## Audit correction

A focused test found that conditional age-KM returned immediate failure probability after the test age exceeded all empirical survival support, while its quantiles used the declared support+24h sentinel. The corrected runtime uses the matching point-mass fallback and flags unsupported ages. This is an unidentifiable baseline region, not calibrated extrapolation. `repair_km_support.py` preserves original artifacts, changes only KM probabilities/Brier diagnostics, and asserts unchanged quantiles and selection losses. Learned models and warning scores are untouched.
