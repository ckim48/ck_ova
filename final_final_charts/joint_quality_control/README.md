# Joint mR/KL control — development results

**Result: the shared round100 FL switch retains mR leadership over the three unchanged comparators on both datasets and lowers KL. It does not achieve joint leadership on fixed-reference KL and mR.**

[Combined six-figure PDF](joint_quality_control_all.pdf) · [Gallery](index.html) · [All122 exact-screen outcomes](data/all_122_exact_screens.csv) · [Every physical candidate](data/all_physical_candidates.csv)

Current published Proposed remains P-KS. `Controlled` denotes the separate `uniform_after100` candidate: use the original priority16 sampler through round100, then uniform16 through200. The original gated FU rule, parallel AirComp, data, request schedule, temperature and power caps are unchanged. This is a fixed training schedule, not an automatic online KL controller.

| Dataset | Configuration | Retained mR (%) | Loss | Common KL | Worst-request KL | Matched KL | Energy (kJ) | Active delay (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 17.59115 | 5.22885 | 0.0182323 | 0.0261308 | 0.0044503 | 204.277 | 121.102 |
| CapERA | Controlled | 17.65625 | 5.27950 | 0.0100569 | 0.0139482 | 0.0034600 | 204.277 | 121.102 |
| CapERA | ISCC-P2 | 16.90104 | 5.34352 | 0.0037886 | 0.0058325 | 0.0033352 | 649.962 | 384.304 |
| CapERA | AFL-OTA | 16.96615 | 5.34348 | 0.0037912 | 0.0058378 | 0.0033372 | 252.882 | 149.912 |
| CapERA | F-SalUn | 17.55208 | 5.28097 | 0.0096676 | 0.0161790 | 0.0089805 | 649.960 | 384.302 |
| AirScape | Proposed | 13.80208 | 5.51150 | 0.0073197 | 0.0137718 | 0.0012347 | 204.267 | 121.096 |
| AirScape | Controlled | 13.80208 | 5.52934 | 0.0034039 | 0.0062087 | 0.0008931 | 204.267 | 121.096 |
| AirScape | ISCC-P2 | 13.73698 | 5.55939 | 0.0008385 | 0.0013707 | 0.0012296 | 649.305 | 383.915 |
| AirScape | AFL-OTA | 13.73698 | 5.55937 | 0.0008385 | 0.0013713 | 0.0012300 | 253.066 | 150.021 |
| AirScape | F-SalUn | 13.73698 | 5.52985 | 0.0031273 | 0.0047710 | 0.0039088 | 649.307 | 383.916 |

The CapERA common KL remains slightly above0.01; its worst-request KL also exceeds0.01. AirScape satisfies both. Training loss increases versus original P-KS on both datasets, despite preserving/improving retained mR. Small mR gains are descriptive seed0 outcomes, not statistical significance.

The common reference is the same fixed retained-only200-round reference for every method. Matched KL uses each learning policy’s own retained-only retraining reference; the new schedule has a fresh200-round matched reference. It is a different counterfactual, not a replacement for common KL. AirScape controlled matched KL is best among the three unchanged baselines; CapERA remains behind ISCC-P2/AFL-OTA.

## FU intervention check

| Dataset | Exact condition | Forget mR (%) | Common KL | Matched KL |
|---|---|---:|---:|---:|
| CapERA | exact_no_fu | 16.46424 | 0.0032181 | 0.0002072 |
| CapERA | exact_fu | 16.97706 | 0.0100397 | 0.0034479 |
| AirScape | exact_no_fu | 11.35940 | 0.0013284 | 0.0007612 |
| AirScape | exact_fu | 11.68529 | 0.0033991 | 0.0008890 |

FU-on still has larger common and matched KL than its paired exact no-FU control. It also has higher forget-set mR. Thus the improvement versus original P-KS supports reducing FL-induced distribution drift; it does **not** establish a positive forgetting effect. C9 bounds transmission perturbation, not equivalence to retraining. No-FU request commits are bookkeeping, not completed deletion.

## Scope and complete search

Four sequential exploratory families contain61 declared configurations including repeated controls, each on both datasets:122 exact-aggregation runs. All outcomes, including worse mR/KL and failed settings, are exported. Exact controls reproduce original exact-P-KS tensors bitwise. Exact screens do not supply physical energy/delay or power-feasibility claims.

Three shortlisted settings are physically rerun on both datasets, using the original AirComp/head/power optimization and actual noisy received updates. `bound100` lost CapERA mR leadership; `uniform_after100` is preferred here because it gives the largest verified KL reduction while maintaining the original mR level on both datasets. The third candidate, `rank0.5`, tests a symmetric CE/hard-negative margin mixture and is retained in the table below. These candidates are alternatives, not additional reference baselines.

| Dataset | Physical candidate | Retained mR (%) | Common KL | Worst-request KL |
|---|---|---:|---:|---:|
| CapERA | bound100 | 17.44792 | 0.0091452 | 0.0125873 |
| CapERA | uniform_after100 | 17.65625 | 0.0100569 | 0.0139482 |
| CapERA | rank0.5 | 17.85156 | 0.0143255 | 0.0199503 |
| AirScape | bound100 | 13.80208 | 0.0044091 | 0.0081501 |
| AirScape | uniform_after100 | 13.80208 | 0.0034039 | 0.0062087 |
| AirScape | rank0.5 | 13.93229 | 0.0054064 | 0.0099507 |

The physical-v1 factory attempt failed before training while constructing an unused proxy-objective factory. Its script and local log are retained; physical-v2 restores the original factory for that unused objective and supplies the original two seed adapters. No failed attempt is silently overwritten.

All candidates use the same200 FL rounds,45 requests and225 prescribed FU cycles, existing known-forget-cohort proxy20, native video captions, full gallery and frozen pretrained encoders with trainable rank10 adapters. No evaluation reference, test caption or evaluation probability is used as a training teacher/gradient. References retrain adapters from initialization, not backbone pretraining from scratch.

This is adaptive development on the previously inspected seed0 cohorts. Shared settings across datasets do not make this an independent held-out confirmation. Further seeds and untouched evaluation cohorts are required before claiming a general lead. No per-dataset model selection, changed comparator, temperature-only calibration, favorable-query filtering or added training rounds is used.

Active energy uses the unchanged communication+computation+hover ledger; shared travel/collection/preflight are excluded. All physical runs retain0.1J network RF and0.32W device caps. Extra gradient/teacher work is charged through executed device MAC counts. Inherited FL_gradient_calls metadata counts nominal participating clients, not additional gradient passes for bounded variants; use work MAC ledgers for costs. Simulated energy is not a hardware measurement.

## Figures

| File | Meaning |
|---|---|
|28_controlled_retained_mr_1x2|Actual final retained mR; original and controlled Proposed against unchanged comparators.|
|29_controlled_common_kl_1x2|Primary fixed-reference forget KL; lower is better.|
|30_controlled_matched_kl_1x2|Secondary policy-matched KL; references differ by learning policy.|
|31_controlled_worst_kl_1x2|Maximum common-reference request KL over all45 requests.|
|32_paired_fu_control_kl_1x2|Exact FU/no-FU causal diagnostic under the new FL schedule.|
|33_development_tradeoff_1x2|All exact-screen candidates; two physical Proposed points identified separately.|

## Reproduction

Run the four `joint_quality_*` screen drivers with `.venv/bin/python`; use the physical-v2 driver for first/focused/context candidates and the loss-v4 physical action for loss candidates. Run `joint_quality_audit_v1.py`, then `joint_quality_report_v1.py` and `joint_quality_check_v1.py`. GPU drivers preserve interrupted trials rather than overwrite them. Protocol hashes, plot arrays, physical outcomes and causal controls are in `data/`; raw checkpoints and licensed feature caches remain local.

Conceptual motivation only: [CLIPErase (ACL2025)](https://aclanthology.org/2025.acl-long.1469/) for visual/text forgetting-retention-consistency terms and [SLUG (ICML2025)](https://proceedings.mlr.press/v267/cai25f.html) for VLM gradient considerations. The new controls are custom adaptations, not full reproductions of either paper.
