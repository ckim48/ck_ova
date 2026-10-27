# Final charts: adaptive retrieval FU

Updated 2026-10-28. Left: CapERA. Right: AirScape. [Browse figures](index.html). Proposed now uses the automatic retention-weight controller in every current primary figure. Comparators keep their recorded objectives, data and physical constraints.

| Dataset | mR change (pp) | KL reduction (%) | Energy change (%) | Delay change (%) |
|---|---:|---:|---:|---:|
| CapERA | -0.573 | 91.59 | +0.093 | +0.092 |
| AirScape | -0.130 | 86.65 | +0.243 | +0.243 |

Changes above are against previous Proposed at 1 MHz, seed 0. Lower KL is better; a negative mR change is a loss of utility. The automatic controller substantially reduces retrain-distance in this development experiment but does not improve both evaluation metrics simultaneously. It is a new tradeoff, not evidence of universal superiority. All paired seed outcomes and all sweep conditions are retained.

## Figures

| Figure | Content | Files |
|---|---|---|
| 01_round_retained_mr_1x2 | Retained mR (%) versus FL rounds | [PNG](01_round_retained_mr_1x2.png) · [PDF](01_round_retained_mr_1x2.pdf) · [SVG](01_round_retained_mr_1x2.svg) |
| 02_time_retained_mr_1x2 | Retained mR (%) versus active service time | [PNG](02_time_retained_mr_1x2.png) · [PDF](02_time_retained_mr_1x2.pdf) · [SVG](02_time_retained_mr_1x2.svg) |
| 03_time_forget_kl_1x2 | Forget-set KL versus active service time | [PNG](03_time_forget_kl_1x2.png) · [PDF](03_time_forget_kl_1x2.pdf) · [SVG](03_time_forget_kl_1x2.svg) |
| 04_bandwidth_total_energy_1x2 | Total energy (kJ) versus bandwidth | [PNG](04_bandwidth_total_energy_1x2.png) · [PDF](04_bandwidth_total_energy_1x2.pdf) · [SVG](04_bandwidth_total_energy_1x2.svg) |
| 05_bandwidth_delay_1x2 | Delay (s) versus bandwidth | [PNG](05_bandwidth_delay_1x2.png) · [PDF](05_bandwidth_delay_1x2.pdf) · [SVG](05_bandwidth_delay_1x2.svg) |
| 06_request_mean_delay_1x2 | Mean FU request delay | [PNG](06_request_mean_delay_1x2.png) · [PDF](06_request_mean_delay_1x2.pdf) · [SVG](06_request_mean_delay_1x2.svg) |
| 07_request_p95_delay_1x2 | 95th-percentile FU request delay | [PNG](07_request_p95_delay_1x2.png) · [PDF](07_request_p95_delay_1x2.pdf) · [SVG](07_request_p95_delay_1x2.svg) |
| 08_energy_to_quality_1x2 | Energy to the common quality target | [PNG](08_energy_to_quality_1x2.png) · [PDF](08_energy_to_quality_1x2.pdf) · [SVG](08_energy_to_quality_1x2.svg) |
| 09_delay_to_quality_1x2 | Time to the common quality target | [PNG](09_delay_to_quality_1x2.png) · [PDF](09_delay_to_quality_1x2.pdf) · [SVG](09_delay_to_quality_1x2.svg) |
| final_quality_table | Final quality and reference models | [PNG](final_quality_table.png) · [PDF](final_quality_table.pdf) · [SVG](final_quality_table.svg) |
| supplementary/adaptive_retention_weight_1x2 | Automatic retention weight: mean and range across active requests | [PNG](supplementary/adaptive_retention_weight_1x2.png) · [PDF](supplementary/adaptive_retention_weight_1x2.pdf) · [SVG](supplementary/adaptive_retention_weight_1x2.svg) |
| supplementary/adaptive_vs_previous_kl_1x2 | Forget-set KL: adaptive versus previous Proposed | [PNG](supplementary/adaptive_vs_previous_kl_1x2.png) · [PDF](supplementary/adaptive_vs_previous_kl_1x2.pdf) · [SVG](supplementary/adaptive_vs_previous_kl_1x2.svg) |
| supplementary/adaptive_vs_previous_mr_1x2 | Retained mR (%): adaptive versus previous Proposed | [PNG](supplementary/adaptive_vs_previous_mr_1x2.png) · [PDF](supplementary/adaptive_vs_previous_mr_1x2.pdf) · [SVG](supplementary/adaptive_vs_previous_mr_1x2.svg) |
| supplementary/round_forget_kl_1x2 | Forget-set KL versus FL rounds | [PNG](supplementary/round_forget_kl_1x2.png) · [PDF](supplementary/round_forget_kl_1x2.pdf) · [SVG](supplementary/round_forget_kl_1x2.svg) |
| supplementary/round_validation_loss_1x2 | Validation loss versus FL rounds | [PNG](supplementary/round_validation_loss_1x2.png) · [PDF](supplementary/round_validation_loss_1x2.pdf) · [SVG](supplementary/round_validation_loss_1x2.svg) |
| supplementary/time_validation_loss_1x2 | Validation loss versus active service time | [PNG](supplementary/time_validation_loss_1x2.png) · [PDF](supplementary/time_validation_loss_1x2.pdf) · [SVG](supplementary/time_validation_loss_1x2.svg) |

## Automatic rule and evaluation

The FU objective is 3 KL_forget(proxy20 || current) + lambda_R CE_retain + 3 KL_retain(request-start teacher || current). At each request step, the rule compares relative retained CE change against the request teacher with relative forget proxy-KL change against the first request step. Their clipped difference is smoothed with EMA 0.5. The previous weight is multiplied by exp(0.5 EMA), bounded to [0.25, 1], with each request initialized at 0.5. If the resulting radio design is infeasible, the rule tries bounded weight floors 0.5 and 1 and stops at the first feasible design; all attempts are logged. No physical constraint is relaxed. Both datasets use exactly the same constants and code.

Training retained CE and proxy KL are surrogates for held-out mR and final retrain KL. Validation mR/KL, the reference200 model and dataset names are never controller inputs. No guarantee that lowering either training surrogate improves its evaluation metric is implied. The clean proxy20 inherits the existing offline assumption that the final forget cohort is known before the mission. This is not a solution for an unknown future deletion cohort. The design follows earlier development results; there is no independent test-set selection claim.

Retained mR averages R@1, R@5 and R@10 in both retrieval directions. Loss is common held-out retained retrieval negative log likelihood after both FL and FU updates. Forget-set KL is KL(reference200 || candidate), averaged over both directions. It starts at the first recorded observation, round 25; no round-0 KL is invented. Early KL also reflects FL progress and future requests. These metrics and the MIA AUC gap are empirical diagnostics, not a deletion or privacy certificate. The Pre-FU table row is shared round20, not a 200-round no-FU control. Retrain is the retained-set round200 reference.

## Costs and comparisons

Total energy sums active FL/FU communication, computing and hovering across ten UAVs. Travel, collection and preflight preparation are excluded. Time is modeled active service time including scheduling waits, not GPU wall time. Extra controller computations, retrieval matrices, teacher forwards and the extra retained backward are charged. Vector repair is conservatively precharged. As for existing methods, resource-optimizer wall time has no calibrated UAV energy model; radio trial counts are disclosed.

Primary charts use post-warm seed0. The supplement reports paired seeds 0, 1, 2 for adaptive and previous Proposed. Only minibatch and radio randomness change; warm20, split, feature cache, proxy20 and reference200 are shared. These are conditional robustness checks, not independent training/split replicates. All methods retain 200 FL rounds and 225 FU steps across 45 requests. The comparison changes both the FU objective and its controller; it does not isolate the causal effect of adaptive weighting from the objective change.

Bandwidth measurements are at 1, 2 and 4 MHz; the tick at 3 is not another measurement. Request-load points are 1, 3, 5, 9 and 15 requests per release, with the same 45 requests and original FL-triggered release schedule. Mean and 95th-percentile request delay run from arrival to final broadcast. S-OF retains its previously authorized soft TX cap with overruns recorded; analog methods keep hard caps. Corrected nonzero head-local broadcasts are used in the affected serial single-request conditions.

The common target remains all 45 requests complete, retained mR >=95% of reference200, mean and worst-request KL <=0.01, confirmed at three distinct consecutive evaluated rounds. See data/common_quality_costs.csv for observed confirmations. This broad acceptance criterion is not a claim that methods have equal quality. Time curves hold their last measured value to the common display horizon without extra training.

## Previous figures and reproducibility

The complete previous final_chart is preserved byte-for-byte in [the archive](../final_chart_archive/20261028_before_adaptive/README.md), including all original radio ablations and the AirScape H-OVA 4 MHz infeasibility. Those radio ablations belong to the previous objective and are not relabeled as the new Proposed. [The previous Git revision](https://github.com/ckim48/ck_ova/tree/3f6316351d7247e1424053329be4e2f8bdc83b31/final_chart) preserves original relative paths.

The frozen protocol, outcome summaries, feedback verification and source mapping are in [results/adaptive_fu_v1_20261028](../results/adaptive_fu_v1_20261028). Numerical plot arrays and hashes are in data/. Raw records, checkpoints, runner log and the verified raw archive remain local in that results directory and are not included in the GitHub publication. The local raw archive omits shared feature caches/checkpoints already referenced by the original experiment; it is not a standalone dataset release.

Run `.venv/bin/python tools/uav_convergence_v2/adaptive_fu.py run`, then `.venv/bin/python tools/uav_convergence_v2/adaptive_final_charts.py build` and `... adaptive_final_charts.py publish`.
