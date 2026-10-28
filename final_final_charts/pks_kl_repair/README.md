# P-KS: reference audit and actual KL-control experiments

**Two different findings must remain separate.** The original brown P-KS is our KL-gate + priority-FL candidate. Its earlier large KL was measured against a uniformly trained common reference. Repeating retained-only training with the candidate’s own FL sampler substantially reduces that measured discrepancy without changing the candidate. This is an evaluation correction/diagnosis, not a newly improved trained model. Both reference definitions remain published for every method.

Separately, the P-KB25/P-KB50/P-AB50 arms are real new training runs. They bound the additional priority gradient relative to the same-step uniform retained gradient. All predeclared outcomes are included, even when mR, loss or worst-request KL is worse.

## Methods

| Label | Change |
|---|---|
| Proposed | Original adaptive-FU, uniform-FL method, unchanged |
| P-KS | Original proxy-KL gate and half-exploration/half-priority FL sampler, unchanged |
| P-KB25 / P-KB50 | P-KS FU; limit the applied gradient’s deviation from the uniform retained gradient to 25% / 50% of the uniform-gradient norm |
| P-AB50 | Same 50% bound, with the original adaptive FU objective instead of the gate |
| P-KU / P-LU | Reuse P-KS / P-LS at 200 rounds; switch to uniform retained sampling only during subsequent FL |
| ISCC-P2 / AFL-OTA / F-SalUn | Existing published paper-core transfers, unchanged |

The gradient rule is `g = g_U + beta*(g_H-g_U)`, where `beta=min(1,rho*||g_U||/||g_H-g_U||)`. It adapts using current training gradients only. Both passes and selection/blend arithmetic are charged. This is a local step constraint, not a theorem bounding final KL. The same constants apply to both datasets. The idea is our adaptation; [SLUG, ICML 2025](https://proceedings.mlr.press/v267/cai25f.html) motivates attention to gradient alignment, and [CLIPErase, ACL 2025](https://aclanthology.org/2025.acl-long.1469/) motivates separate forgetting/retention/consistency terms. We do not claim either paper’s exact algorithm or guarantees.

## What the two KL columns mean

- **Common KL:** the original reference at 200 rounds, then the same uniform continuation to the candidate’s final FL count. This preserves the previous comparison protocol. Float accumulation differences in the independently recomputed continuation are checked within 1e-6; previous published values are also retained in the CSV.
- **Policy-matched KL:** exact retained-only training from theta0, the shared retained 20-round warmup, then each learner’s sampler, FL weighting and LR rule. Uniform sample-weighted for Proposed, priority for P-KS, equal-client for all three literature transfers, and the respective bounded or phase-changing rule for new candidates. All deleted samples are excluded from the start. No FU is performed by the reference because those samples never entered it.
- Every method receives its own matched reference under the same construction rule. No reference parameter or prediction enters candidate training. The original reference at 200 rounds is reproduced bit-for-bit in the independent audit.
- Neither KL is a deletion certificate. The 0.01 threshold is the existing empirical criterion, and both mean and worst-request results are reported. A small matched KL does not establish indistinguishability or justify hiding a large common KL.

## Clocks and figures

All methods complete 45 requests/225 FU cycles and 200 FL rounds, then continue retained-only FL under the previously declared constant 0.0035 LR until the first completed update beyond 400 active seconds. This is a fixed deletion workload with continuation, not a continuous-request capacity experiment.

Figures 01–06 show the last observed completed model at or before 400 seconds. Before 200 rounds evaluations are sparse; afterward every update is evaluated. The exact 400-second budget table is in `data/budget400.csv`. Figures 07–12 use the final saved checkpoint just beyond 400 seconds; actual times and FL counts are explicit. Figures 13–14 compare P-KS and P-KB25 at the last completed checkpoint within 400 seconds, with all corresponding metrics in `data/strict400.csv`. No interpolation of model quality, artificial slope, or hidden unsuccessful arm.

Total energy is actual active communication + adapter computation + hover. Frozen features and optimizer-host exclusions remain as previously declared. Extra evaluation/retraining oracles are reporting work, excluded from every candidate’s service cost. Same-time energy is not a performance metric by itself when all UAVs hover for a similar duration.

## Exact 400-second budget: last completed checkpoint

P-KS was replayed without changing its algorithm to recover the previously unsaved pre-400 checkpoint. Every original observed quality value, modeled time, energy and the final parameter tensor reproduce exactly. The new common-KL and matched-KL measurements below use the same checkpoint as the mR value. Both methods finish 612 FL rounds and 225 FU cycles within 400 seconds. The terminal table below instead uses the next completed checkpoint (613 rounds), so its small mR differences must not be mixed with these values.

| Dataset | Method | mR (%) | Loss | Common KL | Common worst KL | Matched KL | Checkpoint time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| CapERA | P-KS | 24.284 | 4.27001 | 0.0825719 | 0.122596 | 0.00102927 | 399.793 |
| CapERA | P-KB25 | 24.219 | 4.34733 | 0.00471595 | 0.00675115 | 0.000701956 | 399.849 |
| AirScape | P-KS | 16.797 | 4.99559 | 0.0714382 | 0.0963315 | 0.00052054 | 399.787 |
| AirScape | P-KB25 | 16.602 | 5.01936 | 0.00358609 | 0.00495466 | 0.00032422 | 399.844 |

At the same 400-second budget, P-KB25 reduces common KL by 94.29% on CapERA and 94.98% on AirScape, while mR falls by 0.065 and 0.195 percentage points, respectively. Its mean and worst-request common KL are below 0.01 on both datasets. Loss increases by 0.0773 and 0.0238; the additional computation costs about 0.057 seconds and 0.098 kJ at the same completed workload. These are tradeoffs, not an improvement in every metric.

## Final checkpoint results (approximately 400 active seconds)

| Dataset | Method | mR (%) | Loss | Common KL | Common worst KL | Matched KL | Matched worst KL | Actual time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 24.076 | 4.38338 | 0.0001398 | 0.000245678 | 0.000132806 | 0.000226685 | 400.571 |
| CapERA | P-KS | 24.167 | 4.26984 | 0.0825096 | 0.12267 | 0.00103387 | 0.00154546 | 400.469 |
| CapERA | ISCC-P2 | 17.305 | 5.30772 | 0.00445988 | 0.00699749 | 0.00382019 | 0.00571916 | 400.628 |
| CapERA | AFL-OTA | 23.880 | 4.38797 | 0.0135082 | 0.0510025 | 0.00132022 | 0.00196207 | 400.256 |
| CapERA | F-SalUn | 17.721 | 5.24258 | 0.0112352 | 0.0190511 | 0.0102683 | 0.0169891 | 400.626 |
| CapERA | P-KB25 | 24.258 | 4.34638 | 0.00470489 | 0.00674197 | 0.000701077 | 0.00105961 | 400.526 |
| CapERA | P-KB50 | 23.854 | 4.32623 | 0.0122696 | 0.0183623 | 0.000713285 | 0.00104261 | 400.526 |
| CapERA | P-AB50 | 24.128 | 4.33577 | 0.00971672 | 0.0150167 | 0.000160241 | 0.000235676 | 400.587 |
| CapERA | P-KU | 24.049 | 4.34521 | 0.00420703 | 0.00653481 | 0.000857786 | 0.00141839 | 400.469 |
| CapERA | P-LU | 23.945 | 4.35819 | 0.00241844 | 0.00403297 | 0.000226801 | 0.000518961 | 400.141 |
| AirScape | Proposed | 16.667 | 5.04403 | 5.13143e-05 | 6.94105e-05 | 4.35733e-05 | 5.77447e-05 | 400.073 |
| AirScape | P-KS | 16.732 | 4.99491 | 0.0710471 | 0.0960005 | 0.000510912 | 0.000922761 | 400.463 |
| AirScape | ISCC-P2 | 13.802 | 5.54179 | 0.000927828 | 0.00158563 | 0.00141108 | 0.00185954 | 400.239 |
| AirScape | AFL-OTA | 16.536 | 5.05060 | 0.00380704 | 0.00625693 | 0.000468043 | 0.000700676 | 400.170 |
| AirScape | F-SalUn | 13.672 | 5.51134 | 0.00345502 | 0.00544515 | 0.00443152 | 0.00569334 | 400.240 |
| AirScape | P-KB25 | 16.602 | 5.01907 | 0.00359439 | 0.00495054 | 0.000324775 | 0.000456653 | 400.520 |
| AirScape | P-KB50 | 17.122 | 5.00818 | 0.0112535 | 0.0149515 | 0.000333612 | 0.000477278 | 400.520 |
| AirScape | P-AB50 | 16.927 | 5.01576 | 0.0096351 | 0.01233 | 5.66013e-05 | 8.24862e-05 | 400.048 |
| AirScape | P-KU | 16.602 | 5.01440 | 0.00284313 | 0.00518553 | 0.000427434 | 0.00068053 | 400.463 |
| AirScape | P-LU | 16.667 | 5.02348 | 0.00175048 | 0.00329598 | 0.000136312 | 0.000210357 | 400.237 |

The CSV also contains every 200-round endpoint, including the early utility loss of the bounded variants. This is one developmental seed and the existing subset/validation split; no statistical superiority or official full-benchmark claim. A preferred candidate requires paired new-seed and independent-split confirmation.

## Reproduction

With the existing local dataset/cache inputs, run `pks_reference_audit.py --dataset CapERA` and repeat for AirScape; `pks_balance_campaign.py run`; `pks_evaluation_v1.py --dataset CapERA` and repeat for AirScape. Recover the exact 400-second P-KS checkpoints with `pks_balance_v1.py continue --variant control --dataset CapERA` and repeat for AirScape; then run `pks_budget400_fix.py`, `pks_report.py` and `pks_release_check.py`. The earlier budget audit stopped because its assertion included reference-dependent retained KL/disagreement in an equality check; the correction compares the same model’s mR, loss and six recalls exactly while allowing its reference-dependent diagnostics to change. No model or reference training rule changes, and the stopped attempt remains preserved. All scripts are under `tools/uav_convergence_v2`. Previously published input arms are reused exactly. No raw data, learned parameters, or work/transaction traces are published.
