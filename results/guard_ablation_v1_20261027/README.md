# FU-noise-guard ablation and overlapping learning curves

S-AC, AC-OF and S-OF share the same safe-proxy FU objective, sampled data and FL/FU learning-rate schedule. Their communication policies differ. The close round-based loss/mR curves are measured outcomes, not copies of one model: `overlap_audit.json` verifies nonzero differences in all three pairs of final adapter tensors on both datasets. S-AC versus AC-OF differs by about 0.10% (CapERA) and 0.16% (AirScape) of S-AC's adaptation from warm20. Mean recall depends on discrete retrieval ranks, so distinct parameters can give exactly equal mR.

## Registered component removal

**NG-OVA** keeps Proposed's CLIPErase-inspired FU objective, joint FL/FU scheduler, optimized head/resources, data, workload, seeds and physical limits. It removes only the FU-relative error guard and uses the existing joint component-noise budget. This is a component ablation, not a separately published algorithm. No noise coefficient or objective weight was adjusted to force a middle rank.

The ten arms (two datasets x five bandwidth values) were fixed in `protocol.json` before outcomes. All original measurements remain immutable. Numerical candidate screening uses the existing final receiver 1ppm peak-power tolerance; physical constraints remain unchanged. Every new transaction is audited for the unguarded budget, joint scheduling, hard network cap and actual FM/RM/CM loss branch.

New outcomes: {'completed': 10}. Among completed paired conditions, the maximum absolute final-round NG-OVA minus Proposed mR difference is **0.000000 percentage points**, and the maximum final loss difference is **0.00001717**. This measures the component's effect in this scenario; it does not establish that the guard is unnecessary under all channel/error/workload conditions. No intermediate ordering is assumed. At the 1 MHz baseline, the largest mR difference anywhere on the saved round curve is 0.065104 percentage points for CapERA and zero for AirScape; both final mR values are equal.

NG-OVA reduces service energy by up to 0.406185 kJ and service delay by up to 0.240640 s over the tested bandwidth conditions, with no observed final-mR penalty. The guard tightens the mathematical error bound, but this experiment does not demonstrate an empirical retained-utility benefit from that guard. This unfavorable result is retained alongside the communication/scheduling benefits.

## Baseline at 1 MHz

| Dataset | Method | mR (%) | Loss | FL/FU energy (kJ) | FL/FU delay (s) | Forget KL |
|---|---|---:|---:|---:|---:|---:|
| CapERA | Proposed | 16.901042 | 5.36068821 | 204.261065 | 121.092396 | 0.002666711 |
| CapERA | H-OVA | 16.835938 | 5.36071539 | 221.184997 | 131.127596 | 0.002667534 |
| CapERA | S-AC | 16.406250 | 5.44800758 | 255.600774 | 151.528625 | 0.000077445 |
| CapERA | AC-OF | 16.406250 | 5.44800997 | 269.255010 | 159.618225 | 0.000077455 |
| CapERA | S-OF+ | 16.901042 | 5.35958004 | 609.050426 | 361.079596 | 0.002739313 |
| CapERA | S-OF | 16.341146 | 5.44801044 | 609.051089 | 361.079985 | 0.000077322 |
| CapERA | NG-OVA | 16.901042 | 5.36069489 | 204.260734 | 121.092396 | 0.002665813 |
| AirScape | Proposed | 13.671875 | 5.55314684 | 204.254995 | 121.088795 | 0.001754050 |
| AirScape | H-OVA | 13.671875 | 5.55317497 | 216.343766 | 128.256795 | 0.001752560 |
| AirScape | S-AC | 13.541667 | 5.60706615 | 255.594245 | 151.524789 | 0.000303836 |
| AirScape | AC-OF | 13.541667 | 5.60706854 | 268.903100 | 159.409589 | 0.000303935 |
| AirScape | S-OF+ | 13.671875 | 5.55247736 | 608.698929 | 360.871195 | 0.001811430 |
| AirScape | S-OF | 13.541667 | 5.60710001 | 608.699191 | 360.871349 | 0.000305386 |
| AirScape | NG-OVA | 13.671875 | 5.55315304 | 204.254648 | 121.088795 | 0.001753353 |

## Interpretation and figures

The existing main baselines already occupy different energy/delay levels. A method need not lie between the others on round-based accuracy to be a valid communication comparator. Shared-objective ablations test whether communication/head/error-budget changes preserve learning while changing modeled costs. The objective difference between Proposed and safe-proxy baselines prevents attributing all mR gains to parallel transmission. The safe-proxy methods are closer to the retained-set retrain on the baseline forget-set KL, so higher retained mR alone is not overall superiority or a deletion guarantee.

[Main figures](../../selected_graph/README.md) retain Proposed/S-AC/AC-OF/S-OF with their original measured curves. [Ablation figures](../../ablation_graph/README.md) add NG-OVA beside Proposed/H-OVA/S-OF+. The existing AirScape H-OVA 4 MHz infeasibility remains explicit. Curves are neither separated artificially nor selected by rank; 84 prior exported curves retain exactly the same numerical arrays.

Energy covers active FL/FU communication, computing and hovering; delay covers the same service windows. Travel, collection and preflight preparation are excluded. These are single-seed, 200-round, validation-subset results, not a statistical significance claim or a full benchmark reproduction.

All 70 outcomes and quality diagnostics are in `comparison.json`/`.csv`. The ten new raw trajectories are archived in `raw_new_arms.tar.gz` with file hashes in `raw_archive_manifest.json`; shared features and initial/reference checkpoints remain in their existing locations. Extract relative to the repository root. The archive is not a self-contained dataset release.

Run: `.venv/bin/python tools/uav_convergence_v2/guard_ablation.py run`. Rebuild: `.venv/bin/python tools/uav_convergence_v2/guard_ablation_report.py`.
