# Current UAV/VLM comparison: Proposed = P-KS

<!-- tgcn-additions -->
[Energy–mR, paired FU-workload bars and retrieval outcomes with examples](downlink_bandwidth/TGCN_ADDITIONS.md). Main figures10–11 and17; earlier example sheets12–16 remain linked.
<!-- /tgcn-additions -->













<!-- downlink-bandwidth-current -->
[Downlink bandwidth figures, paired resource bars, smaller map and comparison tables](downlink_bandwidth/index.html). Explicit fixed spectral efficiency; 1–5 MHz.

**Parallel advantage:** [Five figures and a matched immediate-serial P-KS experiment](parallel_advantage/README.md).

**Current charts:** [P-KS as Proposed, round/time learning curves and three resource sensitivities](proposed_pks/README.md) · [Gallery](proposed_pks/index.html). The old adaptive Proposed is excluded from this current comparison. P-KB25 remains a separate candidate; its results are not relabeled as P-KS.

Historical bundles retain their original identities: [KL-reference audit and bounded-learning candidates](pks_kl_repair/README.md), [earlier four-method comparison](four_methods/index.html), [learning diagnostics](learning_quality/index.html), and [earlier shared-time continuation](same_time/index.html).

## Historical benchmark notes

The descriptions and results below refer to the earlier adaptive Proposed, not the newly selected P-KS.

**Expanded sweeps:** [23 additional 1×2 figures](expanded_sweeps/index.html) cover bandwidth, downlink rate, request grouping, FL-round budget and packet loss, with separate active/mission delay and quality checks.

CapERA (left), AirScape (right). Main curves: Proposed, ISCC-P2, AFL-OTA. These are explicitly scoped transfers into the common10-UAV scenario, not complete reproductions of the cited papers.

| Method | Published source | Preserved core | Explicit adaptation / omitted part |
|---|---|---|---|
| ISCC-P2 | [Tang et al., IEEE TWC24(4),2025](https://doi.org/10.1109/TWC.2024.3523381); [official code](https://github.com/TheaSherlock/ISCC-UAV) | Equal-client FedSGD, FP32 FDMA and exact fixed-position/batch P2 bandwidth allocation | Cached VLM data; sensing success1; fixed nominal UAV head; no BBPO batch/position/round optimization |
| AFL-OTA | [Zhai et al., IEEE TWC24(3),2025](https://doi.org/10.1109/TWC.2024.3520501); [author manuscript](https://arxiv.org/abs/2403.06653) | Alg.1 cached gradients, ready-device updates, normalized OTA aggregation and overlapping local computation | UAV clients plus fixed nominal head; all-ready selection, channel inversion/repetition; no joint Alg.2 optimizer |
| Proposed | Current repository formulation | Joint FL/FU OVA, optimized head/resources and adaptive retrieval-FU controller | Planned-cohort clean proxy20 available; the external baselines do not use this teacher |

The two cited papers are FL-only. Both receive the same **GA→FT FU extension**: cosine forgetting then retained contrastive SGD, based on the [SLUG ICML2025 official comparison code](https://github.com/CSIPlab/SLUG/blob/main/src/clip/unlearn/GAFT.py). This is a federated adapter adaptation, not an original UAV-FU method from either paper. Learning objectives and radio protocols both differ; the main comparison measures complete adapted systems. It does not isolate a communication effect.

The previous FM-CM/S-AC/AC-OF/S-OF curves remain available in [the ablation bundle](ablation/index.html), with their original names and data. They are not relabeled as literature methods.

**Scope and limitations.** Shared frozen RemoteCLIP B32 mean8 features, rank10 adapters, warm20, same partitions and45 requests. The final round is200 (20 common warmup +180 mission FL updates). FU-cycle sweeps apply the declared cycle count to every method. Single development seed and subset diagnostics, not official dataset scores or statistical superiority. Cached features mean active compute cost is adapter work, not re-encoding/full-VLM training. Methods use different FU training signals, and Proposed has an additional offline known-cohort proxy assumption. Equal-device versus sample-weighted FL aggregation follows the respective method.

Energy is active FL/FU communication + executed computation + hovering. Delay is active service time; travel, collection and preflight are excluded from these figures. All-peer circuit costs are retained even when only selected receivers consume a broadcast. OTA mean/variance metadata is modeled as error-free with zero overhead, following the source convention. Resource-optimizer wall time is not calibrated to UAV energy.

The homogeneous cached-adapter scenario may make all devices ready together: an asynchronous algorithm can then operate synchronously. We do not manufacture staleness or drop clients to separate curves. Actual maximum staleness is in the data table. CPU/batch effects may be small because communication/hover dominates; plotted slopes are not amplified or invented.

Every point is a fresh simulation or the identical pre-existing Proposed arm; no time/energy rescaling of another learning trajectory. Power bounds stay hard. Paper transfers complete their payloads and record network-energy/update-energy/deadline QoS exceedances. `data/metrics.csv` reports these; a completed QoS-violating point must not be described as feasible under all original Proposed constraints. The full digital upload is32bit, not the old Q6 ablation.

Both axes include endpoint ticks at uniform numerical spacing. Panel precision matches. No point values, extra grey captions, smoothing, or fabricated confidence intervals. Time plots hold the terminal model after completion; the appended segment is not additional training. Repeated FL rounds show intervening FU evaluations.

| Dataset | Method | mR (%) | Forget KL | Energy (kJ) | Active delay (s) |
|---|---|---:|---:|---:|---:|
| CapERA | Proposed | 16.328 | 0.000224149 | 204.451 | 121.204 |
| CapERA | ISCC-P2 | 16.901 | 0.00378861 | 649.962 | 384.304 |
| CapERA | AFL-OTA | 16.966 | 0.00379123 | 252.882 | 149.912 |
| AirScape | Proposed | 13.542 | 0.000234091 | 204.751 | 121.383 |
| AirScape | ISCC-P2 | 13.737 | 0.000838523 | 649.305 | 383.915 |
| AirScape | AFL-OTA | 13.737 | 0.000838547 | 253.066 | 150.021 |

Status: complete.

Five asynchronous large-batch arms exposed a terminal timing assertion: already-started local jobs outlasted the final zero-LR update by0.128–0.24576ms. Their failed records remain local and unchanged. The source-frozen `paper_uav_drain_fix.py` reruns include this final compute wait/hover cost. Every transaction, job and work ledger is byte-identical to the original attempt; final learned metrics are identical. See `data/terminal_drain_correction.json`.

Reproduce the complete workflow: `python tools/uav_convergence_v2/paper_uav_release.py run`. It completes the preserved original attempts, accepts only the documented terminal-drain rejection, runs the correction, audits and exports. Rebuild/audit existing results with `paper_uav_release.py finish`. Requires the existing local datasets/caches. Raw samples, checkpoints, transaction logs and local archives are not published. Source hashes and plot arrays are included.
