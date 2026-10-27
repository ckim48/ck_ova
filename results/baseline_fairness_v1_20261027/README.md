# Baseline definitions and matched-objective ablations

The original H-OVA and S-OF+ were intentionally close controls, not independent learning baselines. Sharing the objective/model is appropriate for a communication or head-selection ablation. The new S-OF adds an existing proxy-objective control without removing or weakening those matched controls.

| Method | FU objective | FL/FU transport | Head/resources | Role |
|---|---|---|---|---|
| Proposed | CLIPErase-inspired FM/RM/CM | Joint analog OVA | Optimized | Main and ablation |
| H-OVA | Same as Proposed | Joint analog OVA | Centroid head, uniform inverse noise scale, first feasible frame | Head/resource ablation |
| S-OF+ | Same as Proposed | Serial digital OFDMA, Q6 | Optimized head and integer tones | Transport/scheduler ablation |
| S-AC | Existing safe-proxy FU | Serial AirComp | Existing optimizer | Main baseline |
| AC-OF | Existing safe-proxy FU | AirComp FL, periodic digital OFDMA FU | Existing optimizer | Main baseline |
| S-OF (new) | Existing safe-proxy FU, identical to S-AC/AC-OF | Serial digital OFDMA, Q6 | Identical to S-OF+ | Main baseline |

The FM/RM/CM objective adapts [CLIPErase (ACL 2025)](https://aclanthology.org/2025.acl-long.1469/); it is not claimed as an original objective of this communication method. In the current implementation, the CLIPErase vector function does not use its `crossmodal` argument. Setting that flag to false would not remove CM. The new control explicitly selects the existing safe-proxy vector function instead, with no new coefficients.

Common IoT placement/routes, model initialization and evaluation galleries are controlled scenario inputs. They are deliberately shared rather than removed from a comparator. These are project-defined comparison configurations, not claims of faithful implementations of separately named published baseline algorithms.

## Fixed protocol

Before the new outcomes, `protocol.json` registered both datasets, seed 0 and all five bandwidth values (0.25, 0.5, 1, 2, 4 MHz). All ten new arms ran 200 FL rounds and 45 FU requests with five steps each. Only the FU objective changes relative to S-OF+. The model/features, data/splits, warm20/proxy20/reference200, FL/FU learning-rate schedules, requests, RNG seeds, placement, channels, digital optimization and energy model are unchanged. The new arm uses a distinct method ID and result directory. All original results remain immutable; 72 exported original curves were checked against the previous published plotting arrays.

S-OF and S-OF+ retain the previously authorized soft network-TX-cap fallback: cap-feasible designs are preferred, but an overrun is permitted and logged when needed, subject to unchanged per-user peak power/energy, deadline, coherence and quantization-error bounds. Analog arms retain the hard network cap. Thus network-cap policies differ and should be disclosed when claiming fairness. All new transaction schedules are serial and the recorded FU local statistics confirm that the safe-proxy branch executed.

Ten new arms completed; the combined audit retains all 60 outcomes across six methods. The existing AirScape H-OVA 4 MHz radio infeasibility remains visible in the ablation figures.

## Baseline results at 1 MHz

Energy and delay below cover the active FL/FU service windows, including scheduling waits. Energy includes communication, computing and hovering; travel, collection and preflight preparation are excluded.

| Dataset | Method | mR (%) | Loss | Energy (kJ) | Delay (s) | Forget KL to retrain | MIA AUC gap to retrain |
|---|---|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 16.9010 | 5.360688 | 204.261 | 121.092 | 0.00266671 | 0.00317383 |
| CapERA | H-OVA | 16.8359 | 5.360715 | 221.185 | 131.128 | 0.00266753 | 0.00317383 |
| CapERA | S-AC | 16.4063 | 5.448008 | 255.601 | 151.529 | 0.00007745 | 0.00012207 |
| CapERA | AC-OF | 16.4063 | 5.448010 | 269.255 | 159.618 | 0.00007746 | 0.00012207 |
| CapERA | S-OF+ | 16.9010 | 5.359580 | 609.050 | 361.080 | 0.00273931 | 0.00317383 |
| CapERA | S-OF | 16.3411 | 5.448010 | 609.051 | 361.080 | 0.00007732 | 0.00012207 |
| AirScape | Proposed | 13.6719 | 5.553147 | 204.255 | 121.089 | 0.00175405 | 0.00256348 |
| AirScape | H-OVA | 13.6719 | 5.553175 | 216.344 | 128.257 | 0.00175256 | 0.00256348 |
| AirScape | S-AC | 13.5417 | 5.607066 | 255.594 | 151.525 | 0.00030384 | 0.00012207 |
| AirScape | AC-OF | 13.5417 | 5.607069 | 268.903 | 159.410 | 0.00030393 | 0.00012207 |
| AirScape | S-OF+ | 13.6719 | 5.552477 | 608.699 | 360.871 | 0.00181143 | 0.00262451 |
| AirScape | S-OF | 13.5417 | 5.607100 | 608.699 | 360.871 | 0.00030539 | 0.00012207 |

## Interpretation

Changing the S-OF FU objective reduces retained mR slightly at the baseline, but leaves its modeled energy and delay essentially unchanged. The main benefit of joint scheduling is therefore already visible against the matched-objective S-OF+ control; a different loss is not needed to demonstrate that timing effect.

Proposed has higher retained mR at the baseline, while safe-proxy S-OF is closer to the retained-set retrain on forget-set KL and the fixed MIA AUC gap. This is a utility/forgetting tradeoff, not uniform superiority. KL proximity is a diagnostic, and the fixed retrieval-loss membership attack is descriptive rather than a certified deletion/privacy guarantee. The main comparison varies the objective and cannot isolate the communication contribution; use H-OVA and S-OF+ for matched-objective claims. Single-seed subset results do not establish statistical significance or full benchmark performance.

| Dataset | Comparator | Proposed energy reduction (%) | Proposed delay reduction (%) | mR gap (percentage points) |
|---|---|---:|---:|---:|
| CapERA | H-OVA | 7.651 | 7.653 | 0.0651 |
| CapERA | S-AC | 20.086 | 20.086 | 0.4948 |
| CapERA | AC-OF | 24.138 | 24.136 | 0.4948 |
| CapERA | S-OF+ | 66.462 | 66.464 | 0.0000 |
| CapERA | S-OF | 66.462 | 66.464 | 0.5599 |
| AirScape | H-OVA | 5.588 | 5.589 | 0.0000 |
| AirScape | S-AC | 20.086 | 20.086 | 0.1302 |
| AirScape | AC-OF | 24.041 | 24.039 | 0.1302 |
| AirScape | S-OF+ | 66.444 | 66.445 | 0.0000 |
| AirScape | S-OF | 66.444 | 66.445 | 0.1302 |

See [selected_graph](../../selected_graph/README.md) for the four-method main figures and [ablation_graph](../../ablation_graph/README.md) for the matched-objective controls. `comparison.json` and `.csv` contain every bandwidth outcome and quality diagnostic, not only the baseline.

Reproduce the registered runs with `.venv/bin/python tools/uav_convergence_v2/baseline_fairness.py run`; completed outcomes are preserved. Rebuild figures/audit with `.venv/bin/python tools/uav_convergence_v2/baseline_fairness_report.py`.

`raw_new_arms.tar.gz` contains the new arms' recorded transactions, work attempts, evaluations, checkpoints and scenario inputs, using paths relative to the repository root. `raw_archive_manifest.json` lists hashes. Shared feature caches, initial/reference checkpoints and reused original arms remain in their original locations; they are not duplicated in this archive. The archive alone is not a self-contained dataset release.
