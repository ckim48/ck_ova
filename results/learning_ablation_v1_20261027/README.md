# Learning-component ablation: Proposed without FU retention RM

All compared systems share the standard FL contrastive-learning update. Several also share the full FU objective; their round-based loss/mR can therefore overlap even when service time differs. FM-CM isolates a learning component instead of modifying the radio guard.

## Fixed comparison

| Method | FU objective | FL update | Transport and head/resource selection |
|---|---|---|---|
| Proposed | 3 FM + 1 RM + 3 CM | Shared standard FL | Joint OVA, FU guard, optimized |
| FM-CM | 3 FM + 0 RM + 3 CM | Identical to Proposed | Identical to Proposed |

FM is the forgetting term, RM the retained-data contrastive term, and CM teacher consistency. The FM/RM/CM construction adapts [CLIPErase, ACL 2025](https://aclanthology.org/2025.acl-long.1469/). FM-CM is a component ablation of that adopted FU objective, not an independent published algorithm or a newly invented FL loss. Setting RM exactly to zero was registered before the ten outcomes; no coefficient search or target ranking was used. All original methods and measurements are retained.

The FL loss/gradient rule, frozen backbone, adapters, sampling, teacher construction, learning rate, initialization, 200 FL rounds, 45 requests x 5 FU steps, seed, physical limits, hard TX cap, scheduler and guard remain fixed. A training-only gradient check on each dataset verifies that full minus partial FU gradient equals an independently calculated RM gradient, and the FL-only update vectors are bit-identical at the same input model. Later model states and first-service teacher snapshots may diverge as a consequence of removing RM. Every completed arm verifies 225 FU records with RM coefficient zero.

RM is still evaluated diagnostically; its objective/gradient contribution is zero. CM still needs retained features. The existing compute model counts adapter MACs and does not separately count contrastive-matrix/softmax arithmetic for either method. Consequently these results do not claim measured hardware compute savings from removing RM.

New outcomes: {'completed': 10}. All five bandwidth values (0.25, 0.5, 1, 2, 4 MHz) and both datasets are retained. The plotted bandwidth range remains 1–4 MHz, with measured points at 1, 2 and 4 MHz. The original AirScape H-OVA 4 MHz infeasibility remains marked.

## Baseline results at 1 MHz

| Dataset | Method | Retained mR (%) | Validation loss | Service energy (kJ) | Service delay (s) | Forget KL | MIA AUC gap |
|---|---|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 16.901042 | 5.36068821 | 204.261065 | 121.092396 | 0.002666711 | 0.00317383 |
| CapERA | H-OVA | 16.835938 | 5.36071539 | 221.184997 | 131.127596 | 0.002667534 | 0.00317383 |
| CapERA | S-AC | 16.406250 | 5.44800758 | 255.600774 | 151.528625 | 0.000077445 | 0.00012207 |
| CapERA | AC-OF | 16.406250 | 5.44800997 | 269.255010 | 159.618225 | 0.000077455 | 0.00012207 |
| CapERA | S-OF+ | 16.901042 | 5.35958004 | 609.050426 | 361.079596 | 0.002739313 | 0.00317383 |
| CapERA | S-OF | 16.341146 | 5.44801044 | 609.051089 | 361.079985 | 0.000077322 | 0.00012207 |
| CapERA | NG-OVA | 16.901042 | 5.36069489 | 204.260734 | 121.092396 | 0.002665813 | 0.00317383 |
| CapERA | FM-CM | 16.367188 | 5.45014954 | 207.926754 | 123.263276 | 0.000212863 | 0.00024414 |
| AirScape | Proposed | 13.671875 | 5.55314684 | 204.254995 | 121.088795 | 0.001754050 | 0.00256348 |
| AirScape | H-OVA | 13.671875 | 5.55317497 | 216.343766 | 128.256795 | 0.001752560 | 0.00256348 |
| AirScape | S-AC | 13.541667 | 5.60706615 | 255.594245 | 151.524789 | 0.000303836 | 0.00012207 |
| AirScape | AC-OF | 13.541667 | 5.60706854 | 268.903100 | 159.409589 | 0.000303935 | 0.00012207 |
| AirScape | S-OF+ | 13.671875 | 5.55247736 | 608.698929 | 360.871195 | 0.001811430 | 0.00262451 |
| AirScape | S-OF | 13.541667 | 5.60710001 | 608.699191 | 360.871349 | 0.000305386 | 0.00012207 |
| AirScape | NG-OVA | 13.671875 | 5.55315304 | 204.254648 | 121.088795 | 0.001753353 | 0.00256348 |
| AirScape | FM-CM | 13.476562 | 5.60160732 | 206.122407 | 122.194715 | 0.000274715 | 0.00018311 |

Removing RM produces the following partial-minus-full changes:

- CapERA: retained mR -0.533854 percentage points; validation loss +0.08946133; forget KL -0.002453848.
- AirScape: retained mR -0.195312 percentage points; validation loss +0.04846048; forget KL -0.001479335.

## Interpretation

The matched RM-removal experiment measures the effect of the adopted retained-data FU gradient. The learning improvement cannot be attributed to parallel transmission: FM-CM uses the same parallel communication. It also need not fall between Proposed and the safe-proxy baselines. An intermediate ordering is not a validity criterion, and no curves are adjusted to create one.

At the measured baseline, full RM improves retained utility, while FM-CM is closer to the retained-set retrain on forget-set KL. Higher retained mR alone therefore does not establish overall unlearning superiority. Forget KL, worst-request KL and MIA gaps are included in the full comparison. The existing safe-proxy baselines likewise retain their forgetting diagnostics.

[Selected figures](../../selected_graph/README.md) now include Proposed, FM-CM, S-AC, AC-OF and S-OF. [Radio ablations](../../ablation_graph/README.md) retain Proposed, H-OVA, S-OF+ and NG-OVA. All 96 previously exported numerical curves and display arrays are unchanged. Energy/delay use active FL/FU service; energy includes communication, computing and hovering, excluding travel, collection and preflight. Loss is held-out validation retrieval NLL, not minibatch training loss. Time uses modeled UAV service time, not experiment wall time.

These are single-seed, 200-round validation-subset results, not a significance test or full benchmark reproduction. All 80 method/condition outcomes are in comparison.json/CSV. The ten new raw arms are in raw_new_arms.tar.gz with file hashes in raw_archive_manifest.json. The two training-only gradient checks are also published under cases/baseline. Shared data features and initial/reference checkpoints stay in their existing locations; this is not a self-contained dataset release. Extract the archive relative to the repository root.

Run: `.venv/bin/python tools/uav_convergence_v2/learning_ablation.py run`. Rebuild: `.venv/bin/python tools/uav_convergence_v2/learning_ablation_report.py`.
