# Energy efficiency and actual video–text retrieval

[All figures](index.html) · [FU workload metrics](data/fu_workload_metrics.csv) · [Retrieval metrics](data/retrieval_summary.csv)

Current Proposed is unchanged P-KS. All other algorithms, models and data definitions retain their published identities. No model is selected or retrained to manufacture illustrative differences.

## 10: Retained mR versus cumulative energy

Uses the existing actual completed-model trajectories, with communication, executed adapter computation and active hovering independently reconstructed from transaction ledgers. All methods receive the same650kJ active-work budget. Zero is the common warm20 checkpoint; preflight feature extraction, warmup, travel and IoT collection are excluded consistently with the existing resource figures. No energy is inferred merely from elapsed time.

All45 deletion procedures finish by650kJ for every method. Before completion, points reflect concurrent FL/FU progress; they do not imply that all requests have already finished. After the original200FL/225FU workload, every method continues retained-only FL with the same restarted0.0035 learning rate. No new deletion requests arrive. Curves show the last completed evaluated model using steps-post, with markers at the same energy budgets. No interpolation or unexecuted model updates. All recorded declines are retained.

Energy-to-target CSV requires all45 FU procedures finished; missing targets remain unreached. Procedure completion and retained mR do not certify forgetting quality. Existing P-KS common-retrain KL limitations remain; this figure is an energy/retained-utility comparison, not an equal-certified-unlearning claim.

## 11: Total energy versus FU steps per request

The main figure uses two adjacent bars per algorithm: 1 versus 9 outer FU steps per request, the endpoints of the completed sweep. Method colors match the other resource plots; hatching identifies workload. Both panels use the same zero baseline, with no values above bars. All five measured workloads (1, 3, 5, 7, 9) remain in the full metrics files.

Forty fresh executions:1/3/5/7/9 outer FU steps per request × four methods × two datasets. Same45 requests, sample cohorts, release ordinals,10UAV, five regions,200 totalFL rounds, initial models, features, learning/radio/energy coefficients. Every prescribed FU step finishes. Eight fresh five-step controls exactly reproduce original final parameter tensors, quality, transaction clocks and radio decisions.

GA→FT comparators retain two local optimizer substeps per outer FU cycle; both are charged. More steps do not necessarily improve mR or KL, so all quality values and physical-bound exceedances accompany the energy figure. Flat Proposed curves are retained: co-aggregation may absorb additional FU work within the unchanged180 mission-FL transmissions. Parallel refers to joint FL/FU aggregation/application, not simultaneous local CPU execution.

## 12–13: Earlier fixed-round retrieval diagnostics

[CapERA, 200 rounds](12_capera_text_video_examples.pdf) · [AirScape, 200 rounds](13_airscape_text_video_examples.pdf). These earlier diagnostic figures remain accessible as supporting material; the main gallery uses the combined selected examples in16.

Evaluate the unchanged final200 checkpoints with all45 requests/225FU cycles complete, at the same workload rather than the same energy. Native captions, original gallery and cached eight-frame RemoteCLIP features are unchanged. All six recalls and validation loss are independently reproduced before selecting illustrations. A displayed image is the fourth of the same eight original sampled video frames, downsampled only to fit the figure. It is not a generated image and the task is video retrieval, not caption generation.

Each dataset shows one selected Proposed rank advantage and one disadvantage. Prefer a top1 reversal among unique native caption strings; choose the median rank gap with caption-ID tie breaking. Full selection rules and counts are in the protocol. These two deliberately contrasting cases are not a random or representative sample. Complete retained-query rank/top3-ID tables and better/equal/worse counts prevent a favorable-only interpretation.

Ground truth is the annotated paired video, not a judgment that every other semantically similar video is irrelevant. Green borders indicate the correct top1 video, red an incorrect top1; GT rank gives the annotated video position in the full gallery. Full native query text and media/frame hashes accompany selected cases. Long displayed queries may be explicitly abbreviated, but inference always used the original complete caption. Only derived representative-frame figures are published; raw videos/features/checkpoints stay local.

## 14–15: Selected successful retrievals under the same 650 kJ budget

These use the exact last completed model within 650 kJ from the energy/mR curve. All 45 FU procedures complete. The original continuation was replayed unchanged; every evaluation/energy/time point and final parameter tensor reproduced exactly before intermediate checkpoints were used. Different completed FL counts reflect actual energy costs.

Display revision v2 shows six different ground-truth videos per dataset in a compact six-row grid with one shared header. Every shown Proposed result has ground-truth rank 1 and improves over at least one comparator; comparator rank-1 ties are preserved. The examples were selected after evaluation for favorable outcomes and scene diversity. They illustrate successful cases and do not estimate average success, statistical superiority, or dominance over every comparator. All complete query-rank tables, paired better/equal/worse counts, checkpoints and aggregate metrics are unchanged.

CapERA scenes: baseball, urban fire, harvesting, a crowd, swimming and a bridge. AirScape scenes: elephants, a snowy parking area, pedestrians, horses near water, a lighthouse and a road junction. Original query excerpts are shown with explicit ellipses where the repeated camera-perspective sentence or a long ending is omitted; inference used the full native caption. No generated captions or altered image content. Full captions, selection IDs, candidate counts and frame provenance accompany the figures.

[Earlier CapERA success-only sheet](14_capera_energy_matched_retrieval.pdf) · [Earlier AirScape success-only sheet](15_airscape_energy_matched_retrieval.pdf). These are retained for reference; figure16 is the current combined main illustration.

## 16: Combined examples from both datasets

One sheet contains four CapERA and four AirScape cases, with three Proposed top1 successes and one miss in each dataset. Successes cover urban fire, harvesting, swimming, elephants, horses near water and a road junction. The misses (flood and pagoda) reuse the original median-rank-gap disadvantage examples selected by the frozen inference protocol. This is a deliberate illustrative 3:1 mixture, not an observed success rate. Same 650 kJ checkpoints and complete native-query inference as14–15; no retraining or score changes.

Every method column shows its actual top1 retrieved video. The numeric GT rank is the rank of the annotated paired video in the entire gallery, not the rank of the displayed top1 image. Green means the paired video is top1, red means it is not. Similar-looking unpaired videos may still be semantically plausible. Each image is one original frame from a video scored using eight frames, so the single displayed frame cannot establish temporal understanding.

Interpretation: this figure illustrates retained text-to-video retrieval behavior and typical-looking confusions under a shared energy budget. The full-query mR-versus-energy curve and complete rank tables establish aggregate performance. These selected retained-data cases cannot establish forgetting quality, explain gradient causality or by themselves prove AirComp superiority. A paper caption should state the selection and budget explicitly.

Single development seed and frozen-feature rank10 adapter study, not a full-backbone VLM hardware benchmark or statistical superiority result.

## Reproduction

```sh
.venv/bin/python tools/uav_convergence_v2/tgcn_fu_workload_v1.py run
.venv/bin/python tools/uav_convergence_v2/tgcn_retrieval_examples_v1.py run
.venv/bin/python tools/uav_convergence_v2/tgcn_energy_checkpoint_v1.py
.venv/bin/python tools/uav_convergence_v2/tgcn_energy_retrieval_v1.py
.venv/bin/python tools/uav_convergence_v2/tgcn_additions_report.py
.venv/bin/python tools/uav_convergence_v2/tgcn_additions_check.py
```
