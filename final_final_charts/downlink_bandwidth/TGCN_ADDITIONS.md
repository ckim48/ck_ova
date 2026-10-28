# Energy efficiency and actual video–text retrieval

[All figures](index.html) · [FU workload metrics](data/fu_workload_metrics.csv) · [Retrieval metrics](data/retrieval_summary.csv)

Current Proposed is unchanged P-KS. All other algorithms, models and data definitions retain their published identities. No model is selected or retrained to manufacture illustrative differences.

## 10: Retained mR versus cumulative energy

Uses the existing actual completed-model trajectories, with communication, executed adapter computation and active hovering independently reconstructed from transaction ledgers. All methods receive the same650kJ active-work budget. Zero is the common warm20 checkpoint; preflight feature extraction, warmup, travel and IoT collection are excluded consistently with the existing resource figures. No energy is inferred merely from elapsed time.

All45 deletion procedures finish by650kJ for every method. Before completion, points reflect concurrent FL/FU progress; they do not imply that all requests have already finished. After the original200FL/225FU workload, every method continues retained-only FL with the same restarted0.0035 learning rate. No new deletion requests arrive. Curves show the last completed evaluated model using steps-post, with markers at the same energy budgets. No interpolation or unexecuted model updates. All recorded declines are retained.

Energy-to-target CSV requires all45 FU procedures finished; missing targets remain unreached. Procedure completion and retained mR do not certify forgetting quality. Existing P-KS common-retrain KL limitations remain; this figure is an energy/retained-utility comparison, not an equal-certified-unlearning claim.

## 11: Total energy versus FU steps per request

Forty fresh executions:1/3/5/7/9 outer FU steps per request × four methods × two datasets. Same45 requests, sample cohorts, release ordinals,10UAV, five regions,200 totalFL rounds, initial models, features, learning/radio/energy coefficients. Every prescribed FU step finishes. Eight fresh five-step controls exactly reproduce original final parameter tensors, quality, transaction clocks and radio decisions.

GA→FT comparators retain two local optimizer substeps per outer FU cycle; both are charged. More steps do not necessarily improve mR or KL, so all quality values and physical-bound exceedances accompany the energy figure. Flat Proposed curves are retained: co-aggregation may absorb additional FU work within the unchanged180 mission-FL transmissions. Parallel refers to joint FL/FU aggregation/application, not simultaneous local CPU execution.

## 12–13: Native text-to-video retrieval examples

Evaluate the unchanged final200 checkpoints with all45 requests/225FU cycles complete, at the same workload rather than the same energy. Native captions, original gallery and cached eight-frame RemoteCLIP features are unchanged. All six recalls and validation loss are independently reproduced before selecting illustrations. A displayed image is the fourth of the same eight original sampled video frames, downsampled only to fit the figure. It is not a generated image and the task is video retrieval, not caption generation.

Each dataset shows one selected Proposed rank advantage and one disadvantage. Prefer a top1 reversal among unique native caption strings; choose the median rank gap with caption-ID tie breaking. Full selection rules and counts are in the protocol. These two deliberately contrasting cases are not a random or representative sample. Complete retained-query rank/top3-ID tables and better/equal/worse counts prevent a favorable-only interpretation.

Ground truth is the annotated paired video, not a judgment that every other semantically similar video is irrelevant. Green borders indicate the correct top1 video, red an incorrect top1; GT rank gives the annotated video position in the full gallery. Full native query text and media/frame hashes accompany selected cases. Long displayed queries may be explicitly abbreviated, but inference always used the original complete caption. Only derived representative-frame figures are published; raw videos/features/checkpoints stay local.

## 14–15: Actual retrieval under the same650kJ energy budget

These use the exact last completed model within650kJ from the energy/mR curve, rather than the common200-round checkpoint. All45 FU procedures complete. The original continuation was replayed unchanged; every evaluation/energy/time point and final parameter tensor reproduced exactly before intermediate checkpoints were used. Different completed FL counts are consequences of the actual energy costs, not extra uncharged updates. The same illustration selection rule and full-query reporting apply. The unchanged200-round examples remain above as a separate comparison.

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
