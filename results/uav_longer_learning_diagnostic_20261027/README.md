# Additional-learning diagnostic: CapERA and AirScape

This is a separate, single-seed **FL-only, ideal-aggregation diagnostic**, not a Proposed run or a five-method mission comparison. The completed mission results and their communication/energy/delay values are unchanged.

## Fixed protocol

Both datasets use the same frozen RemoteCLIP ViT-B/32 features, rank-10 residual adapters, captions, split, retrieval gallery, ten client partitions, initialization, per-round batch generation and batch size as the existing FL-only pilots. Rounds 1–200 reproduce the original schedule: learning rate 0.0035 through round 40, then cosine decay to zero at round 200. Every 20-round retained-metric checkpoint through round 200 was verified against the prior pilot.

Rounds 201–1000 continue the same model and batch sequence at a predeclared fixed learning rate of 0.001. All 20-round evaluations are retained; the table reports the final prescribed checkpoint, not the best checkpoint. There is no FU, channel noise, head selection or mission timing in this diagnostic.

## Results

| Dataset | mR at round 200 (%) | mR at round 1000 (%) | Gain (percentage points) | Loss at 200 | Loss at 1000 |
|---|---:|---:|---:|---:|---:|
| CapERA | 16.42 | 22.51 | 6.09 | 5.4247 | 4.5904 |
| AirScape | 13.54 | 15.49 | 1.95 | 5.5993 | 5.1957 |

The current 200-round training configuration leaves additional learning headroom. However, this diagnostic changes both the number of updates and the post-200 learning-rate schedule; it does not identify the causal effect of round count alone. Appending rounds while keeping the final zero learning rate would produce no further learning. The earlier apparent plateau therefore does not establish that the model has reached its best achievable retrieval performance.

The low absolute mR is not resolved by this continuation: even at round 1000, the scores remain modest. The backbone is frozen and video features average eight sampled frames, with only small residual adapters trained. These are potential modeling limitations, not experimentally isolated causes. The metric is the arithmetic mean of image-to-text and text-to-image Recall@1, @5 and @10 (six terms); it is not classification accuracy. Custom retained-validation galleries and exploratory dataset selection also prevent direct comparison with official benchmark scores.

Changing the paper comparison to longer training requires rerunning every method with the same new round budget and schedule, then recalculating FL/FU completion, mission delay and energy. These diagnostic values must not be spliced into the existing mission plots.

`protocol.json` was saved before execution. `capera/rounds.json` and `airscape/rounds.json` contain all checkpoints; each `inputs.json` records original input hashes and gallery sizes. `finished.json` records both outcomes.
