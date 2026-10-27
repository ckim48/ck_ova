# Energy sensitivity and 200-round retrieval audit

## Energy figures

[Downlink rate](figures/downlink_energy_1x2.png) is the primary decreasing-energy example. [IoT collection slot duration](figures/collection_slot_energy_1x2.png) is the increasing-energy example. Both contain CapERA (left), AirScape (right), and all five methods with unchanged measured scenario outcomes. They use the established colors, regular endpoint ticks, matched numeric precision, and no point-value annotations.

Energy is communication + computing + hovering, as in the current report. Common travel energy is excluded. A faster downlink shortens aggregate broadcasts and their hover duration. Longer IoT slots extend hovering; the measured communication contribution is also retained. The 0.625-second slot violates the original IoT power constraint for all arms and remains missing, with its status recorded in `energy_metrics.json`. The complete ten-axis report remains available in the sibling time/energy folder.

## Our 200-round controls

mR is the arithmetic mean of video-to-text and text-to-video Recall@1, @5, @10. The code uses `i2t`/`t2i` names for pooled-video features. Every six-term average was checked. Neither class-average recall, Mean Rank (MnR), nor Median Rank (MdR) is used.

| Condition | CapERA mR (%) | AirScape mR (%) |
|---|---:|---:|
| Frozen encoder / initial adapters | 15.42 | 13.09 |
| Common warm model, round 20, before mission FU | 15.68 | 13.09 |
| FL: full training pool, no FU, ideal aggregation, round 200 | 16.42 | 13.54 |
| RT: retained-only retraining from initialization, ideal aggregation, round 200 | 16.47 | 13.54 |
| Proposed: 200 FL rounds and all prescribed FU work | 16.90 | 13.67 |

[Round-based control curves](figures/mR_round_controls_1x2.png) use the labels above. FL and RT are diagnostic references, not additional mission algorithms. They incur no simulated radio errors, and no mission energy or delay is assigned to them. The same frozen feature archive, adapter initialization, retained queries, gallery and 200-round learning-rate schedule are used. FL trains on the full pool; RT removes every eventual deletion item from the beginning; Proposed handles arriving requests online. The common mission prefix includes actual communication, unlike the ideal controls. Therefore these are not isolated toggles of the FU term, and the small positive mR differences do not prove that FU improves utility.

The same fixed retained evaluation is used throughout: CapERA has 256 video queries, 503 gallery videos and five captions per video; AirScape has 256 queries, 614 gallery videos and one caption per video. These are exploratory held-out validation subsets, not official benchmark test evaluations; seed 0 only.

The local controls do not show a drop from a high-performing 35% model to 16.9% after FU. The underlying 200-round learner itself reaches only 16.42%/13.54%. Proposed matches that limited utility level. It is reasonable to report this as retention relative to local references, but insufficient to claim strong absolute retrieval performance, statistical superiority, or effective deletion from mR alone. Forgetting must be assessed separately against retained-only retraining, including the existing forget-side distribution and privacy diagnostics.

The separate [longer-learning diagnostic](../uav_longer_learning_diagnostic_20261027/) reached 22.51%/15.49% after 1000 FL-only rounds with a restarted learning rate. It shows additional learning headroom, but changes both round budget and schedule; those values are not inserted into this 200-round comparison. The present frozen RemoteCLIP B/32, eight-frame mean pooling and rank-10 adapters remain possible modeling limitations; their individual effects have not been isolated.

## External context

From the CapERA rows of [TCMA v2 Tables I and II](https://arxiv.org/html/2510.10180v2), the six-recall averages are:

| Published model | Computed mR (%) |
|---|---:|
| CLIP ViT-B/32 | 20.92 |
| CLIP ViT-B/16 | 21.47 |
| TCMA ViT-B/32 | 35.25 |
| TCMA ViT-B/16 | 38.28 |

These are contextual values, not a matched FL/FU comparison. TCMA uses CLIP initialization, 12 sampled frames and encoder learning; our frozen-feature setup and custom validation evaluation differ. The raw recall inputs and their arithmetic are retained in `literature.json`.

[AirScape Table 1](https://arxiv.org/html/2507.08885v2) evaluates generated-video quality and intention alignment with FID, FVD and IAR. It does not supply a retrieval-mR baseline. Our 13.67% therefore has no directly matched original-paper mR reference.

The present evidence supports comparing communication efficiency at matched modest retrieval utility. A stronger absolute-accuracy claim needs a stronger or better-trained learner and rerunning every method under the same revised training budget, with energy and delay recomputed. Independent seeds and a final held-out test should be used for confirmatory conclusions.
