# Expanded UAV/VLM sensitivity figures

CapERA (left), AirScape (right). Proposed, ISCC-P2 and AFL-OTA; the [paper-core transfer definitions and limitations](../README.md) remain unchanged. All three use the same intervention at each plotted point. These are scoped adaptations, not complete reproductions of the cited UAV papers.

## Figures

- **01–10:** five axes × total energy / active FL/FU delay.
- **11–16:** retained mR, validation loss and forget KL versus round and active time, at the unchanged baseline.
- **17–18:** final mR and KL versus FL-round budget; fewer rounds change utility and forgetting, not just cost.
- **19–23:** mission delay for the five axes, kept separate from active FL/FU delay.
- Earlier CPU-frequency, per-request FU-cycle and batch-cap results remain [in the parent bundle](../index.html), including nearly flat curves.

## Controlled interventions

| Axis | Values | Fixed / interpretation |
|---|---|---|
| Uplink bandwidth | 1,2,3,4,5 MHz | Existing audited fresh simulations; unchanged channel/noise scaling and all full payloads. |
| Downlink rate | 0.25,0.5,1,2,4 Mbit/s | Existing audited fresh simulations; all model broadcasts included. |
| FU requests per release | 1,3,5,9,15 | Same45 requests and225 FU cycles, regrouped into45,15,9,5,3 releases across the five sites. Release grouping also changes which site serves a request, following the registered existing scenario. |
| FL-round budget | 100,125,150,175,200 |20 common warmup +80/105/130/155/180 mission FL updates, equally divided over five sites. Same45 requests and225 FU cycles. Original cosine schedule is truncated, not retuned. Within-site release ordinals are compressed proportionally. |
| Downlink packet loss | 0,10,20,30,40% | Independent exogenous erasures of whole multicast model+control packets. Actual attempts until success are recorded and charged. |

**ARQ assumptions:** one trial represents delivery to all intended peers; failed trials repeat the whole broadcast and consume real modeled airtime, communication energy and hover time. Shared deterministic uniform draws are coupled across loss settings and methods by transaction ordinal. Feedback/ACK has zero duration and energy in this simplified model. Uplink loss/CSI assumptions are unchanged. Packet loss is an imposed scenario parameter, not measured or SNR-calibrated reliability. Retries do not create additional gradient updates.

**Costs and clocks:** Total energy = active FL/FU communication + executed adapter computation + hovering. Main delay = sum of active service intervals. Mission delay includes travel, IoT collection, all service and return, with no preflight feature extraction. Feature caches are fixed; these are not full-VLM encoding costs. Energy, active delay and mission delay are independently recomputed from recorded work/transaction/timeline ledgers. Active hovering dominates this cached-adapter scenario, so energy and delay profiles are closely related; component energies remain available in the metrics CSV.

**Quality and reference:** all cases use the same data partitions,45-request forget union, warm20 and reporting reference200. Shorter-round cases are intentionally compared against that fixed reference; their KL must not be presented as equivalence to a newly matched shorter retraining oracle. The reference is reporting-only. Proposed retains its previously disclosed offline known-cohort proxy and adaptive controller; paper baselines retain their shared GA→FT FU extension.

**Integrity:** six zero-change control arms reproduce baseline final metrics exactly and preserve transaction order, times and RF energy. The expansion contains64 new intervention arms plus those6 controls and8 identical reused Proposed request-grouping arms. The combined five-axis bundle compares126 unique arms. Single development seed; no confidence intervals or statistical superiority claim. No curve smoothing, rank filtering, artificial separation or added slopes. QoS-cap exceedances remain in metrics; completion does not imply every original Proposed bound was met.

Both axes have equally spaced numeric endpoint ticks, matching precision across panels. No point labels or extra grey captions. Convergence plots retain all observed FU effects; time curves hold the last completed model to the common display horizon.

The two earlier launcher attempts stopped during dynamic function construction before any training arm started; the v3 launcher fixes source tracking for freshly loaded modules. They produced no selected or discarded model results. The100-round AirScape Proposed run later exposed a real regional-departure assertion: its late request had only4 of5 FU cycles when the third site FL quota was reached. A preserved correction performs the remaining FU-only transaction before departure, charges its full costs, and adds no FL updates. Its first48 transactions are identical to the failed attempt except for measured host optimizer wall time, which is excluded from modeled UAV energy/delay; all other completed arms already satisfy regional completion. There are71 new training attempts including this preserved failure,70 completed new arms and8 reused arms. See `data/regional_completion_correction.json`. Source-frozen experiment code, control checks, input hashes, full plot arrays and aggregate metrics are included; raw data/checkpoints/transaction traces remain local.

## Measured endpoint changes

Percent change is from the first to last x value. Nonmonotonic curves retain every intermediate point.

| Axis | Dataset | Method | Energy change | Active-delay change | Shape (energy) |
|---|---|---|---:|---:|---|
| bandwidth | CapERA | Proposed | -2.03% | -2.03% | decreasing |
| bandwidth | CapERA | ISCC-P2 | -38.49% | -38.43% | decreasing |
| bandwidth | CapERA | AFL-OTA | -0.56% | -0.56% | decreasing |
| bandwidth | AirScape | Proposed | -2.09% | -2.09% | decreasing |
| bandwidth | AirScape | ISCC-P2 | -38.53% | -38.47% | decreasing |
| bandwidth | AirScape | AFL-OTA | -0.50% | -0.50% | decreasing |
| downlink | CapERA | Proposed | -93.00% | -93.00% | decreasing |
| downlink | CapERA | ISCC-P2 | -66.69% | -66.78% | decreasing |
| downlink | CapERA | AFL-OTA | -93.27% | -93.27% | decreasing |
| downlink | AirScape | Proposed | -92.96% | -92.96% | decreasing |
| downlink | AirScape | ISCC-P2 | -66.73% | -66.81% | decreasing |
| downlink | AirScape | AFL-OTA | -93.26% | -93.26% | decreasing |
| request_bursts | CapERA | Proposed | -1.10% | -1.10% | nonmonotonic |
| request_bursts | CapERA | ISCC-P2 | -34.63% | -34.70% | decreasing |
| request_bursts | CapERA | AFL-OTA | -54.04% | -54.04% | decreasing |
| request_bursts | AirScape | Proposed | -1.34% | -1.34% | nonmonotonic |
| request_bursts | AirScape | ISCC-P2 | -34.63% | -34.70% | decreasing |
| request_bursts | AirScape | AFL-OTA | -54.03% | -54.03% | decreasing |
| fl_rounds | CapERA | Proposed | +122.79% | +122.80% | increasing |
| fl_rounds | CapERA | ISCC-P2 | +87.21% | +87.16% | increasing |
| fl_rounds | CapERA | AFL-OTA | +78.71% | +78.71% | increasing |
| fl_rounds | AirScape | Proposed | +120.08% | +120.08% | increasing |
| fl_rounds | AirScape | ISCC-P2 | +87.38% | +87.33% | increasing |
| fl_rounds | AirScape | AFL-OTA | +78.65% | +78.65% | increasing |
| packet_loss | CapERA | Proposed | +77.38% | +77.38% | increasing |
| packet_loss | CapERA | ISCC-P2 | +29.11% | +29.18% | increasing |
| packet_loss | CapERA | AFL-OTA | +74.81% | +74.81% | increasing |
| packet_loss | AirScape | Proposed | +77.27% | +77.27% | increasing |
| packet_loss | AirScape | ISCC-P2 | +29.14% | +29.21% | increasing |
| packet_loss | AirScape | AFL-OTA | +74.76% | +74.76% | increasing |

Rebuild after local inputs are available: `python tools/uav_convergence_v2/workload_release.py run`. Audit/export existing completed results: `python tools/uav_convergence_v2/workload_release.py finish`. The existing parent campaign supplies the reused bandwidth/downlink and baseline arms.
