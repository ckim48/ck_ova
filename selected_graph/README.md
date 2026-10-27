# Selected graphs

Six figures only, each exported as PNG, PDF and SVG. Left: CapERA. Right: AirScape. All five existing methods use the established style, regular endpoint ticks and matched decimal precision. No point-value labels. Bandwidth figures explicitly mark radio-infeasible conditions with a cross and a separate status legend.

| No. | x axis | y axis | Preview |
|---|---|---|---|
| 1 | FL round | FL loss | [PNG](01_round_fl_loss_1x2.png) |
| 2 | FL round | mR (%) | [PNG](02_round_mr_1x2.png) |
| 3 | Time (s) | FL loss | [PNG](03_time_fl_loss_1x2.png) |
| 4 | Time (s) | mR (%) | [PNG](04_time_mr_1x2.png) |
| 5 | Bandwidth (MHz) | Total energy (kJ) | [PNG](05_bandwidth_total_energy_1x2.png) |
| 6 | Bandwidth (MHz) | Delay (s) | [PNG](06_bandwidth_delay_1x2.png) |

Item 4 repeated time–FL loss in the request; it is interpreted as time–mR.

## Exact definitions

**FL loss** is the saved symmetric contrastive retrieval loss on held-out retained queries, not the per-client minibatch training objective. **mR** is the average of the six video-to-text/text-to-video Recall@1, @5 and @10 scores. Original metric values are preserved.

**Round plots** cover the shared initial model at round 0, common preparation through round 20, and method-specific FL/FU observations through round 200. FU updates at an unchanged FL round retain their original x coordinate. **Time plots** start at the common round-20 model; time zero begins mission FL/FU service. Only active service windows are accumulated, including compute, uplink, aggregate broadcasts and scheduler waits. Flight, IoT collection and all preflight preparation are excluded. Time is modeled UAV time, not GPU/server runtime. Each final value is held horizontally to the common display endpoint; the extension is not an extra observation or proof of convergence.

**Total energy** is total energy within those same FL/FU service windows: communication + computing + hovering across all ten UAVs. It is not whole-mission battery energy. The original physical coefficients, per-device MACs, recorded transmit bounds and broadcast costs are used. Travel, collection communication/hovering, cold feature extraction, initial20 and proxy20 preparation are excluded. Consequently these numbers differ from the earlier energy subtotal, which retained collection and preflight costs. The original full energy ledger remains untouched. The service sum is independently reconciled against original total minus all excluded components.

**Bandwidth** varies the UAV aggregation uplink symbol rate: 0.25, 0.5, 1, 2, 4 MHz. This is the existing uplink bandwidth sweep, not the downlink-rate sweep. Noise and all other conditions follow the original paired sensitivity protocol. **Delay** uses exactly the active FL/FU service time defined above. All five methods and five bandwidth values are retained for both datasets: 49 completed outcomes and one radio-infeasible outcome (AirScape, H-OVA, 4 MHz). That condition is shown at 4 MHz with a purple cross above the upper plot edge and the legend `Infeasible (H-OVA)`. Its vertical placement is fixed at 104% of the axes height, outside the data area, and does not encode an energy/delay value. The original missing metric remains null; no value is filled or extrapolated. Marker metadata is stored separately in `data/manifest.json`.

All results use the existing seed-0, 200-FL-round experiments and validation subsets. Raw plotting arrays, service components, original/excluded totals, axis settings and source hashes are in `data/`. Rebuild with `.venv/bin/python tools/uav_convergence_v2/selected_graphs.py`.
