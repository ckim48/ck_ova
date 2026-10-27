# Matched-objective ablations

Six figures only, each exported as PNG, PDF and SVG. Left: CapERA. Right: AirScape. Methods: Proposed, H-OVA, S-OF+, NG-OVA. The figures use the established style, regular endpoint ticks and matched decimal precision. No point-value labels. Bandwidth figures explicitly mark radio-infeasible conditions with a cross and a separate status legend.

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

**Bandwidth** varies the UAV aggregation uplink symbol rate: 0.25, 0.5, 1, 2, 4 MHz. This is the existing uplink bandwidth sweep, not the downlink-rate sweep. Noise and all other conditions follow the original paired sensitivity protocol. **Delay** uses exactly the active FL/FU service time defined above. All listed methods and all five bandwidth values are retained for both datasets: 39 completed outcomes and 1 infeasible outcomes. Infeasible conditions, when present, are shown by a cross above the upper plot edge and a separate legend. Its vertical placement is fixed at 104% of the axes height, outside the data area, and does not encode an energy/delay value. The original missing metric remains null; no value is filled or extrapolated. Marker metadata is stored separately in `data/manifest.json`.

All results use seed-0, 200-FL-round experiments and the unchanged validation subsets. Raw plotting arrays, service components, original/excluded totals, axis settings and source hashes are in `data/`. Rebuild with `.venv/bin/python tools/uav_convergence_v2/selected_graphs.py --comparison guard_ablation`.

## Interpretation

All displayed methods use the same CLIPErase-inspired FM/RM/CM FU objective, model and workload. H-OVA replaces optimized head/resource allocation with a centroid head/uniform allocation heuristic. S-OF+ executes FL/FU serially through digital OFDMA with optimized head/tones. These matched-objective controls are retained to isolate head/allocation and transport/scheduling effects; similar retained loss/mR does not invalidate an energy/delay advantage. S-OF+ retains its previously authorized soft network-cap fallback, while Proposed/H-OVA keep hard caps.

AirScape H-OVA at 4 MHz is radio-infeasible and remains explicitly marked; no energy or delay is fabricated. The [main comparison](../selected_graph/README.md) additionally reports the new safe-proxy S-OF baseline.

**NG-OVA** removes only Proposed's FU-relative noise guard, retaining the existing joint component-noise budget, optimized head/resources, objective and parallel scheduler. It is a project-defined component ablation, not an independent published algorithm. All five bandwidth conditions were registered before the new outcomes. A curve is not moved or interpolated to occupy an intermediate rank. See the [guard ablation audit](../results/guard_ablation_v1_20261027/README.md) for all measured outcomes and the diagnostic explaining the overlap of the existing proxy-objective baselines.
