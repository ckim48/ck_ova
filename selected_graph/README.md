# Selected graphs

Six figures only, each exported as PNG, PDF and SVG. Left: CapERA. Right: AirScape. Methods: Proposed, S-AC, AC-OF, S-OF. The figures use the established style, regular endpoint ticks and matched decimal precision. No point-value labels. Bandwidth figures explicitly mark radio-infeasible conditions with a cross and a separate status legend.

| No. | x axis | y axis | Preview |
|---|---|---|---|
| 1 | FL round | Validation loss | [PNG](01_round_fl_loss_1x2.png) |
| 2 | FL round | mR (%) | [PNG](02_round_mr_1x2.png) |
| 3 | Time (s) | Validation loss | [PNG](03_time_fl_loss_1x2.png) |
| 4 | Time (s) | mR (%) | [PNG](04_time_mr_1x2.png) |
| 5 | Bandwidth (MHz) | Total energy (kJ) | [PNG](05_bandwidth_total_energy_1x2.png) |
| 6 | Bandwidth (MHz) | Delay (s) | [PNG](06_bandwidth_delay_1x2.png) |

Item 4 repeated time–FL loss in the request; it is interpreted as time–mR.

## Exact definitions

**Validation loss** is the saved symmetric contrastive retrieval loss on held-out retained queries, not the per-client minibatch training objective. **mR** is the average of the six video-to-text/text-to-video Recall@1, @5 and @10 scores. Original metric values are preserved.

The loss-axis label previously said FL loss. It now says Validation loss to match the actual saved metric. Existing filenames are retained for compatibility; no loss values were replaced, normalized or rescaled.

**Round plots** cover the shared initial model at round 0, common preparation through round 20, and method-specific FL/FU observations through round 200. FU updates at an unchanged FL round retain their original x coordinate. **Time plots** start at the common round-20 model; time zero begins mission FL/FU service. Only active service windows are accumulated, including compute, uplink, aggregate broadcasts and scheduler waits. Flight, IoT collection and all preflight preparation are excluded. Time is modeled UAV time, not GPU/server runtime. Each final value is held horizontally to the common display endpoint; the extension is not an extra observation or proof of convergence.

**Total energy** is total energy within those same FL/FU service windows: communication + computing + hovering across all ten UAVs. It is not whole-mission battery energy. The original physical coefficients, per-device MACs, recorded transmit bounds and broadcast costs are used. Travel, collection communication/hovering, cold feature extraction, initial20 and proxy20 preparation are excluded. Consequently these numbers differ from the earlier energy subtotal, which retained collection and preflight costs. The original full energy ledger remains untouched. The service sum is independently reconciled against original total minus all excluded components.

**Bandwidth** varies the UAV aggregation uplink symbol rate: 0.25, 0.5, 1, 2, 4 MHz. At the user's request, the displayed axis starts at 1 MHz with regular ticks 1, 2, 3, 4. The displayed measured points are 1, 2 and 4 MHz; 3 MHz is an axis tick, not an additional measurement. The original 0.25 and 0.5 MHz outcomes remain in the ledgers and raw plotting arrays; display arrays are stored separately. This is the existing uplink bandwidth sweep, not the downlink-rate sweep. Noise and all other conditions follow the original paired sensitivity protocol. **Delay** uses exactly the active FL/FU service time defined above. All listed methods and all five bandwidth values are retained for both datasets: 40 completed outcomes and 0 infeasible outcomes. Infeasible conditions, when present, are shown by a cross above the upper plot edge and a separate legend. Its vertical placement is fixed at 104% of the axes height, outside the data area, and does not encode an energy/delay value. The original missing metric remains null; no value is filled or extrapolated. Marker metadata is stored separately in `data/manifest.json`.

All results use seed-0, 200-FL-round experiments and the unchanged validation subsets. Raw plotting arrays, service components, original/excluded totals, axis settings and source hashes are in `data/`. Rebuild with `.venv/bin/python tools/uav_convergence_v2/selected_graphs.py --comparison main`.

## Comparison scope

S-OF is a new measured serial digital OFDMA baseline using the same existing safe-proxy FU objective as S-AC and AC-OF. S-OF+ used the CLIPErase-inspired objective shared with Proposed. The new S-OF is not a relabeling of S-OF+. Only the FU objective changes between these two digital controls; optimized digital head/tones and all physical limits remain intact. This main comparison therefore compares whole systems, including the objective difference. It cannot by itself attribute a learning-quality difference to parallel transmission.

[Matched-objective ablations](../ablation_graph/README.md) retain Proposed, H-OVA and S-OF+ with their original outcomes, including the AirScape H-OVA 4 MHz infeasibility. H-OVA isolates the head/allocation heuristic; S-OF+ is the same-objective serial digital control. Similar round-based utility is expected when their objectives match.

S-OF and S-OF+ inherit the previously authorized soft network-TX-cap fallback; overruns are recorded. The analog arms retain their original hard network cap. Per-user power/energy, deadline and error checks remain enforced. See the [protocol and quality audit](../results/baseline_fairness_v1_20261027/README.md) for component definitions, forgetting diagnostics and all ten new outcomes. No hyperparameter was selected based on these results.
