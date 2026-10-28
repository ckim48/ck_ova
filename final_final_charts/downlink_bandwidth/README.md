# Downlink bandwidth, paired resource bars and comparison tables

<!-- tgcn-additions -->
[Energy–mR, paired FU-workload bars and all-method retrieval charts](TGCN_ADDITIONS.md). Main figures10–11 and18–19; example sheets12–17 remain linked.
<!-- /tgcn-additions -->















Current Proposed is the unchanged P-KS. Four methods: Proposed, ISCC-P2, AFL-OTA and F-SalUn. CapERA is left and AirScape right. [Learning curves and definitions](../proposed_pks/README.md) remain available.

## Explicit bandwidth model

`R_dl = B_dl * eta_dl`, with `eta_dl = 1 bit/s/Hz`. Downlink bandwidth is 1, 2, 3, 4, 5 MHz; the derived rates are 1, 2, 3, 4, 5 Mbit/s. Only the downlink bandwidth changes; uplink bandwidth stays 1 MHz. Transmit power stays 0.32 W and every actual full-model broadcast is charged.

This is a declared fixed effective spectral-efficiency service model, conditional on every receiver supporting that efficiency across the sweep. It is not a measured bandwidth-throughput relationship, a Shannon-capacity calculation, or a validation of link feasibility at fixed power. There is no adaptive coding or bandwidth-dependent packet-error model. The new explicit bandwidth model overrides the inherited legacy 1 Mbps broadcast description in the physical-parameter history. An MHz axis would not have been valid without this extra assumption.

All eight 1 MHz controls and all sixteen 3/5 MHz conditions execute the full workload: 24 new runs. The sixteen existing 2/4 Mbit/s runs are reused as equivalent 2/4 MHz conditions only under the declared eta=1 model. Original files remain unchanged; the release audit checks shared input hashes, physical settings, every broadcast duration, and recomputed costs. All eight fresh 1 MHz controls reproduce the original models, final evaluations, radio decisions and clocks. Reuse is explicit in metrics, not claimed as new execution.

## Figures

- 01–03: energy, active FL/FU delay and separate mission delay versus downlink bandwidth.
- 04–05: two adjacent bars per method, at 1 MHz and 5 MHz. Method colors match the line plots; hatching identifies bandwidth. Common zero baseline and bandwidth legend, no bar-top numbers or extra grey text. Both values come from the completed runs, without additional training or rescaling costs.
- 06: each square map panel is another 20% smaller than the previous 72%-scale release, now 57.6% of the original map area width/height. Tick labels and visit numbers are redrawn at the same 12pt size and serif family as the line-chart ticks; they are not scaled with the map. Geometry and snapshot heads are unchanged.
- 07: baseline mR, validation loss, energy and active delay as a LaTeX table. The requested Forget KL column is omitted from this table; aggregate measurement records retain the original diagnostics.
- 08: 1 MHz/5 MHz energy and active-delay comparison table. Both tables include PDF/PNG previews, standalone LaTeX sources and CSV values. Best values are bold within each dataset and metric; a displayed-precision tie is bold for all tied methods. No uncertainty is invented from this single seed.

- 09: aggregation MSE budget versus total energy, from 32 new executions plus 8 shared baseline arms. No amplification of small changes or removal of flat digital curves.

Map panel side: previous 3.2495 inches → current 2.5996 inches. The surrounding margins accommodate full-size text, so the total canvas is not uniformly scaled. Map imagery: Esri, Vantor, Earthstar Geographics, and the GIS User Community. Road data: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). Original attribution, source hashes, typography checks and snapshot provenance are in `data/mission_map.json`.

## Aggregation MSE sweep

The plotted threshold is the dimensionless squared relative-RMSE coefficient, tau = delta^2: 0.010, 0.025, 0.040, 0.055, 0.070. The baseline delta=0.2 corresponds to tau=0.04. Each nonbaseline condition reruns all four methods on both datasets with all other inputs and physical settings fixed. Energy remains communication + computation + active hovering; both links stay at the original 1 MHz/1 Mbit/s condition.

The actual squared-error constraint is conditional E[||reconstruction error||_2^2] <= epsilon_k^2, where epsilon_k follows the original method norm/guard and 1e-8 RMS numerical floor. Proposed keeps its FL block-norm bound and FU-relative total-noise guard; AFL-OTA keeps its raw aggregate norm and coherent repetition rule. Thus tau is a relative MSE-budget coefficient, not an identical absolute or per-coordinate MSE limit across methods. The fixed FP32 ideal digital transports have zero modeled aggregation error and legitimately flat cost curves. All quality outcomes and bound diagnostics are retained in `data/mse_metrics.csv`; numerical definitions and unchanged method assumptions are in `data/mse_protocol.json`.

Energy = active communication + executed adapter computation + active hovering. Travel/IoT collection/preflight energy is excluded. Main delay is active FL/FU service time; mission delay additionally includes travel, collection and return. Same 200 total FL rounds and 225 FU steps for 45 requests. Single development seed, frozen feature caches and scoped literature adaptations; these are not full-VLM hardware measurements, statistical superiority results or certified equal-quality unlearning. Unfavorable quality and QoS-cap exceedances remain in the metrics.

| Dataset | Method | Energy (kJ), 1 MHz | Active delay (s), 1 MHz |
|---|---|---:|---:|
| CapERA | Proposed | 204.277 | 121.102 |
| CapERA | ISCC-P2 | 649.962 | 384.304 |
| CapERA | AFL-OTA | 252.882 | 149.912 |
| CapERA | F-SalUn | 649.960 | 384.302 |
| AirScape | Proposed | 204.267 | 121.096 |
| AirScape | ISCC-P2 | 649.305 | 383.915 |
| AirScape | AFL-OTA | 253.066 | 150.021 |
| AirScape | F-SalUn | 649.307 | 383.916 |

## Reproduction

With the existing local caches and original rate-study arms:

```sh
.venv/bin/python tools/uav_convergence_v2/downlink_bandwidth_v1.py run
.venv/bin/python tools/uav_convergence_v2/aggregation_mse_v1.py run
.venv/bin/python tools/uav_convergence_v2/downlink_bandwidth_report.py
.venv/bin/python tools/uav_convergence_v2/downlink_bandwidth_check.py
```

The frozen runner refuses to overwrite attempts. Only code, figures, aggregate metrics and hashes are published; no raw data, checkpoints or execution logs.
