# Downlink bandwidth, paired resource bars and comparison tables

Current Proposed is the unchanged P-KS. Four methods: Proposed, ISCC-P2, AFL-OTA and F-SalUn. CapERA is left and AirScape right. [Learning curves and definitions](../proposed_pks/README.md) remain available.

## Explicit bandwidth model

`R_dl = B_dl * eta_dl`, with `eta_dl = 1 bit/s/Hz`. Downlink bandwidth is 1, 2, 3, 4, 5 MHz; the derived rates are 1, 2, 3, 4, 5 Mbit/s. Only the downlink bandwidth changes; uplink bandwidth stays 1 MHz. Transmit power stays 0.32 W and every actual full-model broadcast is charged.

This is a declared fixed effective spectral-efficiency service model, conditional on every receiver supporting that efficiency across the sweep. It is not a measured bandwidth-throughput relationship, a Shannon-capacity calculation, or a validation of link feasibility at fixed power. There is no adaptive coding or bandwidth-dependent packet-error model. The new explicit bandwidth model overrides the inherited legacy 1 Mbps broadcast description in the physical-parameter history. An MHz axis would not have been valid without this extra assumption.

All eight 1 MHz controls and all sixteen 3/5 MHz conditions execute the full workload: 24 new runs. The sixteen existing 2/4 Mbit/s runs are reused as equivalent 2/4 MHz conditions only under the declared eta=1 model. Original files remain unchanged; the release audit checks shared input hashes, physical settings, every broadcast duration, and recomputed costs. All eight fresh 1 MHz controls reproduce the original models, final evaluations, radio decisions and clocks. Reuse is explicit in metrics, not claimed as new execution.

## Figures

- 01–03: energy, active FL/FU delay and separate mission delay versus downlink bandwidth.
- 04–05: two adjacent bars per method, at 1 MHz and 5 MHz. Method colors match the line plots; hatching identifies bandwidth. Common zero baseline and bandwidth legend, no bar-top numbers or extra grey text. Both values come from the completed runs, without additional training or rescaling costs.
- 06: the existing satellite/IoT/UAV map reduced by another 20% in width and height relative to the previous 90%-size release. Unchanged geometry and snapshot heads. Its provenance identifies the original snapshot run, not a newly measured bandwidth-dependent trajectory.
- 07: baseline mR, validation loss, common-reference forget KL, energy and active delay as a LaTeX table.
- 08: 1 MHz/5 MHz energy and active-delay comparison table. Both tables include PDF/PNG previews, standalone LaTeX sources and CSV values. Best values are bold within each dataset and metric; a displayed-precision tie is bold for all tied methods. No uncertainty is invented from this single seed.

Map size: 10.4 × 5.35 inches → previous 9.36 × 4.815 inches → current 7.488 × 3.852 inches (72% of the original). Map imagery: Esri, Vantor, Earthstar Geographics, and the GIS User Community. Road data: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). The complete original attribution, source hashes and snapshot provenance are in `data/mission_map.json`.

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
.venv/bin/python tools/uav_convergence_v2/downlink_bandwidth_report.py
.venv/bin/python tools/uav_convergence_v2/downlink_bandwidth_check.py
```

The frozen runner refuses to overwrite attempts. Only code, figures, aggregate metrics and hashes are published; no raw data, checkpoints or execution logs.
