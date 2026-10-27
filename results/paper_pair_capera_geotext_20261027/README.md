# CapERA / GeoText-1652 paired figures

Left: CapERA. Right: GeoText-1652. Five methods, all ten sensitivity axes, paired seed 0.

This presentation pair was chosen after inspecting the completed four-dataset experiment, for UAV relevance and favorable energy/delay with largely maintained retrieval quality. It is not an independent confirmatory benchmark. All original results remain in [the full campaign](../iot_sensitivity_v1_20261027/figures).

[CapERA](https://github.com/yakoubbazi/CapEra) contains real UAV video-caption pairs (Remote Sensing, 2023). [GeoText-1652](https://github.com/MultimodalGeo/GeoText-1652) is an ECCV 2024 language-guided drone benchmark, also used by [HCCM at ACM MM 2025](https://github.com/rhao-hur/HCCM). Its drone views come from Google Earth / University-1652, so they must be described as virtual drone views, not real flight captures.

The retained mR values are our fixed held-out subset diagnostics. They are not the published dataset scores or an official test evaluation. GeoText's absolute mR remains low. The curves have no smoothing, numeric annotations or confidence intervals. Infeasible conditions are gaps, including the 0.625 s IoT slot for every method. S-OF+ can exceed the network TX cap; other methods retain the hard cap.

Energy includes communication, computing and flight/hovering. FL/FU completion is measured from takeoff to the last required broadcast (including preceding collection and travel). Mission delay also includes the return flight; total delay additionally includes preflight work. These are simulated mission costs, not measured UAV flight results.

There are 120 sensitivity figures (10 axes × 12 metrics) and two convergence figures, each in PNG, PDF and SVG. All five methods and every requested setting are retained.

| X axis | Energy | FL/FU completion | Mission delay | Loss | mR |
|---|---|---|---|---|---|
| Uplink bandwidth (MHz) | [Energy](figures/bandwidth_total_energy_j_1x2.png) | [FL/FU](figures/bandwidth_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/bandwidth_mission_delay_s_1x2.png) | [Loss](figures/bandwidth_loss_1x2.png) | [mR](figures/bandwidth_mR_1x2.png) |
| Downlink rate (Mbit/s) | [Energy](figures/downlink_total_energy_j_1x2.png) | [FL/FU](figures/downlink_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/downlink_mission_delay_s_1x2.png) | [Loss](figures/downlink_loss_1x2.png) | [mR](figures/downlink_mR_1x2.png) |
| UAV peak power (W) | [Energy](figures/power_total_energy_j_1x2.png) | [FL/FU](figures/power_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/power_mission_delay_s_1x2.png) | [Loss](figures/power_loss_1x2.png) | [mR](figures/power_mR_1x2.png) |
| Network TX budget (J) | [Energy](figures/network_cap_total_energy_j_1x2.png) | [FL/FU](figures/network_cap_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/network_cap_mission_delay_s_1x2.png) | [Loss](figures/network_cap_loss_1x2.png) | [mR](figures/network_cap_mR_1x2.png) |
| Channel gain offset (dB) | [Energy](figures/channel_gain_total_energy_j_1x2.png) | [FL/FU](figures/channel_gain_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/channel_gain_mission_delay_s_1x2.png) | [Loss](figures/channel_gain_loss_1x2.png) | [mR](figures/channel_gain_mR_1x2.png) |
| FU release groups | [Energy](figures/request_bursts_total_energy_j_1x2.png) | [FL/FU](figures/request_bursts_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/request_bursts_mission_delay_s_1x2.png) | [Loss](figures/request_bursts_loss_1x2.png) | [mR](figures/request_bursts_mR_1x2.png) |
| FU steps per request | [Energy](figures/fu_steps_total_energy_j_1x2.png) | [FL/FU](figures/fu_steps_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/fu_steps_mission_delay_s_1x2.png) | [Loss](figures/fu_steps_loss_1x2.png) | [mR](figures/fu_steps_mR_1x2.png) |
| Cruise speed (m/s) | [Energy](figures/speed_total_energy_j_1x2.png) | [FL/FU](figures/speed_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/speed_mission_delay_s_1x2.png) | [Loss](figures/speed_loss_1x2.png) | [mR](figures/speed_mR_1x2.png) |
| CPU frequency (GHz) | [Energy](figures/frequency_total_energy_j_1x2.png) | [FL/FU](figures/frequency_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/frequency_mission_delay_s_1x2.png) | [Loss](figures/frequency_loss_1x2.png) | [mR](figures/frequency_mR_1x2.png) |
| IoT slot duration (s) | [Energy](figures/collection_slot_total_energy_j_1x2.png) | [FL/FU](figures/collection_slot_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/collection_slot_mission_delay_s_1x2.png) | [Loss](figures/collection_slot_loss_1x2.png) | [mR](figures/collection_slot_mR_1x2.png) |

[Loss convergence](figures/loss_convergence_1x2.png) · [mR convergence](figures/mR_convergence_1x2.png)

For each dataset, Proposed has the lowest energy and FL/FU/mission delays in 35/36 conditions where all four original methods completed. The exception is `request_bursts_45`. Proposed mR is best or tied in 30/36 CapERA conditions and 36/36 GeoText conditions. H-OVA feasibility and all candidate-dataset results are recorded in `selection.json`.

Reproduce with `python3 tools/iot_sensitivity_v1/selected_pair.py`. No experiments are rerun. `metrics.csv`, `plot_series.json` and `summary.json` provide the raw values, displayed series and source hashes.
