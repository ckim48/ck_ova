# CapERA / AirScape paired experiments

Left CapERA; right AirScape (WebUAV3M source subset). Paired seed0, five methods, ten axes.

AirScape is ACM MM2025 real UAV video/intention data. This is an eight-frame mean-pooled retrieval adaptation, not its published world-model benchmark. Its1native text/video differs from CapERA's5captions/video. Absolute mR across datasets is not a matched difficulty comparison. All methods within a dataset share the identical split, feature cache, initialization and request schedule.

The replacement was screened on validation after earlier candidates. This is exploratory; the official test set is excluded. [All pilot outcomes](../uav_convergence_v2_20261027/) remain available. AirScape's +0.4557pp pilot gain did not pass the prospective1.0pp strong-gain gate. This follow-up examines its modest rise and plateau under FU/radio conditions; it is not presented as a passed screen or a CapERA-sized improvement. No selection of favorable sensitivity settings or method results.

Energy includes communication, computing and flight/hovering. FL/FU completion includes preceding movement/collection and ends with the last required broadcast; mission completion also includes return; total completion additionally includes preflight. Costs are modeled, not measured UAV flight values. S-OF+ permits TX-budget overruns.

Curves use raw observations, including repeated round observations and infeasible gaps. Partial convergence traces stop at their actual last evaluation. Single seed; no confidence intervals.

Current terminal arms: 410/410; figures: 122/122.

| Axis | Energy | FL/FU completion | Mission delay | Loss | mR |
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

[loss convergence](figures/loss_convergence_1x2.png)

[mR convergence](figures/mR_convergence_1x2.png)

## Numerical power-check correction

This checked version replaces the five AirScape arms at the -6dB channel-gain scenario with a deterministic rerun. Candidate screening now uses the same1ppm power tolerance as the unchanged final receiver check, instead of2ppm. Physical limits were not relaxed. All four previously nonfailed arms retain their original status and metrics. The original failure and all original results are preserved in `airscape_sensitivity_v2_20261027`. See `../airscape_power_audit_v1_20261027/verification.json`.
