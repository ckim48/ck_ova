# CapERA / AirScape paired experiments

Left CapERA; right AirScape (WebUAV3M source subset). Paired seed0, five methods, ten axes.

AirScape is ACM MM2025 real UAV video/intention data. This is an eight-frame mean-pooled retrieval adaptation, not its published world-model benchmark. Its1native text/video differs from CapERA's5captions/video. Absolute mR across datasets is not a matched difficulty comparison. All methods within a dataset share the identical split, feature cache, initialization and request schedule.

The replacement was screened on validation after earlier candidates. This is exploratory; the official test set is excluded. [All pilot outcomes](../uav_convergence_v2_20261027/) remain available. AirScape's +0.4557pp pilot gain did not pass the prospective1.0pp strong-gain gate. This follow-up examines its modest rise and plateau under FU/radio conditions; it is not presented as a passed screen or a CapERA-sized improvement. No selection of favorable sensitivity settings or method results.

Energy includes communication, computing and flight/hovering. FL/FU completion includes preceding movement/collection and ends with the last required broadcast; mission completion also includes return; total completion additionally includes preflight. Costs are modeled, not measured UAV flight values. S-OF+ permits TX-budget overruns.

Curves use raw observations, including repeated round observations and infeasible gaps. Partial convergence traces stop at their actual last evaluation. Single seed; no confidence intervals.

Current terminal arms: 90/410; figures: 26/122.

| Axis | Energy | FL/FU completion | Mission delay | Loss | mR |
|---|---|---|---|---|---|
| Uplink bandwidth (MHz) | [Energy](figures/bandwidth_total_energy_j_1x2.png) | [FL/FU](figures/bandwidth_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/bandwidth_mission_delay_s_1x2.png) | [Loss](figures/bandwidth_loss_1x2.png) | [mR](figures/bandwidth_mR_1x2.png) |
| Downlink rate (Mbit/s) | [Energy](figures/downlink_total_energy_j_1x2.png) | [FL/FU](figures/downlink_FL_FU_completion_delay_s_1x2.png) | [Mission](figures/downlink_mission_delay_s_1x2.png) | [Loss](figures/downlink_loss_1x2.png) | [mR](figures/downlink_mR_1x2.png) |
| UAV peak power (W) | pending | pending | pending | pending | pending |
| Network TX budget (J) | pending | pending | pending | pending | pending |
| Channel gain offset (dB) | pending | pending | pending | pending | pending |
| FU release groups | pending | pending | pending | pending | pending |
| FU steps per request | pending | pending | pending | pending | pending |
| Cruise speed (m/s) | pending | pending | pending | pending | pending |
| CPU frequency (GHz) | pending | pending | pending | pending | pending |
| IoT slot duration (s) | pending | pending | pending | pending | pending |

[loss convergence](figures/loss_convergence_1x2.png)

[mR convergence](figures/mR_convergence_1x2.png)
