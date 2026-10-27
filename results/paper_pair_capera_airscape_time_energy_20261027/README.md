# CapERA / AirScape: convergence over time and revised energy charts

These plots use the checked, completed experiments. No model, seed, request, route or radio condition is rerun or changed.

## Energy definition

The chart label is **Energy (kJ)**, as requested. Its value is communication + computing + hovering. Inter-site travel and return-flight propulsion energy are subtracted from the former whole-mission total. The retained communication scope includes UAV communications and IoT transmission. Common preflight feature/initial/proxy computing remains included. This subtotal is not the UAV battery's whole-mission energy. Travel energy is common across methods within each scenario, although it changes across cruise-speed settings. The full original components remain in `metrics.csv`; infeasible points remain gaps. The baseline bar has a zero baseline.

## Time definitions

All times are the existing UAV/IoT simulation model, not GPU/server runtime.

- **Elapsed time:** from takeoff after common initial20/proxy20 preparation, includes movement and collection.
- **FL/FU service time:** active service windows only; includes computation, uplink, broadcast and scheduler waits.
- **Total elapsed time:** from cold feature extraction; includes initial20/proxy20 and mission movement/collection.

Every convergence trace ends when its final required FL/FU broadcast is complete. Return flight happens after learning and is not an extra learning period. Mission and service plots begin at the common20-round initial model. Total-time plots additionally show the recorded initial20 evaluations. Step traces carry forward only the last observed metric, avoiding an artificial slope during flight. There is no smoothing, invented evaluation, change to loss/mR, or annotation text on the chart. Different methods stop at their own actual work completion time.

| Time axis | Loss | mR |
|---|---|---|
| Elapsed time (s) | [Loss](figures/loss_mission_time_1x2.png) | [mR](figures/mR_mission_time_1x2.png) |
| FL/FU service time (s) | [Loss](figures/loss_service_time_1x2.png) | [mR](figures/mR_service_time_1x2.png) |
| Total elapsed time (s) | [Loss](figures/loss_total_time_1x2.png) | [mR](figures/mR_total_time_1x2.png) |

[Baseline energy bars](figures/baseline_energy_1x2.png)

| Parameter | Energy |
|---|---|
| Uplink bandwidth (MHz) | [Energy](figures/bandwidth_energy_1x2.png) |
| Downlink rate (Mbit/s) | [Energy](figures/downlink_energy_1x2.png) |
| UAV peak power (W) | [Energy](figures/power_energy_1x2.png) |
| Network TX budget (J) | [Energy](figures/network_cap_energy_1x2.png) |
| Channel gain offset (dB) | [Energy](figures/channel_gain_energy_1x2.png) |
| FU release groups | [Energy](figures/request_bursts_energy_1x2.png) |
| FU steps per request | [Energy](figures/fu_steps_energy_1x2.png) |
| Cruise speed (m/s) | [Energy](figures/speed_energy_1x2.png) |
| CPU frequency (GHz) | [Energy](figures/frequency_energy_1x2.png) |
| IoT slot duration (s) | [Energy](figures/collection_slot_energy_1x2.png) |

Time-to-quality results include every quarter increment of the common reachable range for loss and mR in `time_to_quality.json`. They are evaluated on recorded checkpoints, not continuous measurements. Faster completion of the prescribed workload is distinct from reaching every quality threshold first. Single seed and exploratory dataset selection still apply. The earlier full charts and round-based traces remain in [the checked report](../paper_pair_capera_airscape_checked_20261027/).
