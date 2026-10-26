# IoT-driven UAV deployment and VLM evaluation

The map now models a reason for UAV positions: receiving data from declared ground IoT devices. It uses four mapped service regions, the same ten UAVs over time, and 40 synthetic ground devices per region. The IoT locations and packet demands are generated with seed 48 before optimization. They are not surveyed locations. Real CapERA/RSITMD media remain a logical VLM replay; the datasets do not geolocate these IoT devices.

## Deployment model

`tools/iot_uav_deployment_v1/deployment.py` jointly updates balanced device association, fixed-altitude UAV positions and a nominal planning head. Its objective is ground uplink PA energy plus a nominal equal-size UAV packet-relay PA energy. Constant circuit terms are accounted for later. Propulsion is accounted for, but is not minimized by this deployment objective.

- Ten orthogonal 100 kHz bands, one per UAV; four devices per UAV use four 2.5-second slots. Ground packet sizes are declared scenario inputs in the saved registration.
- Free-space ground links at 50 m altitude; required transmit power is inverted from the Shannon rate, constrained by a 0.1 W device cap.
- Balanced association is solved by linear assignment. Position updates use SLSQP and are accepted only when feasible and non-increasing. Nominal heads are chosen by exhaustive evaluation of all ten candidates. This is a local alternating solution, not a global optimality claim.
- Horizontal service radius is 220 m. UAV separation is constrained to 22 m during hovering and throughout straight-line inter-region transitions, including return. Transition durations enforce at most 5 m/s.
- Inter-UAV channels use the existing noise-normalized free-space model, recomputed at the new positions. Planning heads are not forced on FL/FU: actual methods reselect heads using their own update vectors and radio constraints.

The acquisition/deployment structure is motivated by [Mozaffari et al., IEEE TWC 2017](https://arxiv.org/abs/1703.05401). The fixed-altitude balanced-load objective and numerical assumptions here are this project's scenario, not a reproduction of that paper.

The saved deployment passes independent payload/rate, power, assignment, all-candidate head and dense transition-separation checks. Its nominal variable communication objective changes from 0.102919 J at the original two-row positions to 0.0662045 J. This small deployment objective excludes flight/circuits and is **not** a measured total-energy saving or an FL/FU method comparison.

## Figure

```bash
.venv/bin/python tools/iot_uav_deployment_v1/deployment.py
.venv/bin/python tools/iot_uav_deployment_v1/checks.py
env MPLCONFIGDIR=/tmp/ova_ck_mpl .venv/bin/python tools/plot_mission_map.py --mode iot
```

The left panel is satellite imagery; the right shows IoT circles, UAV stars, a gold head star and a compact legend. Thin lines show device associations. Positions come from optimization, with no decorative jitter. `mission_satellite_iot_uav_planned_1x2` uses nominal planning heads. After each completed Proposed run, the runner writes `mission_satellite_iot_uav_actual_heads_1x2` from the first actual positive-duration transmission in each region. The last completed dataset supplies that actual-head view; the companion JSON identifies its source records and hashes.

An additional reference-style export is available via `python tools/plot_mission_map_reference.py --mode iot`. It keeps the same data while increasing the canvas to 9.4×4.2 inches, using blue UAV stars, orange IoT circles, red head stars, black marker outlines, metric axes and a shared top legend. The dashed red head ring is only visual emphasis, not a modeled coverage boundary. The separate script does not alter registered numerical experiment code.

## New experiment

`tools/iot_vlm_v1/experiment.py` runs fresh RSITMD and CapERA evaluations with the same registered RemoteCLIP/adapters, data partitions, four methods, 200 FL rounds and 45 five-step FU requests as the base campaign. The new geometry and channel matrices are common to all arms. Neither the earlier two-row results nor nominal planning-head costs are relabeled as new method results.

Communication energy now also includes ground-device PA/circuit costs and UAV reception. Each region adds ten seconds of acquisition hovering before FL/FU service. Total energy covers ground transmission plus the ten UAVs' communication, computing and flight. The learning simulator's service-only clock is distinct from postprocessed deployment delay, which includes acquisition and modeled computing time.

```bash
env PYTHONPATH=.cache/uav_experiment_deps .venv/bin/python tools/iot_vlm_v1/launch.py
cat results/iot_uav_remoteclip_v1/status.json
tail -n 30 results/iot_uav_remoteclip_v1/run.log
```

The launcher uses a detached session and persistent log. It refuses a duplicate campaign, uses the shared GPU lock and verifies registered source/input hashes. When all eight dataset/method arms complete, it exports the 1×2 loss/total-energy/total-delay charts and pushes only completed figures and chart summaries to the configured `ckim48/ck_ova` repository. Existing staged work defers automatic publication; failures are recorded rather than substituted with zeros.

The local manuscript extension is `manuscript/iot_uav_extension.tex`; it is inserted into `manuscript/iot_uav_draft.tex` while preserving the original manuscript. The draft distinguishes planning heads from actual FL/FU heads and discloses the synthetic physical scenario versus real-data replay. It does not assert numerical learning results before the new experiment completes.
