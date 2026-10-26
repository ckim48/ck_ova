# IoT-driven UAV deployment and VLM evaluation

## Completion-driven mission revision

The user identified the fixed minimum residence as preventing communication savings from affecting mission completion. `tools/iot_event_mission_v1/` implements a separate scenario in `results/iot_uav_remoteclip_v4_event_mission/`. Each region performs its unchanged 10-second ground collection, 36 FL updates and all assigned five-step FU requests, then departs immediately. The route, formation, flight speeds, data partitions and request ordinals are unchanged. All eight adaptation arms are rerun from the same hash-checked initial/proxy/reference checkpoints; previous comparison-arm results are not reused.

The live mission clock now advances through maximum per-device computation time, uplink and complete model broadcast for each transaction. The queues observe this full time, so periodic decisions and model updates can differ from the former uplink-only clock. FU completion includes the fifth step's broadcast to all UAVs. Flight and collection also advance time; the complete return route is retained. Energy accounting uses the resulting actual hover durations once, with no 60-second floor and no second addition of compute or downlink time. S-OF+ retains the authorized network-energy overrun policy and all other radio limits.

Independent preflight replays check fast and slow service, 36 FL updates in each of five regions, all 225 FU request-steps, correct uplink-only counters, departure/arrival continuity and return travel. A separate numerical fixture checks compute/uplink/downlink timing and that collection and hover are charged once. The request process still follows FL-round triggers and has no active deadline; this revision does not by itself establish exogenous-load capacity or deadline compliance.

## Completed results and figure style

The completion-driven campaign above is now complete on both datasets for all four methods. Its eight arms each passed the 200-FL / 45-request checks. Current figures are in `results/iot_uav_remoteclip_v4_event_mission/figures/`; earlier fixed-residence results below are historical comparisons.

`tools/plot_event_mission_reference_style.py` regenerates the four summary charts and a separate hovering-energy chart using the user's blue/orange reference palette, two additional matched colors, 14-point serif type, dotted grids and full frames. Every panel is square (the complete 1×2 canvas remains wide). There are no numerical bar annotations. All bar axes start at zero; energy, delay and latency use common scales across datasets. The restyle leaves metrics and registered simulation sources unchanged and records rendering checks in `reference_style_manifest.json`.

The accompanying `hover_energy_audit.csv` / `.json` reconcile actual transaction timings with all four energy components. For CapERA, Proposed/S-AC/AC-OF/S-OF+ hover for 171.092/201.529/209.618/411.080 seconds and consume 288.274/339.556/353.186/692.628 kJ in hovering. All methods have the same 1,538.214 kJ travel cost. Proposed saves 51.340/64.994/404.789 kJ in total energy relative to the three baselines, respectively. The S-AC difference is primarily 45 avoided broadcasts (29.514 seconds), plus 0.922 seconds of avoided uplink: at the declared ten-UAV hover power of 1,684.9 W, the total 30.436-second reduction saves 51.282 kJ of hovering. Computation still charges executed work; parallel communication does not remove that computation energy. These are model-based, single-seed results.

## Audit of earlier request service times

`tools/iot_fu_service_audit_v1/audit.py` verifies all 360 dataset/method/request records against their transmitting transactions. Results are in `results/iot_fu_service_audit_v1/`. The earlier request clock includes queue wait and uplink, but excludes compute and downlink. CapERA mean procedural latency was 0.1024/0.1547/1.0536/2.0025 seconds for Proposed/S-AC/AC-OF/S-OF+. A diagnostic that preserves the exact old processing order and adds declared compute/broadcast costs gives 3.3834/5.1117/6.0105/5.8661 seconds. These added-cost estimates do not re-evaluate the scheduler and must not be presented as new end-to-end simulation results.

All methods completed the same 45 requests in nine bursts; no burst overlapped the next and maximum pending requests were five. Therefore this workload does not measure maximum sustainable throughput. Requests have null active deadlines; archived old-calendar deadlines are not applicable. At five-step completion, 41/36/36/39 CapERA requests respectively met the existing KL<=0.01 reference-similarity diagnostic. Procedural completion and reference similarity are distinct, and neither certifies deletion. The source hashes, per-request timing, p95 values, quality diagnostics and observed radio-work efficiency are preserved in the audit artifacts.

## Latest global-IoT scenario

The latest rendering is `python tools/plot_mission_map_cluster.py`. It overlays five translucent snapshot enclosures and ten identity-preserving, piecewise-linear UAV paths including return. Numeric labels 1–5 indicate visit order; there are no textual captions, legends or axis names. Axis ticks remain in ground metres, including both endpoints. Blue stars are UAV positions, red stars are planning heads, and pale orange circles are IoT devices. Ellipses are visual grouping only, not modeled coverage. All positions match the registered deployment; the plotting update does not change the experiment.

`tools/iot_uav_deployment_v3_global_iot/` corrects the earlier assumption that devices are generated near preselected stops. It first samples 200 synthetic infrastructure sensors on cached OSM road, footway and cycleway geometry throughout the map, with spatial stratification across a 10×10 grid and at least 18 m device separation. All 100 spatial cells are represented. These are declared simulated devices, not measured infrastructure.

Only after generating the global device positions and packet demands does the method determine five balanced 40-device collection zones. Alternating balanced assignment and weighted centroids defines the coarse zones, and enumeration of the 24 closed tours from a fixed start chooses the shortest centroid tour. Ten UAV positions and a nominal planning head are then optimized per zone, with a 450 m allowed radius and the existing power/separation constraints. All 200 device IDs are preserved across partitioning. UAV identity is maintained throughout the five transitions, including return.

The latest figure is generated by `python tools/plot_mission_map_reference.py --mode iot --global-iot`; add `--download` once for its separate cache. IoT circles span the whole map. The 50 star placements are five time-separated snapshots of the same ten UAVs, not 50 concurrent UAVs. Heads are planning candidates until an actual learning run finishes. This scenario supersedes the two earlier spatial-demand designs below.

## Active global-IoT experiment

**Completed:** the recovery produced the three original-cap arms on both datasets; the energy-overrun continuation completed S-OF+ on both. Every final arm passed verification of 200 FL rounds and all 45 requests with five FU steps. Final charts and numeric summaries are in `results/iot_uav_remoteclip_v3_energy_overrun/figures/` and were pushed in commit `913f127`. Proposed has the highest retained mR on both datasets in this single seed, but S-OF+ has slightly lower retained loss. Proposed's modeled total-energy reduction is 4.78%/4.90% relative to S-OF+ (RSITMD/CapERA), and below 0.004% relative to either S-AC or AC-OF. The latter three methods have identical modeled total delay because they fit the same fixed service windows. These small quality differences do not establish statistical superiority. S-OF+ records 36 network-cap violations and 0.716907 J summed excess on each dataset. The following paragraphs retain the execution history and explain the changed constraint.

The user subsequently authorized continuing even when the network energy cap is exceeded. `tools/iot_vlm_energy_overrun_v1/launch.py` queues a separate continuation behind the active GPU campaign. It reuses the same frozen feature cache, common initial/proxy/reference checkpoints, and completed original-cap results for the other three methods. S-OF is rerun from the common initial checkpoint on both datasets. Its normal original-cap search is unchanged; if that fails, the continuation evaluates the longest permitted frame and selects the minimum-energy feasible head while allowing network sum-energy above 0.100 J. Individual UAV energy, peak power, coherence, total uplink duration and quantization-error bounds remain enforced.

Every transmission records the original cap, compliance and excess joules. Total-energy accounting uses the actual modeled transmission energy, without clipping to the old cap. Comparison charts label this variant `S-OF+`; their numeric summaries include excess totals, maximum transaction energy and violation counts. This is explicitly an energy-overrun sensitivity comparison, not evidence that all four methods are feasible under the original cap. The formerly failing first-site fixture now transmits at 0.119914 J (0.019914 J excess); a previously feasible site's vector, head, duration, error and energy remain identical. Exact OFDMA allocation results are memoized by complete inputs to avoid repeating unchanged geometry/payload optimizations.

The initial global-IoT attempt stopped before any comparison arm completed. RSITMD finished both common 20-round prefixes, then its all-method preflight stopped at S-OF. With the first site's actual FL payload and the maximum 1.31072-second duration, the exact discrete OFDMA allocator requires at least 0.119914 J over the ten candidate heads, exceeding the unchanged 0.100 J network cap. This is a transmission-feasibility finding, not a learning-quality result. CapERA separately failed random seeking to frame 137 of `Baseball_043 .mp4`; sequential decoding successfully reads all 138 frames, including the requested last frame.

`tools/iot_vlm_recovery_v1/launch.py` starts the separate recovery in `results/iot_uav_remoteclip_v3_recovery/`. Its protocol preserves source/input hashes and hashes every reused completed RSITMD prefix/cache artifact. CapERA uses sequential decoding at the same eight uniform indices. Preflight records each transport case independently, and a failed probe no longer prevents unrelated arms from running. Actual arm constraints and assertions remain unchanged; failure is never assigned a zero loss/energy or replaced by a fabricated successful result. No power, energy, tolerance, deadline, geometry, partition, learning-rate or method setting is changed. Figures requiring all eight completed arms remain incomplete if an arm cannot finish.

`tools/iot_vlm_v3_global_iot/launch.py` starts the fresh RSITMD/CapERA campaign in `results/iot_uav_remoteclip_v3_global_iot/`. Both datasets and all four methods share the registered global-IoT geometry. The schedule assigns 36 adaptation FL updates to each of five zones (180 plus the initial 20) and retains all 45 five-step FU requests. Preflight checks exercise both joint and sequential queues and independently check the fifth-site computing ledger, all five flight legs and 50 seconds of acquisition hovering.

The detached run writes `status.json`, `run.log` and terminal `finished.json` or `failure.json`. Completed eight-arm comparisons produce loss, total-energy and delay figures. Actual-head maps use the cluster rendering above. The publisher pushes only completed chart artifacts and numeric summaries. The prelaunch protocol archive and amendment record retain the visual-callback and accounting-check changes made before any training started; numerical settings and registered inputs are unchanged by that amendment.

```bash
env PYTHONPATH=.cache/uav_experiment_deps .venv/bin/python tools/iot_vlm_v3_global_iot/experiment.py --preflight-only
env PYTHONPATH=.cache/uav_experiment_deps .venv/bin/python tools/iot_vlm_v3_global_iot/launch.py
cat results/iot_uav_remoteclip_v3_global_iot/status.json
```

The launcher refuses a campaign that has already started. The results directory contains immutable registration, a separate log and inputs for reproducibility; it is not a reuse of the earlier four-region training run.

## Earlier five-region scenario

The user requested a wider five-region route after reviewing the four-region figure. The four-region training process was stopped with its records preserved. `tools/iot_uav_deployment_v2_five_sites/` creates an independent five-region deployment: Government Complex Forest Park B, Hanbat Arboretum East, Expo Science Park, Gapcheon West Bank Cycleway, and Mannyeon Neighborhood Park, anchored to cached OSM ways. There are 40 declared IoT devices per region (200 total) and the same ten UAVs across all regions. Optimized centroid spans are approximately 1.40 km east–west and 1.42 km north–south, with a 4.59 km closed centroid route.

`python tools/plot_mission_map_reference.py --mode iot --five-sites` draws the latest square 1×2 map. Add `--download` once for its separate satellite cache. The viewport covers all UAV/IoT positions plus padding and is rounded to 2000×2000 ground metres; ticks include both 0 and 2000 on both axes. The circles/stars/head colors match the accepted reference design. This initial export shows planning heads; actual new-run heads require the five-region learning experiment to finish.

The five-region acquisition, power, head enumeration and continuous transition separation checks passed. The new learning schedule retains 180 post-prefix FL updates, distributed as 36 updates at each region, with the same total of 45 deletion requests and five FU steps per request. Geometry and learning artifacts use distinct versioned directories; no four-region result is relabeled as a five-region result.

## Earlier four-region implementation

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

Add `--square` for a 9.4×5.5-inch canvas with square map panels. This expands the north/south viewport rather than stretching the map. The square satellite image has a separate cache; add `--download` once to populate it. The updated IoT circles use a lighter orange fill and muted outlines. UAV positions, device positions and heads remain identical across the visual variants.

## Earlier four-region experiment

`tools/iot_vlm_v1/experiment.py` runs fresh RSITMD and CapERA evaluations with the same registered RemoteCLIP/adapters, data partitions, four methods, 200 FL rounds and 45 five-step FU requests as the base campaign. The new geometry and channel matrices are common to all arms. Neither the earlier two-row results nor nominal planning-head costs are relabeled as new method results.

Communication energy now also includes ground-device PA/circuit costs and UAV reception. Each region adds ten seconds of acquisition hovering before FL/FU service. Total energy covers ground transmission plus the ten UAVs' communication, computing and flight. The learning simulator's service-only clock is distinct from postprocessed deployment delay, which includes acquisition and modeled computing time.

```bash
env PYTHONPATH=.cache/uav_experiment_deps .venv/bin/python tools/iot_vlm_v1/launch.py
cat results/iot_uav_remoteclip_v1/status.json
tail -n 30 results/iot_uav_remoteclip_v1/run.log
```

The launcher uses a detached session and persistent log. It refuses a duplicate campaign, uses the shared GPU lock and verifies registered source/input hashes. When all eight dataset/method arms complete, it exports the 1×2 loss/total-energy/total-delay charts and pushes only completed figures and chart summaries to the configured `ckim48/ck_ova` repository. Existing staged work defers automatic publication; failures are recorded rather than substituted with zeros.

The local manuscript extension is `manuscript/iot_uav_extension.tex`; it is inserted into `manuscript/iot_uav_draft.tex` while preserving the original manuscript. The draft distinguishes planning heads from actual FL/FU heads and discloses the synthetic physical scenario versus real-data replay. It does not assert numerical learning results before the new experiment completes.
