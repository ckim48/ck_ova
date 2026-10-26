# UAV video-language experiment and mission figure

This snapshot adds the base experiment implementation and mission figures. Its first RemoteCLIP run stopped during initial training and is preserved without final metrics. The active successor uses [IoT-driven deployment](iot_uav_deployment.md) and a separate output directory; final comparison metrics are produced by that campaign.

## Mission figure

`tools/plot_mission_map.py` defaults to `results/figures/mission_satellite_uav_snapshots_1x2.{png,pdf,svg,json}`. The left panel uses cached Esri satellite imagery and the right uses OpenStreetMap geometry over the same extent. Attribution and image provenance are retained in the companion JSON.

The four snapshots select the earliest feasible positive-duration transmission at each region in the completed historical RSITMD/Proposed run (`uav_single_model_two_dataset_v1`), not the ongoing RemoteCLIP/CapERA experiment. Actual prescribed hover positions are checked against every matching row of `trajectory.csv`. The selected head is read from each transaction's `radio.head`; golden double circles highlight it. Display IDs U1–U10 map to logged indices 0–9. Regions show the same ten UAVs at different times, not forty simultaneous UAVs or fixed regional heads. These are simulation records, not flight telemetry. A different completed run can be supplied with `--transactions PATH`.

The clean figure shows only UAV numbers, with light circular markers and sans-serif numerals. Region/time/head text, panel captions and compass/scale labels are omitted; times and all provenance remain in the companion JSON. The recorded formations are two rows of five, rotated and spaced differently at each site. This is a prescribed simulation geometry, not a claim that real UAV swarms must fly in two rows.

The earlier schematic is preserved as `mission_satellite_swarm_cluster_1x2` and can be regenerated with `--mode schematic`. Its recorded four swarm centroids and route are preserved, but its ten circular UAV markers at each stop use a deterministic dispersed schematic layout. Those display positions are not individual simulated positions. Both original formation positions and schematic positions are recorded in its JSON; the simulation inputs are unchanged.

Regeneration requires the local `results/overnight_energy_v2/korea_mission_v1/{mission_spec.json,trajectory.csv}` and `data/korea_mission_daejeon_v1/` map/imagery cache. Run `python tools/plot_mission_map.py`; `--download` refreshes the satellite cache.

## New experiment

- Data: [official CapERA annotations](https://github.com/yakoubbazi/CapEra) with [ERA UAV videos](https://lcmou.github.io/ERA_Dataset/), plus RSITMD as the remote-sensing comparator. CapERA has 1,473 training and 1,391 test clips. A 256-clip validation subset comes only from the official training split; official test clips are not used in adaptation.
- Backbone: [RemoteCLIP ViT-B/32](https://github.com/ChenDelong1999/RemoteCLIP), [IEEE TGRS 2024](https://doi.org/10.1109/TGRS.2024.3390838). Eight-frame normalized mean pooling and rank-10 residual adapters are this project's adaptation. CapERA itself is a Remote Sensing dataset publication, not an IEEE publication.
- Comparison: one paired seed, four existing methods, fresh initial 20 rounds, proxy 20 rounds and reference 200 rounds per dataset. Each arm reaches 200 FL rounds and processes 45 requests with five FU steps each, subject to feasibility checks. This is a whole-system comparison with the existing objectives, not an isolated transport ablation.
- Labels: Proposed, S-AC (separate AirComp), AC-OF (AirComp FL / OFDMA FU), S-OF (sequential OFDMA).
- Accounting: adapter work is counted per data-owning UAV. Cold training-feature construction is charged per independent deployment. `energy.py` declares communication circuit/PA, compute-cycle and rotary-wing propulsion assumptions. Server GPU time is not converted to UAV energy. Hover time expands when modeled work exceeds the prescribed service window. Results are modeled scenario estimates, not hardware measurements.
- Figures: automatic 1×2 bar plots for loss, total energy and total delay once all eight dataset/method combinations complete. Missing or failed arms are recorded instead of plotted as zero. Figure annotations contain values and labels; explanatory footers are omitted.

The implementation imports the existing frozen four-method campaign. Execution needs its local registered schedules, mission/channel inputs, original datasets and the pinned RemoteCLIP checkpoint/provenance. Dataset videos, model weights, caches and live per-round logs are not part of this source/chart snapshot.

The verified environment uses Python 3.11, torch 2.6.0+cu124, torchvision 0.21.0, open_clip_torch 2.26.1, timm 0.9.16, ftfy 6.2.3 and opencv-python-headless 4.10.0.84. Existing baseline dependencies also apply. Local added packages are isolated in `.cache/uav_experiment_deps`.

```bash
env PYTHONPATH=.cache/uav_experiment_deps .venv/bin/python tools/uav_real_vlm_v1/experiment.py --preflight-only
env PYTHONPATH=.cache/uav_experiment_deps .venv/bin/python -u tools/uav_real_vlm_v1/experiment.py
```

The full runner uses a shared GPU lock, registers source/input hashes, refuses duplicate campaigns and writes status into `results/uav_remoteclip_capera_v1/`. The feature-progress callback failure before any training update is preserved locally in a separate preflight-failure directory; its correction did not change training settings or select results.
