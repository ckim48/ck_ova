# Yeouido mission map

[PDF](mission_map_1x2.pdf) · [PNG](mission_map_1x2.png) · [SVG](mission_map_1x2.svg)

Simulation scenario in Yeouido, Seoul, South Korea: satellite imagery (left) and synthetic IoT deployment with planned UAV positions and trajectories (right). Circles denote IoT devices, blue stars denote UAVs, and orange stars denote nominal planning heads. Numbers indicate five sequential collection stops of the same ten UAVs. The 1800 × 1800 m mission region has 100 m map padding on each side. The geometry is newly planned for Yeouido; the existing FL/FU performance results use the previous Daejeon scenario. Imagery: Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community. Map data: © OpenStreetMap contributors.

The 200 synthetic sensors are stratified on OSM park paths before deriving five balanced 40-device collection zones. UAV positions, associations and nominal packet heads are then optimized using the existing placement solver. Heads are planning outputs, not replayed or claimed FL/FU AirComp head decisions. Straight-line transitions are checked for 22 m inter-UAV separation, but building clearance, terrain and flight permissions are not modeled. This is a simulation map, not an operational flight plan. No model-quality or resource results have been rerun.

The old map is retained in `previous_daejeon/`. The new paper alias is `../figures/mission_map_1x2.pdf`. Both square panels retain the prior physical dimensions and 12 pt ticks.

OSM source and license: [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL. Cached imagery metadata and OSM query are in `data/korea_mission_yeouido_v1` at repository root.

Reproduce: `.venv/bin/python tools/uav_convergence_v2/yeouido_mission_map_v1.py --publish`.
