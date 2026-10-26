Paper figures — 2026-10-26

All PNG/PDF/SVG figures in this folder use completed v4 event-mission results.
Summary bars: 7.4 × 3.735 inch canvas; 10% lower than the preceding version.
Total energy uses 1800–2300 kJ with explicit axis breaks; the full-scale file starts at zero.

loss_convergence_1x2 / mr_convergence_1x2: actual saved retained-set evaluations
from FL round 5 through 200. The first 20 rounds are common initialization.
FU-only evaluations may share an FL-round x coordinate and are not removed.
Markers distinguish overlapping methods; values are not shifted or smoothed.
This single-seed trajectory is not proof of asymptotic convergence.
convergence_samples.csv provides every plotted sample and its source file.

mission_map_1x2: larger 10.4 × 5.35 inch map. Blue enclosures are visual
groupings of five snapshots of the same ten UAVs, not radio coverage.
Five centroid arrows show visits 1–2–3–4–5 and return to 1. Faint paths preserve
individual UAV identity. Orange circles are synthetic IoT demand; stars are
recorded optimized positions. Red stars are actual heads from the earliest
positive-duration Proposed CapERA transmission in each region of the v4 run.
Head selections are checked against the completion-driven mission timeline.
Satellite and map data attribution and provenance are in mission_map_1x2.json.

Reproduce: MPLCONFIGDIR=/tmp/ova_mpl_style .venv/bin/python tools/build_paper_figures_20261026.py
