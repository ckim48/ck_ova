# Retrieval rank under limited energy

The current 650 kJ figure remains unchanged. The middle 300 kJ case is the main illustrative variant; 200 and 400 kJ are published alongside it. Existing aggregate curves were inspected before choosing this exploratory grid; it is not an independent confirmation experiment. All four methods and all retained queries are included.

Exact replays reproduce original final parameters and recorded learning trajectories. Snapshots are the last completed transactions within each budget, including transactions without a previous evaluation. Their actual full-gallery scores were recomputed; no mR-to-rank conversion or interpolation is used.

Energy includes active communication, adapter computation and hovering, with the same common warmup and travel exclusions as the prior study. After 200 FL rounds and all 45 FU procedures, the unchanged retained-only continuation restarts the common 0.0035 learning rate. The round-triggered request schedule means equal energy does not imply equal request exposure. FU procedure completion does not certify forgetting, and KL is not matched here.

## 200 kJ

[PDF](200kJ/19_all_method_rank_boxplot_1x2.pdf) · [Caption](200kJ/19_all_method_rank_boxplot_1x2_caption.txt)

| Dataset | Method | Energy (kJ) | FL rounds | FU arrived | FU completed | Retained mR (%) | Median rank |
|---|---|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 199.71 | 195 | 45 | 45 / 45 | 17.591 | 25 |
| CapERA | ISCC-P2 | 198.78 | 73 | 15 | 15 / 45 | 15.911 | 27 |
| CapERA | AFL-OTA | 199.73 | 162 | 35 | 35 / 45 | 16.615 | 26 |
| CapERA | F-SalUn | 198.78 | 73 | 15 | 15 / 45 | 16.367 | 27 |
| AirScape | Proposed | 199.70 | 195 | 45 | 45 / 45 | 13.802 | 60 |
| AirScape | ISCC-P2 | 198.78 | 73 | 15 | 15 / 45 | 13.477 | 63.5 |
| AirScape | AFL-OTA | 199.92 | 162 | 35 | 35 / 45 | 13.737 | 58.5 |
| AirScape | F-SalUn | 198.78 | 73 | 15 | 15 / 45 | 13.542 | 63 |

## 300 kJ

[PDF](300kJ/19_all_method_rank_boxplot_1x2.pdf) · [Caption](300kJ/19_all_method_rank_boxplot_1x2_caption.txt)

| Dataset | Method | Energy (kJ) | FL rounds | FU arrived | FU completed | Retained mR (%) | Median rank |
|---|---|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 298.98 | 283 | 45 | 45 / 45 | 21.094 | 19 |
| CapERA | ISCC-P2 | 298.11 | 103 | 20 | 20 / 45 | 16.406 | 27 |
| CapERA | AFL-OTA | 298.96 | 241 | 45 | 45 / 45 | 18.294 | 24 |
| CapERA | F-SalUn | 298.11 | 103 | 20 | 20 / 45 | 16.602 | 26 |
| AirScape | Proposed | 298.97 | 283 | 45 | 45 / 45 | 14.714 | 45.5 |
| AirScape | ISCC-P2 | 299.80 | 100 | 25 | 25 / 45 | 13.542 | 61 |
| AirScape | AFL-OTA | 299.23 | 241 | 45 | 45 / 45 | 13.737 | 55.5 |
| AirScape | F-SalUn | 299.80 | 100 | 25 | 25 / 45 | 13.607 | 60.5 |

## 400 kJ

[PDF](400kJ/19_all_method_rank_boxplot_1x2.pdf) · [Caption](400kJ/19_all_method_rank_boxplot_1x2_caption.txt)

| Dataset | Method | Energy (kJ) | FL rounds | FU arrived | FU completed | Retained mR (%) | Median rank |
|---|---|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 399.39 | 371 | 45 | 45 / 45 | 22.995 | 16 |
| CapERA | ISCC-P2 | 397.86 | 132 | 25 | 25 / 45 | 16.771 | 26 |
| CapERA | AFL-OTA | 399.15 | 330 | 45 | 45 / 45 | 20.521 | 19 |
| CapERA | F-SalUn | 397.86 | 132 | 25 | 25 / 45 | 17.227 | 26 |
| AirScape | Proposed | 399.38 | 371 | 45 | 45 / 45 | 15.169 | 38.5 |
| AirScape | ISCC-P2 | 399.48 | 129 | 30 | 30 / 45 | 13.607 | 60 |
| AirScape | AFL-OTA | 399.37 | 330 | 45 | 45 / 45 | 14.909 | 45.5 |
| AirScape | F-SalUn | 399.48 | 129 | 30 | 30 / 45 | 13.737 | 56.5 |

Reproduce checkpoints, inference and figures with the three `limited_energy_*_v1.py` scripts in `tools/uav_convergence_v2`. Raw checkpoints, full score matrices and features remain local.
