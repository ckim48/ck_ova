# Same-time FL/FU continuation

Each method first completes the original200 FL rounds and45 requests (225 FU cycles), then spends remaining active service time on retained-only FL. Every method uses the same restarted constant0.0035 LR; the original cosine schedule ended at zero. Final-site channels remain fixed. P-KS is the predeclared KL-gate + selective-FL16 candidate, not the current Proposed. It is not promoted to the main method because its fixed-workload CapERA mean KL and worst-request KL exceed0.01.

No new deletion requests arrive during continuation. This tests use of saved time, not an online request-capacity benchmark. The FU-progress chart counts executed prescribed steps, not certified successful deletions.

Quality curves use the last observed completed checkpoint, without interpolation. Before200 rounds the original evaluations are sparse; after200 every completed model is evaluated. CSV includes actual completed FL/FU counts, quality-checkpoint time and lag. At400 seconds all methods have completed all deletions and have dense continuation observations.

Energy includes all active communication, compute and hover. At the same hover duration energy is similar; energy benefits must be assessed by time/energy to a quality target. Time-to-target tables require all45 FU procedures to have finished and retain unreached targets as missing. These are descriptive developmental thresholds, not tuning or stopping inputs.

The old reference200 KL is not a fair standalone forgetting score after hundreds of additional FL updates. Thus no such KL curve is used to claim forgetting superiority here. A separate final table extends the common retained-only reference to each actual final FL count using exact sample-weighted updates and the identical continuation LR. This is a round-matched common reference, not a method-specific oracle for selective/equal-client FL, and provides no deletion certificate. It is reporting-only. Final checkpoints are the first completed update beyond400 seconds; their actual timestamps are explicit. The common400-second utility table uses the last checkpoint at or before400.

Single development seed and frozen-feature adapter simulation; no official full-dataset score or statistical superiority claim.

| Dataset | Method | mR at400s (%) | Loss | Completed FL rounds | FU cycles |
|---|---|---:|---:|---:|---:|
| CapERA | Proposed | 24.010 | 4.38452 | 612 | 225 |
| CapERA | P-KS | 24.284 | 4.27001 | 612 | 225 |
| CapERA | ISCC-P2 | 17.161 | 5.31122 | 208 | 225 |
| CapERA | AFL-OTA | 23.893 | 4.38927 | 572 | 225 |
| CapERA | F-SalUn | 17.461 | 5.24633 | 208 | 225 |
| AirScape | Proposed | 16.667 | 5.04427 | 611 | 225 |
| AirScape | P-KS | 16.797 | 4.99559 | 612 | 225 |
| AirScape | ISCC-P2 | 13.802 | 5.54370 | 208 | 225 |
| AirScape | AFL-OTA | 16.536 | 5.05108 | 572 | 225 |
| AirScape | F-SalUn | 13.672 | 5.51333 | 208 | 225 |

## Final round-matched common-reference diagnostics

| Dataset | Method | FL rounds | Actual time (s) | Mean KL | Worst request KL |
|---|---|---:|---:|---:|---:|
| CapERA | ISCC-P2 | 209 | 400.628 | 0.00445988 | 0.00699749 |
| CapERA | F-SalUn | 209 | 400.626 | 0.0112352 | 0.0190511 |
| CapERA | AFL-OTA | 573 | 400.256 | 0.0135082 | 0.0510025 |
| CapERA | Proposed | 613 | 400.571 | 0.000139811 | 0.000245619 |
| CapERA | P-KS | 613 | 400.469 | 0.0825097 | 0.12267 |
| AirScape | ISCC-P2 | 209 | 400.239 | 0.000927828 | 0.00158564 |
| AirScape | F-SalUn | 209 | 400.240 | 0.00345502 | 0.00544516 |
| AirScape | AFL-OTA | 573 | 400.170 | 0.00380704 | 0.00625695 |
| AirScape | Proposed | 612 | 400.073 | 5.13125e-05 | 6.95185e-05 |
| AirScape | P-KS | 613 | 400.463 | 0.0710471 | 0.0960005 |

[Time and energy to each descriptive target](data/time_to_target.csv). Rebuild: `python tools/uav_convergence_v2/learning_time_report.py`.
