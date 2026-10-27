# Bandwidth display, hybrid energy and validation loss audit

Bandwidth axes now start at 1 MHz, with regular ticks 1, 2, 3, 4. Measured displayed points are 1, 2 and 4 MHz. The 0.25/0.5 MHz outcomes remain in the raw data; no value was invented for the 3 MHz tick. The AirScape H-OVA 4 MHz infeasibility remains marked.

## AC-OF energy

AC-OF uses AirComp for FL and integer-subcarrier OFDMA only for FU. S-OF uses digital OFDMA for both. At 1 MHz, their FU transaction counts, uplink times and RF energy are equal on both datasets. No OFDMA cost was omitted from AC-OF. Its lower energy comes from its analog FL phase. Each transaction's radio energy, PA/circuit/broadcast cost, recorded MAC cost and hover duration was independently summed and reconciled with the plotted ledger.

| Dataset | Method | FL uplink (s) | FU uplink (s) | Joint uplink (s) | Communication (J) | Hover share (%) | Service energy (kJ) | Service delay (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 2.744 | 0.000 | 0.922 | 229.633 | 99.8862 | 204.261 | 121.092 |
| CapERA | S-AC | 3.666 | 0.922 | 0.000 | 286.872 | 99.8865 | 255.601 | 151.529 |
| CapERA | AC-OF | 3.666 | 9.011 | 0.000 | 306.896 | 99.8833 | 269.255 | 159.618 |
| CapERA | S-OF | 205.128 | 9.011 | 0.000 | 559.325 | 99.8904 | 609.051 | 361.080 |
| AirScape | Proposed | 2.744 | 0.000 | 0.922 | 229.685 | 99.8862 | 204.255 | 121.089 |
| AirScape | S-AC | 3.666 | 0.922 | 0.000 | 286.865 | 99.8865 | 255.594 | 151.525 |
| AirScape | AC-OF | 3.666 | 8.806 | 0.000 | 306.678 | 99.8833 | 268.903 | 159.410 |
| AirScape | S-OF | 205.128 | 8.806 | 0.000 | 559.119 | 99.8904 | 608.699 | 360.871 |

The modeled ten-UAV hover power is 1,684.9 W. Over 99.8% of displayed service energy is hovering, so energy follows service duration closely. This is not an RF-only energy comparison. A 32-bit aggregate broadcast at 1 Mbps is charged after every nonzero transaction. For baseline Proposed, broadcasts occupy 117.401 s of about 121.09 s service; that downlink assumption, the reduced number of broadcasts under joint scheduling, and the fixed hover-power model strongly influence the reported advantage. These are modeled scenario costs, not hardware-calibrated measurements. Travel/collection/preflight are excluded.

## Loss definition and learning progress

The old FL loss label referred to held-out retained-query retrieval negative log likelihood, not the local minibatch training loss. Its label is now Validation loss; all values and filenames are preserved. Initial and final checkpoints were reevaluated and the loss independently recomputed from raw gallery scores. Image-to-text uses the probability mass of all positive captions, text-to-image uses the correct video probability, and the two directional mean negative log probabilities are averaged. With N videos and equal caption counts, uniform predictions give log(N): 6.22059 for CapERA and 6.41999 for AirScape. A value near 5 is not by itself a bug, nor is contrastive loss meaningless for a VLM. [CLIP (ICML 2021)](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) uses symmetric contrastive cross entropy; this project's multi-caption validation reduction is defined by its own evaluator.

| Dataset | Uniform loss | Initial validation loss | Final validation loss | Initial mR (%) | Final mR (%) | Fixed train-batch loss before / after |
|---|---:|---:|---:|---:|---:|---:|
| CapERA | 6.2206 | 5.7128 | 5.3607 | 15.4167 | 16.9010 | 2.4827 / 2.2849 |
| AirScape | 6.4200 | 5.7524 | 5.5531 | 13.0859 | 13.6719 | 2.3154 / 2.1527 |

A decrease in this log loss indicates improved correct-match probability, but the observed retained-mR gains are modest. This experiment freezes RemoteCLIP, pools eight frame features, and trains only 20,480 rank-10 adapter coordinates. It does not fine-tune the entire VLM. The fixed learning-rate schedule reaches zero at round 200, so a flat endpoint is not evidence that the task is fully optimized. Logit scale is frozen and unchanged, so loss improvement is not obtained by tuning temperature.

An existing separate FL-only, ideal-aggregation diagnostic restarted LR at 0.001 after round 200 and continued to round 1000: CapERA mR increased 16.4193→22.5130%, and AirScape 13.5417→15.4948%. This indicates remaining learning potential under that changed training setup. It changes both duration and LR, omits FU and radio, and cannot substitute for the main 200-round FL/FU method comparison. No new training or parameter updates were performed for this audit.

All 96 original plotted numerical series remain unchanged. Source hashes, phase-level accounting and checkpoint reevaluation are in `audit.json`. Rebuild with `.venv/bin/python tools/uav_convergence_v2/energy_loss_audit.py`.
