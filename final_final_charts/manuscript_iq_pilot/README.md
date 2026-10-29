# Manuscript I/Q pilot

Same 200 FL rounds and 45 requests × 5 FU steps; paired development seed 0. Existing Proposed and its published charts are unchanged.

IQ-only keeps the current P-KS learning policy and changes to the manuscript I/Q transceiver and empirical task-distortion design. Manuscript-IQ also uses the manuscript cosine-forgetting/embedding-consistency objective, uniform FL sampling, and mini-batch-size aggregation weights.

The manuscript omits numerical curvature bounds, loss-perturbation budgets, expansion grid and initialization. The operational choices are recorded in protocol.json, before observing the candidate results. HVP-based curvature is an empirical surrogate, not a certified Hessian bound or a forgetting guarantee. Both I/Q streams count toward the SAME block power limit. BCD solves are local, not globally optimal.

Primary energy/delay use the existing active FL/FU scope including downlink, circuits and all executed curvature costs; travel/collection/common preflight are excluded for every method. The narrower manuscript formula is exported separately in metrics.csv; mixing these scopes would be misleading. Solver wall time is reported separately and is not calibrated to UAV energy. All KL values compare with the same reporting-only retained retraining reference200.

Curvature and update-norm control information is assumed available to the head; its signaling payload is not modeled. The simulated curvature probes access the local training objectives jointly, so this pilot is not a fully decentralized curvature-estimation protocol.

| Dataset | Method | Status | mR (%) | Loss | Forget KL | Energy (kJ) | Delay (s) | Uplink (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | completed | 17.591 | 5.22885 | 0.018232 | 204.277 | 121.102 | 3.666 |
| CapERA | IQ-only | completed | 17.786 | 5.22928 | 0.018157 | 214.582 | 127.208 | 9.626 |
| CapERA | Manuscript-IQ | completed | 17.044 | 5.34996 | 0.003341 | 214.013 | 126.871 | 9.298 |
| AirScape | Proposed | completed | 13.802 | 5.51150 | 0.007320 | 204.267 | 121.096 | 3.666 |
| AirScape | IQ-only | completed | 13.802 | 5.51147 | 0.007327 | 211.461 | 125.358 | 7.803 |
| AirScape | Manuscript-IQ | completed | 13.672 | 5.55784 | 0.001062 | 211.414 | 125.330 | 7.782 |

The radio-only change leaves model quality close to Proposed in this seed. The full manuscript changes the learning/forgetting tradeoff: lower reference KL accompanies lower retained mR. It does not dominate Proposed in energy or delay under the declared constants and BCD initialization. The two interventions do not isolate the individual effects of the FU objective, FL sampler and aggregation weights.

Reproduce with `manuscript_iq_study_v1.py run --variant iq_only`, `manuscript_iq_study_v1.py run --variant paper_iq`, then `manuscript_iq_report_v1.py` in `tools/uav_convergence_v2`, using the registered local source experiments.
