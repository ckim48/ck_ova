# Four-method UAV/VLM comparison

The original Proposed, ISCC-P2 and AFL-OTA results are unchanged. F-SalUn is added as a real FU-algorithm transfer; it is not a weakened version of Proposed. See [the existing transport and workload definitions](../expanded_sweeps/README.md).

**F-SalUn:** [SalUn, ICLR 2024, Algorithm 1](https://proceedings.iclr.cc/paper_files/paper/2024/file/ec4d2e436794d1bf55ca83f5ebb31887-Paper-Conference.pdf), transferred to federated rank10 VLM adapters with ISCC-P2 digital FDMA. At each request start, form a median hard mask from the forget-set contrastive gradient; keep that mask for five prescribed FU cycles. Wrong-label classification is adapted to randomly shifted video–text assignments, combined with retained CE. The backbone and temperature stay frozen. Equal-client FedSGD and dense FP32 transmissions remain unchanged. Saliency, backward passes and mask operations are charged. SalUn is an ICLR FU paper, not a UAV-FL paper; this is an explicit adaptation, not an official benchmark reproduction.

All four methods use identical data, initial weights, partitions, requests, routes and physical coefficients within each condition. All 45 requests and225 FU cycles finish. A digital QoS overrun is retained and reported; completion does not certify every original bound. The fifth sweep changes total FL budget, not the reporting reference200.

Total energy is active communication + computation + hovering. Fixed cached features imply adapter-only computation. The main delay is active FL/FU service time; mission delay is separate. No raw samples, checkpoints or transaction records are published.

Single development seed. SalUn has higher retained mR but also larger retrain KL in the baseline; its CapERA worst-request KL exceeds0.01. Mean KL below0.01 must not be mistaken for every-request compliance. All measured outcomes and original comparators remain visible. A separate [learning diagnosis and continuation bundle](../learning_quality/index.html) tests changes to Proposed; they are not silently substituted into these fixed-workload curves.

| Dataset | Method | mR (%) | Loss | Mean KL | Worst request KL | Energy (kJ) | Delay (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 16.328 | 5.42157 | 0.000224149 | 0.000342769 | 204.451 | 121.204 |
| CapERA | ISCC-P2 | 16.901 | 5.34352 | 0.00378861 | 0.00583252 | 649.962 | 384.304 |
| CapERA | AFL-OTA | 16.966 | 5.34348 | 0.00379123 | 0.00583789 | 252.882 | 149.912 |
| CapERA | F-SalUn | 17.552 | 5.28097 | 0.00966762 | 0.0161791 | 649.960 | 384.302 |
| AirScape | Proposed | 13.542 | 5.59488 | 0.000234091 | 0.000290354 | 204.751 | 121.383 |
| AirScape | ISCC-P2 | 13.737 | 5.55939 | 0.000838523 | 0.00137075 | 649.305 | 383.915 |
| AirScape | AFL-OTA | 13.737 | 5.55937 | 0.000838547 | 0.00137128 | 253.066 | 150.021 |
| AirScape | F-SalUn | 13.737 | 5.52985 | 0.00312732 | 0.00477099 | 649.307 | 383.916 |

Rebuild: `python tools/uav_convergence_v2/learning_quality_report.py four`. Requires local registered source experiments.
