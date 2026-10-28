# Parallel FL/FU advantage: Proposed (P-KS)

The parallelism is **FL/FU co-aggregation and model application in the same service transaction**. Local FL and FU gradient computation is charged as executed work on each UAV; the simulator does not assume two gradients execute simultaneously on one processor.

## Matched scheduling ablation

**S-PKS** uses the identical P-KS objectives, training-only sample selection, initialization, data, requests, optimized AirComp/head-selection algorithm, learning-rate rules and physical coefficients. Only scheduling changes: pending FU is served immediately in separate transactions (`parallel=False, period=0`), then FL resumes. Multiple FU clients still aggregate via AirComp. No artificial periodic wait or OFDMA substitution is introduced. Scheduling changes model states and optimized physical decisions as consequences. Both methods finish 200 total FL rounds and 225 request-local FU steps for 45 requests.

Four fresh runs cover the two modes and two datasets. The new parallel controls reproduce the original P-KS final parameters, evaluation, transaction order and physical timings/energy exactly. All serial outcomes remain reported. Gains are for the same prescribed work, not proof of equal final quality or certified unlearning.

The mean latency of an individual FU request is approximately 3.384 seconds in both modes. Immediate serial service also begins FU promptly; its disadvantage here is pausing FL during that service. Thus the approximately 50% first-window reduction refers to completing FU and five FL updates together, not halving FU-only latency. The full-workload reduction is approximately 20.1%.

## Figures

- 01: the first chronological FU release, containing five requests × five FU steps, plus five additional FL updates. Both modes start after FL round 25. Colored intervals are actual service transactions including compute, uplink and broadcast. Coincident FL/FU lanes mean the updates are jointly served and applied; they do not claim simultaneous CPU execution. Every mode completes exactly five FL updates and 25 request-local FU steps in this window. This local window is about half as long under parallel service; it is not the total-mission speedup.
- 02–03: actual full-workload delay and energy for Proposed versus S-PKS.
- 04–05: completed FL rounds and prescribed FU steps versus active time for the four main methods. After all initial work finishes, each uses remaining time for retained-only FL under the already declared common continuation learning rate. The FL axis starts with 20 shared warmup rounds already complete. No new FU requests arrive during continuation. Completed steps describe service, not deletion quality.

In the baseline, Proposed uses 135 FL-only plus 45 joint transactions. S-PKS and the literature transfers use 180 FL-only plus 45 FU-only transactions. Actual broadcasts are 179 versus 224 because the terminal zero-learning-rate FL update does not send a model. Joint service covers all 225 request-local FU steps. A transaction serving five requests counts as five FU steps, not five broadcasts.

The literature curves compare complete adapted methods with different learning and radio rules. Their entire gap cannot be assigned solely to parallelism. The matched S-PKS comparison isolates this scheduling change within the P-KS design. Arrival releases are tied to FL ordinals, not a common exogenous wall-clock process; this is not a stochastic-arrival capacity benchmark.

Total energy is active communication + executed adapter computation + hover. Delay excludes flight, collection and preflight. Fixed feature caching and the known-cohort training-proxy assumption remain. Single development seed; no statistical superiority or deletion guarantee. The reported KL remains D_KL(retrain || candidate) against the common retained-only round-200 reference.

| Dataset | Method | Delay (s) | Energy (kJ) | First-release window (s) | mR (%) | Loss | Mean KL | Worst KL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 121.102 | 204.277 | 3.384 | 17.591 | 5.22885 | 0.0182323 | 0.0261308 |
| CapERA | S-PKS | 151.538 | 255.616 | 6.766 | 17.487 | 5.22791 | 0.0186361 | 0.0266084 |
| AirScape | Proposed | 121.096 | 204.267 | 3.384 | 13.802 | 5.51150 | 0.00731969 | 0.0137718 |
| AirScape | S-PKS | 151.532 | 255.606 | 6.766 | 13.867 | 5.50903 | 0.00768377 | 0.0143767 |

| Dataset | Full-workload delay reduction | Energy reduction | First-release window reduction | Broadcast reduction |
|---|---:|---:|---:|---:|
| CapERA | 20.08% | 20.08% | 49.98% | 20.09% |
| AirScape | 20.09% | 20.09% | 49.99% | 20.09% |

## Reproduction

```sh
.venv/bin/python tools/uav_convergence_v2/pks_parallel_ablation.py run
.venv/bin/python tools/uav_convergence_v2/parallel_advantage_report.py
.venv/bin/python tools/uav_convergence_v2/parallel_advantage_check.py
```

Requires the existing local data/cache and previous main/continuation runs. Raw inputs, parameters and execution traces remain local. Published outputs contain code, figures, aggregated plot values and provenance.
