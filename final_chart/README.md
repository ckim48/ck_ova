# Final paper charts

Left: CapERA. Right: AirScape. PNG previews, PDF for publication, SVG for editing. Open [index.html](index.html) to browse the complete set. Main methods: Proposed, FM-CM, S-AC, AC-OF, S-OF.

| Figure | Content | Files |
|---|---|---|
| 01_round_retained_mr_1x2 | Retained mR versus FL rounds | [PNG](01_round_retained_mr_1x2.png) · [PDF](01_round_retained_mr_1x2.pdf) · [SVG](01_round_retained_mr_1x2.svg) |
| 02_time_retained_mr_1x2 | Retained mR versus active service time | [PNG](02_time_retained_mr_1x2.png) · [PDF](02_time_retained_mr_1x2.pdf) · [SVG](02_time_retained_mr_1x2.svg) |
| 03_time_forget_kl_1x2 | Forget-set KL versus active service time | [PNG](03_time_forget_kl_1x2.png) · [PDF](03_time_forget_kl_1x2.pdf) · [SVG](03_time_forget_kl_1x2.svg) |
| 04_bandwidth_total_energy_1x2 | Total service energy versus bandwidth | [PNG](04_bandwidth_total_energy_1x2.png) · [PDF](04_bandwidth_total_energy_1x2.pdf) · [SVG](04_bandwidth_total_energy_1x2.svg) |
| 05_bandwidth_delay_1x2 | FL/FU service delay versus bandwidth | [PNG](05_bandwidth_delay_1x2.png) · [PDF](05_bandwidth_delay_1x2.pdf) · [SVG](05_bandwidth_delay_1x2.svg) |
| 06_request_mean_delay_1x2 | Mean FU request delay | [PNG](06_request_mean_delay_1x2.png) · [PDF](06_request_mean_delay_1x2.pdf) · [SVG](06_request_mean_delay_1x2.svg) |
| 07_request_p95_delay_1x2 | 95th-percentile FU request delay | [PNG](07_request_p95_delay_1x2.png) · [PDF](07_request_p95_delay_1x2.pdf) · [SVG](07_request_p95_delay_1x2.svg) |
| 08_energy_to_quality_1x2 | Energy to the common quality target | [PNG](08_energy_to_quality_1x2.png) · [PDF](08_energy_to_quality_1x2.pdf) · [SVG](08_energy_to_quality_1x2.svg) |
| 09_delay_to_quality_1x2 | Time to the common quality target | [PNG](09_delay_to_quality_1x2.png) · [PDF](09_delay_to_quality_1x2.pdf) · [SVG](09_delay_to_quality_1x2.svg) |
| final_quality_table | Final quality and reference models | [PNG](final_quality_table.png) · [PDF](final_quality_table.pdf) · [SVG](final_quality_table.svg) |

## Definitions and comparisons

Retained mR evaluates models after both FL and FU updates. Forget-set KL compares their retrieval distributions with the final retained-set retrain. Early KL also reflects FL progress and future requests. The curves do not sum the different training objectives or certify deletion. The full-RM Proposed has better retained utility in these records, while FM-CM and safe-proxy methods have smaller final forget-set KL. MIA is a descriptive fixed loss-score attack on 128 members and 128 nonmembers; the table reports its absolute AUC gap to retrain, not a proof of privacy. Raw and orientation-free AUC, validation loss and worst-request KL are retained in final_quality.csv.

The Pre-FU reference is the shared round-20 model, evaluated again without modifying its checkpoint. It is explicitly not a 200-round no-unlearning baseline. Retrain is the existing round-200 model trained without the final forgotten-data union. Its self-KL and MIA gap are zero by definition.

Time and energy cover active FL/FU service, including computation, uplink, broadcasts and queue waits. Energy sums communication, computing and hovering across ten UAVs; travel, collection and preflight are excluded. Time is modeled UAV service time, not GPU wall time. These are single-seed validation results; no cross-seed confidence interval is implied. End-of-training values in time plots are held to the shared display horizon without claiming additional learning.

Bandwidth axes show 1–4 MHz with measurements at 1, 2, 4 MHz. The tick at 3 is not another run; 0.25/0.5 MHz outcomes remain in the data. S-OF retains the previously authorized soft network TX cap with overruns recorded; analog methods retain hard caps.

## Request-load experiment

The x axis is requests per release: 1, 3, 5, 9, 15, corresponding to 45, 15, 9, 5, 3 release groups. All conditions retain the same 45 requests, 225 local FU steps and 200 total FL rounds. Releases are triggered by FL progress, not a fixed wall-clock arrival rate. Changing group count also changes the release-round/site distribution according to the original schedule. Thus this is the existing burst-schedule sensitivity, not a sweep of total requests or independent Poisson arrival rates.

Each request delay is final-broadcast completion minus actual arrival. The mean and empirical 95th percentile are computed across all 45 requests, including waits. The 95th percentile is not a confidence interval. These curves measure prescribed-step completion, not time to a quality/deletion guarantee. Final retained mR, mean/worst KL and the quality-confirmation flag for every load condition are included in data/request_latency.json and CSV. Original Proposed/S-AC/AC-OF runs are reused, and the missing 16 FM-CM/S-OF runs are added without tuning. All 50 main-method conditions are retained: {'completed': 50}.

A cost-accounting defect was identified in the single-request condition: a nonzero head-local FU update had no uplink and was incorrectly assigned no aggregate broadcast. The six affected S-AC/AC-OF/S-OF arms (two datasets) were rerun with broadcasts required for nonzero decoded updates, using the existing 0.655872 s broadcast duration and power coefficients. The final request curves use those corrected runs. True zero updates still require no broadcast. An audit of all 50 main-method burst arms found no other affected conditions. Earlier raw outcomes are preserved; curves were not repaired by shifting values. The baseline/bandwidth charts are unaffected. Correction protocol, verification and raw records are in ../results/final_broadcast_fix_v1_20261027.

## Common quality target

The criterion was already defined in the original simulation: all 45 requests complete, retained mR at least 95% of the same dataset's retrain mR, mean forget KL at most 0.01, worst-request KL at most 0.01, confirmed at three distinct consecutive evaluated FL rounds. No threshold was chosen to separate the methods in this report. The cost charts use the 1 MHz baseline and the confirmation time, including those confirmation observations. Costs are independently rebuilt from the recorded transactions up to that time and use the same service-energy scope.

For these ten baseline arms, confirmation occurs at round 200, coinciding with the fixed-workload completion. Consequently the target-cost bars equal the corresponding 1 MHz service totals; they verify a common acceptance condition and do not establish an additional speedup. The first passing observation (round 190) is also retained in the data. Passing this empirical criterion is not a formal deletion guarantee, and does not mean all methods have identical quality.

## Supplementary figures

The supplementary folder retains round/time validation loss, round forget-set KL, and all six matched-objective radio ablation figures (Proposed/H-OVA/S-OF+/NG-OVA). The original AirScape H-OVA 4 MHz infeasibility stays marked. These controls distinguish transport/head effects from differences in the FU objective. No earlier figure or outcome was overwritten.

## Data and reproduction

Numerical plot arrays, request latencies, quality targets, components, axes and hashes are in data/. The new raw runs are archived under ../results/final_request_sweep_v1_20261027, with the protocol, verification and source hashes. A module-import failure before any new experiment began was preserved along with its pre-correction protocol; only import ordering was corrected. Existing shared features and checkpoints remain in their original locations.

Run `.venv/bin/python tools/uav_convergence_v2/final_request_sweep.py run`, `.venv/bin/python tools/uav_convergence_v2/final_reference_audit.py`, then `.venv/bin/python tools/uav_convergence_v2/final_broadcast_fix.py`, then `.venv/bin/python tools/uav_convergence_v2/final_charts.py`.
