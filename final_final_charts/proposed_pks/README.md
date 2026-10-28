# Proposed = P-KS: learning and resource comparisons

<!-- tgcn-additions -->
[Energy–mR, paired FU-workload bars and combined retrieval examples](../downlink_bandwidth/TGCN_ADDITIONS.md). Main figures10–11 and16; earlier diagnostic/success sheets12–15 remain linked.
<!-- /tgcn-additions -->









<!-- downlink-bandwidth-current -->
[Downlink bandwidth figures, paired resource bars, smaller map and comparison tables](../downlink_bandwidth/index.html). Explicit fixed spectral efficiency; 1–5 MHz.

**Parallel advantage:** [Five figures and a matched immediate-serial P-KS experiment](../parallel_advantage/README.md).

The user-selected P-KS is now the only method labeled **Proposed** in this bundle. It is the unchanged proxy-KL-gated FU + half-exploration/half-loss-priority FL16 method. The previous adaptive-FU/uniform-FL Proposed is removed from these comparisons. P-KB25 is a separate experiment and is not substituted for P-KS. Historical bundles retain their original identities.

Methods: Proposed, ISCC-P2, AFL-OTA and F-SalUn. The [paper-transfer definitions](../four_methods/README.md) remain unchanged. Main line color/marker for Proposed is blue/circle. Every chart is CapERA left and AirScape right, with uniform endpoint ticks, matching panel precision, and no value labels or extra captions. Round-curve markers share the same 25-round grid across all methods; the last observed evaluation is marked when FU produces multiple observations at one FL round. All original line values remain unchanged.

## Figures

- 01–03: validation loss, retained mR and forget-set KL versus FL round, through the common 200-round workload. FU updates occur during this workload; the loss is retained retrieval validation CE after the actual FL/FU updates, not a sum of incompatible local FL and FU objectives.
- 04–05: loss and mR versus active time, with actual retained-only FL continuation after all 45 requests finish. All methods receive the same continuation learning rate and 400-second budget. Display the last evaluated completed checkpoint; no invented additional progress.
- 06–11: total energy and active FL/FU delay versus downlink rate, total FL-round budget and packet loss.
- 12–14: mission delay for those same three axes, kept separate from active service delay.

## Why these axes change costs

In this cached-adapter scenario, full-model broadcasts and active hovering dominate cost. Increasing uplink bandwidth changes only one part of service time. At a fixed 400-second hover budget, total hover energy is almost fixed by construction; resource comparisons therefore use completed prescribed work, not equal elapsed-time budgets.

| Axis | Values | Intervention |
|---|---|---|
| Downlink rate | 0.25, 0.5, 1, 2, 4 Mbit/s | Same full update payload and work; change the model broadcast rate. |
| FL-round budget | 100, 125, 150, 175, 200 | Same 20 warmup rounds and all 45 requests/225 FU cycles; change subsequent FL work. Original cosine schedule is truncated, not retuned. Release ordinals scale within each site. |
| Downlink packet loss | 0, 10, 20, 30, 40% | Repeat failed whole-model broadcasts until success; charge actual attempts. This is a declared erasure model, not a measured link. |

Every condition uses identical scenario inputs, physical coefficients and prescribed workload across the four methods. All 24 nonbaseline P-KS dataset/condition arms are fresh executions of the frozen registered algorithm. The matching 78 paper-transfer arms and two original P-KS baseline arms are reused unchanged: 104 unique arms total. Neither flat curves nor unfavorable results are rescaled or removed. Energy is active communication + executed adapter computation + hovering; travel, IoT collection and preflight are excluded. Mission delay separately includes travel, collection and return. The fixed backbone feature cache excludes full-VLM encoding costs.

## KL interpretation and limits

The round-wise KL compares bidirectional retrieval output distributions on the forget set against the common retained-only retraining model at round 200, using D_KL(reference || candidate). It is not the training proxy KL. Early-round distances also reflect the different training horizons, so they are descriptive and do not certify current-request unlearning. Shorter FL-budget cases use the same fixed reference; there is no newly matched shorter reference claim.

P-KS has the previously reported larger common-reference KL; renaming it does not improve it. The much smaller P-KB25 values from the [separate KL repair study](../pks_kl_repair/README.md) do not belong to this Proposed. Both common and policy-matched reference diagnostics remain available there under their historical method names.

These are single-seed development results with the original validation subsets. The known-cohort training proxy assumption remains; the reporting retrain is never a training teacher. Completed requests mean all prescribed FU steps finished, not proof of deletion. Fewer FL rounds also change quality. All final utility/KL and physical bound exceedances remain in the aggregate metrics.

## Baseline cost components

| Dataset | Method | Energy (kJ) | Active delay (s) | Broadcast share of delay | Hover share of energy |
|---|---|---:|---:|---:|---:|
| CapERA | Proposed | 204.277 | 121.102 | 96.94% | 99.89% |
| CapERA | ISCC-P2 | 649.962 | 384.304 | 38.23% | 99.62% |
| CapERA | AFL-OTA | 252.882 | 149.912 | 98.00% | 99.88% |
| CapERA | F-SalUn | 649.960 | 384.302 | 38.23% | 99.62% |
| AirScape | Proposed | 204.267 | 121.096 | 96.95% | 99.89% |
| AirScape | ISCC-P2 | 649.305 | 383.915 | 38.27% | 99.62% |
| AirScape | AFL-OTA | 253.066 | 150.021 | 97.93% | 99.88% |
| AirScape | F-SalUn | 649.307 | 383.916 | 38.27% | 99.62% |

## Measured endpoint changes for Proposed

| Axis | Dataset | Energy change | Active-delay change | Energy shape |
|---|---|---:|---:|---|
| downlink | CapERA | -93.02% | -93.02% | decreasing |
| downlink | AirScape | -93.02% | -93.02% | decreasing |
| fl_rounds | CapERA | +123.70% | +123.70% | increasing |
| fl_rounds | AirScape | +120.96% | +120.96% | increasing |
| packet_loss | CapERA | +77.45% | +77.45% | increasing |
| packet_loss | AirScape | +77.45% | +77.45% | increasing |

## Reproduction

With existing local datasets/caches and the baseline/transfer experiments, run:

```sh
.venv/bin/python tools/uav_convergence_v2/learning_quality_runner_v2.py run --corrected --variants gate_select16 --cases downlink_0p25 downlink_0p5 downlink_2 downlink_4 fl_rounds_100 fl_rounds_125 fl_rounds_150 fl_rounds_175 packet_loss_0p1 packet_loss_0p2 packet_loss_0p3 packet_loss_0p4
.venv/bin/python tools/uav_convergence_v2/proposed_pks_report.py
.venv/bin/python tools/uav_convergence_v2/proposed_pks_check.py
```

The frozen runner preserves earlier experiments. Only code, charts, aggregate metrics and provenance are published. Checkpoints, raw samples and execution logs remain local.
