# Learning diagnosis and saved-time continuation

All ten predeclared diagnostics are reported below. The current Proposed is preserved; the KL-gate and selective-FL settings are developmental candidates. P-LS changes only FL sampling; P-KS adds the KL-gate objective. No validation or reporting-reference predictions select samples, gradients or weights. Single seed; this is diagnosis, not held-out statistical confirmation.

The exact-aggregation diagnostic removes received-update noise while retaining its original cost. It barely changes utility, whereas changing the FU objective recovers much of the retained performance. The old adaptive retained-CE coefficient was usually about0.5. Fixed1 and gated variants explicitly change that learning tradeoff.

**KL gate:** `3*relu(local_forget_proxy_KL-0.01) + retained_CE + 3*request_teacher_KL`. This avoids continuing to minimize a local proxy term below the declared tolerance. The training proxy and request teacher are the original disclosed assumptions. This local surrogate is not the final retrain KL and is not a deletion certificate.

**Selection:** half of each local batch explores uniformly; the remainder prioritizes high cached per-example training loss, scored only during earlier gradient computations. Deleted samples are removed as requests arrive. The fixed16 and explicit8 caps are separate experiments. Extra selection/scoring work is included. Smaller batches still send the same20480-coordinate model update, so reduced computation alone need not reduce total energy.

The original selection attempts and two setup corrections failed before the first transaction. They are preserved locally. The final correction only retains the factory’s seeded RNG call sites and binds its registered selection configuration; it does not change the sampling rule.

| Variant | Dataset | mR (%) | Loss | Mean KL | Worst KL | Energy (kJ) | Delay (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Current Proposed | CapERA | 16.328 | 5.42157 | 0.000224149 | 0.000342769 | 204.451 | 121.204 |
| Exact aggregation diagnostic | CapERA | 16.393 | 5.42163 | 0.000223222 | 0.000340468 | 204.451 | 121.204 |
| Equal-client FL diagnostic | CapERA | 16.589 | 5.41235 | 0.00051956 | 0.00109336 | 204.727 | 121.368 |
| Joint GAFT diagnostic | CapERA | 16.940 | 5.35571 | 0.00288692 | 0.00397576 | 204.262 | 121.093 |
| Retention floor1 | CapERA | 16.706 | 5.37406 | 0.00167777 | 0.00233552 | 204.277 | 121.102 |
| Proxy-KL gate | CapERA | 16.914 | 5.36249 | 0.00238797 | 0.00332776 | 204.277 | 121.102 |
| Loss-priority FL16 | CapERA | 17.591 | 5.30013 | 0.00848689 | 0.0124375 | 204.866 | 121.450 |
| KL gate + FL16 selection | CapERA | 17.591 | 5.22885 | 0.0182323 | 0.0261308 | 204.277 | 121.102 |
| KL gate + FL8 selection | CapERA | 17.318 | 5.33302 | 0.00715799 | 0.0107297 | 204.299 | 121.115 |
| F-SalUn | CapERA | 17.552 | 5.28097 | 0.00966762 | 0.0161791 | 649.960 | 384.302 |
| Current Proposed | AirScape | 13.542 | 5.59488 | 0.000234091 | 0.000290354 | 204.751 | 121.383 |
| Exact aggregation diagnostic | AirScape | 13.542 | 5.59494 | 0.000234307 | 0.000291065 | 204.751 | 121.383 |
| Equal-client FL diagnostic | AirScape | 13.542 | 5.59864 | 0.000643676 | 0.000839854 | 204.613 | 121.301 |
| Joint GAFT diagnostic | AirScape | 13.737 | 5.55531 | 0.00136452 | 0.00184218 | 204.256 | 121.089 |
| Retention floor1 | AirScape | 13.672 | 5.57061 | 0.000451914 | 0.000673189 | 204.267 | 121.096 |
| Proxy-KL gate | AirScape | 13.737 | 5.56219 | 0.000779285 | 0.00101183 | 204.267 | 121.096 |
| Loss-priority FL16 | AirScape | 13.672 | 5.54909 | 0.00427209 | 0.00746798 | 205.028 | 121.547 |
| KL gate + FL16 selection | AirScape | 13.802 | 5.51150 | 0.00731969 | 0.0137718 | 204.267 | 121.096 |
| KL gate + FL8 selection | AirScape | 13.672 | 5.51957 | 0.0114637 | 0.0198043 | 204.255 | 121.089 |
| F-SalUn | AirScape | 13.737 | 5.52985 | 0.00312732 | 0.00477099 | 649.307 | 383.916 |

## Reproduction

Requires the original local feature caches and scenario inputs. Use `learning_quality_v3.py --variant control`, then `learning_quality_campaign.py control` and `learning_quality_unit.py`. Run the remaining non-selection variants with `learning_quality_runner.py run --variants exact equal gaft retain1 gate salun`. Run selection with `learning_quality_runner_v2.py run --corrected --variants select16 gate_select16 gate_select8`. Add the40 SalUn sensitivity arms with `learning_quality_runner.py run --variants salun --all-sweeps`. All named scripts are under `tools/uav_convergence_v2`.

Next run `learning_time_budget.py --dataset CapERA` and repeat for AirScape; run `learning_continuation_reference.py` for both datasets. Export with `learning_quality_report.py four`, `learning_quality_report.py diagnostics`, and `learning_time_report.py`, then verify using `learning_release_check.py`. The final corrected sampler is used without rerunning the intentionally preserved setup failures. Raw data, checkpoints and transaction logs remain local.
