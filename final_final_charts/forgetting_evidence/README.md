# Forgetting-quality evidence: one figure

[PDF](34_forget_matched_retrain_1x2.pdf) · [PNG](34_forget_matched_retrain_1x2.png) · [All values, including common-reference KL](values.csv)

This is a scoped comparison of forget-set prediction distributions to policy-matched retained-only retraining. It is not proof that Proposed has the best forgetting quality overall.

The user explicitly requested the label `Proposed`. In this figure it denotes the physically verified `retain0.95` configuration: priority16 FL sampling through round100, uniform16 afterwards, and FU retained-CE coefficient0.95. This is a real training change from the previous `uniform_after100` setting, not a relabeling of its measurements. The same configuration is used on both datasets; all three comparison algorithms are unchanged. Older paper bundles and the FU-workload energy chart retain their original P-KS version until separately rerun.

| Dataset | Proposed matched KL | Best baseline matched KL | Proposed rank |
|---|---:|---:|---:|
| CapERA | 0.0031155 | 0.0033352 (ISCC-P2) | 1 |
| AirScape | 0.0007957 | 0.0012296 (ISCC-P2) | 1 |

The defensible positive claim is: **Proposed achieves6.6%/35.3% lower forget-set KL to policy-matched retraining references than ISCC-P2 on CapERA/AirScape, and also leads worst-request matched KL and retained mR among the three unchanged comparators.** This is descriptive for the existing development seed, not statistically established superiority or an independent confirmation after adaptive tuning. Retained mR is17.61719%/13.80208%; CapERA is0.03906 percentage points below the preceding late-uniform configuration, while still above all three comparators and original P-KS.

Policies have separate matched references. The shared FL-only policy is unchanged by the FU coefficient, so the same frozen matched reference is used for all late-uniform variants. Under the unchanged common reference, Proposed has KL0.0094830 on CapERA and0.0031926 on AirScape and does not rank first. These primary outcomes remain in `values.csv` and the [complete refinement](refinement/README.md); the secondary matched comparison does not supersede them. Earlier configurations and failed candidates remain available.

Every deleted video and all native captions are included:247 CapERA and358 AirScape videos; galleries503/614 include the same256 retained validation videos per dataset. KL is D_KL(P_retrain || P_candidate), averaged across both retrieval directions and positive captions. Retraining starts the adapters from initialization with the deletion union excluded; the pretrained encoders remain frozen. This does not mean full-backbone pretraining from scratch.

The paired exact-FU versus exact-no-FU controls still have higher KL with FU. `paired_controls.csv` preserves those results. Accordingly, this figure supports only a scoped retraining-similarity observation; it does **not** establish that FU causes forgetting, that private associations are removed, or that forgetting quality is best across both datasets. Retained mR leadership and the C9 aggregation bound do not resolve this limitation.

The refinement comprises20 new exact-aggregation development runs and two actual wireless runs. All outcomes are in `refinement/`; the previous34 chart is preserved in `archive/before_retention095/`. Each physical run completes200 FL rounds/225 prescribed FU steps with unchanged0.1J network RF and0.32W device caps. Both datasets, all queries, alternate-reference values, method identities, source hashes and figure geometry are retained. The shared late-uniform and original exact controls reproduce their earlier tensors. Other figures are unchanged and must not be described as results of this updated coefficient without rerunning them.

Rebuild with `MPLCONFIGDIR=/tmp/ova-mpl .venv/bin/python tools/uav_convergence_v2/forgetting_evidence_figure_v1.py`.
