# Forgetting-quality evidence: one figure

[PDF](34_forget_matched_retrain_1x2.pdf) · [PNG](34_forget_matched_retrain_1x2.png) · [All values, including common-reference KL](values.csv)

This is a scoped comparison of forget-set prediction distributions to policy-matched retained-only retraining. It is not proof that Proposed has the best forgetting quality overall.

`Controlled` is the new `uniform_after100` candidate, not the original P-KS used in the existing paper figures. The blue label explicitly preserves that distinction. All three comparison algorithms are unchanged. This figure does not silently replace Proposed or relabel a candidate as the original method.

| Dataset | Controlled matched KL | Best baseline matched KL | Controlled rank |
|---|---:|---:|---:|
| CapERA | 0.0034600 | 0.0033352 (ISCC-P2) | 3 |
| AirScape | 0.0008931 | 0.0012296 (ISCC-P2) | 1 |

The defensible positive claim is: **On AirScape, the controlled schedule achieves 27.4% lower forget-set KL to its policy-matched retraining reference than ISCC-P2, the best comparator under this metric.** This is descriptive for the existing development seed, not statistically established superiority or an independent confirmation after adaptive tuning.

CapERA is also plotted, including its non-leading result. Policies have separate matched references. Under the unchanged common reference, Controlled has KL0.0100569 on CapERA and0.0034039 on AirScape and ranks fourth on both. These primary outcomes remain in `values.csv` and the [complete comparison](../joint_quality_control/README.md); the secondary matched comparison does not supersede them.

Every deleted video and all native captions are included:247 CapERA and358 AirScape videos; galleries503/614 include the same256 retained validation videos per dataset. KL is D_KL(P_retrain || P_candidate), averaged across both retrieval directions and positive captions. Retraining starts the adapters from initialization with the deletion union excluded; the pretrained encoders remain frozen. This does not mean full-backbone pretraining from scratch.

The paired exact-FU versus exact-no-FU controls still have higher KL with FU. `paired_controls.csv` preserves those results. Accordingly, this figure supports only a scoped retraining-similarity observation; it does **not** establish that FU causes forgetting, that private associations are removed, or that forgetting quality is best across both datasets. Retained mR leadership and the C9 aggregation bound do not resolve this limitation.

No new training or selection of favorable queries was performed for this chart. It presents the already evaluated shared candidate,200 FL rounds/225 prescribed FU steps, seed0, and the unchanged comparators. Both datasets, alternate-reference values, method identities, source hashes and figure geometry are retained. The existing main Proposed figures are unchanged.

Rebuild with `MPLCONFIGDIR=/tmp/ova-mpl .venv/bin/python tools/uav_convergence_v2/forgetting_evidence_figure_v1.py`.
