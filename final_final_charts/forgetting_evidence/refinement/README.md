# FU retention refinement for Figure34

The updated Proposed uses priority16 FL sampling through round100 and uniform16 afterwards, and changes only the retained-CE term in the FU gradient from coefficient1.0 to0.95. The forget proxy gate0.01, forgetting coefficient3, consistency coefficient3, temperature,200 FL rounds,45 requests,225 prescribed FU cycles, radio caps, datasets and complete query galleries stay fixed. No reporting model/probability enters training.

Twenty exact-aggregation development runs (including controls) and two actual wireless runs are exported. Seven initial settings per dataset were followed by three per dataset (two smaller coefficient changes plus the unchanged control). The physical0.95 result is selected on the same already-inspected development cohorts, with no independent-seed or statistical-superiority claim. This follows the142 total exact screens across the earlier122 and these20; no unsuccessful result is hidden.

| Dataset | Matched KL | Reduction vs best comparator | Retained mR (%) |
|---|---:|---:|---:|
| CapERA | 0.0031155 | 6.59% | 17.61719 |
| AirScape | 0.0007957 | 35.29% | 13.80208 |

Mean and worst-request **policy-matched** KL, and retained mR, lead the three unchanged comparators in both datasets at the measured physical endpoint. The common-reference KL does not lead. CapERA retained mR drops from the preceding late-uniform variant17.65625% to17.61719%, while remaining above the original P-KS17.59115% and comparator maximum17.55208%. AirScape retained mR remains13.80208%.

The0.95 setting has exact-aggregation CapERA mR17.46094% but physical mR17.61719%. Noise affects subsequent cached-loss sampling; physical and exact results must not be conflated. Report both, and validate additional seeds before claiming stable superiority.

All new settings share the same FL-only policy, so the already frozen uniform_after100 retained-only reference and no-FU control remain valid. No reference was changed to obtain a favorable rank. Exact-FU still has larger common and matched KL than its paired exact-no-FU control. Consequently, final policy-matched similarity leadership does not establish a causal forgetting improvement or deletion guarantee.

Selected checkpoints and all per-video metrics, all45 request KLs, mean/worst KLs, alternate common-reference values, energy/delay, raw checkpoint hashes and protocols are retained. Native caches and raw model tensors remain local. Original0.1J network RF/0.32W device limits and full200/225 completion pass for the two physical runs.

The two intermediate0.97/0.98 settings are exploratory exact results only; no physical energy/delay claim is made for them. No need to rerun additional candidates once the disclosed Figure34 criterion was met.

Figure34 is updated to this specific variant. Existing learning curves, other figures and FU workload sweeps refer to their original method versions unless separately rerun; their values cannot be described as results from this updated coefficient.
