# Paired cross-modal ablation protocol

## Question

Does the cross-modal update coupling change the communication-energy/feasibility comparison among Hybrid NOMA-AirComp, AirComp-only, phase-separated AirComp+NOMA, AirComp+OMA, and full OMA when multiple deletion requests arrive together?

## Paired variants

- `crossmodal`: apply the existing low-rank cross-modal Hessian coupling to each active user's forget gradient.
- `vanilla`: use the same raw forget gradient without that coupling.

The frozen CLIP checkpoint, round-60 state, retained and forget examples, batch seeds, active requesters, learning updates, radio channels, Q8 quantizer, request-controller rule (`alpha=1`), radio solvers, power and battery limits, and frame grid are held fixed. The comparison changes only whether the cross-modal coupling transform is applied to the unlearning gradient. The variant-specific request-controller target is recomputed from that variant's update vectors; it is not forcibly held equal because it is a function of the update being ablated.

The absolute application-update RMSE budget is first set from the cross-modal variant's predeclared 20%-of-update-norm rule, then reused unchanged for vanilla. This prevents the ablation from receiving a looser radio-error allowance merely because its update has a different norm. The sensitivity bound is recomputed for each variant's request anchors, as required by the same controller's local error analysis.

## Workloads and radio methods

Use the same seeded simultaneous-request construction as the previous pilot, evaluating `K=5` and `K=8`. The fixed symbol grids are 8 symbols per coordinate at K=5, and 8 and 4 at K=8. They are comparison design points, not optimized global time allocations.

The radio methods are Hybrid NOMA-AirComp, AirComp-only additive reference, phase-separated AirComp+NOMA, AirComp+OMA, and full OMA. AirComp-only cannot implement individualized deletion control and is reported as task-ineligible, not ranked as a same-task winner. OMA-only uses ideal-Shannon continuous symbol allocation and is a lower-energy relaxation. Communication values are model estimates; this is not a measured radio experiment.

## Run order

On the remote project, place the copied scripts in a new `tools/overnight_energy_v2/crossmodal_ablation_v1` directory and keep all output in `results/overnight_energy_v2/crossmodal_ablation_v1`. Do not modify the source scripts or output folders from earlier runs. Run one variant at a time:

```text
python tools/overnight_energy_v2/crossmodal_ablation_v1/prepare_paired.py --variant crossmodal
python tools/overnight_energy_v2/crossmodal_ablation_v1/prepare_paired.py --variant vanilla --reference-budget-dir results/overnight_energy_v2/crossmodal_ablation_v1/prepared/crossmodal
python tools/overnight_energy_v2/crossmodal_ablation_v1/audit_paired.py
python tools/overnight_energy_v2/crossmodal_ablation_v1/radio_paired.py --variant crossmodal
python tools/overnight_energy_v2/crossmodal_ablation_v1/radio_paired.py --variant vanilla
```

The audit must pass before radio runs are interpreted. Keep GPU work serial. The result supports only a single frozen-round communication comparison; it does not establish retained VLM utility, verified deletion, membership-inference resistance, or a full federated training trajectory. A low update MSE or a directional constraint is not a deletion guarantee.
