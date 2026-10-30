Head-side FL/FU fusion: complete development comparison

The current manuscript recovers FL and FU separately but adds them linearly.
Its c_L and c_U are channel/noise design coefficients. Under exact alignment,
they do not become task-combination weights. The new experiments explicitly
change the receiver update, preserving the previous results separately.

Methods:
  C-Sum: one real combined AirComp stream, linear sum.
  IQ-Sum: separate I/Q aggregates, linear sum.
  C-Local: two-task PCGrad at each UAV, followed by combined transmission.
  IQ-Head: two-task PCGrad after noisy global I/Q recovery.
  IQ-NR: fixed noise-margin and variance-regularized head PCGrad.
All five use a parallel transmission interval; this is not a serial-vs-parallel
comparison. PCGrad is an existing algorithm, credited in references.bib.

Families are reported separately:
  current: latest proxy-KL-gated FU, retention weight0.95, priority16 FL through
    round100 then uniform16, retained-pool weights.
  manuscript: supplied cosine-forgetting/retained-CE/embedding-consistency
    equations, uniform16 FL and minibatch weights. The numeric3/1/3 FU weights
    were predeclared from the earlier pilot; the pasted text did not give them.
Do not attribute a difference BETWEEN these families to I/Q or head fusion.

Each arm has200 total FL rounds (a shared20-round warm start and180 active
updates),45 requests and225 FU steps,10 UAVs and5 regions. Four current-family
sum controls are reused from the frozen experiment A;16 complete fresh models
were scheduled. Every attempted condition is retained. This is one previously
inspected development seed, not independent statistical confirmation.

Evaluation uses all247 deleted CapERA videos and358 deleted AirScape videos,
all native captions, and256 retained-validation videos per dataset. The full
galleries contain503 and614 videos. mR averages bidirectional R@1/5/10.
KL is D_KL(P_retrained || P_candidate) over both retrieval directions, averaged
by video/native captions. Both fixed common and policy-matched references are
provided. The manuscript matched reference uses uniform16/equal-client updates;
every retained client pool has at least16 samples, making minibatch weighting
equal to1/10. References never enter head decisions. These are adapter-only
retained-data retraining references; the pretrained encoders remain frozen.

The current-family inherited training proxy assumes the future deletion cohort
is known. That assumption is unchanged and is not a new head capability.

Local correction changes the actual waveform and its RF energy, including the
true cross term for a combined stream. Its error budgets still derive from the
uncorrected local updates. Head correction uses only received estimates and
designed branch-noise variances; clean aggregates are diagnostic-only.

Accounting includes active computation, uplink, broadcast, circuits and hovering.
Common warm-start and fixed travel energy remain excluded, as in experiment A.
Head fusion runs AFTER uplink and BEFORE broadcast, adding CPU and hovering time.
The metrics.csv total adds its explicit CPU energy to the inherited ledger;
legacy_ledger_total_j is retained for audit. Raw inherited energy_and_delay.json
does not include this new explicit CPU term; use the corrected exported totals.
Local/head arithmetic uses conservative scalar MAC equivalents under the existing
processor model, not measured onboard runtimes. The older generic physical_mode
metadata is overwritten by the parent runner; the actual transport is specified
by fusion_method and the implementation, and independently checked in
physical_checks.json. This metadata quirk does not change transmitted signals.

Pre-fusion component MSE and curvature constraints are checked on every round.
Nonlinear fusion can bias or amplify noise, so those constraints do not certify
post-fusion objective distortion. Transactions export both pre/post realized
errors. Their sums are realized diagnostics, not estimated conditional MSEs.
The addendum gives the changed ideal/global updates and explains why the old
zero-mean linear-sum theorem cannot be reused. The old SOCP remains a transport
design; no convexity claim is made for a new nonlinear output-quality constraint.

Files47--52 show the current family;53--58 show the manuscript family. Each
family includes matched/common KL, energy, delay, mR and loss. Exactly overlapping
curves are preserved without artificial offsets. Bars start at zero. The graphs
have identical panel geometry, regular endpoint ticks and no outcome labels.
The baseline and candidate names are deliberately not renamed Proposed.
Figure59 shows the matched-KL change relative to IQ-Sum; positive means worse.
Zero manuscript bars represent exactly unchanged KL, not missing runs.
The fitted-model equality audit is in model_equivalence_checks.json.

Reproduce:
  .venv/bin/python tools/uav_convergence_v2/head_fusion_study_v1.py register
  .venv/bin/python tools/uav_convergence_v2/head_fusion_study_v1.py queue
  .venv/bin/python tools/uav_convergence_v2/head_fusion_check_v1.py
  .venv/bin/python tools/uav_convergence_v2/head_fusion_report_v1.py evaluate
  .venv/bin/python tools/uav_convergence_v2/head_fusion_report_v1.py report
Completed or interrupted attempts are not silently overwritten. Existing native
features, plans, warm-start models and references are required; evaluation source
JSONs and the protocol preserve hashes. Original scripts were not modified.

Endpoints (mR in percent, energy in kJ, active delay in seconds):

current
Dataset  Method  mR  matched_KL  common_KL  energy_kJ  delay_s  head_corrections
CapERA  C-Sum  17.617187  0.003109719772  0.009475286644  212.370743  125.897267  0
CapERA  IQ-Sum  17.682292  0.003118142434  0.009488043146  214.339893  127.064627  0
CapERA  C-Local  17.617187  0.003131440588  0.009514676901  212.374956  125.899725  0
CapERA  IQ-Head  17.682292  0.003122168764  0.00949360223  214.344048  127.067085  4
CapERA  IQ-NR  17.682292  0.003121912404  0.009493295979  214.348212  127.069549  4
AirScape  C-Sum  13.802083  0.0007943126877  0.003190697858  209.699323  124.313325  0
AirScape  IQ-Sum  13.802083  0.0007972506529  0.003196417558  211.392192  125.316845  0
AirScape  C-Local  13.802083  0.0008035082756  0.003213572666  209.772612  124.356743  0
AirScape  IQ-Head  13.802083  0.0008019005995  0.003203154866  211.396347  125.319303  5
AirScape  IQ-NR  13.802083  0.0008015959773  0.00320275902  211.400511  125.321766  5

manuscript
Dataset  Method  mR  matched_KL  common_KL  energy_kJ  delay_s  head_corrections
CapERA  C-Sum  17.044271  0.002962669812  0.003333228651  212.541931  125.998748  0
CapERA  IQ-Sum  17.044271  0.002969579865  0.003340781568  214.027437  126.879388  0
CapERA  C-Local  17.044271  0.002962669812  0.003333228651  212.546139  126.001206  0
CapERA  IQ-Head  17.044271  0.002969579865  0.003340781568  214.031592  126.881846  0
CapERA  IQ-NR  17.044271  0.002969579865  0.003340781568  214.035756  126.884310  0
AirScape  C-Sum  13.671875  0.001560771853  0.001058669439  209.594540  124.251193  0
AirScape  IQ-Sum  13.671875  0.00156489541  0.001061581567  211.425487  125.336633  0
AirScape  C-Local  13.671875  0.001563826095  0.001061193644  209.598748  124.253651  0
AirScape  IQ-Head  13.671875  0.00156489541  0.001061581567  211.429642  125.339091  0
AirScape  IQ-NR  13.671875  0.00156489541  0.001061581567  211.433806  125.341555  0

Head changes relative to the same-family IQ-Sum (positive KL reduction is better):
current CapERA IQ-Head: mR change +0.000000 percentage points; matched KL reduction -0.129126%; energy reduction -0.001938%.
current CapERA IQ-NR: mR change +0.000000 percentage points; matched KL reduction -0.120904%; energy reduction -0.003881%.
current AirScape IQ-Head: mR change +0.000000 percentage points; matched KL reduction -0.583248%; energy reduction -0.001965%.
current AirScape IQ-NR: mR change +0.000000 percentage points; matched KL reduction -0.545039%; energy reduction -0.003935%.
manuscript CapERA IQ-Head: mR change +0.000000 percentage points; matched KL reduction +0.000000%; energy reduction -0.001941%.
manuscript CapERA IQ-NR: mR change +0.000000 percentage points; matched KL reduction +0.000000%; energy reduction -0.003887%.
manuscript AirScape IQ-Head: mR change +0.000000 percentage points; matched KL reduction +0.000000%; energy reduction -0.001965%.
manuscript AirScape IQ-NR: mR change +0.000000 percentage points; matched KL reduction +0.000000%; energy reduction -0.003935%.

Interpretation:
The new rule really uses separately recovered task vectors. In the current family, it does not improve final retained mR and slightly increases matched KL in this seed. In the manuscript family, neither head rule triggers; the final parameter tensors are exactly equal to IQ-Sum on both datasets. The task directions are mostly aligned in the current policy and always aligned at the global level in the manuscript policy. Conflict-only fusion therefore has little or no opportunity to act. These results do not support promoting this fusion rule as a demonstrated I/Q quality advantage. They also do not rule out other head policies or other operating conditions.
