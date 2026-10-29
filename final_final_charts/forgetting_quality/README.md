# Forgetting quality validation

Current Proposed = P-KS. All original numerical charts and model checkpoints are unchanged.
Two datasets, four methods, 200 FL rounds and 45 deletion requests (225 FU steps). All forgotten videos and their native captions are evaluated: CapERA 247 videos (five captions each); AirScape 358 videos (one caption each). Each retained evaluation set contains 256 videos. This remains a seed0 development experiment, not a new independent test benchmark.

## Figures

Open `forgetting_quality_all.pdf` for all eight figures in one file. Individual PDF/PNG/SVG files and LaTeX snippets are also included.

| File | What it measures |
|---|---|
| 20_forget_retrieval_1x2 | Actual wireless final-model forget mR and policy-matched retraining mR. Lower forget mR alone does not establish unlearning. |
| 21_forget_retraining_kl_1x2 | Actual forget-set KL to the original common and policy-matched retraining references. Both counterfactuals are retained. |
| 22_forget_association_change_1x2 | Paired reduction of positive-vs-negative cosine margin: exact no-FU minus exact FU. Positive means weaker associations due to the FU intervention; negative means strengthening. Error bars: 2,000 paired video-bootstrap replicates, conditional on the fitted models. |
| 23_forget_retrieval_fu_control_1x2 | Same200-round no-FU vs FU comparison with exact aggregation and identical original transaction schedules. |
| 24_wireless_error_forgetting_1x2 | KL to common retraining for exact vs actual wireless FU trajectories. This separates aggregation sensitivity from unlearning success. |
| 25_request_retraining_kl_1x2 | Distribution of policy-matched KL over all45 requests. The0.01 line is the existing diagnostic tolerance, not a proven deletion criterion. |
| 26_forget_association_before_after_1x2 | Actual deleted-pair cosine before requests (warm20) and after200-round FL/FU. Different training horizons: descriptive only; use22/23 for the paired intervention. |
| 27_forget_kl_fu_control_1x2 | Paired exact no-FU and exact FU distance to policy-matched retraining, at the identical200-round horizon. This tests whether FU improves that particular unlearning criterion. |

## Results

| Dataset | Method | Retained mR (%) | Forget mR (%) | Retrain forget mR (%) | Common KL | Matched KL | Worst request matched KL | Association reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CapERA | Proposed | 17.591 | 17.301 | 16.626 | 0.018232 | 0.004450 | 0.006476 | 0.000008 |
| CapERA | ISCC-P2 | 16.901 | 16.613 | 16.343 | 0.003789 | 0.003335 | 0.004926 | -0.000020 |
| CapERA | AFL-OTA | 16.966 | 16.613 | 16.343 | 0.003791 | 0.003337 | 0.004931 | -0.000020 |
| CapERA | F-SalUn | 17.552 | 17.031 | 16.343 | 0.009668 | 0.008981 | 0.014702 | 0.000172 |
| AirScape | Proposed | 13.802 | 11.546 | 11.266 | 0.007320 | 0.001235 | 0.002176 | -0.000024 |
| AirScape | ISCC-P2 | 13.737 | 11.359 | 11.173 | 0.000839 | 0.001230 | 0.001625 | -0.000059 |
| AirScape | AFL-OTA | 13.737 | 11.359 | 11.173 | 0.000839 | 0.001230 | 0.001625 | -0.000059 |
| AirScape | F-SalUn | 13.737 | 11.499 | 11.173 | 0.003127 | 0.003909 | 0.005007 | 0.000096 |

## Interpretation and limits

**The current results do not validate strong forgetting.** In both datasets, every method has higher forget mR after explicit FU than its exact no-FU control, and its policy-matched retraining KL is also higher. Proposed changes forget mR from16.842% to17.301% on CapERA and from11.220% to11.546% on AirScape. Its exact matched KL changes from0.000312 to0.004448 on CapERA and from0.001055 to0.001231 on AirScape (full precision in the tables). A low absolute KL threshold alone therefore does not demonstrate a benefit over simply excluding requested samples from subsequent FL.
Proposed has only a very small average association-margin reduction on CapERA and a small increase on AirScape. Read the conditional intervals and all-video outcomes in22 rather than claiming uniformly weakened associations. The mean raw positive-pair cosine increases from warm20 on CapERA and decreases on AirScape;26 is descriptive because intervening FL also changes representations.
Wireless vs exact trajectories have very similar forget mR and retraining KL in these runs. That observation points to the current learning/unlearning objectives and reference definition as the substantive issue, rather than establishing that AirComp distortion explains the remaining forgetting gap. No causal guarantee outside these runs is implied.
Do not infer successful unlearning from retained mR, completed FU steps, a small aggregation MSE, or C9. C9 compares distorted and error-free aggregation; it does not bound distance to a deletion-trained reference. Figures21/25 test that distance empirically, while22/23 test the FU intervention at fixed exact transport.
Retraining means resetting and retraining the rank10 visual/text adapters from their original initialization, using the frozen pretrained RemoteCLIP encoders and fixed temperature. It does not mean retraining the full pretrained VLM. The complete deletion union is excluded from every reference round. All six fresh200-round references reproduce their previously saved counterparts bitwise.
The common reference preserves the original sampler. The policy-matched reference changes the retained-only sampling/weighting counterfactual for every method; it never changes a candidate model or feeds evaluation predictions into training. Publish both: agreement with a convenient reference alone is insufficient.
The Proposed training proxy is the existing20-round retained-only proxy with the full future deletion cohort known. Its local request KL gate is not the200-round full-gallery reporting KL. That assumption remains unchanged and should be disclosed; these experiments do not validate unknown future deletion cohorts.
The16 fresh control runs start from the same warm20 checkpoint as the original models. Requested samples leave subsequent FL at the original release times. Exact-no-FU does zero explicit unlearning updates; its request-step counters only preserve the original schedule. It must not be reported as successfully processing deletion requests. All original digital FU controls reproduce the original final parameter tensors bitwise. AFL-OTA has all10 ready and zero staleness in these particular traces.
No energy or delay claim is made for exact-aggregation controls. They isolate FU conditional on the original schedule; comparisons with actual wireless models additionally change communication error. Request before/after intervals can include FL updates; both conditions are exported.
Retrieval can remain above zero after valid retraining because of pretrained/generalizable knowledge. Excess retrieval and probability divergence relative to retraining are more informative than demanding zero recall. Conversely, suppressing all similarities can destroy utility without removing the learned influence of specific samples.
All forget queries are included. No favorable-subset selection, modified baseline, evaluation-driven tuning, or withheld adverse result. Frozen feature caches and raw data remain local for audit; there is no claim of physical data erasure or certified unlearning.

## Reproduction

Run `tools/uav_convergence_v2/forgetting_quality_v1.py run` with the local native data/features and GPU, then `tools/uav_convergence_v2/forgetting_quality_report_v1.py`. Completed runs are reused; input hashes are checked. Raw features, models and logs are intentionally not published.
Detailed per-video metrics, all request events, complete summaries, plot values and provenance accompany the figures in `data/`.

Background: [CLIPErase (ACL2025)](https://aclanthology.org/2025.acl-long.1469/) motivates explicitly assessing visual–textual associations, retention and consistency. [SalUn (ICLR2024)](https://openreview.net/pdf?id=gn0mIhQGNM) is the source of the existing F-SalUn transfer; no method has been modified for this audit.
