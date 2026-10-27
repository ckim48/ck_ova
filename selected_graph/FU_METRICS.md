# Retained loss and forgetting evaluation

The existing loss curves already evaluate the model after both FL and FU updates. They select the retained-query retrieval negative log likelihood from each saved evaluation. FU is not omitted from the model; the plotted scalar simply measures retention rather than forgetting. FL still changes the model throughout training, even when its update rule is shared across methods.

Training objectives and common evaluation criteria are different quantities. Proposed's FU objective uses 3 FM + RM + 3 CM; FM-CM removes RM, and the safe-proxy baselines use another objective. Comparing each method's own FL-plus-FU scalar would therefore compare different functions, coefficients and batch reductions. Removing a term can lower that sum without improving the model. No arbitrary combined loss has been introduced.

Two additional figures use the already recorded common evaluation criterion:

- [Round versus forget-set KL](07_round_forget_kl_1x2.png)
- [Active service time versus forget-set KL](08_time_forget_kl_1x2.png)

Both directions of retrieval are averaged: KL(P_retrain || P_candidate), evaluated on the fixed forget-query set. The same final retained-set retrain reference is used for all methods within a dataset. Lower values indicate closer predictions to that reference on these probes. At early rounds the difference also includes FL progress and future requests, since the reference omits the final union of forgotten samples. This curve is descriptive and is neither a pure causal measure of FU nor a deletion/privacy certificate. Final comparisons are after all 45 requests complete 225 local FU steps.

The source audit verifies nonzero transmitted FU vectors and confirms that the original retained loss curves are drawn from the same post-update evaluation records. The first saved reference KL is at FL round 25; no round-0/20 KL is fabricated. The time plot holds the final observation to the common endpoint, as in the existing charts. Both axes retain regular endpoint ticks, matching decimal precision, and the established colors and markers.

All six existing figures and their numerical exports remain byte-for-byte unchanged. Use retained loss/mR together with forget-set KL and the existing final MIA diagnostics. The current full-RM method has better retained utility but larger final forget-set KL than FM-CM and the safe-proxy methods. These results do not support claiming best performance on every learning/unlearning criterion.

Rebuild after the six primary figures: `.venv/bin/python tools/uav_convergence_v2/forgetting_graphs.py`.
