# Adaptive retrieval FU (2026-10-28)

Training-only feedback adjusts the retained CE weight in a clean-proxy/retrieval-distillation FU objective. Both datasets use the same rule. Validation mR and reference200 are evaluation-only; retained CE and proxy KL are surrogates, not guarantees of improvement in held-out mR or retrain KL.

All 22 runs completed 200 FL rounds and 225 FU steps. Fourteen primary runs cover baseline, 2/4 MHz bandwidth and four request-release schedules. Eight additional runs pair adaptive and previous Proposed on post-warm sampling/radio seeds 1/2. Seed0 is the common original initialization; these are not independent data-split or initial-training repetitions.

[Current final charts](../../final_chart/README.md) contain all updated primary figures, old-versus-new comparisons, controller weights, paired seed CSVs, assumptions and actual tradeoffs. KL decreases substantially, while mR is not consistently improved. The previous bundle is preserved byte-for-byte in [the archive](../../final_chart_archive/20261028_before_adaptive/README.md).

`protocol.json` freezes source/input hashes and the old chart snapshot. `verification.json` checks completion and controller bounds. `feedback_audit.json` independently reconstructs all 4,050 adaptive local-step feedback states and verifies charged extra computation. `gradient_check.json` compares the actual callback with the direct combined-loss gradient. `publication.json` verifies the final bundle and archive.

The raw archive, its entry manifest and the runner log remain local and are not published to GitHub. The published files contain code, charts, plot data and validation summaries. Locally, `raw_new_arms.tar.gz` contains all 22 raw arm records, checkpoints and scenario JSON inputs, with every entry hashed and verified in `raw_archive_manifest.json`. Extract relative to the repository root. Existing shared feature caches, datasets and common warm/proxy/reference checkpoints are referenced from their original locations and are not duplicated. The existing clean proxy assumes the planned final forget cohort was known before the mission.

Reproduce with `.venv/bin/python tools/uav_convergence_v2/adaptive_fu.py run`, then `.venv/bin/python tools/uav_convergence_v2/adaptive_final_charts.py build` and `... adaptive_final_charts.py publish`. Publishing checks the registered previous bundle before moving it to the archive; it will not replace unrelated edits.
