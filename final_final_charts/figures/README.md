# Matching paper figures

`34_forget_matched_retrain_1x2.pdf` uses the updated Proposed: uniform FL sampling after round100 and FU retention coefficient0.95. Both dataset panels lead policy-matched KL among the unchanged comparators; see [all outcomes and limits](../forgetting_evidence/refinement/README.md). Existing learning curves and FU-workload energy figures still refer to their original P-KS parameters and were not relabeled as this new variant.

Panel(a) of Figure34 now includes an inset of the3.0–3.5 ×10^-3 interval. The zero-baseline main bars and panel(b) are unchanged. `34_forget_matched_retrain_1x2_full_scale.pdf` retains the version without the inset; both use exactly the same measured values.

Figures `20_...` through `27_...` add [forgetting-quality validation](../forgetting_quality/README.md). [All eight in one PDF](forgetting_quality_all.pdf). They use the same 532.8 pt canvas width and 205.128 × 174.798 pt panels as the paired bars. The FU/no-FU diagnostics use exact aggregation at the same 200-round horizon; they are separate from the unchanged actual wireless results.

Use every PDF with `width=\linewidth`. All four have the same 532.8 pt canvas width, 12 pt numeric ticks, and 14 pt axis titles and dataset labels.

- `all_method_rank_boxplot_1x2`: all retained-query ranks at **300 kJ**, copied from `../downlink_bandwidth/limited_energy/300kJ/19_all_method_rank_boxplot_1x2.pdf`. See the accompanying caption for the experiment conditions.
- `fu_steps_energy_1x2`: current Figure 11, paired 3-step and 9-step FU workloads.
- `01_round_loss_1x2`: current P-KS round validation loss.
- `05_time_mr_1x2`: current P-KS retained mR under the shared time budget.

The rank and FU bar figures have identical 7.4 × 3.735 inch canvases and 205.128 × 174.798 pt plotting panels. Both use single-line 12 pt algorithm names, with at least 6 pt between adjacent names. Both figures show all four panel borders. The FU legend remains 12 pt and 8 pt above the panels. All measured values and outliers are unchanged. The 200/400 kJ rank alternatives use the same layout; the historic 650 kJ export is unchanged.

Reapply this focused presentation pass after rebuilding plots: `.venv/bin/python tools/uav_convergence_v2/match_rank_fu_layout_v1.py`.

<!-- yeouido-map -->

The current mission map now illustrates Yeouido, Seoul. See [the map folder](../yeouido_map/README.md) for the new geometry and sources. Existing FL/FU result tables and curves remain based on Daejeon; they have not been relabeled or rerun.
