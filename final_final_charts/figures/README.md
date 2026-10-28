# Matching paper figures

These aliases use the current P-KS experiments and match the requested LaTeX filenames. All three PDFs have the same 532.8 pt width, 12 pt numeric ticks and legends, and 14 pt category names/axis titles. Four-method category names wrap at the hyphen to preserve font size without overlap. Legends sit 8 pt above the plots. Thus width=\linewidth gives matching apparent font sizes.

- fu_steps_energy_1x2: current 11, paired 1-step and 9-step FU workloads.
- 01_round_loss_1x2: current P-KS round validation loss.
- 05_time_mr_1x2: current P-KS retained mR under the shared time budget.

The active proposed_pks and downlink_bandwidth chart files have also been updated. Maps, tables and earlier qualitative sheets keep their existing layout. Reapply this presentation pass after rebuilding plots: `.venv/bin/python tools/uav_convergence_v2/paper_typography_v1.py`.
