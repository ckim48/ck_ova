# Final request-load sweep

Sixteen additional FM-CM/S-OF runs complete the existing paired release-schedule experiment. All retain 45 requests, 225 local FU steps and 200 total FL rounds. Outcomes: {'completed': 16}.

[Final charts and definitions](../../final_chart/README.md) include the existing models and all new outcomes. The final reference audit is read-only and does not retrain a model.

The module-import error before any new experiment started is preserved in bootstrap_failure.log, bootstrap_failed_runner.py and protocol.before_import_fix.json. The protocol records the import-order correction; no outcome was erased or chosen by rank.

raw_new_arms.tar.gz contains all new arm records, checkpoints and scenario inputs, with hashes in raw_archive_manifest.json. Extract relative to the repository root. Shared features, baseline models and earlier-method records remain in their existing locations; this is not a self-contained data release.
