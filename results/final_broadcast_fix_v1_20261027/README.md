# Nonzero head-local update broadcast correction

Six single-request release cases were rerun after detecting omitted downlink broadcasts for nonzero local-only aggregation. A nonzero decoded update now triggers the existing broadcast payload, duration and PA/circuit energy even if its uplink is zero. True zero updates remain free. No learning objective, radio constraint, dataset or seed changed.

The protocol audits all 50 main-method release-schedule conditions and identifies exactly these six affected arms. Verification checks the added broadcasts, full 225-step FU workload and 200 FL rounds. Previous outcomes remain preserved.

[Final charts](../../final_chart/README.md) use the corrected single-request points. All baseline and bandwidth curves remain unchanged. Raw corrected records are in raw_new_arms.tar.gz, with hashes in raw_archive_manifest.json. Extract relative to the repository root; shared datasets and prefix/reference checkpoints are not duplicated.

Run `.venv/bin/python tools/uav_convergence_v2/final_broadcast_fix.py` after the original request sweep.
