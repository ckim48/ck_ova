"""Audit paired cross-modal/vanilla conditions before radio comparisons."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/overnight_energy_v2/crossmodal_ablation_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = {"status": "passed", "cases": [], "runner_hashes": {
        name: sha(Path(__file__).parent / name)
        for name in ("prepare_paired.py", "radio_paired.py", "audit_paired.py")}}
    for k in (5, 8):
        loaded = {}
        geometry = {}
        for variant in ("crossmodal", "vanilla"):
            folder = BASE / "prepared" / variant / f"k{k}"
            geometry[variant] = json.loads((folder / "geometry.json").read_text())
            with np.load(folder / "prepared.npz") as data:
                loaded[variant] = {key: data[key].copy() for key in data.files}
        cm, vn = loaded["crossmodal"], loaded["vanilla"]
        gm, gv = geometry["crossmodal"], geometry["vanilla"]

        max_learning_delta = float(np.max(np.abs(cm["learning"] - vn["learning"])))
        max_channel_delta = float(np.max(np.abs(cm["channels"] - vn["channels"])))
        max_gradient_delta = float(np.max(np.abs(cm["gradient"] - vn["gradient"])))
        assert np.array_equal(cm["active"], vn["active"]), "active requesters differ"
        assert max_learning_delta <= 1e-8, f"learning updates differ: {max_learning_delta}"
        assert max_channel_delta == 0.0, f"channel realization differs: {max_channel_delta}"
        assert max_gradient_delta <= 1e-8, f"held-out gradient differs: {max_gradient_delta}"
        assert gm["active_users"] == gv["active_users"]
        assert gm["forget_counts"] == gv["forget_counts"]
        assert gm["retained_counts"] == gv["retained_counts"]
        assert gm["learning_rate"] == gv["learning_rate"]
        assert gm["frozen_checkpoint_sha256"] == gv["frozen_checkpoint_sha256"]
        assert gm["source_hashes"] == gv["source_hashes"]
        assert abs(gm["application_rmse_budget"] - gv["application_rmse_budget"]) <= 1e-12
        assert gm["controller"]["alpha"] == gv["controller"]["alpha"] == 1.0
        for item in (gm, gv):
            assert min(item["controller"]["normalized_constraint_slack"]) >= -1e-8
            assert min(item["controller"]["request_directional_progress"]) >= 1.0 - 1e-7

        coupled = gm["coupling"]
        vanilla = gv["coupling"]
        assert all(x["kappa"] > 0 for x in coupled), "cross-modal branch did not apply coupling"
        assert all(x["kappa"] == 0 for x in vanilla), "vanilla branch has nonzero coupling"
        update_delta = float(np.linalg.norm(cm["unlearning"] - vn["unlearning"]))
        assert update_delta > 1e-12, "the variants produced identical unlearning updates"
        payload_cm = int(np.sum(cm["payload"]))
        payload_vn = int(np.sum(vn["payload"]))
        record = {
            "K": k,
            "status": "paired_conditions_verified",
            "active_users": gm["active_users"],
            "max_abs_learning_update_delta": max_learning_delta,
            "max_abs_channel_delta": max_channel_delta,
            "max_abs_heldout_gradient_delta": max_gradient_delta,
            "common_application_rmse_budget": gm["application_rmse_budget"],
            "crossmodal_kappa": [x["kappa"] for x in coupled],
            "vanilla_kappa": [x["kappa"] for x in vanilla],
            "unlearning_update_l2_delta": update_delta,
            "q8_payload_bits_crossmodal": payload_cm,
            "q8_payload_bits_vanilla": payload_vn,
            "payload_bits_equal": payload_cm == payload_vn,
            "target_l2_delta": float(np.linalg.norm(cm["target"] - vn["target"])),
            "controller_rule": "same alpha=1 request-wise controller; targets are recomputed from each variant's updates",
        }
        record["radio_modes_ready"] = all(
            (BASE / "radio" / variant / f"k{k}" / "radio_results.json").is_file()
            for variant in ("crossmodal", "vanilla"))
        report["cases"].append(record)
    (BASE / "paired_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
