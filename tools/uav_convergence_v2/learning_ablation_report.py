"""Report the fixed RM-removal experiment, including non-intermediate outcomes."""
import collections
import csv
import json
from pathlib import Path
import subprocess

import learning_ablation as study
from learning_ablation import ROOT, OUT, NEW, LABEL, PROPOSED, DATASETS, CASES, read, write, sha
import baseline_fairness as previous
import selected_graphs as plots

PREVIOUS_COMMIT = "122a6471105b228cc15d004b2e88740e22382af2"


def generate():
    study.register()  # Recheck the source/input hashes frozen before training.
    verification = study.verify()
    plots.generate("learning")
    plots.generate("guard_ablation")
    index, sources = {}, {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    git = ["git", "--git-dir=.git-push-ready/code-charts.git", "--work-tree=."]
    unchanged = 0
    for folder in ["selected_graph", "ablation_graph"]:
        old = json.loads(subprocess.check_output(git + ["show", f"{PREVIOUS_COMMIT}:{folder}/data/plot_series.json"], cwd=ROOT))
        old = {(r["figure"], r["dataset"], r["method"]): r for r in old}
        for row in read(ROOT / folder / "data/plot_series.json"):
            if row["method"] == NEW:
                continue
            before = old[(row["figure"], row["dataset"], row["method"])]
            for field in ["x", "y", "status", "display_x", "display_y"]:
                assert row.get(field) == before.get(field)
            unchanged += 1
        for row in read(ROOT / folder / "data/service_energy_delay.json"):
            key = (row["bandwidth_MHz"], row["dataset"], row["method"])
            if key in index:
                assert index[key] == row
            index[key] = row
        manifest = read(ROOT / folder / "data/manifest.json")
        for path, digest in manifest["source_sha256"].items():
            assert sha(ROOT/path) == digest, path
        for figure in manifest["figures"]:
            for file in figure["files"]:
                assert sha(ROOT/folder/file["file"]) == file["sha256"]
    assert unchanged == 96
    checks = {}
    for dataset in DATASETS:
        path = OUT / "cases/baseline" / (dataset.lower()+"_seed0") / "learning_component_check.json"
        checks[dataset] = read(path)
        assert checks[dataset]["passed"]
        sources[str(path.relative_to(ROOT))] = sha(path)
    methods = dict(previous.METHODS, **{previous.NEW: "S-OF", "unguarded_ova": "NG-OVA", NEW: LABEL})
    rows, comparisons = [], []
    for case in CASES:
        bandwidth = case["value"] if case["axis"] == "bandwidth" else 1
        for dataset in DATASETS:
            for method, label in methods.items():
                arm = OUT / "cases" / case["id"] / (dataset.lower()+"_seed0") / method
                ledger = index[(bandwidth, dataset, method)]
                row = dict(case=case["id"], bandwidth_MHz=bandwidth, dataset=dataset, method=method,
                           label=label, status=ledger["status"])
                for filename in ["outcome.json", "manifest.json", "started.json"]:
                    path = arm / filename
                    sources[str(path.relative_to(ROOT))] = sha(path)
                if ledger["status"] == "completed":
                    evaluation = read(arm / "manifest.json")["evaluation"]
                    mia = read(arm / "mia.json")
                    sources[str((arm/"mia.json").relative_to(ROOT))] = sha(arm/"mia.json")
                    row.update(retained_mR_percent=100*evaluation["retained"]["mR"],
                        retained_loss=evaluation["retained"]["loss"],
                        forget_KL_to_retrain=evaluation["forget"]["KL_retrain_to_candidate"],
                        worst_request_KL=evaluation["worst_request_KL"],
                        MIA_AUC_gap_vs_retrain=mia["auc_gap_vs_retrain"],
                        total_energy_kJ=ledger["total_energy_j"]/1000, active_service_delay_s=ledger["delay_s"])
                rows.append(row)
            group = {r["method"]: r for r in rows if r["case"]==case["id"] and r["dataset"]==dataset}
            full, partial = group[PROPOSED], group[NEW]
            comparison = dict(case=case["id"], dataset=dataset, status=partial["status"])
            if full["status"] == partial["status"] == "completed":
                comparison.update({key+"_FM_CM_minus_Proposed": partial[key]-full[key] for key in [
                    "retained_mR_percent", "retained_loss", "forget_KL_to_retrain", "worst_request_KL",
                    "MIA_AUC_gap_vs_retrain", "total_energy_kJ", "active_service_delay_s"]})
            comparisons.append(comparison)
    assert len(rows) == 80 and len(comparisons) == 10
    write(OUT / "comparison.json", dict(rows=rows, learning_comparisons=comparisons,
        gradient_checks=checks, source_sha256=sources, unchanged_previous_exported_curves=unchanged,
        statuses=dict(collections.Counter(r["status"] for r in rows))))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    baseline = [r for r in rows if r["case"] == "baseline"]
    paired = [r for r in comparisons if r["case"] == "baseline"]
    lines = ["# Learning-component ablation: Proposed without FU retention RM", "",
        "All compared systems share the standard FL contrastive-learning update. Several also share the full "
        "FU objective; their round-based loss/mR can therefore overlap even when service time differs. "
        "FM-CM isolates a learning component instead of modifying the radio guard.", "",
        "## Fixed comparison", "",
        "| Method | FU objective | FL update | Transport and head/resource selection |",
        "|---|---|---|---|",
        "| Proposed | 3 FM + 1 RM + 3 CM | Shared standard FL | Joint OVA, FU guard, optimized |",
        "| FM-CM | 3 FM + 0 RM + 3 CM | Identical to Proposed | Identical to Proposed |", "",
        "FM is the forgetting term, RM the retained-data contrastive term, and CM teacher consistency. "
        "The FM/RM/CM construction adapts [CLIPErase, ACL 2025](https://aclanthology.org/2025.acl-long.1469/). "
        "FM-CM is a component ablation of that adopted FU objective, not an independent published algorithm "
        "or a newly invented FL loss. Setting RM exactly to zero was registered before the ten outcomes; "
        "no coefficient search or target ranking was used. All original methods and measurements are retained.", "",
        "The FL loss/gradient rule, frozen backbone, adapters, sampling, teacher construction, learning rate, initialization, "
        "200 FL rounds, 45 requests x 5 FU steps, seed, physical limits, hard TX cap, scheduler and guard "
        "remain fixed. A training-only gradient check on each dataset verifies that full minus partial FU "
        "gradient equals an independently calculated RM gradient, and the FL-only update vectors are "
        "bit-identical at the same input model. Later model states and first-service teacher snapshots may "
        "diverge as a consequence of removing RM. Every completed arm verifies 225 FU records with RM coefficient zero.", "",
        "RM is still evaluated diagnostically; its objective/gradient contribution is zero. CM still needs "
        "retained features. The existing compute model counts adapter MACs and does not separately count "
        "contrastive-matrix/softmax arithmetic for either method. Consequently these results do not claim "
        "measured hardware compute savings from removing RM.", "",
        f"New outcomes: {verification['outcomes']}. All five bandwidth values (0.25, 0.5, 1, 2, 4 MHz) "
        "and both datasets are retained. The plotted bandwidth range remains 1–4 MHz, with measured "
        "points at 1, 2 and 4 MHz. The original AirScape H-OVA 4 MHz infeasibility remains marked.", "",
        "## Baseline results at 1 MHz", "",
        "| Dataset | Method | Retained mR (%) | Validation loss | Service energy (kJ) | Service delay (s) | Forget KL | MIA AUC gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in baseline:
        lines.append(f"| {row['dataset']} | {row['label']} | {row['retained_mR_percent']:.6f} | "
            f"{row['retained_loss']:.8f} | {row['total_energy_kJ']:.6f} | {row['active_service_delay_s']:.6f} | "
            f"{row['forget_KL_to_retrain']:.9f} | {row['MIA_AUC_gap_vs_retrain']:.8f} |")
    lines += ["", "Removing RM produces the following partial-minus-full changes:", ""]
    for row in paired:
        lines.append(f"- {row['dataset']}: retained mR {row['retained_mR_percent_FM_CM_minus_Proposed']:+.6f} "
            f"percentage points; validation loss {row['retained_loss_FM_CM_minus_Proposed']:+.8f}; "
            f"forget KL {row['forget_KL_to_retrain_FM_CM_minus_Proposed']:+.9f}.")
    lines += ["", "## Interpretation", "",
        "The matched RM-removal experiment measures the effect of the adopted retained-data FU gradient. "
        "The learning improvement cannot be attributed to parallel transmission: FM-CM uses the same "
        "parallel communication. It also need not fall between Proposed and the safe-proxy baselines. "
        "An intermediate ordering is not a validity criterion, and no curves are adjusted to create one.", "",
        "At the measured baseline, full RM improves retained utility, while FM-CM is closer to the "
        "retained-set retrain on forget-set KL. Higher retained mR alone therefore does not establish "
        "overall unlearning superiority. Forget KL, worst-request KL and MIA gaps are included in the "
        "full comparison. The existing safe-proxy baselines likewise retain their forgetting diagnostics.", "",
        "[Selected figures](../../selected_graph/README.md) now include Proposed, FM-CM, S-AC, AC-OF and S-OF. "
        "[Radio ablations](../../ablation_graph/README.md) retain Proposed, H-OVA, S-OF+ and NG-OVA. "
        f"All {unchanged} previously exported numerical curves and display arrays are unchanged. "
        "Energy/delay use active FL/FU service; energy includes communication, computing and hovering, "
        "excluding travel, collection and preflight. Loss is held-out validation retrieval NLL, not "
        "minibatch training loss. Time uses modeled UAV service time, not experiment wall time.", "",
        "These are single-seed, 200-round validation-subset results, not a significance test or full "
        "benchmark reproduction. All 80 method/condition outcomes are in comparison.json/CSV. "
        "The ten new raw arms are in raw_new_arms.tar.gz with file hashes in raw_archive_manifest.json. "
        "The two training-only gradient checks are also published under cases/baseline. Shared data "
        "features and initial/reference checkpoints stay in their existing locations; this is not a "
        "self-contained dataset release. Extract the archive relative to the repository root.", "",
        "Run: `.venv/bin/python tools/uav_convergence_v2/learning_ablation.py run`. "
        "Rebuild: `.venv/bin/python tools/uav_convergence_v2/learning_ablation_report.py`.", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    import baseline_fairness_report as archive
    archive.OUT, archive.NEW = OUT, NEW
    archive.archive_new_arms()
    print(dict(new_outcomes=verification["outcomes"], unchanged_curves=unchanged,
               baseline_changes=paired), flush=True)


if __name__ == "__main__":
    generate()
