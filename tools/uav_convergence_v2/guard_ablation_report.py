"""Audit overlap and report the registered no-guard control without rank selection."""
import collections
import csv
import itertools
import json
from pathlib import Path
import subprocess

import guard_ablation as study
from guard_ablation import ROOT, OUT, SOURCE, NEW, LABEL, PROPOSED, DATASETS, CASES, read, write, sha
import baseline_fairness as previous
import selected_graphs as plots


def overlap_audit():
    import torch
    rows, sources = [], {}
    for dataset in DATASETS:
        directory = SOURCE / "cases/baseline" / (dataset.lower()+"_seed0")
        warm_path = directory / "initial20/final.pt"
        sources[str(warm_path.relative_to(ROOT))] = sha(warm_path)
        warm = torch.load(warm_path, map_location="cpu", weights_only=False)["theta"][:-1].double()
        models = {}
        for method in ["separate_aircomp_periodic", "aircomp_ofdma_periodic", previous.NEW]:
            path = directory / method / "final.pt"
            sources[str(path.relative_to(ROOT))] = sha(path)
            models[method] = torch.load(path, map_location="cpu", weights_only=False)["theta"][:-1].double()
        for first, second in itertools.combinations(models, 2):
            a, b = models[first], models[second]
            difference = float((a-b).norm())
            assert difference > 0 and not torch.equal(a, b)
            rows.append(dict(dataset=dataset, first=first, second=second, parameters_exactly_equal=False,
                parameter_difference_L2=difference,
                difference_relative_to_first_adaptation_percent=100*difference/float((a-warm).norm())))
    result = dict(rows=rows, source_sha256=sources,
        definition="Euclidean difference of the 20,480 trained adapter coordinates, divided by first model minus shared warm20. The frozen logit-scale coordinate is excluded.",
        explanation="These independently simulated models differ, but shared objective/sampling/LR and tight transport-error bounds yield very similar retrieval rankings. Equal discrete Recall values do not imply identical models.")
    write(OUT / "overlap_audit.json", result)
    return result


def generate():
    verification = study.verify()
    overlap = overlap_audit()
    plots.generate("main")
    plots.generate("guard_ablation")
    index, sources = {}, {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    git = ["git", "--git-dir=.git-push-ready/code-charts.git", "--work-tree=."]
    unchanged = 0
    for folder in ["selected_graph", "ablation_graph"]:
        before = json.loads(subprocess.check_output(git + ["show", f"2dcded8f2c945ff5f7697bc45d934a36bfe7b59e:{folder}/data/plot_series.json"], cwd=ROOT))
        before = {(r["figure"],r["dataset"],r["method"]):r for r in before}
        for row in read(ROOT / folder / "data/plot_series.json"):
            if row["method"] == NEW:
                continue
            old = before[(row["figure"],row["dataset"],row["method"])]
            for field in ["x", "y", "status"]:
                assert row.get(field) == old.get(field)
            unchanged += 1
        for row in read(ROOT / folder / "data/service_energy_delay.json"):
            key = (row["bandwidth_MHz"],row["dataset"],row["method"])
            if key in index:
                assert index[key] == row
            index[key] = row
        manifest = read(ROOT / folder / "data/manifest.json")
        for path, digest in manifest["source_sha256"].items():
            assert sha(ROOT/path) == digest, path
        for figure in manifest["figures"]:
            for file in figure["files"]:
                assert sha(ROOT/folder/file["file"]) == file["sha256"]
    convergence_gaps = []
    curves = read(ROOT / "ablation_graph/data/plot_series.json")
    for dataset in DATASETS:
        for figure in ["01_round_fl_loss_1x2", "02_round_mr_1x2"]:
            a = next(r for r in curves if r["dataset"]==dataset and r["figure"]==figure and r["method"]==PROPOSED)
            b = next(r for r in curves if r["dataset"]==dataset and r["figure"]==figure and r["method"]==NEW)
            assert a["x"] == b["x"]
            gaps = [abs(y-z) for y,z in zip(a["y"],b["y"])]
            convergence_gaps.append(dict(dataset=dataset, figure=figure, maximum_absolute_gap=max(gaps),
                different_observations=sum(g>0 for g in gaps), observations=len(gaps)))
    methods = dict(previous.METHODS, **{previous.NEW: "S-OF", NEW: LABEL})
    rows, comparisons = [], []
    for case in CASES:
        bandwidth = case["value"] if case["axis"] == "bandwidth" else 1
        for dataset in DATASETS:
            for method, label in methods.items():
                arm = OUT / "cases" / case["id"] / (dataset.lower()+"_seed0") / method
                ledger = index[(bandwidth,dataset,method)]
                row = dict(case=case["id"], bandwidth_MHz=bandwidth, dataset=dataset, method=method, label=label, status=ledger["status"])
                for filename in ["outcome.json", "manifest.json", "started.json"]:
                    path = arm / filename; sources[str(path.relative_to(ROOT))] = sha(path)
                if ledger["status"] == "completed":
                    evaluation = read(arm / "manifest.json")["evaluation"]
                    mia = read(arm / "mia.json")
                    sources[str((arm/"mia.json").relative_to(ROOT))] = sha(arm/"mia.json")
                    row.update(retained_mR_percent=100*evaluation["retained"]["mR"], retained_loss=evaluation["retained"]["loss"],
                        forget_KL_to_retrain=evaluation["forget"]["KL_retrain_to_candidate"], worst_request_KL=evaluation["worst_request_KL"],
                        MIA_AUC_gap_vs_retrain=mia["auc_gap_vs_retrain"],
                        total_energy_kJ=ledger["total_energy_j"]/1000, active_service_delay_s=ledger["delay_s"])
                rows.append(row)
            group = {r["method"]:r for r in rows if r["case"]==case["id"] and r["dataset"]==dataset}
            a,b = group[PROPOSED],group[NEW]
            comparison = dict(case=case["id"],dataset=dataset,status=b["status"])
            if a["status"] == b["status"] == "completed":
                comparison.update({key+"_NG_minus_Proposed":b[key]-a[key] for key in [
                    "retained_mR_percent", "retained_loss", "forget_KL_to_retrain", "worst_request_KL", "total_energy_kJ", "active_service_delay_s"]})
            comparisons.append(comparison)
    assert len(rows) == 70 and len(comparisons) == 10
    write(OUT / "comparison.json", dict(rows=rows, guard_comparisons=comparisons, baseline_convergence_gaps=convergence_gaps, source_sha256=sources,
        unchanged_previous_exported_curves=unchanged, statuses=dict(collections.Counter(r["status"] for r in rows))))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    complete = [r for r in comparisons if r["status"]=="completed"]
    maximum_mr = max(abs(r["retained_mR_percent_NG_minus_Proposed"]) for r in complete)
    maximum_loss = max(abs(r["retained_loss_NG_minus_Proposed"]) for r in complete)
    baseline = [r for r in rows if r["case"] == "baseline"]
    lines = ["# FU-noise-guard ablation and overlapping learning curves", "",
        "S-AC, AC-OF and S-OF share the same safe-proxy FU objective, sampled data and FL/FU learning-rate schedule. "
        "Their communication policies differ. The close round-based loss/mR curves are measured outcomes, not copies "
        "of one model: `overlap_audit.json` verifies nonzero differences in all three pairs of final adapter tensors "
        "on both datasets. S-AC versus AC-OF differs by about 0.10% (CapERA) and 0.16% (AirScape) of S-AC's adaptation "
        "from warm20. Mean recall depends on discrete retrieval ranks, so distinct parameters can give exactly equal mR.", "",
        "## Registered component removal", "",
        "**NG-OVA** keeps Proposed's CLIPErase-inspired FU objective, joint FL/FU scheduler, optimized head/resources, "
        "data, workload, seeds and physical limits. It removes only the FU-relative error guard and uses the existing "
        "joint component-noise budget. This is a component ablation, not a separately published algorithm. "
        "No noise coefficient or objective weight was adjusted to force a middle rank.", "",
        "The ten arms (two datasets x five bandwidth values) were fixed in `protocol.json` before outcomes. "
        "All original measurements remain immutable. Numerical candidate screening uses the existing final receiver "
        "1ppm peak-power tolerance; physical constraints remain unchanged. Every new transaction is audited for the "
        "unguarded budget, joint scheduling, hard network cap and actual FM/RM/CM loss branch.", "",
        f"New outcomes: {verification['outcomes']}. Among completed paired conditions, the maximum absolute final-round NG-OVA "
        f"minus Proposed mR difference is **{maximum_mr:.6f} percentage points**, and the maximum final loss difference "
        f"is **{maximum_loss:.8f}**. This measures the component's effect in this scenario; it does not establish "
        "that the guard is unnecessary under all channel/error/workload conditions. No intermediate ordering is assumed. "
        "At the 1 MHz baseline, the largest mR difference anywhere on the saved round curve is 0.065104 percentage "
        "points for CapERA and zero for AirScape; both final mR values are equal.", "",
        f"NG-OVA reduces service energy by up to {-min(r['total_energy_kJ_NG_minus_Proposed'] for r in complete):.6f} kJ "
        f"and service delay by up to {-min(r['active_service_delay_s_NG_minus_Proposed'] for r in complete):.6f} s "
        "over the tested bandwidth conditions, with no observed final-mR penalty. The guard tightens the mathematical "
        "error bound, but this experiment does not demonstrate an empirical retained-utility benefit from that guard. "
        "This unfavorable result is retained alongside the communication/scheduling benefits.", "",
        "## Baseline at 1 MHz", "",
        "| Dataset | Method | mR (%) | Loss | FL/FU energy (kJ) | FL/FU delay (s) | Forget KL |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in baseline:
        lines.append(f"| {row['dataset']} | {row['label']} | {row['retained_mR_percent']:.6f} | {row['retained_loss']:.8f} | "
                     f"{row['total_energy_kJ']:.6f} | {row['active_service_delay_s']:.6f} | {row['forget_KL_to_retrain']:.9f} |")
    lines += ["", "## Interpretation and figures", "",
        "The existing main baselines already occupy different energy/delay levels. A method need not lie between "
        "the others on round-based accuracy to be a valid communication comparator. Shared-objective ablations "
        "test whether communication/head/error-budget changes preserve learning while changing modeled costs. "
        "The objective difference between Proposed and safe-proxy baselines prevents attributing all mR gains to "
        "parallel transmission. The safe-proxy methods are closer to the retained-set retrain on the baseline "
        "forget-set KL, so higher retained mR alone is not overall superiority or a deletion guarantee.", "",
        "[Main figures](../../selected_graph/README.md) retain Proposed/S-AC/AC-OF/S-OF with their original measured "
        "curves. [Ablation figures](../../ablation_graph/README.md) add NG-OVA beside Proposed/H-OVA/S-OF+. "
        "The existing AirScape H-OVA 4 MHz infeasibility remains explicit. Curves are neither separated artificially "
        f"nor selected by rank; {unchanged} prior exported curves retain exactly the same numerical arrays.", "",
        "Energy covers active FL/FU communication, computing and hovering; delay covers the same service windows. "
        "Travel, collection and preflight preparation are excluded. These are single-seed, 200-round, validation-subset "
        "results, not a statistical significance claim or a full benchmark reproduction.", "",
        "All 70 outcomes and quality diagnostics are in `comparison.json`/`.csv`. The ten new raw trajectories "
        "are archived in `raw_new_arms.tar.gz` with file hashes in `raw_archive_manifest.json`; shared features and "
        "initial/reference checkpoints remain in their existing locations. Extract relative to the repository root. "
        "The archive is not a self-contained dataset release.", "",
        "Run: `.venv/bin/python tools/uav_convergence_v2/guard_ablation.py run`. "
        "Rebuild: `.venv/bin/python tools/uav_convergence_v2/guard_ablation_report.py`.", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    import baseline_fairness_report as archive
    archive.OUT, archive.NEW = OUT, NEW
    archive.archive_new_arms()
    print(dict(new_outcomes=verification["outcomes"], maximum_mR_gap_percentage_points=maximum_mr,
        maximum_loss_gap=maximum_loss, unchanged_previous_curves=unchanged, guard_comparisons=comparisons), flush=True)


if __name__ == "__main__":
    generate()
