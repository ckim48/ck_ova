"""Report every registered new outcome and retain the matched-objective controls."""
import csv
import math
from pathlib import Path
import subprocess
import tarfile

from baseline_fairness import ROOT, OUT, CHECKED, MAIN, METHODS, NEW, OLD, DATASETS, CASES, read, write, sha, verify
import selected_graphs as plots


def archive_new_arms():
    paths = []
    for case in CASES:
        directory = OUT / "cases" / case["id"]
        paths += [p for p in directory.glob("*.json") if not p.is_symlink()]
        for dataset in DATASETS:
            data = directory / (dataset.lower() + "_seed0")
            paths += [data / "plan.json", data / "requests.json"]
            paths += [p for p in (data / NEW).rglob("*") if p.is_file() and not p.is_symlink()]
    paths = sorted(set(paths))
    manifest = {str(p.relative_to(ROOT)): dict(sha256=sha(p), size_bytes=p.stat().st_size) for p in paths}
    destination = OUT / "raw_new_arms.tar.gz"
    with tarfile.open(destination, "w:gz", compresslevel=6) as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    with tarfile.open(destination) as archive:
        assert set(archive.getnames()) == set(manifest)
        assert all(member.isfile() for member in archive.getmembers())
    write(OUT / "raw_archive_manifest.json", dict(file=destination.name, sha256=sha(destination),
        archive_bytes=destination.stat().st_size, entries=manifest))


def generate():
    verify()
    plots.generate("main")
    plots.generate("ablation")
    ledgers = {}
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for folder in ["selected_graph", "ablation_graph"]:
        manifest = read(ROOT / folder / "data/manifest.json")
        for p, digest in manifest["source_sha256"].items():
            assert sha(ROOT / p) == digest, p
        for figure in manifest["figures"]:
            for file in figure["files"]:
                assert sha(ROOT / folder / file["file"]) == file["sha256"]
        for row in read(ROOT / folder / "data/service_energy_delay.json"):
            key = (row["bandwidth_MHz"], row["dataset"], row["method"])
            if key in ledgers:
                assert row == ledgers[key]
            ledgers[key] = row
    # The old measured trajectories and bandwidth metrics are immutable.
    git = ["git", "--git-dir=.git-push-ready/code-charts.git", "--work-tree=."]
    import json
    previous = json.loads(subprocess.check_output(git + ["show", "760f11132af77d532fed18379c58610520dfb4e8:selected_graph/data/plot_series.json"], cwd=ROOT))
    previous = {(r["figure"], r["dataset"], r["method"]): r for r in previous}
    unchanged = 0
    for folder in ["selected_graph", "ablation_graph"]:
        for row in read(ROOT / folder / "data/plot_series.json"):
            key = (row["figure"], row["dataset"], row["method"])
            if row["method"] == NEW:
                continue
            for field in ["x", "y", "status"]:
                assert row.get(field) == previous[key].get(field), (key, field)
            unchanged += 1
    all_methods = dict(METHODS, **{NEW: "S-OF"})
    rows = []
    for case in CASES:
        bandwidth = case["value"] if case["axis"] == "bandwidth" else 1
        for dataset in DATASETS:
            directory = OUT / "cases" / case["id"] / (dataset.lower() + "_seed0")
            for method, label in all_methods.items():
                arm = directory / method
                ledger = ledgers[(bandwidth, dataset, method)]
                row = dict(case=case["id"], bandwidth_MHz=bandwidth, dataset=dataset, method=method, label=label,
                    role="main and ablation" if label == "Proposed" else "matched-objective ablation" if method in ["heuristic_ova", OLD] else "main",
                    status=ledger["status"], objective=read(arm / "started.json")["policy"]["objective"])
                for filename in ["outcome.json", "started.json", "manifest.json"]:
                    path = arm / filename
                    sources[str(path.relative_to(ROOT))] = sha(path)
                if ledger["status"] == "completed":
                    evaluation = read(arm / "manifest.json")["evaluation"]
                    mia = read(arm / "mia.json")
                    sources[str((arm / "mia.json").relative_to(ROOT))] = sha(arm / "mia.json")
                    metrics = read(arm / "outcome.json")["metrics"]
                    row.update(retained_mR_percent=100 * evaluation["retained"]["mR"],
                        retained_loss=evaluation["retained"]["loss"], forget_mR_percent=100 * evaluation["forget"]["mR"],
                        forget_KL_to_retrain=evaluation["forget"]["KL_retrain_to_candidate"],
                        worst_request_KL=evaluation["worst_request_KL"],
                        MIA_orientation_free_AUC=mia["orientation_free_auc"],
                        MIA_AUC_gap_vs_retrain=mia["auc_gap_vs_retrain"],
                        total_energy_kJ=ledger["total_energy_j"] / 1000, active_service_delay_s=ledger["delay_s"],
                        cap_overruns=metrics["network_cap_overrun_transactions"], cap_excess_J=metrics["network_cap_total_excess_j"])
                rows.append(row)
    assert len(rows) == 60 and sum(r["status"] == "completed" for r in rows) == 59
    baseline = [r for r in rows if r["case"] == "baseline"]
    improvements = []
    for dataset in DATASETS:
        index = {r["method"]: r for r in baseline if r["dataset"] == dataset}
        proposed = index["proposed_fu_guard_cliperase"]
        for method, row in index.items():
            if method == "proposed_fu_guard_cliperase":
                continue
            improvements.append(dict(dataset=dataset, method=method, label=row["label"],
                energy_reduction_percent=100 * (1 - proposed["total_energy_kJ"] / row["total_energy_kJ"]),
                delay_reduction_percent=100 * (1 - proposed["active_service_delay_s"] / row["active_service_delay_s"]),
                retained_mR_gap_percentage_points=proposed["retained_mR_percent"] - row["retained_mR_percent"]))
    write(OUT / "comparison.json", dict(rows=rows, baseline_improvements=improvements,
        source_sha256=sources, original_curves_verified_unchanged=unchanged))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Baseline definitions and matched-objective ablations", "",
        "The original H-OVA and S-OF+ were intentionally close controls, not independent learning baselines. "
        "Sharing the objective/model is appropriate for a communication or head-selection ablation. "
        "The new S-OF adds an existing proxy-objective control without removing or weakening those matched controls.", "",
        "| Method | FU objective | FL/FU transport | Head/resources | Role |",
        "|---|---|---|---|---|",
        "| Proposed | CLIPErase-inspired FM/RM/CM | Joint analog OVA | Optimized | Main and ablation |",
        "| H-OVA | Same as Proposed | Joint analog OVA | Centroid head, uniform inverse noise scale, first feasible frame | Head/resource ablation |",
        "| S-OF+ | Same as Proposed | Serial digital OFDMA, Q6 | Optimized head and integer tones | Transport/scheduler ablation |",
        "| S-AC | Existing safe-proxy FU | Serial AirComp | Existing optimizer | Main baseline |",
        "| AC-OF | Existing safe-proxy FU | AirComp FL, periodic digital OFDMA FU | Existing optimizer | Main baseline |",
        "| S-OF (new) | Existing safe-proxy FU, identical to S-AC/AC-OF | Serial digital OFDMA, Q6 | Identical to S-OF+ | Main baseline |", "",
        "The FM/RM/CM objective adapts [CLIPErase (ACL 2025)](https://aclanthology.org/2025.acl-long.1469/); "
        "it is not claimed as an original objective of this communication method. In the current implementation, "
        "the CLIPErase vector function does not use its `crossmodal` argument. Setting that flag to false would not remove CM. "
        "The new control explicitly selects the existing safe-proxy vector function instead, with no new coefficients.", "",
        "Common IoT placement/routes, model initialization and evaluation galleries are controlled scenario inputs. "
        "They are deliberately shared rather than removed from a comparator. These are project-defined comparison "
        "configurations, not claims of faithful implementations of separately named published baseline algorithms.", "",
        "## Fixed protocol", "",
        "Before the new outcomes, `protocol.json` registered both datasets, seed 0 and all five bandwidth values "
        "(0.25, 0.5, 1, 2, 4 MHz). All ten new arms ran 200 FL rounds and 45 FU requests with five steps each. "
        "Only the FU objective changes relative to S-OF+. The model/features, data/splits, warm20/proxy20/reference200, "
        "FL/FU learning-rate schedules, requests, RNG seeds, placement, channels, digital optimization and energy model are unchanged. "
        "The new arm uses a distinct method ID and result directory. All original results remain immutable; "
        f"{unchanged} exported original curves were checked against the previous published plotting arrays.", "",
        "S-OF and S-OF+ retain the previously authorized soft network-TX-cap fallback: cap-feasible designs are preferred, "
        "but an overrun is permitted and logged when needed, subject to unchanged per-user peak power/energy, deadline, "
        "coherence and quantization-error bounds. Analog arms retain the hard network cap. Thus network-cap policies differ "
        "and should be disclosed when claiming fairness. All new transaction schedules are serial and the recorded FU "
        "local statistics confirm that the safe-proxy branch executed.", "",
        "Ten new arms completed; the combined audit retains all 60 outcomes across six methods. "
        "The existing AirScape H-OVA 4 MHz radio infeasibility remains visible in the ablation figures.", "",
        "## Baseline results at 1 MHz", "",
        "Energy and delay below cover the active FL/FU service windows, including scheduling waits. "
        "Energy includes communication, computing and hovering; travel, collection and preflight preparation are excluded.", "",
        "| Dataset | Method | mR (%) | Loss | Energy (kJ) | Delay (s) | Forget KL to retrain | MIA AUC gap to retrain |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in baseline:
        lines.append(f"| {row['dataset']} | {row['label']} | {row['retained_mR_percent']:.4f} | {row['retained_loss']:.6f} | "
            f"{row['total_energy_kJ']:.3f} | {row['active_service_delay_s']:.3f} | {row['forget_KL_to_retrain']:.8f} | {row['MIA_AUC_gap_vs_retrain']:.8f} |")
    lines += ["", "## Interpretation", "",
        "Changing the S-OF FU objective reduces retained mR slightly at the baseline, but leaves its modeled energy "
        "and delay essentially unchanged. The main benefit of joint scheduling is therefore already visible against "
        "the matched-objective S-OF+ control; a different loss is not needed to demonstrate that timing effect.", "",
        "Proposed has higher retained mR at the baseline, while safe-proxy S-OF is closer to the retained-set retrain "
        "on forget-set KL and the fixed MIA AUC gap. This is a utility/forgetting tradeoff, not uniform superiority. "
        "KL proximity is a diagnostic, and the fixed retrieval-loss membership attack is descriptive rather than a "
        "certified deletion/privacy guarantee. The main comparison varies the objective and cannot isolate the "
        "communication contribution; use H-OVA and S-OF+ for matched-objective claims. Single-seed subset results "
        "do not establish statistical significance or full benchmark performance.", "",
        "| Dataset | Comparator | Proposed energy reduction (%) | Proposed delay reduction (%) | mR gap (percentage points) |",
        "|---|---|---:|---:|---:|"]
    for row in improvements:
        lines.append(f"| {row['dataset']} | {row['label']} | {row['energy_reduction_percent']:.3f} | {row['delay_reduction_percent']:.3f} | {row['retained_mR_gap_percentage_points']:.4f} |")
    lines += ["", "See [selected_graph](../../selected_graph/README.md) for the four-method main figures and "
        "[ablation_graph](../../ablation_graph/README.md) for the matched-objective controls. "
        "`comparison.json` and `.csv` contain every bandwidth outcome and quality diagnostic, not only the baseline.", "",
        "Reproduce the registered runs with `.venv/bin/python tools/uav_convergence_v2/baseline_fairness.py run`; "
        "completed outcomes are preserved. Rebuild figures/audit with "
        "`.venv/bin/python tools/uav_convergence_v2/baseline_fairness_report.py`.", "",
        "`raw_new_arms.tar.gz` contains the new arms' recorded transactions, work attempts, evaluations, checkpoints "
        "and scenario inputs, using paths relative to the repository root. `raw_archive_manifest.json` lists hashes. "
        "Shared feature caches, initial/reference checkpoints and reused original arms remain in their original "
        "locations; they are not duplicated in this archive. The archive alone is not a self-contained dataset release.", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    archive_new_arms()
    print(dict(outcomes=len(rows), unchanged_original_curves=unchanged, baseline_improvements=improvements), flush=True)


if __name__ == "__main__":
    generate()
