"""Report every screened learning pilot, including unsuccessful candidates."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/iot_sensitivity_v1"))
from common import read, write, sha
from report import configure
import selected_pair as style

OUT = ROOT / "results/uav_convergence_v2_20261027"
PILOTS = {"CapERA": "capera_pilot", "UAV-Construction": "construction_pilot", "AirScape": "airscape_pilot"}


def generate():
    results, rows, sources = {}, [], {}
    for dataset, folder in PILOTS.items():
        dest = OUT / folder
        if not (dest / "finished.json").exists():
            continue
        result = read(dest / "finished.json")
        trajectory = read(dest / "rounds.json")
        assert trajectory[0]["round"] == 0 and trajectory[-1]["round"] == 200
        assert trajectory[-1]["evaluation"]["retained"]["mR"] == result["final_mR"]
        results[dataset] = result
        for row in trajectory:
            values = row["evaluation"]["retained"]
            rows.append(dict(dataset=dataset, round=row["round"], mR_percent=100 * values["mR"], loss=values["loss"]))
        for name in ["rounds.json", "protocol.json", "finished.json"]:
            sources[f"{folder}/{name}"] = sha(dest / name)
    write(OUT / "pilot_summary.json", dict(results=results, source_sha256=sources,
        scope="Exploratory ideal-link FL-only screening. Not Proposed, not a five-method ranking.",
        limitation="Different native caption counts and galleries; compare learning change within "
                   "each dataset, not absolute cross-dataset difficulty. Single seed; validation is "
                   "used for screening, so any selected replacement needs independent confirmation."))
    if rows:
        with (OUT / "pilot_curves.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    figures = []
    if "AirScape" in results and "CapERA" in results:
        (OUT / "figures").mkdir(exist_ok=True)
        style.DEST, style.DATASETS = OUT, ["CapERA", "AirScape"]
        configure()
        for key, ylabel in [("mR_percent", "Retained mR (%)"), ("loss", "Retained loss")]:
            fig, axes = style.canvas(ylabel)
            for ax, dataset in zip(axes, style.DATASETS):
                selected = [r for r in rows if r["dataset"] == dataset]
                style.curve(ax, [r["round"] for r in selected], [r[key] for r in selected], 0, "FL pilot")
                ax.set_xlabel("Completed FL rounds")
                ax.set_xlim(0, 200)
                ax.set_xticks([0, 50, 100, 150, 200])
            figures += style.save(fig, axes, "pilot_" + key + "_capera_airscape_1x2")
        write(OUT / "pilot_figures.json", figures)
    lines = ["# UAV visual-language convergence screening", "",
        "The user requested a more suitable actual UAV dataset whose retained mR improves with "
        "training. Every completed pilot below is retained, including negative results.", "",
        "All pilots use the unchanged frozen RemoteCLIP ViT-B/32 encoder, rank10 residual adapters, "
        "paired seed0, 10 clients, batch16, SGD and original200-round learning-rate schedule. "
        "They use ideal aggregation and FL only: no radio noise, FU, mission costs or method ranking. "
        "A pilot curve must never be labeled as Proposed.", "",
        "| Dataset | Initial mR (%) | Final mR (%) | Change (percentage points) |",
        "|---|---:|---:|---:|"]
    for dataset, r in results.items():
        lines.append(f"| {dataset} | {r['initial_mR']*100:.4f} | {r['final_mR']*100:.4f} | {r['gain_percentage_points']:+.4f} |")
    lines += ["", "## Data and scope", "",
        "- CapERA is the unchanged existing video-caption subset, five native captions/video.",
        "- [UAV-Construction](https://data.mendeley.com/datasets/4h68fmktwh/1) comes from "
        "Bang and Kim, Automation in Construction2020, DOI10.1016/j.autcon.2020.103116. "
        "The official archive is checksum verified. These are native bounding-box region captions; "
        "this pilot uses two largest valid regions/image, not whole-image captions. Filename blocks "
        "reduce adjacent-frame overlap but do not establish geographic independence. One text/region.",
        "- [AirScape](https://huggingface.co/datasets/EmbodiedCity/AirScape-Dataset), "
        "[ACM MM2025](https://doi.org/10.1145/3746027.3758180), contains real UAV video/intention pairs. "
        "The selected subset uses original WebUAV3M-source footage, one text/video, one clip/source "
        "sequence, 1536 train and256 validation sequences. Entire sequences with duplicate media "
        "or overlap with official-test source IDs/hashes are excluded. No test video is downloaded. "
        "Mean pooling of eight frames is our retrieval adaptation; it does not model temporal order "
        "and is not the published world-model task. Captions are untouched. AirScape metadata states MIT "
        "and lists GPLv3 for source footage; the current original WebUAV3M repository instead specifies "
        "CC BY-NC-SA4.0 for its dataset. Original media and caption text are not redistributed here. "
        "Named-sequence and file-hash checks do not establish geographical independence.", "",
        "## Caption and implementation audit", "",
        "With the existing77-token CLIP tokenizer, all8412 GeoText captions exceed context "
        "(median162 tokens), compared with17.27% of UAVReason and0.0136% of CapERA training captions. "
        "Truncation is a plausible contributor, not a proven sole explanation for flat mR. "
        "See `existing_caption_audit.json` and `airscape_data_audit.json`. "
        "AirScape sequential decoding was verified pixel-exact against the original seek-based "
        "decoder on the first10 predetermined samples at all eight selected frames.", "",
        "The AirScape continuation gate was saved before its first learning run: at least1.0pp "
        "final improvement and0.8pp improvement in the last-five-evaluation mean. This is an "
        "exploratory screen against the matched CapERA gain, not an independent significance test. "
        "AirScape did not pass that strong-gain gate (+0.4557pp). A subsequent decision continues "
        "the five-method sensitivity study only as an exploratory follow-up: its raw mR rises "
        "and settles, but the increase is smaller than CapERA's. The gate is not lowered or "
        "reported as passed. See `airscape_continuation_decision.json`. "
        "No favorable seed, checkpoint, caption rewrite, smoothed curve or official-test feedback is selected.", "",
        "UAV-Captions original images were not located; AeroCaps requires manual gated access. "
        "These are availability limitations, not negative learning results. See `candidate_search.json`.", "",
        "`pilot_curves.csv` includes all raw observations. `pilot_summary.json` links source hashes. "
        "The two panel figures, when present, show matched learning-only CapERA/AirScape pilots; "
        "the five-method figures belong to the separate sensitivity campaign.", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    return results


if __name__ == "__main__":
    print(generate())
