"""Restyle completed mission results without changing registered experiments.

Uses the user's plot_zone_pred_1x2.py palette, serif typography, dotted grid,
and full frame. Each panel is square; bar values and in-panel notes are omitted.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/iot_uav_remoteclip_v4_event_mission"
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
FIGSIZE = (10.2, 5.4)
CHARTS = [
    ("loss", "Retained loss", 1.0, "loss", 0.5),
    ("total_energy_j", "Total energy (kJ)", 1000.0, "total_energy", 500.0),
    ("total_delay_s", "Total delay (s)", 1.0, "total_delay", 500.0),
    ("FU_mean_completion_latency_s", "Mean FU completion latency (s)",
     1.0, "fu_completion_latency", 1.0),
    ("energy_hover_j", "Hovering energy (kJ)", 1000.0, "hover_energy", 100.0),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def configure():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 14,
        "axes.labelsize": 14, "axes.labelweight": "normal",
        "axes.titlesize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.linewidth": 0.9, "axes.spines.top": True,
        "axes.spines.right": True, "grid.linewidth": 0.5,
        "grid.color": "#C4C4C4", "lines.linewidth": 1.6,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "legend.frameon": False, "savefig.pad_inches": 0.08,
    })


def plot(rows, protocol, output):
    configure()
    labels = protocol["plotting"]["labels"]
    datasets = protocol["datasets"]
    assert len(labels) == 4 and len(datasets) == 2
    rendered = []
    for key, ylabel, scale, name, tick_step in CHARTS:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
        fig.subplots_adjust(left=0.095, right=0.985, bottom=0.16,
                            top=0.91, wspace=0.32)
        common_max = max(r[key] / scale for r in rows)
        limits = []
        for ax, dataset in zip(axes, datasets):
            subset = {r["method"]: r for r in rows if r["dataset"] == dataset}
            values = np.array([subset[m][key] / scale for m in labels])
            assert np.all(np.isfinite(values)) and np.all(values >= 0)
            ax.bar(range(4), values, width=0.58, color=COLORS,
                   edgecolor=COLORS, linewidth=0.9, zorder=3)
            ax.set_xticks(range(4), list(labels.values()))
            ax.set_xlim(-0.6, 3.6)
            # All bar axes start at zero. Use a common scale where meaningful.
            ymax = float(values.max()) if key == "loss" else common_max
            step = 1.0 if key == "loss" and ymax >= 4 else tick_step
            ymax = math.ceil(ymax * 1.07 / step) * step
            ax.set_ylim(0, ymax)
            ax.yaxis.set_major_locator(MultipleLocator(step))
            ax.set_ylabel(ylabel, labelpad=9)
            ax.set_title(dataset, pad=10)
            ax.set_box_aspect(1)
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", linestyle=":", zorder=0)
            ax.tick_params(direction="out", pad=6)
            assert not ax.texts  # No values on/above bars or stray annotations.
            assert np.allclose([p.get_height() for p in ax.patches], values)
            limits.append([0, ymax])
        fig.canvas.draw()
        panel_sizes = []
        for ax in axes:
            box = ax.get_window_extent()
            assert math.isclose(box.width, box.height, abs_tol=1e-6)
            panel_sizes.append([box.width / fig.dpi, box.height / fig.dpi])
        files = []
        for extension in ["png", "pdf", "svg"]:
            path = output / f"{name}_1x2.{extension}"
            fig.savefig(path, dpi=400, bbox_inches="tight")
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in
                                          path.read_text().splitlines()) + "\n")
            files.append({"file": path.name, "sha256": sha(path)})
        rendered.append({"metric": key, "panel_size_inches": panel_sizes,
                         "ylim": limits, "bar_annotations": 0, "files": files})
        plt.close(fig)
    return rendered


def energy_audit(campaign, rows, protocol, output):
    """Reconcile energy differences with actual transaction and mission clocks."""
    labels = protocol["plotting"]["labels"]
    hardware = protocol["energy_model"]["propulsion"]
    hover_w = protocol["config"]["users"] * (hardware["P0_w"] + hardware["Pi_w"])
    result, evidence = [], {}
    for dataset in protocol["datasets"]:
        subset = {r["method"]: r for r in rows if r["dataset"] == dataset}
        proposed = subset[next(iter(labels))]
        for method, label in labels.items():
            r = subset[method]
            folder = campaign / (dataset.lower() + "_seed0") / method
            paths = sorted((folder / "transactions").glob("*.json"))
            tx = [read(p) for p in paths]
            timeline_path = folder / "mission_timeline.json"
            timeline = read(timeline_path)
            for path in [*paths, timeline_path, folder / "energy_and_delay.json"]:
                evidence[str(path.relative_to(campaign))] = sha(path)
            timing = {k: sum(t["timing"][k] for t in tx)
                      for k in ["compute_s", "uplink_s", "downlink_s", "duration_s"]}
            hover_s = sum(r["hover_by_site_s"])
            wait_s = hover_s - r["iot_collection_seconds"] - timing["duration_s"]
            assert abs(wait_s) < 1e-7, (dataset, method, wait_s)
            assert math.isclose(r["energy_hover_j"], hover_w * hover_s, abs_tol=1e-6)
            assert math.isclose(r["total_energy_j"], sum(r[k] for k in
                ["energy_hover_j", "energy_travel_j", "energy_computing_j",
                 "energy_communication_j"]), abs_tol=1e-6)
            assert math.isclose(r["energy_travel_j"], proposed["energy_travel_j"], abs_tol=1e-6)
            result.append({
                "dataset": dataset, "method": label,
                "transactions": len(tx),
                "nonzero_transactions": sum(t["radio"]["uplink_s"] > 0 for t in tx),
                "hover_s": hover_s, "compute_s": timing["compute_s"],
                "uplink_s": timing["uplink_s"], "downlink_s": timing["downlink_s"],
                "collection_s": r["iot_collection_seconds"],
                "travel_s": sum(t["duration_s"] for t in timeline["legs"]),
                "hover_kJ": r["energy_hover_j"] / 1000,
                "travel_kJ": r["energy_travel_j"] / 1000,
                "compute_kJ": r["energy_computing_j"] / 1000,
                "communication_kJ": r["energy_communication_j"] / 1000,
                "total_kJ": r["total_energy_j"] / 1000,
                "hover_s_saved_by_proposed": hover_s - sum(proposed["hover_by_site_s"]),
                "hover_kJ_saved_by_proposed": (r["energy_hover_j"] - proposed["energy_hover_j"]) / 1000,
                "total_kJ_saved_by_proposed": (r["total_energy_j"] - proposed["total_energy_j"]) / 1000,
                "hover_reduction_percent": 100 * (1 - proposed["energy_hover_j"] / r["energy_hover_j"]),
                "total_reduction_percent": 100 * (1 - proposed["total_energy_j"] / r["total_energy_j"]),
            })
    save_json(output / "hover_energy_audit.json", {
        "scope": "Declared simulation model; ten UAVs. Energy includes ground IoT communication. No power constants changed.",
        "hover_power_per_uav_W": hover_w / protocol["config"]["users"],
        "hover_power_total_W": hover_w, "checks_passed": True,
        "rows": result, "input_sha256": evidence,
    })
    with (output / "hover_energy_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = campaign / "figures"
    paths = [campaign / "protocol.json", output / "metrics.json", output / "metrics.csv"]
    before = {str(path.relative_to(campaign)): sha(path) for path in paths}
    protocol, metrics = read(paths[0]), read(paths[1])
    assert not metrics["missing"] and len(metrics["rows"]) == 8
    rendered = plot(metrics["rows"], protocol, output)
    audit = energy_audit(campaign, metrics["rows"], protocol, output)
    assert before == {str(path.relative_to(campaign)): sha(path) for path in paths}
    save_json(output / "reference_style_manifest.json", {
        "reference": "User attachment plot_zone_pred_1x2.py (2026-10-26)",
        "figure_size_inches": FIGSIZE, "colors": COLORS,
        "font_points": 14, "grid": "dotted", "full_frame": True,
        "bar_value_labels": False, "input_sha256": before,
        "source": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": sha(Path(__file__)), "rendered": rendered,
    })
    print(json.dumps({"figures": len(rendered), "square_panels": True,
                      "bar_value_labels": False, "unchanged_experiment_inputs": True,
                      "CapERA_energy_audit": [r for r in audit if r["dataset"] == "CapERA"]}, indent=2))


if __name__ == "__main__":
    main()
