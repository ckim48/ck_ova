"""Render scientific plots from measured audit records without interpolation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"hybrid": "Hybrid NOMA-AirComp", "orthogonal_equal_time": "Phase-separated (50/50)",
          "mixed_aircomp": "Summed-update AirComp"}
COLORS = {"hybrid": "#2463a6", "orthogonal_equal_time": "#b85f17", "mixed_aircomp": "#207a55"}


def radio(directory):
    with (directory / "radio_audit.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads((directory / "manifest.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for method, label in LABELS.items():
        attempted = [r for r in rows if r["method"] == method]
        if attempted and not any(r["status"] == "feasible" for r in attempted):
            axes[0].scatter([], [], color=COLORS[method], marker="x",
                            label=f"{label}: 0/{len(attempted)} designs found")
        for i, starts in enumerate(manifest["receive_initializations"]):
            selected = [r for r in rows if r["method"] == method and int(r["receive_initializations"]) == starts
                        and r["status"] == "feasible"]
            selected.sort(key=lambda r: float(r["relative_tolerance"]))
            if not selected:
                continue
            x = [float(r["relative_tolerance"]) for r in selected]
            energy = [1000 * float(r["energy_bound_j"]) for r in selected]
            rmse = [float(r["analytic_total_rmse"]) for r in selected]
            ratio = [float(r["analytic_total_rmse"]) / float(r["epsilon"]) for r in selected]
            style = {"color": COLORS[method], "marker": "o" if i == 0 else "s", "linestyle": "-" if i == 0 else "--",
                     "label": f"{label}, {starts} starts", "linewidth": 1.4, "markersize": 5}
            axes[0].plot(x, energy, **style)
            axes[1].plot(x, ratio, **style)
            # Scatter-only: no implied achievable continuum between configurations.
            axes[2].scatter(rmse, energy, color=COLORS[method], marker=style["marker"], s=35)
    axes[0].set(xlabel="Relative update-error tolerance", ylabel="Communication energy bound (mJ)", yscale="log")
    axes[1].set(xlabel="Relative update-error tolerance", ylabel="Analytic update RMSE / allowed RMSE")
    axes[1].axhline(1, color="0.5", linestyle=":", linewidth=1)
    axes[2].set(xlabel="Analytic total-update RMSE", ylabel="Communication energy bound (mJ)", xscale="log", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize=7)
    failed = sum(r["status"] != "feasible" for r in rows)
    fig.suptitle(f"Fixed actual CLIP updates, legacy round {manifest['round']} | {len(rows)} attempts, {failed} failed searches\n"
                 "Same channels and total time; CPU reconstruction; no task-quality conclusion", fontsize=10)
    for extension in ("png", "svg"):
        fig.savefig(directory / ("paired_radio_audit." + extension), dpi=180)
    plt.close(fig)


def quality(directory):
    report = json.loads((directory / "energy_quality_report.json").read_text())
    eligible = [x for x in report["methods"] if x.get("status") == "completed"
                and x.get("energy_role") == "physical_model" and x.get("forget_kl_reference_to_candidate") is not None]
    if not eligible:
        print("QUALITY_PLOT_PENDING_REFERENCE", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for row in eligible:
        energy = 1000 * row["energy_bound_j"]
        for ax, metric in zip(axes, ("validation_loss", "forget_kl_reference_to_candidate")):
            ax.scatter(row[metric], energy, s=45)
            ax.annotate(row["method"], (row[metric], energy), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].set(xlabel="Retained-task utility validation loss", ylabel="Cumulative communication energy bound (mJ)")
    axes[1].set(xlabel="Forget-cohort KL(reference || candidate)", ylabel="Cumulative communication energy bound (mJ)")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle("Observed final-time energy/quality points | physical-link methods only\n"
                 "Lower-left is preferable; one seed; no certified forgetting claim", fontsize=10)
    for extension in ("png", "svg"):
        fig.savefig(directory / ("energy_quality." + extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio", type=Path)
    parser.add_argument("--quality", type=Path)
    args = parser.parse_args()
    if args.radio:
        radio(args.radio)
    if args.quality:
        quality(args.quality)


if __name__ == "__main__":
    main()
