import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, StrMethodFormatter


ROOT = Path(__file__).resolve().parent
STYLES = [
    ("coupled", "Proposed", "#6B92DE", "d", "-"),
    ("uncoupled", "w/o Coupling", "#F2A460", "o", "-"),
    ("puf_regular_adapted", "PUF (adapted)", "#ED7277", "s", "--"),
    ("retain_only", "Retain-only", "#80BC80", "^", ":"),
]


def main():
    source = ROOT / "results/learning/round_metrics.csv"
    destination = ROOT / "results/figures"
    destination.mkdir(parents=True, exist_ok=True)
    with source.open() as handle:
        records = list(csv.DictReader(handle))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.linewidth": 0.85,
        "axes.edgecolor": "#444444",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    figure.subplots_adjust(left=0.10, right=0.975, bottom=0.30, top=0.80, wspace=0.40)
    exported = []
    panels = [
        ("test_loss", "Test loss", "(a) Retained-task evaluation", (0, 100), [0, 25, 50, 75, 100]),
        ("forget_loss", "Forget-set loss", "(b) Cumulative forget-set evaluation",
         (50, 100), [50, 60, 70, 80, 90, 100]),
    ]
    for axis, (metric, ylabel, caption, limits, ticks) in zip(axes, panels):
        all_values = []
        for style_index, (method, label, color, marker, linestyle) in enumerate(STYLES):
            matching = [row for row in records if row["method"] == method and row[metric]]
            rounds = sorted({int(row["round"]) for row in matching})
            means = []
            for round_number in rounds:
                selected = [row for row in matching if int(row["round"]) == round_number]
                seeds = {int(row["seed"]) for row in selected}
                if seeds != {0, 1, 2} or len(selected) != 3:
                    raise ValueError("Each plotted point must contain exactly seeds 0, 1, 2.")
                values = np.array([float(row[metric]) for row in selected])
                if not np.all(np.isfinite(values)) or np.any(values <= 0):
                    raise ValueError("Logarithmic loss plots require finite positive values.")
                means.append(float(values.mean()))
                exported.append({
                    "metric": metric, "method": method, "round": round_number,
                    "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                    "seeds": 3,
                })
            all_values.extend(means)
            axis.plot(
                rounds, means, color=color, linestyle=linestyle, linewidth=1.25,
                marker=marker, markersize=4.3, markerfacecolor="white",
                markeredgewidth=0.95, markevery=(style_index, 4),
                label=label, zorder=6 - style_index,
            )
        lower, upper = min(all_values), max(all_values)
        padding = max((upper - lower) * 0.16, lower * 0.001)
        lower = np.floor((lower - padding) * 1000) / 1000
        upper = np.ceil((upper + padding) * 1000) / 1000
        axis.set_yscale("log")
        axis.set_ylim(lower, upper)
        axis.yaxis.set_major_locator(FixedLocator(np.linspace(lower, upper, 4)))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_xlim(*limits)
        axis.set_xticks(ticks)
        axis.set_xlabel("Round", labelpad=3)
        axis.set_ylabel(ylabel, labelpad=5)
        axis.grid(True, which="major", color="#E2E2E2", linestyle="--", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.text(0.5, -0.32, caption, transform=axis.transAxes,
                  ha="center", va="top", fontsize=10, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.975),
        ncol=4, frameon=False, fontsize=10,
        columnspacing=1.2, handletextpad=0.4, handlelength=2.0,
    )
    figure.text(
        0.54, 0.025, "Synthetic pilot · mean of 3 seeds · logarithmic y-axes",
        ha="center", va="center", fontsize=9, color="#555555",
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"learning_paper_style.{suffix}", dpi=400)
    plt.close(figure)
    with (destination / "learning_paper_style_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    (destination / "learning_paper_style_metadata.json").write_text(json.dumps({
        "source": str(source.relative_to(ROOT)),
        "aggregation": "arithmetic mean of seeds 0, 1, 2 at each observed round",
        "axes": "logarithmic; narrowly bounded to the actual measured losses",
        "interpolation": "straight plot segments only; no smoothing or invented observations",
        "markers": "staggered among observed rounds to distinguish overlapping curves",
        "forget_set": "grows at deletion request rounds; composition changes over time",
        "uncertainty": "sample SD exported to CSV; not drawn",
        "proposed": "coupled update on synthetic 33-parameter model, ideal communication",
        "not_clip_or_original_paper_reproduction": True,
    }, indent=2) + "\n")
    print(destination / "learning_paper_style.png")


if __name__ == "__main__":
    main()
