"""Paired CapERA/AirScape charts from terminal, auditable raw outcomes only."""
import collections
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/iot_sensitivity_v1"))
from common import read, write, sha, AXES, METHODS, case_id
from report import METRICS, configure
import selected_pair as style

OUT = ROOT / "results/airscape_sensitivity_v2_20261027"
DEST = ROOT / "results/paper_pair_capera_airscape_20261027"
DATASETS = ["CapERA", "AirScape"]
style.DEST, style.DATASETS = DEST, DATASETS


def generate():
    (DEST / "figures").mkdir(parents=True, exist_ok=True)
    rows, sources = [], {}
    for case in ["baseline"] + [case_id(a, v) for a, (_, values, baseline) in AXES.items()
                                 for v in values if v != baseline]:
        for dataset in DATASETS:
            for method in METHODS:
                path = OUT / "cases" / case / (dataset.lower() + "_seed0") / method / "outcome.json"
                if not path.exists():
                    continue
                raw = read(path)
                sources[str(path.relative_to(ROOT))] = sha(path)
                rows.append(dict(case=case, dataset=dataset, method=method, status=raw["status"],
                    **{key: raw.get("metrics", {}).get(key) for key in METRICS}))
    index = {(r["case"], r["dataset"], r["method"]): r for r in rows}
    assert len(index) == len(rows)
    write(DEST / "metrics.json", dict(rows=rows, seed=0, source=str(OUT.relative_to(ROOT))))
    if rows:
        with (DEST / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    configure()
    rendered, series = [], []
    for axis, (xlabel, values, _) in AXES.items():
        if not all((case_id(axis, v), d, m) in index for v in values for d in DATASETS for m in METHODS):
            continue
        for key, (ylabel, divisor) in METRICS.items():
            name = f"{axis}_{key}_1x2"
            fig, axes = style.canvas(ylabel)
            for ax, dataset in zip(axes, DATASETS):
                for number, (method, label) in enumerate(METHODS.items()):
                    records = [index[(case_id(axis, v), dataset, method)] for v in values]
                    y = [r[key] / divisor if r["status"] == "completed" else None for r in records]
                    style.curve(ax, values, [v if v is not None else math.nan for v in y], number, label)
                    series.append(dict(figure=name, dataset=dataset, method=method, x=values,
                        y=y, status=[r["status"] for r in records]))
                ax.set_xlabel(xlabel)
                if axis in ["bandwidth", "downlink", "power", "network_cap", "collection_slot"]:
                    ax.set_xscale("log", base=2)
                    ax.minorticks_off()
                ax.set_xticks(values)
                ax.set_xticklabels([f"{v:g}" for v in values])
                ax.set_xlim(values[0], values[-1])
            rendered.append(dict(name=name, files=style.save(fig, axes, name)))
    if all(("baseline", d, m) in index for d in DATASETS for m in METHODS):
        for key in ["loss", "mR"]:
            ylabel, divisor = METRICS[key]
            name = f"{key}_convergence_1x2"
            fig, axes = style.canvas(ylabel)
            for ax, dataset in zip(axes, DATASETS):
                directory = OUT / "cases/baseline" / (dataset.lower() + "_seed0")
                prefix = directory / "initial20/rounds.json"
                sources[str(prefix.relative_to(ROOT))] = sha(prefix)
                for number, (method, label) in enumerate(METHODS.items()):
                    points = [(r["round"], r["evaluation"]["retained"][key] / divisor)
                              for r in read(prefix) if r.get("evaluation")]
                    for path in sorted((directory / method / "evaluations").glob("*.json")):
                        row = read(path)
                        points.append((row["completed_FL_rounds"], row["evaluation"]["retained"][key] / divisor))
                        sources[str(path.relative_to(ROOT))] = sha(path)
                    points.sort(key=lambda p: p[0])
                    status = index[("baseline", dataset, method)]["status"]
                    if status == "completed":
                        assert points[-1][0] == 200
                        assert math.isclose(points[-1][1], index[("baseline", dataset, method)][key] / divisor,
                                            rel_tol=1e-10, abs_tol=1e-10)
                    x, y = map(list, zip(*points))
                    style.curve(ax, x, y, number, label)
                    series.append(dict(figure=name, dataset=dataset, method=method, x=x, y=y, status=status))
                ax.set_xlim(0, 200)
                ax.set_xticks([0, 50, 100, 150, 200])
                ax.set_xlabel("Completed FL rounds")
            rendered.append(dict(name=name, files=style.save(fig, axes, name)))
    counts = dict(collections.Counter(r["status"] for r in rows))
    summary = dict(datasets=DATASETS, seed=0, terminal_arms=len(rows), expected_arms=410,
        status_counts=counts, figure_count=len(rendered), figures=rendered,
        source_sha256=sources, complete=len(rows) == 410)
    write(DEST / "plot_series.json", series)
    write(DEST / "summary.json", summary)
    lines = ["# CapERA / AirScape paired experiments", "",
        "Left CapERA; right AirScape (WebUAV3M source subset). Paired seed0, five methods, ten axes.", "",
        "AirScape is ACM MM2025 real UAV video/intention data. This is an eight-frame "
        "mean-pooled retrieval adaptation, not its published world-model benchmark. "
        "Its1native text/video differs from CapERA's5captions/video. Absolute mR across "
        "datasets is not a matched difficulty comparison. All methods within a dataset share "
        "the identical split, feature cache, initialization and request schedule.", "",
        "The replacement was screened on validation after earlier candidates. This is exploratory; "
        "the official test set is excluded. [All pilot outcomes](../uav_convergence_v2_20261027/) "
        "remain available. AirScape's +0.4557pp pilot gain did not pass the prospective1.0pp "
        "strong-gain gate. This follow-up examines its modest rise and plateau under FU/radio "
        "conditions; it is not presented as a passed screen or a CapERA-sized improvement. "
        "No selection of favorable sensitivity settings or method results.", "",
        "Energy includes communication, computing and flight/hovering. FL/FU completion "
        "includes preceding movement/collection and ends with the last required broadcast; "
        "mission completion also includes return; total completion additionally includes preflight. "
        "Costs are modeled, not measured UAV flight values. S-OF+ permits TX-budget overruns.", "",
        "Curves use raw observations, including repeated round observations and infeasible gaps. "
        "Partial convergence traces stop at their actual last evaluation. Single seed; no confidence intervals.", "",
        f"Current terminal arms: {len(rows)}/410; figures: {len(rendered)}/122.", "",
        "| Axis | Energy | FL/FU completion | Mission delay | Loss | mR |",
        "|---|---|---|---|---|---|"]
    names = {r["name"] for r in rendered}
    for axis, (label, _, _) in AXES.items():
        links = []
        for key, title in [("total_energy_j", "Energy"), ("FL_FU_completion_delay_s", "FL/FU"),
                           ("mission_delay_s", "Mission"), ("loss", "Loss"), ("mR", "mR")]:
            name = f"{axis}_{key}_1x2"
            links.append(f"[{title}](figures/{name}.png)" if name in names else "pending")
        lines.append("| " + " | ".join([label] + links) + " |")
    for key in ["loss", "mR"]:
        name = f"{key}_convergence_1x2"
        if name in names:
            lines += ["", f"[{key} convergence](figures/{name}.png)"]
    (DEST / "README.md").write_text("\n".join(lines) + "\n")
    return {k: v for k, v in summary.items() if k not in ["figures", "source_sha256"]}


if __name__ == "__main__":
    print(generate())
