"""Replot the completed CapERA/GeoText pair without changing experiment inputs."""
import collections
import csv
import math

from common import AXES, METHODS, OUT, ROOT, case_id, read, sha, write
from report import COLORS, MARKERS, METRICS, STYLES, configure, plt

DEST = ROOT / "results/paper_pair_capera_geotext_20261027"
DATASETS = ["CapERA", "GeoText-1652"]
ORIGINAL_METHODS = [m for m in METHODS if m != "heuristic_ova"]
PROPOSED = "proposed_fu_guard_cliperase"


def canvas(ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.735))
    fig.subplots_adjust(left=.115, right=.985, bottom=.22, top=.80, wspace=.46)
    for ax in axes:
        ax.set_box_aspect(.9)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.grid(True, ls=":", lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", pad=3, labelsize=12)
    return fig, axes


def curve(ax, x, y, number, label):
    return ax.plot(x, y, color=COLORS[number], marker=MARKERS[number],
                   mfc="white", mew=1., ms=5, linestyle=STYLES[number],
                   markevery=max(1, len(x) // 8), label=label)[0]


def save(fig, axes, name):
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=5,
               loc="upper center", bbox_to_anchor=(.5, 1.), fontsize=12,
               handlelength=1.6, columnspacing=.75)
    for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, .015,
                 f"({chr(97+i)}) {dataset}", ha="center", va="bottom", fontsize=14)
        assert not ax.texts  # No numbers on points or additional grey annotations.
    files = []
    for extension in ["png", "pdf", "svg"]:
        path = DEST / "figures" / f"{name}.{extension}"
        fig.savefig(path, dpi=350, bbox_inches="tight")
        files.append(dict(file=str(path.relative_to(DEST)), sha256=sha(path)))
    plt.close(fig)
    return files


def statistics(rows):
    groups = collections.defaultdict(dict)
    for r in rows:
        groups[(r["case"], r["dataset"])][r["method"]] = r
    result = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = {case: group for (case, d), group in groups.items() if d == dataset}
        complete = {case: g for case, g in subset.items()
                    if all(g[m]["status"] == "completed" for m in ORIGINAL_METHODS)}
        wins, exceptions = {}, {}
        for key in ["total_energy_j", "FL_FU_completion_delay_s", "mission_delay_s", "mR"]:
            higher = key == "mR"
            tolerance = 1e-7 if higher else 1e-6
            losing = []
            for case, g in complete.items():
                values = [g[m][key] for m in ORIGINAL_METHODS]
                best = max(values) if higher else min(values)
                if (g[PROPOSED][key] < best - tolerance if higher
                        else g[PROPOSED][key] > best + tolerance):
                    losing.append(case)
            wins[key] = len(complete) - len(losing)
            exceptions[key] = losing
        result[dataset] = dict(
            completed_original_four_comparisons=len(complete),
            proposed_best_or_tied=wins, exceptions=exceptions,
            status_by_method={label: dict(collections.Counter(g[m]["status"]
                for g in subset.values())) for m, label in METHODS.items()},
            baseline_mR_percent={METHODS[m]: r["mR"] * 100
                if r["status"] == "completed" else None
                for m, r in subset["baseline"].items()})
    return result


def main():
    (DEST / "figures").mkdir(parents=True, exist_ok=True)
    source = OUT / "figures/metrics.json"
    all_rows = read(source)["rows"]
    rows = [r for r in all_rows if r["dataset"] in DATASETS]
    assert len(rows) == 410
    index = {(r["case"], r["dataset"], r["method"]): r for r in rows}
    assert len(index) == len(rows)
    assert all(r["status"] == "completed" or "feasible" in r["status"] for r in rows)
    # Verify the publication table against every selected raw arm before plotting.
    sources = {str(source.relative_to(ROOT)): sha(source)}
    for r in rows:
        path = OUT / "cases" / r["case"] / (r["dataset"].lower() + "_seed0") / r["method"] / "outcome.json"
        actual = read(path)
        assert r["status"] == actual["status"]
        for key in METRICS:
            assert r[key] == actual.get("metrics", {}).get(key), (path, key)
        sources[str(path.relative_to(ROOT))] = sha(path)
    write(DEST / "metrics.json", dict(rows=rows, source=str(source.relative_to(ROOT)), seed=0))
    with (DEST / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    configure()
    rendered, series = [], []
    for axis, (xlabel, values, _) in AXES.items():
        for key, (ylabel, divisor) in METRICS.items():
            name = f"{axis}_{key}_1x2"
            fig, axes = canvas(ylabel)
            for ax, dataset in zip(axes, DATASETS):
                for number, (method, label) in enumerate(METHODS.items()):
                    records = [index[(case_id(axis, v), dataset, method)] for v in values]
                    y = [r[key] / divisor if r["status"] == "completed" else None for r in records]
                    curve(ax, values, [v if v is not None else math.nan for v in y], number, label)
                    series.append(dict(figure=name, dataset=dataset, method=method,
                        x=values, y=y, status=[r["status"] for r in records]))
                ax.set_xlabel(xlabel)
                if axis in ["bandwidth", "downlink", "power", "network_cap", "collection_slot"]:
                    ax.set_xscale("log", base=2)
                    ax.minorticks_off()
                ax.set_xticks(values)
                ax.set_xticklabels([f"{v:g}" for v in values])
                ax.set_xlim(values[0], values[-1])
            rendered.append(dict(name=name, files=save(fig, axes, name)))
        print(f"Rendered {axis}: {len(METRICS)} paired figures", flush=True)
    for key in ["loss", "mR"]:
        ylabel, divisor = METRICS[key]
        name = f"{key}_convergence_1x2"
        fig, axes = canvas(ylabel)
        for ax, dataset in zip(axes, DATASETS):
            directory = OUT / "cases/baseline" / (dataset.lower() + "_seed0")
            prefix = directory / "initial20/rounds.json"
            sources[str(prefix.relative_to(ROOT))] = sha(prefix)
            for number, (method, label) in enumerate(METHODS.items()):
                points = [(r["round"], r["evaluation"]["retained"][key] / divisor)
                          for r in read(prefix) if r.get("evaluation")]
                for path in sorted((directory / method / "evaluations").glob("*.json")):
                    r = read(path)
                    points.append((r["completed_FL_rounds"], r["evaluation"]["retained"][key] / divisor))
                    sources[str(path.relative_to(ROOT))] = sha(path)
                points.sort(key=lambda point: point[0])  # Keep repeated round observations.
                assert points and points[-1][0] == 200
                assert math.isclose(points[-1][1], index[("baseline", dataset, method)][key] / divisor,
                                    rel_tol=1e-10, abs_tol=1e-10)
                x, y = map(list, zip(*points))
                curve(ax, x, y, number, label)
                series.append(dict(figure=name, dataset=dataset, method=method, x=x, y=y))
            ax.set_xlim(0, 200)
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_xlabel("Completed FL rounds")
        rendered.append(dict(name=name, files=save(fig, axes, name)))
    assert len(rendered) == 122
    write(DEST / "plot_series.json", series)
    summary = dict(datasets=DATASETS, seed=0, terminal_arms=len(rows),
        status_counts=dict(collections.Counter(r["status"] for r in rows)),
        statistics=statistics(rows), figures=rendered, source_sha256=sources)
    write(DEST / "summary.json", summary)
    write(DEST / "selection.json", dict(
        datasets=DATASETS, panel_order="Left CapERA; right GeoText-1652",
        timing="Selected after the four-dataset seed0 sweep, at the user's request",
        rationale="UAV-oriented data plus favorable energy/delay and retained retrieval results. "
                  "CapERA provides actual UAV footage; GeoText provides geographically grounded virtual drone views.",
        candidate_statistics=statistics(all_rows),
        omissions="RSITMD is general remote sensing. UAVReason has real UAV imagery, "
                  "but lower absolute mR and Proposed trails S-AC/AC-OF on most evaluated mR conditions.",
        full_results=str((OUT / "figures").relative_to(ROOT)),
        limitations=["Post-hoc presentation subset, not independent confirmatory evidence.",
            "All ten axes, five methods, losing conditions and infeasible gaps retained for both selected datasets.",
            "GeoText-1652 uses Google Earth virtual drone views, not recorded UAV flight imagery.",
            "RemoteCLIP adapter experiments on held-out subsets, not official dataset benchmark scores.",
            "Single paired seed; no confidence intervals. GeoText absolute mR remains low.",
            "UAV/IoT mission energy and time are modeled, not measured flight values.",
            "S-OF+ permits network TX cap overruns; the other methods use the hard cap."],
        sources={"CapERA": ["https://github.com/yakoubbazi/CapEra", "https://www.mdpi.com/2072-4292/15/8/2139"],
                 "GeoText-1652": ["https://github.com/MultimodalGeo/GeoText-1652",
                    "https://arxiv.org/abs/2002.12186", "https://github.com/rhao-hur/HCCM"]}))
    lines = ["# CapERA / GeoText-1652 paired figures", "",
        "Left: CapERA. Right: GeoText-1652. Five methods, all ten sensitivity axes, paired seed 0.", "",
        "This presentation pair was chosen after inspecting the completed four-dataset experiment, "
        "for UAV relevance and favorable energy/delay with largely maintained retrieval quality. "
        "It is not an independent confirmatory benchmark. All original results remain in "
        "[the full campaign](../iot_sensitivity_v1_20261027/figures).", "",
        "[CapERA](https://github.com/yakoubbazi/CapEra) contains real UAV video-caption pairs "
        "(Remote Sensing, 2023). [GeoText-1652](https://github.com/MultimodalGeo/GeoText-1652) "
        "is an ECCV 2024 language-guided drone benchmark, also used by "
        "[HCCM at ACM MM 2025](https://github.com/rhao-hur/HCCM). Its drone views come from "
        "Google Earth / University-1652, so they must be described as virtual drone views, not real flight captures.", "",
        "The retained mR values are our fixed held-out subset diagnostics. They are not the published "
        "dataset scores or an official test evaluation. GeoText's absolute mR remains low. "
        "The curves have no smoothing, numeric annotations or confidence intervals. "
        "Infeasible conditions are gaps, including the 0.625 s IoT slot for every method. "
        "S-OF+ can exceed the network TX cap; other methods retain the hard cap.", "",
        "Energy includes communication, computing and flight/hovering. FL/FU completion is measured "
        "from takeoff to the last required broadcast (including preceding collection and travel). "
        "Mission delay also includes the return flight; total delay additionally includes preflight work. "
        "These are simulated mission costs, not measured UAV flight results.", "",
        "There are 120 sensitivity figures (10 axes × 12 metrics) and two convergence figures, "
        "each in PNG, PDF and SVG. All five methods and every requested setting are retained.", "",
        "| X axis | Energy | FL/FU completion | Mission delay | Loss | mR |",
        "|---|---|---|---|---|---|"]
    for axis, (label, _, _) in AXES.items():
        links = [f"[{title}](figures/{axis}_{key}_1x2.png)" for key, title in [
            ("total_energy_j", "Energy"), ("FL_FU_completion_delay_s", "FL/FU"),
            ("mission_delay_s", "Mission"), ("loss", "Loss"), ("mR", "mR")]]
        lines.append("| " + " | ".join([label, *links]) + " |")
    lines += ["", "[Loss convergence](figures/loss_convergence_1x2.png) · "
        "[mR convergence](figures/mR_convergence_1x2.png)", "",
        "For each dataset, Proposed has the lowest energy and FL/FU/mission delays in 35/36 "
        "conditions where all four original methods completed. The exception is `request_bursts_45`. "
        "Proposed mR is best or tied in 30/36 CapERA conditions and 36/36 GeoText conditions. "
        "H-OVA feasibility and all candidate-dataset results are recorded in `selection.json`.", "",
        "Reproduce with `python3 tools/iot_sensitivity_v1/selected_pair.py`. "
        "No experiments are rerun. `metrics.csv`, `plot_series.json` and `summary.json` provide "
        "the raw values, displayed series and source hashes.", ""]
    (DEST / "README.md").write_text("\n".join(lines))
    print(f"Complete: {len(rendered)} figures, {len(rows)} original outcomes -> {DEST}")


if __name__ == "__main__":
    main()
