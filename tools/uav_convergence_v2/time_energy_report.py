"""Time-based convergence and communication/computing/hover energy views.

Postprocessing of checked measurements: no training or mission inputs change.
"""
import collections
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/iot_sensitivity_v1"))
from common import read, write, sha, AXES, METHODS, case_id
from report import configure, COLORS, MARKERS, STYLES, plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import selected_pair as style

SOURCE = ROOT / "results/airscape_power_audit_v1_20261027"
DEST = ROOT / "results/paper_pair_capera_airscape_time_energy_20261027"
DATASETS = ["CapERA", "AirScape"]
style.DEST, style.DATASETS = DEST, DATASETS


def prefix_timing(directory, parameters):
    """Rebuild the recorded common prefix using its native caption count."""
    plan = read(directory / "plan.json")
    compute, comm = parameters["compute"], parameters["communication"]
    frequency = compute["frequency_hz"] * compute["macs_per_cycle"]
    def encoder(tokens, width):
        return 12 * (12 * tokens * width**2 + 2 * tokens**2 * width)
    image = encoder(50, 768) + 49 * 768 * (32 * 32 * 3) + 768 * 512
    text = encoder(77, 512) + 512 * 512
    owner = {i: user for user, part in enumerate(plan["partitions"]) for i in part}
    device_macs = [0.] * 10
    for row in read(directory / "feature_cache/manifest.json")["lineage"]:
        if row["training_sample"]:
            device_macs[owner[row["sample_id"]]] += len(row["frames"]) * image + plan.get("caption_count", 5) * text
    cold = max(device_macs) / frequency
    time = cold
    points, stages = [], {}
    for stage in ["initial20", "proxy20"]:
        start = time
        for row in read(directory / stage / "rounds.json"):
            uplink = row["radio"]["uplink_s"]
            downlink = ((comm["update_coordinates"] * comm["broadcast_bits_per_coordinate"] +
                         comm["control_bits_per_update"]) / comm["downlink_rate_bps"] if uplink else 0.)
            time += max(row["compute"]["device_macs"]) / frequency + uplink + downlink
            if stage == "initial20" and row.get("evaluation"):
                points.append(dict(time_s=time, round=row["round"], evaluation=row["evaluation"]["retained"]))
        stages[stage] = time - start
    return dict(cold_feature_s=cold, total_s=time, stages=stages, initial_points=points)


def active_time(timeline, instant):
    return sum(max(0., min(instant, s["service_end_s"]) - s["service_start_s"])
               for s in timeline["sites"])


def convergence(dataset, method, sources):
    directory = SOURCE / "cases/baseline" / (dataset.lower() + "_seed0")
    arm = directory / method
    settings = read(SOURCE / "cases/baseline/settings.json")
    costs = read(arm / "energy_and_delay.json")
    timeline = read(arm / "mission_timeline.json")
    prefix = prefix_timing(directory, settings["energy_model"])
    assert math.isclose(prefix["total_s"], costs["prefix_delay_s"], rel_tol=1e-12, abs_tol=1e-8)
    initial = prefix["initial_points"][-1]
    assert initial["round"] == 20
    events = [dict(sequence=0, round=20, mission_s=0., active_s=0.,
                   total_s=prefix["total_s"], evaluation=initial["evaluation"])]
    for path in sorted((arm / "evaluations").glob("*.json")):
        row = read(path)
        time = row["mission_clock_s"]
        assert time <= timeline["service_end_s"] + 1e-8
        transaction = read(arm / "transactions" / f"{row['sequence']:04d}.json")
        assert math.isclose(time, transaction["end_s"], abs_tol=1e-9)
        events.append(dict(sequence=row["sequence"], round=row["completed_FL_rounds"],
            mission_s=time, active_s=active_time(timeline, time), total_s=prefix["total_s"] + time,
            evaluation=row["evaluation"]["retained"]))
        sources[str(path.relative_to(ROOT))] = sha(path)
    final = read(arm / "service_completion.json")
    assert math.isclose(final["mission_clock_s"], costs["FL_FU_completion_delay_s"], abs_tol=1e-8)
    if events[-1]["sequence"] != final["sequence"]:
        instant = final["mission_clock_s"]
        events.append(dict(sequence=final["sequence"], round=final["completed_FL_rounds"],
            mission_s=instant, active_s=active_time(timeline, instant),
            total_s=prefix["total_s"] + instant, evaluation=final["evaluation"]["retained"]))
    assert events[-1]["round"] == 200
    assert math.isclose(events[-1]["active_s"], costs["FL_FU_active_service_s"], abs_tol=1e-8)
    assert math.isclose(events[-1]["total_s"] + costs["return_flight_s"], costs["total_delay_s"], abs_tol=1e-8)
    for metric in ["loss", "mR"]:
        assert events[-1]["evaluation"][metric] == costs[metric]
    for clock in ["mission_s", "active_s", "total_s"]:
        assert all(a[clock] <= b[clock] for a, b in zip(events, events[1:]))
    for path in [directory / "plan.json", directory / "initial20/rounds.json",
                 directory / "proxy20/rounds.json", directory / "feature_cache/manifest.json",
                 arm / "energy_and_delay.json", arm / "mission_timeline.json", arm / "service_completion.json"]:
        sources[str(path.relative_to(ROOT))] = sha(path)
    return events, prefix, costs


def nice_ticks(values, padding=0., bins=4, simple=False):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    low, high = min(finite), max(finite)
    margin = (high - low) * padding if high > low else max(abs(low) * .05, .01)
    steps = [1, 2, 2.5, 5, 10] if simple else [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
    return MaxNLocator(nbins=bins, steps=steps).tick_values(
        low - margin, high + margin).tolist()


def tick_decimals(ticks):
    return next(n for n in range(7)
                if all(math.isclose(v, round(v, n), abs_tol=1e-8) for v in ticks))


def regular_axis(ax, dimension, ticks, decimals):
    axis = getattr(ax, dimension + "axis")
    getattr(ax, "set_" + dimension + "lim")(ticks[0], ticks[-1])
    getattr(ax, "set_" + dimension + "ticks")(ticks)
    axis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    step = ticks[1] - ticks[0]
    assert all(math.isclose(b - a, step, rel_tol=1e-10, abs_tol=1e-10)
               for a, b in zip(ticks, ticks[1:]))


def finish_axes(axes, y_values, decimals=None):
    ticks = [nice_ticks(values, padding=.025, bins=5, simple=True) for values in y_values]
    precision = max(tick_decimals(t) for t in ticks) if decimals is None else decimals
    for ax, values in zip(axes, ticks):
        regular_axis(ax, "y", values, precision)


def save_figure(fig, axes, name):
    # Record the rendered axes so endpoint, spacing and formatting are auditable.
    settings = []
    for ax, dataset in zip(axes, DATASETS):
        entry = dict(dataset=dataset)
        for dim in ["x", "y"]:
            ticks = getattr(ax, "get_" + dim + "ticks")().tolist()
            bounds = list(getattr(ax, "get_" + dim + "lim")())
            if ticks:  # The energy bar chart has a categorical, unlabeled x axis.
                assert math.isclose(ticks[0], bounds[0], abs_tol=1e-10)
                assert math.isclose(ticks[-1], bounds[-1], abs_tol=1e-10)
                step = ticks[1] - ticks[0]
                assert all(math.isclose(b - a, step, abs_tol=1e-9)
                           for a, b in zip(ticks, ticks[1:]))
            formatter = getattr(ax, dim + "axis").get_major_formatter()
            entry[dim] = dict(limits=bounds, ticks=ticks, labels=[formatter(t) for t in ticks])
        settings.append(entry)
    return dict(name=name, axes=settings, files=style.save(fig, axes, name))


def draw_time(ax, x, y, number, label, horizon):
    # Last-observed-value steps: no claim of learning during a travel interval.
    assert horizon >= x[-1]
    display_x, display_y = x + [horizon], y + [y[-1]]
    # No extra marker at the artificial display endpoint.
    ax.plot(display_x, display_y, drawstyle="steps-post", color=COLORS[number], marker=MARKERS[number],
            mfc="white", ms=4.5, mew=1., linestyle=STYLES[number],
            markevery=list(range(0, len(x), max(1, len(x) // 7))), label=label)
    return display_x, display_y


def quality_times(trajectories):
    rows = []
    for dataset in DATASETS:
        for metric, higher in [("mR", True), ("loss", False)]:
            paths = {m: trajectories[(dataset, m)][0] for m in METHODS}
            start = paths[next(iter(METHODS))][0]["evaluation"][metric]
            endpoint = (min if higher else max)(p[-1]["evaluation"][metric] for p in paths.values())
            for fraction in [.25, .5, .75, 1.]:
                target = start + fraction * (endpoint - start)
                for method, path in paths.items():
                    good = [p["evaluation"][metric] >= target - 1e-12 if higher
                            else p["evaluation"][metric] <= target + 1e-12 for p in path]
                    first = next(i for i, okay in enumerate(good) if okay)
                    sustained = next(i for i in range(len(good)) if all(good[i:]))
                    rows.append(dict(dataset=dataset, metric=metric, target=target,
                        common_gain_fraction=fraction, method=method,
                        first_observed_mission_s=path[first]["mission_s"],
                        first_observed_active_s=path[first]["active_s"],
                        sustained_observed_mission_s=path[sustained]["mission_s"],
                        sustained_observed_active_s=path[sustained]["active_s"]))
    return rows


def generate():
    (DEST / "figures").mkdir(parents=True, exist_ok=True)
    configure()
    rows, sources = [], {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for directory in sorted((SOURCE / "cases").iterdir()):
        for dataset in DATASETS:
            for method in METHODS:
                path = directory / (dataset.lower() + "_seed0") / method / "outcome.json"
                record = read(path)
                metrics = record.get("metrics", {})
                row = dict(case=directory.name, dataset=dataset, method=method, status=record["status"])
                keys = ["total_energy_j", "energy_travel_j", "energy_hover_j", "energy_communication_j",
                        "energy_computing_j", "FL_FU_completion_delay_s", "FL_FU_active_service_s",
                        "mission_delay_s", "total_delay_s", "mR", "loss"]
                row.update({key: metrics.get(key) for key in keys})
                row["energy_j"] = None
                if record["status"] == "completed":
                    row["energy_j"] = sum(metrics[k] for k in ["energy_communication_j", "energy_computing_j", "energy_hover_j"])
                    assert math.isclose(row["energy_j"], metrics["total_energy_j"] - metrics["energy_travel_j"],
                                        rel_tol=1e-12, abs_tol=1e-7)
                rows.append(row)
                sources[str(path.relative_to(ROOT))] = sha(path)
    assert len(rows) == 410
    index = {(r["case"], r["dataset"], r["method"]): r for r in rows}
    write(DEST / "metrics.json", dict(rows=rows,
        displayed_energy="Communication + computing + hovering; inter-site travel and return-flight energy removed. "
                         "This is a reporting subtotal, not whole-mission total energy.",
        user_instruction="Remove common travel energy from the energy calculation; simple Energy(kJ) chart label."))
    with (DEST / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    figures, series = [], []
    for axis, (xlabel, values, _) in AXES.items():
        name = f"{axis}_energy_1x2"
        fig, axes = style.canvas("Energy (kJ)")
        y_values = []
        xticks = nice_ticks(values)
        for ax, dataset in zip(axes, DATASETS):
            panel_y = []
            for number, (method, label) in enumerate(METHODS.items()):
                records = [index[(case_id(axis, v), dataset, method)] for v in values]
                y = [r["energy_j"] / 1000 if r["energy_j"] is not None else None for r in records]
                style.curve(ax, values, [v if v is not None else math.nan for v in y], number, label)
                series.append(dict(figure=name, dataset=dataset, method=method, x=values, y=y,
                                   status=[r["status"] for r in records]))
                panel_y += y
            ax.set_xlabel(xlabel)
            # Physical x values stay unchanged; linear axes use regular numeric ticks.
            regular_axis(ax, "x", xticks, tick_decimals(xticks))
            y_values.append(panel_y)
        finish_axes(axes, y_values)
        figures.append(save_figure(fig, axes, name))

    trajectories = {(dataset, method): convergence(dataset, method, sources)
                    for dataset in DATASETS for method in METHODS}
    clocks = {"mission": ("Elapsed time (s)", "mission_s"),
              "service": ("FL/FU service time (s)", "active_s"),
              "total": ("Total elapsed time (s)", "total_s")}
    for clock, (xlabel, field) in clocks.items():
        starts = [p[1]["initial_points"][0]["time_s"] if clock == "total" else 0.
                  for p in trajectories.values()]
        ends = [p[0][-1][field] for p in trajectories.values()]
        step = 100 if clock == "service" else 300
        lower, upper = math.floor(min(starts) / step) * step, math.ceil(max(ends) / step) * step
        xticks = list(range(lower, upper + 1, step))
        for metric, ylabel, scale in [("loss", "Retained loss", 1.), ("mR", "Retained mR (%)", 100.)]:
            name = f"{metric}_{clock}_time_1x2"
            fig, axes = style.canvas(ylabel)
            y_values = []
            for ax, dataset in zip(axes, DATASETS):
                panel_y = []
                for number, (method, label) in enumerate(METHODS.items()):
                    events, prefix, costs = trajectories[(dataset, method)]
                    x = [e[field] for e in events]
                    y = [e["evaluation"][metric] * scale for e in events]
                    if clock == "total":
                        initial = prefix["initial_points"]
                        x = [p["time_s"] for p in initial] + x
                        y = [p["evaluation"][metric] * scale for p in initial] + y
                    display_x, display_y = draw_time(ax, x, y, number, label, upper)
                    panel_y += y
                    series.append(dict(figure=name, dataset=dataset, method=method, x=x, y=y,
                        display_x=display_x, display_y=display_y, work_completion_s=x[-1],
                        terminal_hold_from_s=x[-1], terminal_hold_to_s=upper,
                        plotting="Last-observed-value steps with a horizontal final-value hold through the display horizon. "
                                 "The added endpoint is not a new measurement or additional learning."))
                regular_axis(ax, "x", xticks, 0)
                ax.set_xlabel(xlabel)
                y_values.append(panel_y)
            finish_axes(axes, y_values, decimals=2 if metric == "loss" else 1)
            figures.append(save_figure(fig, axes, name))

    # User's hatched-bar style, without numeric annotations.
    name = "baseline_energy_1x2"
    fig, axes = style.canvas("Energy (kJ)")
    for ax, dataset in zip(axes, DATASETS):
        for number, (method, label) in enumerate(METHODS.items()):
            value = index[("baseline", dataset, method)]["energy_j"] / 1000
            ax.bar(number, value, width=.66, color=COLORS[number], edgecolor="black", linewidth=.75,
                   hatch=["//", "++", "\\\\", "xx", ".."][number], label=label)
            series.append(dict(figure=name, dataset=dataset, method=method, x=[number], y=[value]))
        ax.set_xticks([]); ax.set_xlim(-.6, 4.6)
        regular_axis(ax, "y", [0, 200, 400, 600, 800], 0)
    figures.append(save_figure(fig, axes, name))
    quality = quality_times(trajectories)
    write(DEST / "time_to_quality.json", dict(rows=quality,
        target_policy="Quarter increments from the common initial20 model toward the worst final method value. "
                      "All four targets for both metrics retained; no favorable target selection.",
        interpretation="First/sustained observed attainment on the saved evaluation grid, not exact crossing time. "
                       "Sustained means all later recorded evaluations meet the target; single seed."))
    write(DEST / "plot_series.json", series)
    summary = dict(datasets=DATASETS, status_counts=dict(collections.Counter(r["status"] for r in rows)),
        figure_count=len(figures), figures=figures, source_sha256=sources,
        energy_scope="Communication + computing + hovering; travel excluded.",
        clocks=dict(mission="From takeoff, including travel and IoT collection, ending at final FL/FU broadcast.",
            service="Active mission service windows: compute/uplink/broadcast/scheduler wait. Excludes common prefix, travel, IoT collection.",
            total="From cold feature extraction, includes initial20/proxy20 and mission time through final FL/FU broadcast. "
                  "Return flight occurs after learning ends and is excluded from convergence time."),
        baseline={d: {METHODS[m]: dict(energy_kJ=index[("baseline", d, m)]["energy_j"] / 1000,
            mission_completion_s=trajectories[(d, m)][2]["FL_FU_completion_delay_s"],
            active_service_s=trajectories[(d, m)][2]["FL_FU_active_service_s"],
            total_to_work_completion_s=trajectories[(d, m)][0][-1]["total_s"],
            mR_percent=trajectories[(d, m)][2]["mR"] * 100) for m in METHODS} for d in DATASETS})
    write(DEST / "summary.json", summary)
    lines = ["# CapERA / AirScape: convergence over time and revised energy charts", "",
        "These plots use the checked, completed experiments. No model, seed, request, route or radio condition is rerun or changed.", "",
        "## Energy definition", "",
        "The chart label is **Energy (kJ)**, as requested. Its value is communication + computing + hovering. "
        "Inter-site travel and return-flight propulsion energy are subtracted from the former whole-mission total. "
        "The retained communication scope includes UAV communications and IoT transmission. Common preflight "
        "feature/initial/proxy computing remains included. This subtotal is not the UAV battery's whole-mission energy. "
        "Travel energy is common across methods within each scenario, although it changes across cruise-speed settings. "
        "The full original components remain in `metrics.csv`; infeasible points remain gaps. The baseline bar has a zero baseline.", "",
        "## Time definitions", "",
        "All times are the existing UAV/IoT simulation model, not GPU/server runtime.", "",
        "- **Elapsed time:** from takeoff after common initial20/proxy20 preparation, includes movement and collection.",
        "- **FL/FU service time:** active service windows only; includes computation, uplink, broadcast and scheduler waits.",
        "- **Total elapsed time:** from cold feature extraction; includes initial20/proxy20 and mission movement/collection.", "",
        "Every measured convergence trace ends when its final required FL/FU broadcast is complete. Return flight happens "
        "after learning and is not an extra learning period. Mission and service plots begin at the common20-round "
        "initial model. Total-time plots additionally show the recorded initial20 evaluations. "
        "Step traces carry forward only the last observed metric, avoiding an artificial slope during flight. "
        "There is no smoothing, invented evaluation, change to loss/mR, or annotation text on the chart. "
        "For display, each final value is held horizontally through the common right-hand axis limit. "
        "This tail is not evidence of further learning or convergence: a completed run can still have room to improve. "
        "Raw x/y observations, actual completion times and separate display_x/display_y arrays are retained in `plot_series.json`. "
        "Time-to-quality calculations use only actual observations, never the added endpoints.", "",
        "All numeric axes include both endpoint labels with constant arithmetic tick spacing. "
        "Parameter sweeps use linear physical coordinates; original irregularly sampled x values are not moved. "
        "Both panels use the same decimal precision (loss: two places; mR: one). "
        "Axis bounds, ticks and labels are saved in `summary.json`.", "",
        "| Time axis | Loss | mR |", "|---|---|---|"]
    for clock, (label, _) in clocks.items():
        lines.append(f"| {label} | [Loss](figures/loss_{clock}_time_1x2.png) | [mR](figures/mR_{clock}_time_1x2.png) |")
    lines += ["", "[Baseline energy bars](figures/baseline_energy_1x2.png)", "", "| Parameter | Energy |", "|---|---|"]
    for axis, (label, _, _) in AXES.items():
        lines.append(f"| {label} | [Energy](figures/{axis}_energy_1x2.png) |")
    lines += ["", "Time-to-quality results include every quarter increment of the common reachable range "
        "for loss and mR in `time_to_quality.json`. They are evaluated on recorded checkpoints, not continuous "
        "measurements. Faster completion of the prescribed workload is distinct from reaching every quality "
        "threshold first. Single seed and exploratory dataset selection still apply. "
        "The earlier full charts and round-based traces remain in "
        "[the checked report](../paper_pair_capera_airscape_checked_20261027/).", ""]
    (DEST / "README.md").write_text("\n".join(lines))
    print({"figures": len(figures), "baseline": summary["baseline"]}, flush=True)


def publish():
    git = ["git", "--git-dir=.git-push-ready/code-charts.git", "--work-tree=."]
    def call(args):
        return subprocess.check_output(git + args, cwd=ROOT, text=True).strip()
    assert call(["remote", "get-url", "origin"]) == "https://github.com/ckim48/ck_ova.git"
    assert not call(["diff", "--cached", "--name-only"])
    paths = [Path(__file__)] + [p for p in DEST.rglob("*") if p.is_file()]
    subprocess.run(git + ["add", "--"] + [str(p.relative_to(ROOT)) for p in paths], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(git + ["commit", "-m", "Use regular endpoint ticks and retain completed convergence traces"], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(git + ["push", "origin", "main"], cwd=ROOT, check=True, capture_output=True, timeout=180)
    commit = call(["rev-parse", "HEAD"])
    assert call(["ls-remote", "origin", "refs/heads/main"]).split()[0] == commit
    print("Pushed", commit, flush=True)


if __name__ == "__main__":
    generate() if len(sys.argv) == 1 else publish()
