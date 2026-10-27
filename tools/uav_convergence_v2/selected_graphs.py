"""Six requested 1x2 figures, using actual mission FL/FU service ledgers."""
import csv
import argparse
import importlib.util
import math
from pathlib import Path

import time_energy_report as chart
from common import read, write, sha, METHODS, AXES, case_id
from matplotlib.lines import Line2D

ROOT = chart.ROOT
DEST = ROOT / "selected_graph"
DATASETS = chart.DATASETS
spec = importlib.util.spec_from_file_location("selected_energy_model", ROOT / "tools/uav_real_vlm_v1/energy.py")
physical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physical)


def prefix_energy(directory, parameters, sources):
    physical.PARAMETERS = parameters
    def load(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return read(path)
    plan = load(directory / "plan.json")
    owner = {i: u for u, part in enumerate(plan["partitions"]) for i in part}
    image = physical.encoder_macs(50, 768, 12) + 49 * 768 * 32 * 32 * 3 + 768 * 512
    text = physical.encoder_macs(77, 512, 12) + 512 * 512
    macs = [0.] * len(plan["partitions"])
    for row in load(directory / "feature_cache/manifest.json")["lineage"]:
        if row["training_sample"]:
            macs[owner[row["sample_id"]]] += len(row["frames"]) * image + plan.get("caption_count", 5) * text
    computing = float(physical.compute_cost(macs)[0].sum())
    communication = 0.
    for stage in ["initial20", "proxy20"]:
        for row in load(directory / stage / "rounds.json"):
            computing += float(physical.compute_cost(row["compute"]["device_macs"])[0].sum())
            computing += parameters["communication"]["head_processing_w"] * row["radio"]["uplink_s"]
            communication += physical.communication(row["radio"])[0]
    return dict(computing_j=computing, communication_j=communication)


def service_ledger(directory, method, parameters, prefix, sources):
    physical.PARAMETERS = parameters
    arm = directory / method
    def load(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return read(path)
    outcome = load(arm / "outcome.json")
    if outcome["status"] != "completed":
        assert outcome["status"] == "incomplete_no_feasible_radio_design"
        return dict(status=outcome["status"])
    costs = load(arm / "energy_and_delay.json")
    timeline = load(arm / "mission_timeline.json")
    service = sum(s["service_duration_s"] for s in timeline["sites"])
    assert math.isclose(service, costs["FL_FU_active_service_s"], abs_tol=1e-8)
    computing = communication = transaction_s = 0.
    work_count = tx_count = 0
    for path in sorted((arm / "work_attempts").glob("*.json")):
        row = load(path)
        assert row["status"] == "transmitted"
        computing += float(physical.compute_cost(row["compute"]["device_macs"])[0].sum())
        work_count += 1
    for path in sorted((arm / "transactions").glob("*.json")):
        row = load(path)
        comm, downlink = physical.communication(row["radio"])
        assert math.isclose(downlink, row["timing"]["downlink_s"], abs_tol=1e-10)
        communication += comm
        computing += parameters["communication"]["head_processing_w"] * row["radio"]["uplink_s"]
        transaction_s += row["timing"]["duration_s"]
        tx_count += 1
    assert tx_count == work_count
    waiting = sum(w["duration_s"] for w in timeline["waits"])
    assert math.isclose(service, transaction_s + waiting, abs_tol=1e-8)
    uavs = parameters["iot_collection"]["uavs"]
    hover_power = uavs * float(physical.propulsion_power(0.))
    hovering = hover_power * service
    collection_hover = hover_power * costs["iot_collection_seconds"]
    assert math.isclose(costs["energy_hover_j"], hovering + collection_hover, abs_tol=1e-7)
    assert math.isclose(costs["energy_computing_j"], computing + prefix["computing_j"], abs_tol=1e-7)
    assert math.isclose(costs["energy_communication_j"], communication + prefix["communication_j"] +
                        costs["iot_collection_communication_energy_j"], abs_tol=1e-7)
    total = computing + communication + hovering
    removed = (costs["energy_travel_j"] + collection_hover + costs["iot_collection_communication_energy_j"]
               + prefix["computing_j"] + prefix["communication_j"])
    assert math.isclose(total, costs["total_energy_j"] - removed, abs_tol=1e-7)
    return dict(status="completed", service_computing_j=computing, service_communication_j=communication,
        service_hovering_j=hovering, total_energy_j=total, delay_s=service,
        transaction_s=transaction_s, scheduler_wait_s=waiting, transactions=tx_count,
        original_whole_mission_energy_j=costs["total_energy_j"], excluded_energy_j=removed,
        prefix_energy_j=sum(prefix.values()), collection_hover_energy_j=collection_hover,
        collection_communication_energy_j=costs["iot_collection_communication_energy_j"],
        travel_energy_j=costs["energy_travel_j"])


def save(fig, axes, name, status_markers=None):
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(METHODS), loc="upper center",
               bbox_to_anchor=(.5, 1.), fontsize=12, handlelength=1.6, columnspacing=.75)
    if status_markers:
        methods = list(dict.fromkeys(r["method"] for r in status_markers))
        handles = [Line2D([], [], color=chart.COLORS[list(METHODS).index(method)],
                          marker="x", markersize=7, markeredgewidth=1.6, linestyle="none",
                          label=f"Infeasible ({METHODS[method]})") for method in methods]
        fig.legend(handles=handles, ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(.5, .9), fontsize=11, handlelength=1.)
    settings = []
    for number, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, .015, f"({chr(97+number)}) {dataset}",
                 ha="center", va="bottom", fontsize=14)
        assert not ax.texts
        row = dict(dataset=dataset)
        for dim in ["x", "y"]:
            ticks = getattr(ax, f"get_{dim}ticks")().tolist()
            limits = list(getattr(ax, f"get_{dim}lim")())
            assert limits == [ticks[0], ticks[-1]]
            assert all(math.isclose(b - a, ticks[1] - ticks[0], abs_tol=1e-9)
                       for a, b in zip(ticks, ticks[1:]))
            formatter = getattr(ax, dim + "axis").get_major_formatter()
            row[dim] = dict(limits=limits, ticks=ticks, labels=[formatter(t) for t in ticks])
        settings.append(row)
    files = []
    for extension in ["png", "pdf", "svg"]:
        path = DEST / f"{name}.{extension}"
        fig.savefig(path, dpi=350, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        files.append(dict(file=path.name, sha256=sha(path)))
    chart.plt.close(fig)
    return dict(name=name, axes=settings, files=files, status_markers=status_markers or [])


def generate(comparison="main"):
    global METHODS, DEST
    from baseline_fairness import MAIN, OUT, CHECKED, OLD, NEW
    from common import METHODS as original_methods
    from report import COLORS, MARKERS, STYLES
    if comparison == "main":
        assert read(OUT / "finished.json")["passed"]
        METHODS, DEST, chart.SOURCE = MAIN, ROOT / "selected_graph", OUT
        indices = [0, 2, 3, 4]
    elif comparison == "ablation":
        METHODS = {m: original_methods[m] for m in ["proposed_fu_guard_cliperase", "heuristic_ova", OLD]}
        DEST, chart.SOURCE, indices = ROOT / "ablation_graph", CHECKED, [0, 1, 4]
    else:
        assert comparison == "legacy"
        METHODS, DEST, chart.SOURCE = original_methods, ROOT / "results/selected_graph_legacy_20261027", CHECKED
        indices = list(range(5))
    for attribute, values in [("COLORS", COLORS), ("MARKERS", MARKERS), ("STYLES", STYLES)]:
        setattr(chart, attribute, [values[i] for i in indices])
        setattr(chart.style, attribute, [values[i] for i in indices])
    DEST.mkdir(exist_ok=True)
    chart.configure()
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
               "tools/uav_real_vlm_v1/energy.py": sha(ROOT / "tools/uav_real_vlm_v1/energy.py"),
               "tools/uav_convergence_v2/time_energy_report.py": sha(ROOT / "tools/uav_convergence_v2/time_energy_report.py")}
    if comparison == "main":
        for path in [OUT / "protocol.json", OUT / "verification.json", Path(__file__).with_name("baseline_fairness.py")]:
            sources[str(path.relative_to(ROOT))] = sha(path)
    trajectories = {(d, m): chart.convergence(d, m, sources) for d in DATASETS for m in METHODS}
    initial = {}
    for d in DATASETS:
        path = ROOT / "results/uav_convergence_v2_20261027" / (d.lower() + "_pilot/rounds.json")
        sources[str(path.relative_to(ROOT))] = sha(path)
        initial[d] = read(path)[0]["evaluation"]["retained"]
    figures, series = [], []
    specs = [("01_round_fl_loss_1x2", "round", "loss", "FL loss", 1.),
             ("02_round_mr_1x2", "round", "mR", "mR (%)", 100.),
             ("03_time_fl_loss_1x2", "time", "loss", "FL loss", 1.),
             ("04_time_mr_1x2", "time", "mR", "mR (%)", 100.)]
    horizon = math.ceil(max(p[0][-1]["active_s"] for p in trajectories.values()) / 100) * 100
    for name, domain, metric, ylabel, scale in specs:
        fig, axes = chart.style.canvas(ylabel)
        panel_values = []
        for ax, dataset in zip(axes, DATASETS):
            values = []
            for number, (method, label) in enumerate(METHODS.items()):
                events, prefix, _ = trajectories[(dataset, method)]
                if domain == "round":
                    points = [(0, initial[dataset])] + [(p["round"], p["evaluation"]) for p in prefix["initial_points"]]
                    points += [(e["round"], e["evaluation"]) for e in events[1:]]
                    x = [p[0] for p in points]
                    y = [p[1][metric] * scale for p in points]
                    assert x[0] == 0 and x[-1] == 200
                    assert all(a <= b for a, b in zip(x, x[1:]))
                    chart.style.curve(ax, x, y, number, label)
                    display_x, display_y = x, y
                else:
                    x = [e["active_s"] for e in events]
                    y = [e["evaluation"][metric] * scale for e in events]
                    display_x, display_y = chart.draw_time(ax, x, y, number, label, horizon)
                series.append(dict(figure=name, dataset=dataset, method=method, x=x, y=y,
                    display_x=display_x, display_y=display_y,
                    completion_x=x[-1], final_value_held=domain == "time"))
                values += y
            chart.regular_axis(ax, "x", [0, 50, 100, 150, 200] if domain == "round"
                               else list(range(0, horizon + 1, 100)), 0)
            ax.set_xlabel("FL round" if domain == "round" else "Time (s)")
            panel_values.append(values)
        chart.finish_axes(axes, panel_values, decimals=2 if metric == "loss" else 1)
        figures.append(save(fig, axes, name))

    values = AXES["bandwidth"][1]
    ledgers = []
    for value in values:
        case = chart.SOURCE / "cases" / case_id("bandwidth", value)
        settings_path = case / "settings.json"
        sources[str(settings_path.relative_to(ROOT))] = sha(settings_path)
        parameters = read(settings_path)["energy_model"]
        for dataset in DATASETS:
            directory = case / (dataset.lower() + "_seed0")
            prefix = prefix_energy(directory, parameters, sources)
            for method in METHODS:
                ledgers.append(dict(bandwidth_MHz=value, dataset=dataset, method=method,
                                    **service_ledger(directory, method, parameters, prefix, sources)))
    index = {(r["bandwidth_MHz"], r["dataset"], r["method"]): r for r in ledgers}
    for name, metric, ylabel, scale in [
        ("05_bandwidth_total_energy_1x2", "total_energy_j", "Total energy (kJ)", .001),
        ("06_bandwidth_delay_1x2", "delay_s", "Delay (s)", 1.)]:
        fig, axes = chart.style.canvas(ylabel)
        panel_values = []
        status_markers = []
        for ax, dataset in zip(axes, DATASETS):
            ys = []
            for number, (method, label) in enumerate(METHODS.items()):
                records = [index[(v, dataset, method)] for v in values]
                y = [r[metric] * scale if r["status"] == "completed" else None for r in records]
                chart.style.curve(ax, values, [v if v is not None else math.nan for v in y], number, label)
                series.append(dict(figure=name, dataset=dataset, method=method, x=values, y=y,
                                   status=[r["status"] for r in records]))
                for value, record in zip(values, records):
                    if record["status"] != "completed":
                        # Non-numeric status marker: x is physical bandwidth;
                        # vertical placement is an axes fraction, never a measured y.
                        ax.plot([value], [1.04], transform=ax.get_xaxis_transform(),
                                marker="x", ms=7, mew=1.6, color=chart.COLORS[number],
                                linestyle="none", clip_on=False, zorder=8,
                                label="_nolegend_", gid=f"infeasible_{method}_{value}")
                        status_markers.append(dict(dataset=dataset, method=method, x=value,
                            status=record["status"], axes_y_fraction=1.04, metric_y=None))
                ys += y
            chart.regular_axis(ax, "x", [0, 1, 2, 3, 4], 0)
            ax.set_xlabel("Bandwidth (MHz)")
            panel_values.append(ys)
        if metric == "total_energy_j":
            for ax, ys in zip(axes, panel_values):
                chart.regular_axis(ax, "y", chart.nice_ticks(ys, padding=.025, bins=6, simple=True), 0)
        else:
            chart.finish_axes(axes, panel_values, decimals=0)
        figures.append(save(fig, axes, name, status_markers))

    assert len(figures) == 6 and len(series) == 6 * len(DATASETS) * len(METHODS)
    assert len(ledgers) == len(values) * len(DATASETS) * len(METHODS)
    write(DEST / "data/plot_series.json", series)
    write(DEST / "data/service_energy_delay.json", ledgers)
    with (DEST / "data/service_energy_delay.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledgers[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(ledgers)
    write(DEST / "data/manifest.json", dict(datasets=DATASETS, methods=METHODS, figures=figures, comparison=comparison,
        source_sha256=sources, energy_scope="FL/FU service communication + computing + hovering only",
        time_scope="Active mission FL/FU service, including scheduling waits; excludes travel, collection and preflight preparation",
        assumption="The duplicated time-loss item 4 is interpreted as time-mR, completing the round/time by loss/mR set.",
        status="Completed simulation records, including the registered new S-OF in the main view; plotting preserves raw measurements"))
    lines = ["# " + ("Matched-objective ablations" if comparison == "ablation" else "Selected graphs"), "", "Six figures only, each exported as PNG, PDF and SVG. "
        f"Left: CapERA. Right: AirScape. Methods: {', '.join(METHODS.values())}. The figures use the established style, regular endpoint ticks "
        "and matched decimal precision. No point-value labels. Bandwidth figures explicitly mark "
        "radio-infeasible conditions with a cross and a separate status legend.", "",
        "| No. | x axis | y axis | Preview |", "|---|---|---|---|"]
    for name, x, y in [(spec[0], "FL round" if spec[1] == "round" else "Time (s)", spec[3]) for spec in specs] + [
        ("05_bandwidth_total_energy_1x2", "Bandwidth (MHz)", "Total energy (kJ)"),
        ("06_bandwidth_delay_1x2", "Bandwidth (MHz)", "Delay (s)")]:
        lines.append(f"| {int(name[:2])} | {x} | {y} | [PNG]({name}.png) |")
    lines += ["", "Item 4 repeated time–FL loss in the request; it is interpreted as time–mR.", "",
        "## Exact definitions", "",
        "**FL loss** is the saved symmetric contrastive retrieval loss on held-out retained queries, not "
        "the per-client minibatch training objective. **mR** is the average of the six video-to-text/text-to-video "
        "Recall@1, @5 and @10 scores. Original metric values are preserved.", "",
        "**Round plots** cover the shared initial model at round 0, common preparation through round 20, "
        "and method-specific FL/FU observations through round 200. FU updates at an unchanged FL round retain "
        "their original x coordinate. **Time plots** start at the common round-20 model; time zero begins "
        "mission FL/FU service. Only active service windows are accumulated, including compute, uplink, "
        "aggregate broadcasts and scheduler waits. Flight, IoT collection and all preflight preparation "
        "are excluded. Time is modeled UAV time, not GPU/server runtime. Each final value is held horizontally "
        "to the common display endpoint; the extension is not an extra observation or proof of convergence.", "",
        "**Total energy** is total energy within those same FL/FU service windows: communication + computing "
        "+ hovering across all ten UAVs. It is not whole-mission battery energy. The original physical "
        "coefficients, per-device MACs, recorded transmit bounds and broadcast costs are used. "
        "Travel, collection communication/hovering, cold feature extraction, initial20 and proxy20 preparation "
        "are excluded. Consequently these numbers differ from the earlier energy subtotal, which retained "
        "collection and preflight costs. The original full energy ledger remains untouched. "
        "The service sum is independently reconciled against original total minus all excluded components.", "",
        "**Bandwidth** varies the UAV aggregation uplink symbol rate: 0.25, 0.5, 1, 2, 4 MHz. "
        "This is the existing uplink bandwidth sweep, not the downlink-rate sweep. Noise and all other "
        "conditions follow the original paired sensitivity protocol. **Delay** uses exactly the active "
        "FL/FU service time defined above. All listed methods and all five bandwidth values are retained for "
        f"both datasets: {sum(r['status'] == 'completed' for r in ledgers)} completed outcomes and "
        f"{sum(r['status'] != 'completed' for r in ledgers)} infeasible outcomes. "
        "Infeasible conditions, when present, are shown by a cross above the upper plot edge and a separate legend. "
        "Its vertical placement is fixed at 104% of the axes height, outside the data area, and does not "
        "encode an energy/delay value. The original missing metric remains null; no value is filled "
        "or extrapolated. Marker metadata is stored separately in `data/manifest.json`.", "",
        "All results use seed-0, 200-FL-round experiments and the unchanged validation subsets. "
        "Raw plotting arrays, service components, original/excluded totals, axis settings and source hashes "
        f"are in `data/`. Rebuild with `.venv/bin/python tools/uav_convergence_v2/selected_graphs.py --comparison {comparison}`.", ""]
    if comparison == "main":
        lines += ["## Comparison scope", "",
            "S-OF is a new measured serial digital OFDMA baseline using the same existing safe-proxy FU objective as S-AC and AC-OF. "
            "S-OF+ used the CLIPErase-inspired objective shared with Proposed. The new S-OF is not a relabeling of S-OF+. "
            "Only the FU objective changes between these two digital controls; optimized digital head/tones and all physical limits remain intact. "
            "This main comparison therefore compares whole systems, including the objective difference. "
            "It cannot by itself attribute a learning-quality difference to parallel transmission.", "",
            "[Matched-objective ablations](../ablation_graph/README.md) retain Proposed, H-OVA and S-OF+ with their original outcomes, "
            "including the AirScape H-OVA 4 MHz infeasibility. H-OVA isolates the head/allocation heuristic; "
            "S-OF+ is the same-objective serial digital control. Similar round-based utility is expected when their objectives match.", "",
            "S-OF and S-OF+ inherit the previously authorized soft network-TX-cap fallback; overruns are recorded. "
            "The analog arms retain their original hard network cap. Per-user power/energy, deadline and error checks remain enforced. "
            "See the [protocol and quality audit](../results/baseline_fairness_v1_20261027/README.md) for component definitions, "
            "forgetting diagnostics and all ten new outcomes. No hyperparameter was selected based on these results.", ""]
    elif comparison == "ablation":
        lines += ["## Interpretation", "",
            "All three methods use the same CLIPErase-inspired FM/RM/CM FU objective, model and workload. "
            "H-OVA replaces optimized head/resource allocation with a centroid head/uniform allocation heuristic. "
            "S-OF+ executes FL/FU serially through digital OFDMA with optimized head/tones. "
            "These matched-objective controls are retained to isolate head/allocation and transport/scheduling effects; "
            "similar retained loss/mR does not invalidate an energy/delay advantage. "
            "S-OF+ retains its previously authorized soft network-cap fallback, while Proposed/H-OVA keep hard caps.", "",
            "AirScape H-OVA at 4 MHz is radio-infeasible and remains explicitly marked; no energy or delay is fabricated. "
            "The [main comparison](../selected_graph/README.md) additionally reports the new safe-proxy S-OF baseline.", ""]
    (DEST / "README.md").write_text("\n".join(lines))
    print(dict(figures=len(figures), series=len(series), ledger_rows=len(ledgers),
               baseline=[r for r in ledgers if r["bandwidth_MHz"] == 1 and r["method"] == "proposed_fu_guard_cliperase"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", choices=["main", "ablation", "legacy"], default="main")
    generate(parser.parse_args().comparison)
