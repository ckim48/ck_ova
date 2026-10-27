"""Energy trends and a source-checked 200-round retained-retrieval audit."""
import csv
import math
from pathlib import Path

import time_energy_report as chart
from common import read, write, sha, METHODS, AXES, case_id

ROOT = chart.ROOT
DEST = ROOT / "results/paper_energy_trends_mr_audit_20261027"
PILOTS = ROOT / "results/uav_convergence_v2_20261027"
PAPER = "https://arxiv.org/html/2510.10180v2"
AIRSCAPE_PAPER = "https://arxiv.org/html/2507.08885v2"
PROPOSED = "proposed_fu_guard_cliperase"
RECALLS = [f"{direction}_R{k}" for direction in ["t2i", "i2t"] for k in [1, 5, 10]]


def generate():
    (DEST / "figures").mkdir(parents=True, exist_ok=True)
    chart.DEST = chart.style.DEST = DEST
    chart.configure()
    sources = {}

    def source(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    def check_metric(metric):
        assert math.isclose(sum(metric[k] for k in RECALLS) / 6, metric["mR"], abs_tol=1e-12)
        return metric

    figures, series, energy = [], [], []
    original = ROOT / "results/paper_pair_capera_airscape_time_energy_20261027/metrics.json"
    index = {(r["case"], r["dataset"], r["method"]): r for r in source(original)["rows"]}
    # Select physical controls for visible energy sensitivity, retain all methods/points.
    for axis in ["downlink", "collection_slot"]:
        xlabel, values, _ = AXES[axis]
        name = f"{axis}_energy_1x2"
        fig, axes = chart.style.canvas("Energy (kJ)")
        ys = []
        for ax, dataset in zip(axes, chart.DATASETS):
            panel = []
            for number, (method, label) in enumerate(METHODS.items()):
                records = [index[(case_id(axis, v), dataset, method)] for v in values]
                y = []
                for value, record in zip(values, records):
                    raw_path = chart.SOURCE / "cases" / record["case"] / (dataset.lower() + "_seed0") / method / "outcome.json"
                    raw = source(raw_path)
                    assert raw["status"] == record["status"]
                    if raw["status"] == "completed":
                        m = raw["metrics"]
                        subtotal = sum(m[k] for k in ["energy_communication_j", "energy_computing_j", "energy_hover_j"])
                        assert math.isclose(subtotal, record["energy_j"], abs_tol=1e-7)
                        y.append(subtotal / 1000)
                    else:
                        y.append(None)
                    energy.append(dict(axis=axis, x=value, **record))
                chart.style.curve(ax, values, [math.nan if v is None else v for v in y], number, label)
                series.append(dict(figure=name, dataset=dataset, method=method, x=values, y=y,
                                   status=[r["status"] for r in records]))
                panel += y
            xticks = chart.nice_ticks(values)
            chart.regular_axis(ax, "x", xticks, chart.tick_decimals(xticks))
            ax.set_xlabel(xlabel)
            ys.append(panel)
        chart.finish_axes(axes, ys)
        figures.append(chart.save_figure(fig, axes, name))

    audit, paths = {}, {}
    for dataset in chart.DATASETS:
        directory = chart.SOURCE / "cases/baseline" / (dataset.lower() + "_seed0")
        plan = source(directory / "plan.json")
        pilot = source(PILOTS / (dataset.lower() + "_pilot/rounds.json"))
        # Verify the pilot uses the same plan and frozen features as the mission.
        pilot_protocol = source(PILOTS / (dataset.lower() + "_pilot/protocol.json"))
        if dataset == "CapERA":
            original_plan = ROOT / "results/iot_uav_remoteclip_v4_event_mission/capera_seed0/plan.json"
            assert pilot_protocol["plan_sha256"] == sha(original_plan)
            assert source(original_plan) == plan  # Scenario copy uses different JSON formatting.
            assert pilot_protocol["features_sha256"] == sha(directory / "feature_cache/features.pt")
        else:
            pilot_plan = source(PILOTS / "airscape_pilot/plan.json")
            assert pilot_plan == plan
            assert sha(PILOTS / "airscape_pilot/feature_cache/features.pt") == sha(directory / "feature_cache/features.pt")
        warm = source(directory / "initial20/rounds.json")
        ref = source(directory / "reference200/rounds.json")
        reference = source(directory / "reference_metrics.json")["metrics"]["retained"]
        events, _, _ = chart.convergence(dataset, PROPOSED, sources)
        initial = check_metric(pilot[0]["evaluation"]["retained"])
        p = [(0, initial)] + [(r["round"], check_metric(r["evaluation"]["retained"]))
                              for r in warm if r.get("evaluation")]
        p += [(r["round"], check_metric(r["evaluation"])) for r in events[1:]]
        fl = [(r["round"], check_metric(r["evaluation"]["retained"])) for r in pilot]
        rt = [(0, initial)] + [(r["round"], check_metric(r["evaluation"]["retained"]))
                               for r in ref if r.get("evaluation")]
        assert p[-1][0] == fl[-1][0] == rt[-1][0] == 200
        assert rt[-1][1] == reference
        paths[dataset] = {"Proposed": p, "FL": fl, "RT": rt}
        metrics = dict(frozen=initial, before_mission_FU=warm[-1]["evaluation"]["retained"],
                       no_FU_ideal_FL200=fl[-1][1], retained_only_ideal_retrain200=reference,
                       proposed_FL_FU200=p[-1][1])
        audit[dataset] = dict(metrics=metrics,
            proposed_minus_no_FU_pp=100 * (p[-1][1]["mR"] - fl[-1][1]["mR"]),
            proposed_minus_retrain_pp=100 * (p[-1][1]["mR"] - reference["mR"]),
            evaluation=dict(validation_video_queries=len(plan["validation_images"]),
                gallery_videos=len(plan["gallery_images"]), captions_per_video=plan.get("caption_count", 5),
                official_test_used=False, seed=0),
            training_videos=sum(map(len, plan["partitions"])), forgotten_videos=len(plan["forgotten_images"]),
            interpretation="No large retained-mR deficit relative to these local controls. "
                "This does not establish the causal FU effect: controls differ in data availability, "
                "batch sequence and communication; ongoing FL prevents a simple before/after retention interpretation.")

    name = "mR_round_controls_1x2"
    fig, axes = chart.style.canvas("Retained mR (%)")
    ys = []
    for ax, dataset in zip(axes, chart.DATASETS):
        panel = []
        for number, (label, path) in enumerate(paths[dataset].items()):
            x = [p[0] for p in path]
            y = [100 * p[1]["mR"] for p in path]
            chart.style.curve(ax, x, y, number, label)
            series.append(dict(figure=name, dataset=dataset, method=label, x=x, y=y))
            panel += y
        chart.regular_axis(ax, "x", [0, 50, 100, 150, 200], 0)
        ax.set_xlabel("FL round")
        ys.append(panel)
    chart.finish_axes(axes, ys, decimals=1)
    figures.append(chart.save_figure(fig, axes, name))

    # Factual numerical transcription from CapERA rows of TCMA v2 Tables I/II.
    papers = []
    for label, t2v, v2t in [
        ("CLIP ViT-B/32", [8.8, 24.7, 36.8], [4.6, 18.6, 32.0]),
        ("CLIP ViT-B/16", [8.8, 26.0, 38.1], [4.5, 18.8, 32.6]),
        ("TCMA ViT-B/32", [14.9, 41.9, 57.8], [12.7, 34.6, 49.6]),
        ("TCMA ViT-B/16", [16.1, 44.6, 61.2], [15.0, 39.3, 53.5]),
    ]:
        papers.append(dict(model=label, text_to_video_R1_R5_R10=t2v, video_to_text_R1_R5_R10=v2t,
                           computed_mR_percent=sum(t2v + v2t) / 6, source=PAPER,
                           tables=["I", "II"], direct_comparison=False))
    write(DEST / "literature.json", dict(capera=papers,
        meaning="mR calculated from the six published recall numbers, not a separately reported column; "
                "neither MdR nor MnR is mean recall.",
        airscape=dict(source=AIRSCAPE_PAPER, table=1, metrics=["FID", "FVD", "IAR"],
                      retrieval_mR_reported=False)))
    write(DEST / "mr_audit.json", audit)
    write(DEST / "energy_metrics.json", energy)
    write(DEST / "plot_series.json", series)
    with (DEST / "mr_controls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "control"] + RECALLS + ["mR", "loss"])
        writer.writeheader()
        for dataset, result in audit.items():
            for control, metrics in result["metrics"].items():
                writer.writerow(dict(dataset=dataset, control=control,
                                     **{key: metrics[key] for key in RECALLS + ["mR", "loss"]}))
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    write(DEST / "summary.json", dict(figures=figures, source_sha256=sources,
        policy="Existing observations only. Select energy axes for physical sensitivity; retain every method, "
               "sample and infeasible status. No new training, metric tuning, interpolation over infeasible points or smoothing."))
    lines = ["# Energy sensitivity and 200-round retrieval audit", "",
        "## Energy figures", "",
        "[Downlink rate](figures/downlink_energy_1x2.png) is the primary decreasing-energy example. "
        "[IoT collection slot duration](figures/collection_slot_energy_1x2.png) is the increasing-energy example. "
        "Both contain CapERA (left), AirScape (right), and all five methods with unchanged measured scenario outcomes. "
        "They use the established colors, regular endpoint ticks, matched numeric precision, and no point-value annotations.", "",
        "Energy is communication + computing + hovering, as in the current report. Common travel energy is excluded. "
        "A faster downlink shortens aggregate broadcasts and their hover duration. Longer IoT slots extend hovering; "
        "the measured communication contribution is also retained. The 0.625-second slot violates the original IoT "
        "power constraint for all arms and remains missing, with its status recorded in `energy_metrics.json`. "
        "The complete ten-axis report remains available in the sibling time/energy folder.", "",
        "## Our 200-round controls", "",
        "mR is the arithmetic mean of video-to-text and text-to-video Recall@1, @5, @10. "
        "The code uses `i2t`/`t2i` names for pooled-video features. Every six-term average was checked. "
        "Neither class-average recall, Mean Rank (MnR), nor Median Rank (MdR) is used.", "",
        "| Condition | CapERA mR (%) | AirScape mR (%) |", "|---|---:|---:|"]
    for key, label in [("frozen", "Frozen encoder / initial adapters"),
                       ("before_mission_FU", "Common warm model, round 20, before mission FU"),
                       ("no_FU_ideal_FL200", "FL: full training pool, no FU, ideal aggregation, round 200"),
                       ("retained_only_ideal_retrain200", "RT: retained-only retraining from initialization, ideal aggregation, round 200"),
                       ("proposed_FL_FU200", "Proposed: 200 FL rounds and all prescribed FU work")]:
        lines.append(f"| {label} | {audit['CapERA']['metrics'][key]['mR']*100:.2f} | {audit['AirScape']['metrics'][key]['mR']*100:.2f} |")
    lines += ["", "[Round-based control curves](figures/mR_round_controls_1x2.png) use the labels above. "
        "FL and RT are diagnostic references, not additional mission algorithms. They incur no simulated radio errors, "
        "and no mission energy or delay is assigned to them. The same frozen feature archive, adapter initialization, "
        "retained queries, gallery and 200-round learning-rate schedule are used. FL trains on the full pool; RT "
        "removes every eventual deletion item from the beginning; Proposed handles arriving requests online. "
        "The common mission prefix includes actual communication, unlike the ideal controls. Therefore these are "
        "not isolated toggles of the FU term, and the small positive mR differences do not prove that FU improves utility.", "",
        "The same fixed retained evaluation is used throughout: CapERA has 256 video queries, 503 gallery videos "
        "and five captions per video; AirScape has 256 queries, 614 gallery videos and one caption per video. "
        "These are exploratory held-out validation subsets, not official benchmark test evaluations; seed 0 only.", "",
        "The local controls do not show a drop from a high-performing 35% model to 16.9% after FU. The underlying "
        "200-round learner itself reaches only 16.42%/13.54%. Proposed matches that limited utility level. "
        "It is reasonable to report this as retention relative to local references, but insufficient to claim "
        "strong absolute retrieval performance, statistical superiority, or effective deletion from mR alone. "
        "Forgetting must be assessed separately against retained-only retraining, including the existing forget-side "
        "distribution and privacy diagnostics.", "",
        "The separate [longer-learning diagnostic](../uav_longer_learning_diagnostic_20261027/) reached "
        "22.51%/15.49% after 1000 FL-only rounds with a restarted learning rate. It shows additional learning headroom, "
        "but changes both round budget and schedule; those values are not inserted into this 200-round comparison. "
        "The present frozen RemoteCLIP B/32, eight-frame mean pooling and rank-10 adapters remain possible "
        "modeling limitations; their individual effects have not been isolated.", "",
        "## External context", "",
        f"From the CapERA rows of [TCMA v2 Tables I and II]({PAPER}), the six-recall averages are:", "",
        "| Published model | Computed mR (%) |", "|---|---:|"]
    for row in papers:
        lines.append(f"| {row['model']} | {row['computed_mR_percent']:.2f} |")
    lines += ["", "These are contextual values, not a matched FL/FU comparison. TCMA uses CLIP initialization, "
        "12 sampled frames and encoder learning; our frozen-feature setup and custom validation evaluation differ. "
        "The raw recall inputs and their arithmetic are retained in `literature.json`.", "",
        f"[AirScape Table 1]({AIRSCAPE_PAPER}) evaluates generated-video quality and intention alignment with FID, FVD and IAR. "
        "It does not supply a retrieval-mR baseline. Our 13.67% therefore has no directly matched original-paper mR reference.", "",
        "The present evidence supports comparing communication efficiency at matched modest retrieval utility. "
        "A stronger absolute-accuracy claim needs a stronger or better-trained learner and rerunning every method "
        "under the same revised training budget, with energy and delay recomputed. Independent seeds and a final "
        "held-out test should be used for confirmatory conclusions.", ""]
    (DEST / "README.md").write_text("\n".join(lines))
    print(dict(figures=len(figures), controls={d: {k: round(v["mR"] * 100, 4)
          for k, v in a["metrics"].items()} for d, a in audit.items()}), flush=True)


if __name__ == "__main__":
    generate()
