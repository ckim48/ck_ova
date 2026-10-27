"""Add common forget-set evaluation curves; preserve all six existing figures."""
import collections
import csv
import math
from pathlib import Path

import learning_ablation as study
import selected_graphs as plots
from learning_ablation import ROOT, OUT, read, write, sha

DEST = ROOT / "selected_graph"
MARKER = "## FU evaluation supplement"


def generate():
    main = read(DEST / "data/manifest.json")
    assert main["comparison"] == "learning"
    methods = main["methods"]
    assert list(methods) == [study.PROPOSED, study.NEW, "separate_aircomp_periodic",
                             "aircomp_ofdma_periodic", "sequential_digital_proxy"]
    preserved = {str(p.relative_to(ROOT)): sha(p)
                 for p in DEST.iterdir() if p.suffix in [".png", ".pdf", ".svg"] and p.name[:2] in ["01", "02", "03", "04", "05", "06"]}
    preserved.update({str((DEST/"data"/name).relative_to(ROOT)): sha(DEST/"data"/name)
                      for name in ["manifest.json", "plot_series.json", "service_energy_delay.json", "service_energy_delay.csv"]})
    chart = plots.chart
    chart.configure()
    from report import COLORS, MARKERS, STYLES
    for attr, values in [
        ("COLORS", [COLORS[0], "#8C6D31", COLORS[2], COLORS[3], COLORS[4]]),
        ("MARKERS", [MARKERS[0], "d", MARKERS[2], MARKERS[3], MARKERS[4]]),
        ("STYLES", [STYLES[0], (0, (5, 2)), STYLES[2], STYLES[3], STYLES[4]])]:
        setattr(chart, attr, values); setattr(chart.style, attr, values)
    plots.METHODS, plots.DEST = methods, DEST
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for name in ["selected_graphs.py", "time_energy_report.py"]:
        p = Path(__file__).with_name(name); sources[str(p.relative_to(ROOT))] = sha(p)

    def load(p):
        sources[str(p.relative_to(ROOT))] = sha(p)
        return read(p)

    saved = read(DEST / "data/plot_series.json")
    points, audits = {}, []
    for dataset in main["datasets"]:
        directory = OUT / "cases/baseline" / (dataset.lower()+"_seed0")
        reference = directory / "reference200/final.pt"
        sources[str(reference.relative_to(ROOT))] = sha(reference)
        for method in methods:
            arm = directory / method
            timeline = load(arm / "mission_timeline.json")
            records = [load(p) for p in sorted((arm / "evaluations").glob("*.json"))]
            final = load(arm / "service_completion.json")
            if final["sequence"] != records[-1]["sequence"]:
                records.append(final)
            sequence = []
            for r in records:
                tx = load(arm / "transactions" / f"{r['sequence']:04d}.json")
                assert r["evaluation"] == tx["evaluation"]
                assert math.isclose(r["mission_clock_s"], tx["end_s"], abs_tol=1e-9)
                value = r["evaluation"]["forget"]["KL_retrain_to_candidate"]
                assert math.isfinite(value) and value >= 0
                sequence.append(dict(sequence=r["sequence"], round=r["completed_FL_rounds"],
                    active_s=chart.active_time(timeline, r["mission_clock_s"]), forget_KL=value,
                    retained_loss=r["evaluation"]["retained"]["loss"],
                    completed_requests=r["requests"]["procedure_completed"]))
            assert sequence[-1]["round"] == 200 and sequence[-1]["completed_requests"] == 45
            old = next(r for r in saved if r["figure"]=="01_round_fl_loss_1x2"
                       and r["dataset"]==dataset and r["method"]==method)
            mission = [(x,y) for x,y in zip(old["x"], old["y"]) if x>20]
            assert mission == [(p["round"], p["retained_loss"]) for p in sequence]
            attempts = [load(p) for p in sorted((arm / "work_attempts").glob("*.json"))]
            fu = [r for r in attempts if r["kind"] in ["fu", "joint"]]
            assert all(r["status"]=="transmitted" and r["FU_vector_norm"]>0 for r in fu)
            steps = sum(len(r["local"]) for r in fu)
            assert steps == 225
            audits.append(dict(dataset=dataset, method=method, evaluation_points=len(sequence),
                transaction_kinds=dict(collections.Counter(r["kind"] for r in attempts)),
                nonzero_transmitted_FU_transactions=len(fu), local_FU_steps=steps,
                retained_curve_matches_same_post_update_evaluations=True,
                first_recorded_round=sequence[0]["round"], final_forget_KL=sequence[-1]["forget_KL"],
                final_retained_loss=sequence[-1]["retained_loss"]))
            points[(dataset, method)] = sequence
    horizon = max(r["display_x"][-1] for r in saved if r["figure"]=="03_time_fl_loss_1x2")
    assert horizon % 100 == 0
    series, figures = [], []
    for name, domain in [("07_round_forget_kl_1x2", "round"), ("08_time_forget_kl_1x2", "time")]:
        fig, axes = chart.style.canvas("Forget-set KL")
        all_ticks = []
        for ax, dataset in zip(axes, main["datasets"]):
            values = []
            for number, (method, label) in enumerate(methods.items()):
                rows = points[(dataset,method)]
                x = [p["round" if domain=="round" else "active_s"] for p in rows]
                y = [p["forget_KL"] for p in rows]
                assert all(a<=b for a,b in zip(x,x[1:]))
                if domain=="round":
                    chart.style.curve(ax,x,y,number,label); dx,dy=x,y
                else:
                    dx,dy=chart.draw_time(ax,x,y,number,label,horizon)
                series.append(dict(figure=name,dataset=dataset,method=method,x=x,y=y,
                    display_x=dx,display_y=dy,final_value_held=domain=="time",
                    metric="KL_retrain_to_candidate",evaluation_group="forget",
                    observation_sequences=[p["sequence"] for p in rows]))
                values+=y
            chart.regular_axis(ax,"x",[0,50,100,150,200] if domain=="round" else list(range(0,int(horizon)+1,100)),0)
            ax.set_xlabel("FL round" if domain=="round" else "Time (s)")
            all_ticks.append(chart.nice_ticks([0.,*values],bins=4,simple=True))
        decimals=max(chart.tick_decimals(ticks) for ticks in all_ticks)
        for ax,ticks in zip(axes,all_ticks):chart.regular_axis(ax,"y",ticks,decimals)
        figures.append(plots.save(fig,axes,name))
    write(DEST / "data/forgetting_plot_series.json",series)
    write(DEST / "data/forgetting_manifest.json",dict(figures=figures,methods=methods,datasets=main["datasets"],
        source_sha256=sources,retained_curve_audit=audits,unchanged_existing_files=preserved,
        definition="Mean video-to-text/text-to-video KL(reference || candidate) on the fixed forget-query set; the reference is the final retained-set retrain model.",
        model_scope="Both FL and FU updates are applied before evaluation. The existing retained loss and this KL use the same recorded model state.",
        interpretation="Smaller KL means closer predictions to the reference on these probes. Early differences also include FL progress and requests not yet arrived; this is not an isolated FU training loss or deletion/privacy certificate.",
        missing_initial="Initial20 has no saved reference KL. Curves start at the first recorded value (FL round 25); no initial value is fabricated.",
        time_scope=main["time_scope"],status="Existing evaluation records only; no new training or metric rescaling."))
    with (DEST / "data/forgetting_summary.csv").open("w",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=list(audits[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(audits)
    text = ["# Retained loss and forgetting evaluation", "",
        "The existing loss curves already evaluate the model after both FL and FU updates. They select "
        "the retained-query retrieval negative log likelihood from each saved evaluation. FU is not omitted "
        "from the model; the plotted scalar simply measures retention rather than forgetting. FL still "
        "changes the model throughout training, even when its update rule is shared across methods.", "",
        "Training objectives and common evaluation criteria are different quantities. Proposed's FU "
        "objective uses 3 FM + RM + 3 CM; FM-CM removes RM, and the safe-proxy baselines use another "
        "objective. Comparing each method's own FL-plus-FU scalar would therefore compare different "
        "functions, coefficients and batch reductions. Removing a term can lower that sum without "
        "improving the model. No arbitrary combined loss has been introduced.", "",
        "Two additional figures use the already recorded common evaluation criterion:", "",
        "- [Round versus forget-set KL](07_round_forget_kl_1x2.png)",
        "- [Active service time versus forget-set KL](08_time_forget_kl_1x2.png)", "",
        "Both directions of retrieval are averaged: KL(P_retrain || P_candidate), evaluated on the fixed "
        "forget-query set. The same final retained-set retrain reference is used for all methods within "
        "a dataset. Lower values indicate closer predictions to that reference on these probes. At early "
        "rounds the difference also includes FL progress and future requests, since the reference omits "
        "the final union of forgotten samples. This curve is descriptive and is neither a pure causal "
        "measure of FU nor a deletion/privacy certificate. Final comparisons are after all 45 requests "
        "complete 225 local FU steps.", "",
        "The source audit verifies nonzero transmitted FU vectors and confirms that the original retained "
        "loss curves are drawn from the same post-update evaluation records. The first saved reference "
        "KL is at FL round 25; no round-0/20 KL is fabricated. The time plot holds the final observation "
        "to the common endpoint, as in the existing charts. Both axes retain regular endpoint ticks, "
        "matching decimal precision, and the established colors and markers.", "",
        "All six existing figures and their numerical exports remain byte-for-byte unchanged. "
        "Use retained loss/mR together with forget-set KL and the existing final MIA diagnostics. "
        "The current full-RM method has better retained utility but larger final forget-set KL than "
        "FM-CM and the safe-proxy methods. These results do not support claiming best performance "
        "on every learning/unlearning criterion.", "",
        "Rebuild after the six primary figures: "
        "`.venv/bin/python tools/uav_convergence_v2/forgetting_graphs.py`.", ""]
    (DEST / "FU_METRICS.md").write_text("\n".join(text))
    readme=DEST / "README.md"
    original=readme.read_text().split(MARKER)[0].rstrip()
    original=original.replace("Six figures only, each exported", "Six primary figures, each exported")
    readme.write_text(original+"\n\n"+MARKER+"\n\nThe original loss already includes the effect of FL and FU updates on retained queries. "
        "Two additional figures show the common forget-set KL metric from the same models: "
        "[round](07_round_forget_kl_1x2.png) and [time](08_time_forget_kl_1x2.png). "
        "See [metric definitions and audit](FU_METRICS.md). The original six figures are unchanged.\n")
    for p,digest in preserved.items():assert sha(ROOT/p)==digest,p
    for p,digest in sources.items():assert sha(ROOT/p)==digest,p
    print(dict(figures=len(figures),series=len(series),verified_models=len(audits),
               verified_FU_steps=sum(r["local_FU_steps"] for r in audits),unchanged_files=len(preserved)),flush=True)


if __name__ == "__main__":
    generate()
