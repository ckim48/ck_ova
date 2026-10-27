"""Assemble the final paper figures from immutable records and the completed sweep."""
import collections
import copy
import csv
import html
import math
from pathlib import Path
import shutil
import tarfile

import numpy as np
import final_request_sweep as sweep
import final_broadcast_fix as broadcast_fix
import selected_graphs as plots
from final_request_sweep import ROOT, OUT, BASELINE, DATASETS, NEW, CASES, read, write, sha

DEST = ROOT / "final_chart"
METHODS = read(ROOT/"selected_graph/data/manifest.json")["methods"]
SOURCES = {}
SERIES = []
FIGURES = []
VALIDATED = set()
chart = plots.chart


def load(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return read(path)


def audited_manifest(path):
    result=load(path)
    if path not in VALIDATED:
        for name,digest in result["source_sha256"].items():assert sha(ROOT/name)==digest,name
        SOURCES.update(result["source_sha256"])
        for figure in result["figures"]:
            for f in figure["files"]:assert sha(path.parents[1]/f["file"])==f["sha256"]
        VALIDATED.add(path)
    return result


def csv_file(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)


def configure():
    chart.configure()
    from report import COLORS, MARKERS, STYLES
    for attr,values in [("COLORS",[COLORS[0],"#8C6D31",COLORS[2],COLORS[3],COLORS[4]]),
                        ("MARKERS",[MARKERS[0],"d",MARKERS[2],MARKERS[3],MARKERS[4]]),
                        ("STYLES",[STYLES[0],(0,(5,2)),STYLES[2],STYLES[3],STYLES[4]])]:
        setattr(chart,attr,values);setattr(chart.style,attr,values)
    plots.METHODS,plots.DEST=METHODS,DEST


def copy_figure(folder, old_name, name, title, supplementary=False):
    main = audited_manifest(folder/"data/manifest.json")
    old = next(f for f in main["figures"] if f["name"]==old_name)
    source_series = load(folder/"data/plot_series.json")
    metadata=copy.deepcopy(old);metadata.update(name=name,title=title,supplementary=supplementary,methods=main["methods"],files=[])
    for ext in ["png","pdf","svg"]:
        source=folder/(old_name+"."+ext);target=DEST/(name+"."+ext)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        SOURCES[str(source.relative_to(ROOT))]=sha(source)
        assert sha(source)==sha(target)
        metadata["files"].append(dict(file=str(target.relative_to(DEST)),sha256=sha(target)))
    for row in source_series:
        if row["figure"]==old_name:SERIES.append(dict(row,figure=name))
    FIGURES.append(metadata)


def copy_forget(old_name,name,title,supplementary=False):
    folder=ROOT/"selected_graph"
    manifest=audited_manifest(folder/"data/forgetting_manifest.json")
    old=next(f for f in manifest["figures"] if f["name"]==old_name)
    metadata=copy.deepcopy(old);metadata.update(name=name,title=title,supplementary=supplementary,methods=manifest["methods"],files=[])
    for ext in ["png","pdf","svg"]:
        source=folder/(old_name+"."+ext);target=DEST/(name+"."+ext)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        SOURCES[str(source.relative_to(ROOT))]=sha(source)
        assert sha(source)==sha(target)
        metadata["files"].append(dict(file=str(target.relative_to(DEST)),sha256=sha(target)))
    for row in load(folder/"data/forgetting_plot_series.json"):
        if row["figure"]==old_name:SERIES.append(dict(row,figure=name))
    FIGURES.append(metadata)


def request_curves():
    result=[]
    for groups in [45,15,9,5,3]:
        case="baseline" if groups==9 else f"request_bursts_{groups}"
        for dataset in DATASETS:
            directory=OUT/"cases"/case/(dataset.lower()+"_seed0")
            original_requests=load(directory/"requests.json")
            releases=collections.Counter(r["release_after_adaptation_FL"] for r in original_requests)
            assert len(original_requests)==45 and len(releases)==groups and set(releases.values())=={45//groups}
            for method,label in METHODS.items():
                corrected=groups==45 and method in broadcast_fix.METHODS
                arm=(broadcast_fix.OUT/"cases"/case/directory.name if corrected else directory)/method
                outcome=load(arm/"outcome.json")
                row=dict(case=case,dataset=dataset,method=method,label=label,release_groups=groups,
                         requests_per_release=45//groups,total_requests=45,status=outcome["status"],
                         broadcast_correction=corrected,source=str(arm.relative_to(ROOT)))
                if outcome["status"]=="completed":
                    requests=load(arm/"request_records.json")
                    assert len(requests)==45 and all(r["steps"]==5 for r in requests)
                    delays=[r["completed_s"]-r["arrival_s"] for r in requests]
                    assert all(v>=0 for v in delays)
                    mean,p95=float(np.mean(delays)),float(np.percentile(delays,95))
                    metrics=outcome["metrics"]
                    assert math.isclose(mean,metrics["FU_mean_completion_latency_s"],abs_tol=1e-9)
                    assert math.isclose(p95,metrics["FU_p95_completion_latency_s"],abs_tol=1e-9)
                    manifest=load(arm/"manifest.json");evaluation=manifest["evaluation"]
                    row.update(mean_latency_s=mean,p95_latency_s=p95,max_latency_s=max(delays),
                        individual_latency_s=delays,cap_overruns=metrics["network_cap_overrun_transactions"],
                        retained_mR_percent=100*evaluation["retained"]["mR"],retained_loss=evaluation["retained"]["loss"],
                        forget_KL=evaluation["forget"]["KL_retrain_to_candidate"],worst_request_KL=evaluation["worst_request_KL"],
                        common_quality_confirmed=manifest["joint_target"] is not None)
                else:assert "feasible" in outcome["status"],outcome
                result.append(row)
    assert len(result)==50
    write(DEST/"data/request_latency.json",result)
    csv_file(DEST/"data/request_latency.csv",[{k:v for k,v in r.items() if k!="individual_latency_s"} for r in result])
    for name,metric,ylabel,title in [
        ("06_request_mean_delay_1x2","mean_latency_s","Mean FU delay (s)","Mean FU request delay"),
        ("07_request_p95_delay_1x2","p95_latency_s","95th-percentile FU delay (s)","95th-percentile FU request delay")]:
        fig,axes=chart.style.canvas(ylabel);panel=[];status_markers=[]
        for ax,dataset in zip(axes,DATASETS):
            ys=[]
            for n,(method,label) in enumerate(METHODS.items()):
                rows=[r for r in result if r["dataset"]==dataset and r["method"]==method]
                x=[r["requests_per_release"] for r in rows]
                y=[r.get(metric) for r in rows]
                chart.style.curve(ax,x,[v if v is not None else math.nan for v in y],n,label)
                for r in rows:
                    if r["status"]!="completed":
                        ax.plot([r["requests_per_release"]],[1.04],transform=ax.get_xaxis_transform(),
                            marker="x",ms=7,mew=1.6,color=chart.COLORS[n],linestyle="none",clip_on=False)
                        status_markers.append(dict(dataset=dataset,method=method,x=r["requests_per_release"],
                            status=r["status"],axes_y_fraction=1.04,metric_y=None))
                SERIES.append(dict(figure=name,dataset=dataset,method=method,x=x,y=y,
                    display_x=x,display_y=y,status=[r["status"] for r in rows]))
                ys+=y
            chart.regular_axis(ax,"x",list(range(1,16,2)),0)
            ax.set_xlabel("FU requests per release");panel.append(ys)
        chart.finish_axes(axes,panel)
        record=plots.save(fig,axes,name,status_markers);record.update(title=title,supplementary=False);FIGURES.append(record)
    return result


def quality_costs():
    records=[]
    ledgers=load(ROOT/"selected_graph/data/service_energy_delay.json")
    for dataset in DATASETS:
        directory=BASELINE/"cases/baseline"/(dataset.lower()+"_seed0")
        reference=load(directory/"reference_metrics.json")
        threshold=reference["retained_mR_threshold"]
        assert math.isclose(threshold,.95*reference["metrics"]["retained"]["mR"],abs_tol=1e-12)
        params=load(directory.parent/"settings.json")["energy_model"]
        plots.physical.PARAMETERS=params
        for method,label in METHODS.items():
            arm=directory/method
            final=load(arm/"service_completion.json");timeline=load(arm/"mission_timeline.json")
            points=[load(p) for p in sorted((arm/"evaluations").glob("*.json"))]
            hits=[];confirmation=None
            for p in points:
                k=p["completed_FL_rounds"]
                if k%5:continue
                e=p["evaluation"]
                good=p["requests"]["procedure_completed"]==45 and e["retained"]["mR"]>=threshold and e["forget"]["KL_retrain_to_candidate"]<=.01 and e["worst_request_KL"]<=.01
                if not hits or hits[-1]["completed_FL_rounds"]!=k:hits=hits+[p] if good else []
                elif not good:hits=[]
                if len(hits)>=3:confirmation=p;break
            row=dict(dataset=dataset,method=method,label=label,retained_mR_threshold_percent=100*threshold,
                forget_KL_limit=.01,worst_request_KL_limit=.01,required_completed_requests=45,
                confirmation_observations=3,status="met" if confirmation else "not_met")
            target=final["joint_target"]
            if confirmation is None:assert target is None
            else:
                assert target is not None
                t=confirmation["mission_clock_s"]
                assert math.isclose(t,target["mission_clock_s"],abs_tol=1e-9)
                assert confirmation["completed_FL_rounds"]==target["confirmation_FL_round"]
                assert hits[0]["completed_FL_rounds"]==target["first_pass"]["fl_round"]
                active=chart.active_time(timeline,t);compute=communication=0.
                tx={r["sequence"]:r for r in [load(p) for p in sorted((arm/"transactions").glob("*.json"))]}
                for p in sorted((arm/"work_attempts").glob("*.json")):
                    work=load(p);tr=tx[work["transaction"]]
                    assert work["status"]=="transmitted"
                    if tr["end_s"]>t+1e-9:continue
                    compute+=float(plots.physical.compute_cost(work["compute"]["device_macs"])[0].sum())
                    compute+=params["communication"]["head_processing_w"]*tr["radio"]["uplink_s"]
                    communication+=plots.physical.communication(tr["radio"])[0]
                hover=params["iot_collection"]["uavs"]*float(plots.physical.propulsion_power(0.))*active
                energy=compute+communication+hover
                ledger=next(r for r in ledgers if r["bandwidth_MHz"]==1 and r["dataset"]==dataset and r["method"]==method)
                at_completion=math.isclose(t,final["mission_clock_s"],abs_tol=1e-9)
                if at_completion:
                    assert math.isclose(energy,ledger["total_energy_j"],abs_tol=1e-7)
                    assert math.isclose(active,ledger["delay_s"],abs_tol=1e-8)
                row.update(confirmation_FL_round=confirmation["completed_FL_rounds"],first_pass_FL_round=hits[0]["completed_FL_rounds"],
                    confirmation_mission_s=t,first_pass_mission_s=hits[0]["mission_clock_s"],
                    delay_s=active,energy_kJ=energy/1000,computing_j=compute,communication_j=communication,
                    hovering_j=hover,coincides_with_fixed_workload_completion=at_completion)
            records.append(row)
    write(DEST/"data/common_quality_costs.json",records);csv_file(DEST/"data/common_quality_costs.csv",records)
    for name,metric,ylabel,title in [
        ("08_energy_to_quality_1x2","energy_kJ","Energy to target (kJ)","Energy to the common quality target"),
        ("09_delay_to_quality_1x2","delay_s","Time to target (s)","Time to the common quality target")]:
        fig,axes=chart.style.canvas(ylabel);settings=[]
        for ax,dataset in zip(axes,DATASETS):
            values=[]
            for n,(method,label) in enumerate(METHODS.items()):
                r=next(r for r in records if r["dataset"]==dataset and r["method"]==method)
                value=r.get(metric);values.append(value)
                ax.bar(n,value if value is not None else 0,width=.65,color=chart.COLORS[n],edgecolor="black",linewidth=.6,
                       hatch=["///","..","\\\\","xx","--"][n],label=label)
                if value is None:ax.plot([n],[1.04],transform=ax.get_xaxis_transform(),marker="x",color=chart.COLORS[n],clip_on=False)
                SERIES.append(dict(figure=name,dataset=dataset,method=method,x=[n],y=[value],status=[r["status"]],categorical=True))
            ax.set_xlim(-.6,4.6);ax.set_xticks([]);ax.grid(False,axis="x")
            chart.regular_axis(ax,"y",chart.nice_ticks([0.,*values],bins=4,simple=True),0)
            settings.append(dict(dataset=dataset,x=dict(kind="categorical",positions=list(range(5)),labels=list(METHODS.values())),
                y=dict(limits=list(ax.get_ylim()),ticks=ax.get_yticks().tolist()),ylabel=ylabel))
        fig.legend(*axes[0].get_legend_handles_labels(),ncol=5,loc="upper center",bbox_to_anchor=(.5,1.),fontsize=12,
                   handlelength=1.6,columnspacing=.75)
        for i,(ax,dataset) in enumerate(zip(axes,DATASETS)):
            pos=ax.get_position();fig.text((pos.x0+pos.x1)/2,.015,f"({chr(97+i)}) {dataset}",ha="center",va="bottom",fontsize=14)
            assert not ax.texts
        files=save_exports(fig,name)
        FIGURES.append(dict(name=name,title=title,axes=settings,files=files,supplementary=False))
    return records


def save_exports(fig,name):
    files=[]
    for ext in ["png","pdf","svg"]:
        p=DEST/(name+"."+ext);fig.savefig(p,dpi=350,bbox_inches="tight")
        if ext=="svg":p.write_text("\n".join(s.rstrip() for s in p.read_text().splitlines())+"\n")
        files.append(dict(file=str(p.relative_to(DEST)),sha256=sha(p)))
    chart.plt.close(fig);return files


def quality_table():
    references=load(OUT/"pre_fu_reference.json");assert references["passed"]
    for name,digest in references["source_sha256"].items():assert sha(ROOT/name)==digest,name
    SOURCES.update(references["source_sha256"])
    rows=[]
    for dataset in DATASETS:
        directory=BASELINE/"cases/baseline"/(dataset.lower()+"_seed0")
        for stage,label in [("pre_FU_round20","Pre-FU (round 20)"),("retained_set_retrain_round200","Retrain (round 200)")]:
            r=next(r for r in references["rows"] if r["dataset"]==dataset and r["stage"]==stage)
            e=r["evaluation"]
            rows.append(dict(dataset=dataset,method=stage,label=label,retained_mR_percent=100*e["retained"]["mR"],
                retained_loss=e["retained"]["loss"],forget_KL=e["forget"].get("KL_retrain_to_candidate",0.),
                MIA_AUC=r["mia"]["loss_score_auc"],MIA_orientation_free_AUC=r["mia"]["orientation_free_auc"],
                MIA_AUC_gap_vs_retrain=r["MIA_AUC_gap_vs_retrain"]))
        for method,label in METHODS.items():
            e=load(directory/method/"manifest.json")["evaluation"];mia=load(directory/method/"mia.json")
            rows.append(dict(dataset=dataset,method=method,label=label,retained_mR_percent=100*e["retained"]["mR"],
                retained_loss=e["retained"]["loss"],forget_KL=e["forget"]["KL_retrain_to_candidate"],
                worst_request_KL=e["worst_request_KL"],MIA_AUC=mia["loss_score_auc"],
                MIA_orientation_free_AUC=mia["orientation_free_auc"],MIA_AUC_gap_vs_retrain=mia["auc_gap_vs_retrain"]))
    write(DEST/"data/final_quality.json",rows);csv_file(DEST/"final_quality.csv",rows)
    # A compact paper table; numeric cells are data, not point-value annotations.
    fig,axes=chart.plt.subplots(1,2,figsize=(10.,2.8))
    fig.subplots_adjust(left=.01,right=.99,top=.98,bottom=.11,wspace=.08)
    for ax,dataset in zip(axes,DATASETS):
        ax.axis("off");group=[r for r in rows if r["dataset"]==dataset]
        values=[[r["label"],f"{r['retained_mR_percent']:.2f}",f"{r['forget_KL']:.6f}",f"{r['MIA_AUC_gap_vs_retrain']:.5f}"] for r in group]
        table=ax.table(cellText=values,colLabels=["Method","Retained\nmR (%)","Forget-set\nKL","MIA AUC\ngap"],
                       colWidths=[.41,.17,.21,.21],cellLoc="center",bbox=[0,0,1,1])
        table.auto_set_font_size(False);table.set_fontsize(10)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor("#c0c0c0");cell.set_linewidth(.35)
            if r==0:cell.set_facecolor("#eaf0f6");cell.set_text_props(weight="bold")
            elif r%2==0:cell.set_facecolor("#f7f8fa")
        pos=ax.get_position();fig.text((pos.x0+pos.x1)/2,.02,dataset,ha="center",fontsize=12)
    FIGURES.append(dict(name="final_quality_table",title="Final quality and reference models",files=save_exports(fig,"final_quality_table"),table=True,supplementary=False))
    return rows


def archive_new_runs(out=OUT,cases=CASES,methods=NEW):
    files=[]
    for case in cases:
        directory=out/"cases"/case["id"]
        files += [p for p in directory.glob("*.json") if not p.is_symlink()]
        for dataset in DATASETS:
            dd=directory/(dataset.lower()+"_seed0");files += [dd/"plan.json",dd/"requests.json"]
            for method in methods:files += [p for p in (dd/method).rglob("*") if p.is_file() and not p.is_symlink()]
    entries={str(p.relative_to(ROOT)):dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in sorted(set(files))}
    destination=out/"raw_new_arms.tar.gz"
    with tarfile.open(destination,"w:gz",compresslevel=6) as archive:
        for name in entries:archive.add(ROOT/name,arcname=name,recursive=False)
    write(out/"raw_archive_manifest.json",dict(file=destination.name,sha256=sha(destination),archive_bytes=destination.stat().st_size,entries=entries))


def documentation(requests,costs,quality):
    primary=[f for f in FIGURES if not f["supplementary"]]
    lines=["# Final paper charts", "", "Left: CapERA. Right: AirScape. PNG previews, PDF for publication, SVG for editing. "
        "Open [index.html](index.html) to browse the complete set. Main methods: Proposed, FM-CM, S-AC, AC-OF, S-OF.", "",
        "| Figure | Content | Files |", "|---|---|---|"]
    for f in primary:
        n=f["name"];lines.append(f"| {n} | {f['title']} | [PNG]({n}.png) · [PDF]({n}.pdf) · [SVG]({n}.svg) |")
    lines += ["", "## Definitions and comparisons", "",
        "Retained mR evaluates models after both FL and FU updates. Forget-set KL compares their retrieval "
        "distributions with the final retained-set retrain. Early KL also reflects FL progress and future requests. "
        "The curves do not sum the different training objectives or certify deletion. The full-RM Proposed "
        "has better retained utility in these records, while FM-CM and safe-proxy methods have smaller final "
        "forget-set KL. MIA is a descriptive fixed loss-score attack on 128 members and 128 nonmembers; "
        "the table reports its absolute AUC gap to retrain, not a proof of privacy. Raw and orientation-free "
        "AUC, validation loss and worst-request KL are retained in final_quality.csv.", "",
        "The Pre-FU reference is the shared round-20 model, evaluated again without modifying its checkpoint. "
        "It is explicitly not a 200-round no-unlearning baseline. Retrain is the existing round-200 model "
        "trained without the final forgotten-data union. Its self-KL and MIA gap are zero by definition.", "",
        "Time and energy cover active FL/FU service, including computation, uplink, broadcasts and queue waits. "
        "Energy sums communication, computing and hovering across ten UAVs; travel, collection and preflight "
        "are excluded. Time is modeled UAV service time, not GPU wall time. These are single-seed validation "
        "results; no cross-seed confidence interval is implied. End-of-training values in time plots are "
        "held to the shared display horizon without claiming additional learning.", "",
        "Bandwidth axes show 1–4 MHz with measurements at 1, 2, 4 MHz. The tick at 3 is not another run; "
        "0.25/0.5 MHz outcomes remain in the data. S-OF retains the previously authorized soft network TX "
        "cap with overruns recorded; analog methods retain hard caps.", "",
        "## Request-load experiment", "",
        "The x axis is requests per release: 1, 3, 5, 9, 15, corresponding to 45, 15, 9, 5, 3 release groups. "
        "All conditions retain the same 45 requests, 225 local FU steps and 200 total FL rounds. Releases "
        "are triggered by FL progress, not a fixed wall-clock arrival rate. Changing group count also changes "
        "the release-round/site distribution according to the original schedule. Thus this is the existing "
        "burst-schedule sensitivity, not a sweep of total requests or independent Poisson arrival rates.", "",
        "Each request delay is final-broadcast completion minus actual arrival. The mean and empirical 95th "
        "percentile are computed across all 45 requests, including waits. The 95th percentile is not a confidence "
        "interval. These curves measure prescribed-step completion, not time to a quality/deletion guarantee. "
        "Final retained mR, mean/worst KL and the quality-confirmation flag for every load condition are included "
        "in data/request_latency.json and CSV. Original Proposed/S-AC/AC-OF runs are reused, and the missing 16 FM-CM/S-OF runs are added "
        f"without tuning. All 50 main-method conditions are retained: {dict(collections.Counter(r['status'] for r in requests))}.", "",
        "A cost-accounting defect was identified in the single-request condition: a nonzero head-local FU "
        "update had no uplink and was incorrectly assigned no aggregate broadcast. The six affected "
        "S-AC/AC-OF/S-OF arms (two datasets) were rerun with broadcasts required for nonzero decoded updates, "
        "using the existing 0.655872 s broadcast duration and power coefficients. The final request curves "
        "use those corrected runs. True zero updates still require no broadcast. An audit of all 50 "
        "main-method burst arms found no other affected conditions. Earlier raw outcomes are preserved; "
        "curves were not repaired by shifting values. The baseline/bandwidth charts are unaffected. "
        "Correction protocol, verification and raw records are in ../results/final_broadcast_fix_v1_20261027.", "",
        "## Common quality target", "",
        "The criterion was already defined in the original simulation: all 45 requests complete, retained "
        "mR at least 95% of the same dataset's retrain mR, mean forget KL at most 0.01, worst-request KL at "
        "most 0.01, confirmed at three distinct consecutive evaluated FL rounds. No threshold was chosen "
        "to separate the methods in this report. The cost charts use the 1 MHz baseline and the confirmation "
        "time, including those confirmation observations. Costs are independently rebuilt from the recorded "
        "transactions up to that time and use the same service-energy scope.", "",
        "For these ten baseline arms, confirmation occurs at round 200, coinciding with the fixed-workload "
        "completion. Consequently the target-cost bars equal the corresponding 1 MHz service totals; "
        "they verify a common acceptance condition and do not establish an additional speedup. The first "
        "passing observation (round 190) is also retained in the data. Passing this empirical criterion is "
        "not a formal deletion guarantee, and does not mean all methods have identical quality.", "",
        "## Supplementary figures", "",
        "The supplementary folder retains round/time validation loss, round forget-set KL, and all six "
        "matched-objective radio ablation figures (Proposed/H-OVA/S-OF+/NG-OVA). The original AirScape "
        "H-OVA 4 MHz infeasibility stays marked. These controls distinguish transport/head effects "
        "from differences in the FU objective. No earlier figure or outcome was overwritten.", "",
        "## Data and reproduction", "",
        "Numerical plot arrays, request latencies, quality targets, components, axes and hashes are in data/. "
        "The new raw runs are archived under ../results/final_request_sweep_v1_20261027, with the protocol, "
        "verification and source hashes. A module-import failure before any new experiment began was "
        "preserved along with its pre-correction protocol; only import ordering was corrected. "
        "Existing shared features and checkpoints remain in their original locations.", "",
        "Run `.venv/bin/python tools/uav_convergence_v2/final_request_sweep.py run`, "
        "`.venv/bin/python tools/uav_convergence_v2/final_reference_audit.py`, then "
        "`.venv/bin/python tools/uav_convergence_v2/final_broadcast_fix.py`, then "
        "`.venv/bin/python tools/uav_convergence_v2/final_charts.py`.", ""]
    (DEST/"README.md").write_text("\n".join(lines))
    cards=[]
    for f in FIGURES:
        n=f["name"];title=html.escape(f["title"])
        cards.append(f'<article><h2>{title}</h2><a href="{n}.png"><img loading="lazy" src="{n}.png" alt="{title}"></a>'
                     f'<p><a href="{n}.pdf">PDF</a> · <a href="{n}.svg">SVG</a> · <a href="{n}.png">PNG</a></p></article>')
    (DEST/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Final charts — CapERA and AirScape</title><style>body{margin:0;background:#f3f5f8;color:#182338;font:16px system-ui,sans-serif}'
        'main{max-width:1450px;margin:auto;padding:30px}h1{font-size:28px}h2{font-size:17px;margin:0 0 15px}a{color:#315c96}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,580px),1fr));gap:24px}'
        'article{background:white;padding:20px;border-radius:10px;border:1px solid #dfe4eb}img{width:100%;height:auto}p{line-height:1.6}</style>'
        '<main><h1>Final charts</h1><p>CapERA · AirScape | <a href="README.md">Definitions and comparison scope</a> · '
        '<a href="final_quality.csv">Quality table CSV</a></p><div class="grid">'+''.join(cards)+'</div></main></html>\n')


def generate():
    global SOURCES
    assert read(OUT/"finished.json")["passed"]
    assert read(broadcast_fix.OUT/"finished.json")["passed"]
    broadcast_fix.register()
    verification=sweep.verify()
    DEST.mkdir(exist_ok=True);(DEST/"data").mkdir(exist_ok=True)
    SOURCES={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for p in [OUT/"protocol.json",OUT/"verification.json",Path(sweep.__file__),Path(plots.__file__)]:SOURCES[str(p.relative_to(ROOT))]=sha(p)
    for p in [broadcast_fix.OUT/"protocol.json",broadcast_fix.OUT/"verification.json",Path(broadcast_fix.__file__)]:SOURCES[str(p.relative_to(ROOT))]=sha(p)
    configure()
    for old,new,title in [
        ("02_round_mr_1x2","01_round_retained_mr_1x2","Retained mR versus FL rounds"),
        ("04_time_mr_1x2","02_time_retained_mr_1x2","Retained mR versus active service time"),
        ("05_bandwidth_total_energy_1x2","04_bandwidth_total_energy_1x2","Total service energy versus bandwidth"),
        ("06_bandwidth_delay_1x2","05_bandwidth_delay_1x2","FL/FU service delay versus bandwidth")]:
        copy_figure(ROOT/"selected_graph",old,new,title)
    copy_forget("08_time_forget_kl_1x2","03_time_forget_kl_1x2","Forget-set KL versus active service time")
    requests=request_curves();costs=quality_costs();quality=quality_table()
    for old,new,title in [
        ("01_round_fl_loss_1x2","supplementary/round_validation_loss_1x2","Validation loss versus FL rounds"),
        ("03_time_fl_loss_1x2","supplementary/time_validation_loss_1x2","Validation loss versus active service time")]:
        copy_figure(ROOT/"selected_graph",old,new,title,True)
    copy_forget("07_round_forget_kl_1x2","supplementary/round_forget_kl_1x2","Forget-set KL versus FL rounds",True)
    for f in read(ROOT/"ablation_graph/data/manifest.json")["figures"]:
        copy_figure(ROOT/"ablation_graph",f["name"],"supplementary/ablation/"+f["name"],"Radio ablation: "+f["name"][3:].replace("_"," "),True)
    FIGURES.sort(key=lambda f:f["name"])
    documentation(requests,costs,quality)
    write(DEST/"data/plot_series.json",SERIES)
    write(DEST/"data/manifest.json",dict(figures=FIGURES,methods=METHODS,datasets=DATASETS,source_sha256=SOURCES,
        new_sweep_outcomes=verification["outcomes"],seed=0,quality_definition="Existing 95%-retrain mR / 0.01 mean and worst KL / all45 requests / three-observation confirmation",
        copied_figures="Existing exports copied byte-for-byte; extra figures use actual completed records only."))
    for name,digest in SOURCES.items():assert sha(ROOT/name)==digest,name
    for f in FIGURES:
        for file in f["files"]:assert sha(DEST/file["file"])==file["sha256"]
    archive_new_runs()
    archive_new_runs(broadcast_fix.OUT,[broadcast_fix.CASE],broadcast_fix.METHODS)
    write(OUT/"publication.json",dict(passed=True,folder="final_chart",primary_plots=9,quality_tables=1,supplementary_figures=9,
        new_arms=verification["outcomes"],broadcast_correction_arms=read(broadcast_fix.OUT/"verification.json")["outcomes"],
        request_conditions=len(requests),quality_confirmations=sum(r["status"]=="met" for r in costs)))
    (OUT/"README.md").write_text("# Final request-load sweep\n\n"
        "Sixteen additional FM-CM/S-OF runs complete the existing paired release-schedule experiment. "
        "All retain 45 requests, 225 local FU steps and 200 total FL rounds. "
        f"Outcomes: {verification['outcomes']}.\n\n"
        "[Final charts and definitions](../../final_chart/README.md) include the existing models and all new outcomes. "
        "The final reference audit is read-only and does not retrain a model.\n\n"
        "The module-import error before any new experiment started is preserved in bootstrap_failure.log, "
        "bootstrap_failed_runner.py and protocol.before_import_fix.json. The protocol records the import-order "
        "correction; no outcome was erased or chosen by rank.\n\n"
        "raw_new_arms.tar.gz contains all new arm records, checkpoints and scenario inputs, with hashes in "
        "raw_archive_manifest.json. Extract relative to the repository root. Shared features, baseline models "
        "and earlier-method records remain in their existing locations; this is not a self-contained data release.\n")
    (broadcast_fix.OUT/"README.md").write_text("# Nonzero head-local update broadcast correction\n\n"
        "Six single-request release cases were rerun after detecting omitted downlink broadcasts for "
        "nonzero local-only aggregation. A nonzero decoded update now triggers the existing broadcast "
        "payload, duration and PA/circuit energy even if its uplink is zero. True zero updates remain free. "
        "No learning objective, radio constraint, dataset or seed changed.\n\n"
        "The protocol audits all 50 main-method release-schedule conditions and identifies exactly these "
        "six affected arms. Verification checks the added broadcasts, full 225-step FU workload and "
        "200 FL rounds. Previous outcomes remain preserved.\n\n"
        "[Final charts](../../final_chart/README.md) use the corrected single-request points. "
        "All baseline and bandwidth curves remain unchanged. Raw corrected records are in "
        "raw_new_arms.tar.gz, with hashes in raw_archive_manifest.json. Extract relative to the repository "
        "root; shared datasets and prefix/reference checkpoints are not duplicated.\n\n"
        "Run `.venv/bin/python tools/uav_convergence_v2/final_broadcast_fix.py` after the original request sweep.\n")
    print(read(OUT/"publication.json"),flush=True)


if __name__=="__main__":generate()
