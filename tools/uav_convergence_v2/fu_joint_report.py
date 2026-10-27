"""Report all preregistered FU pilot outcomes without changing final_chart."""
import collections
import csv
import html
import math
from pathlib import Path
import tarfile

import numpy as np
import fu_joint_optimization as study
from fu_joint_optimization import ROOT, OUT, SOURCE, PROPOSED, DATASETS, VARIANTS, read, write, sha
import selected_graphs as plots

DEST=ROOT/"fu_joint_pilot"
SOURCES={}
METHODS={PROPOSED:"Proposed",**{k:v["label"] for k,v in VARIANTS.items()}}


def load(p):
    SOURCES[str(p.relative_to(ROOT))]=sha(p)
    return read(p)


def directory(dataset,method):
    root=SOURCE if method==PROPOSED else OUT
    return root/"cases/baseline"/(dataset.lower()+"_seed0")


def csv_file(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)


def finish_axes(axes,values,metric):
    chart=plots.chart
    chart.finish_axes(axes,values)
    if metric=="kl":
        # KL is nonnegative; retain regular endpoint ticks without negative padding.
        ticks=[[float(v) for v in ax.get_yticks() if v>=0] for ax in axes]
        precision=max(chart.tick_decimals(t) for t in ticks)
        for ax,t in zip(axes,ticks):chart.regular_axis(ax,"y",t,precision)


def references_html():
    return ('<p>방법 참고: <a href="https://aclanthology.org/2025.acl-long.1469/">CLIPErase (ACL 2025)</a>의 FM/RM/CM 구분과 '
        '<a href="https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html">PCGrad (NeurIPS 2020)</a>의 충돌 방향 투영을 참고했다. '
        '여기서는 UAV 로컬 업데이트에 단방향 투영을 적용했으며, 두 논문의 원래 실험을 재현했다는 의미는 아니다.</p>')


def gather():
    rows=[];points={}
    for dataset in DATASETS:
        for method,label in METHODS.items():
            dd=directory(dataset,method);arm=dd/method;o=load(arm/"outcome.json")
            row=dict(dataset=dataset,method=method,label=label,status=o["status"],
                diagnostic_only=VARIANTS.get(method,{}).get("ideal",False),source=str(arm.relative_to(ROOT)))
            if o["status"]!="completed":
                row["failure_reason"]=o.get("error",o["status"]);rows.append(row);continue
            manifest=load(arm/"manifest.json");e=manifest["evaluation"];mia=load(arm/"mia.json")
            params=load(dd.parent/"settings.json")["energy_model"]
            prefix=plots.prefix_energy(dd,params,SOURCES)
            ledger=plots.service_ledger(dd,method,params,prefix,SOURCES)
            row.update(retained_mR_percent=100*e["retained"]["mR"],retained_loss=e["retained"]["loss"],
                forget_KL=e["forget"]["KL_retrain_to_candidate"],worst_request_KL=e["worst_request_KL"],
                MIA_AUC_gap=mia["auc_gap_vs_retrain"],MIA_AUC=mia["loss_score_auc"],
                energy_kJ=ledger["total_energy_j"]/1000,delay_s=ledger["delay_s"],
                communication_j=ledger["service_communication_j"],computing_j=ledger["service_computing_j"],hovering_j=ledger["service_hovering_j"],
                joint_target_confirmed=manifest["joint_target"] is not None)
            work=[load(p) for p in sorted((arm/"work_attempts").glob("*.json"))]
            local=[r for w in work for r in w["local"]]
            assert len(local)==225
            row["FU_control_scalar_ops"]=sum(sum(w["compute"].get("FU_control_scalar_ops",[])) for w in work)
            row["FU_learning_rate_min"]=min(r["FU_learning_rate"] for r in local)
            row["FU_learning_rate_max"]=max(r["FU_learning_rate"] for r in local)
            if VARIANTS.get(method,{}).get("adaptive"):
                row["adaptive_weight_median"]=float(np.median([r["adaptive_preservation_weight"] for r in local]))
            if VARIANTS.get(method,{}).get("fusion"):
                row["local_projected_fraction"]=float(np.mean([r["fusion_projected"] for r in local]))
                row["local_negative_dot_after_count"]=sum(r["FL_FU_dot_after"]<-1e-12 for r in local)
                # Only one observation per transmitted aggregate, not one per request.
                first=[w["local"][0] for w in work if w["local"]]
                row["global_conflicting_transactions_before"]=sum(r["global_FL_FU_dot_before"]<0 for r in first)
                row["global_conflicting_transactions_after"]=sum(r["global_FL_FU_dot_after"]<0 for r in first)
                row["global_conflict_transactions"]=len(first)
            if VARIANTS.get(method,{}).get("ideal"):
                row["discarded_receiver_error_L2_max"]=max(w["radio"]["discarded_noisy_receiver_error_L2"] for w in work)
                assert all(w["radio"]["applied_error_to_raw_L2"]==0 for w in work)
            sequence=[]
            for p in sorted((arm/"evaluations").glob("*.json")):
                r=load(p);ev=r["evaluation"]
                sequence.append(dict(round=r["completed_FL_rounds"],sequence=r["sequence"],
                    mr=100*ev["retained"]["mR"],kl=ev["forget"]["KL_retrain_to_candidate"],
                    completed_requests=r["requests"]["procedure_completed"]))
            assert sequence[-1]["round"]==200 and sequence[-1]["completed_requests"]==45
            points[(dataset,method)]=sequence
            rows.append(row)
    for r in rows:
        if r["status"]!="completed":continue
        p=next(x for x in rows if x["dataset"]==r["dataset"] and x["method"]==PROPOSED)
        r["mR_change_pp"]=r["retained_mR_percent"]-p["retained_mR_percent"]
        r["KL_reduction_percent"]=100*(1-r["forget_KL"]/p["forget_KL"])
        r["energy_change_percent"]=100*(r["energy_kJ"]/p["energy_kJ"]-1)
        r["delay_change_percent"]=100*(r["delay_s"]/p["delay_s"]-1)
        r["MIA_gap_change"]=r["MIA_AUC_gap"]-p["MIA_AUC_gap"]
        r["mR_not_lower_and_KL_lower"]=r["mR_change_pp"]>=-1e-8 and r["KL_reduction_percent"]>1e-8
        r["all_reported_quality_and_cost_not_worse"]=(r["mR_change_pp"]>=-1e-8 and r["forget_KL"]<=p["forget_KL"]+1e-12 and
            r["worst_request_KL"]<=p["worst_request_KL"]+1e-12 and r["MIA_gap_change"]<=1e-12 and
            r["energy_change_percent"]<=1e-8 and r["delay_change_percent"]<=1e-8)
    return rows,points


def figures(rows,points):
    chart=plots.chart;chart.configure();plots.DEST=DEST
    for key,value in [("COLORS",["#4C72B0","#8C6D31","#F4A64B","#E15759","#2993A8","#60AE62","#8D6BB8"]),
                      ("MARKERS",["o","d","s","^","P","v","X"]),
                      ("STYLES",["-",(0,(5,2)),"-.",":",(0,(2,2)),"--",(0,(4,1,1,1))])]:
        setattr(chart,key,value);setattr(chart.style,key,value)
    groups=[("weight_rate",[PROPOSED,"fu_balanced","fu_request_lr","fu_balanced_request_lr"]),
            ("fusion",[PROPOSED,"fu_direction_fusion","fu_balanced_request_lr","fu_balanced_request_lr_fusion"]),
            ("receiver",[PROPOSED,"fu_exact_aggregate"])]
    figures=[];series=[];number=0
    for group,methods in groups:
        plots.METHODS={m:METHODS[m] for m in methods}
        for metric,ylabel in [("mr","Retained mR (%)"),("kl","Forget-set KL")]:
            number+=1;name=f"{number:02d}_{group}_{metric}_1x2"
            fig,axes=chart.style.canvas(ylabel);values=[]
            for ax,dataset in zip(axes,DATASETS):
                ys=[]
                for i,method in enumerate(methods):
                    if (dataset,method) not in points:continue
                    rr=points[(dataset,method)];x=[r["round"] for r in rr];y=[r[metric] for r in rr]
                    chart.style.curve(ax,x,y,list(METHODS).index(method),METHODS[method]);ys+=y
                    series.append(dict(figure=name,dataset=dataset,method=method,x=x,y=y))
                chart.regular_axis(ax,"x",[0,50,100,150,200],0);ax.set_xlabel("FL round")
                values.append(([0.]+ys) if metric=="kl" else ys)
            finish_axes(axes,values,metric)
            figures.append(plots.save(fig,axes,name))
    write(DEST/"data/plot_series.json",series)
    return figures


def archive():
    files=[OUT/"cases/baseline"/p for p in ["settings.json","mission_spec.json","deployment.json","preflight.json"]]
    for ds in DATASETS:
        dd=OUT/"cases/baseline"/(ds.lower()+"_seed0")
        files += [dd/"plan.json",dd/"requests.json"]
        for method in VARIANTS:
            files += [p for p in (dd/method).rglob("*") if p.is_file() and not p.is_symlink()]
    entries={str(p.relative_to(ROOT)):dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in sorted(set(files))}
    path=OUT/"raw_new_arms.tar.gz"
    with tarfile.open(path,"w:gz",compresslevel=6) as tf:
        for name in entries:tf.add(ROOT/name,arcname=name,recursive=False)
    write(OUT/"raw_archive_manifest.json",dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,entries=entries))


def report_html(rows,figs):
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FU optimization pilot</title>',
        '<style>body{font:16px system-ui;background:#f4f6f9;color:#172438;margin:0}main{max-width:1300px;margin:auto;padding:28px}p{line-height:1.7}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #dce1e8;text-align:right}th:first-child,td:first-child{text-align:left}a{color:#315c96}article{background:white;padding:18px;margin:24px 0}img{max-width:100%}.scroll{overflow:auto}</style><main>',
        '<h1>FU optimization pilot</h1><p>CapERA · AirScape | 1 MHz · seed 0 · 200 FL rounds · 45 requests × 5 FU steps. '
        '<a href="comparison.csv">Full CSV</a> · <a href="../results/fu_joint_optimization_v1_20261027/protocol.json">Frozen protocol</a></p>',
        '<p>후속 성분 분리 비교: <a href="component_fusion/index.html">순수 망각에 대한 엄격한 무선 보호</a> · '
        '<a href="component_existing_budget/index.html">기존 무선 제약을 유지한 방향 보정</a></p>',
        '<p><b>이번 단일 seed 탐색에서 두 데이터셋의 mR 유지와 KL 감소를 함께 달성한 후보는 없었다.</b> '
        '가중치 조정은 KL과 mR 사이의 교환관계를 보였고, 수신 오차만 제거한 대조군의 KL은 거의 변하지 않았다.</p>',
        '<p>AW: FU 내부에서 3FM과 RM+3CM의 gradient 크기를 맞추는 적응형 가중치. RL: 모든 요청의 FU 학습률을 기존 초기값 0.0035로 유지. '
        'DF: 요청 UAV에서 FU와 반대인 FL 성분만 제거한 뒤 합산. Exact aggregate: 실제 수신 벡터만 오차 없는 합으로 바꾼 진단 대조군.</p>',
        '<p>DF의 식: L′ = L − min(0, L·U)/‖U‖² × U, 전송 벡터는 L′+U. '
        'L/U는 이미 학습률과 표본 가중치가 적용된 업데이트다. U=0이면 FL을 그대로 유지한다. '
        '각 UAV의 FU 방향 보호이며, 다른 UAV 사이의 충돌이나 전체 KL 개선을 보장하지 않는다. '
        'FU 방향은 FM만이 아니라 해당 FU 목적함수 전체의 방향이다. PCGrad에서 착안한 단방향 투영으로, 투영 자체를 새로운 발명으로 주장하지 않는다.</p>']
    for ds in DATASETS:
        parts += [f'<h2>{ds}</h2><div class="scroll"><table><tr><th>Method</th><th>mR (%) ↑</th><th>ΔmR (pp)</th><th>Forget KL ↓</th><th>KL reduction (%)</th><th>MIA gap ↓</th><th>Energy (kJ)</th><th>Delay (s)</th></tr>']
        for r in rows:
            if r["dataset"]!=ds:continue
            if r["status"]!="completed":parts.append(f'<tr><td>{r["label"]}</td><td colspan="7">{html.escape(r["status"])}</td></tr>');continue
            parts.append(f'<tr><td>{r["label"]}</td><td>{r["retained_mR_percent"]:.4f}</td><td>{r["mR_change_pp"]:+.4f}</td><td>{r["forget_KL"]:.7f}</td><td>{r["KL_reduction_percent"]:+.2f}</td><td>{r["MIA_AUC_gap"]:.6f}</td><td>{r["energy_kJ"]:.3f}</td><td>{r["delay_s"]:.3f}</td></tr>')
        parts.append('</table></div>')
    parts += ['<p>각 후보의 설정은 실행 전에 고정했고 모든 결과를 남겼다. 기존 검증셋의 단일 seed 탐색 실험이며, '
        '독립 테스트나 통계적 유의성 검증이 아니다. mR 변화와 KL 변화는 별개로 보고한다. '
        'retrain은 평가에만 사용했고 gradient/teacher/설정 선택에 사용하지 않았다. 원래 final_chart는 변경하지 않았다.</p>',
        '<p>에너지는 FL/FU 처리의 통신+연산+호버이며, 이동·수집·공통 준비 비용은 제외한다. '
        '추가 gradient 제어 연산은 기존 연산 모델에 보수적인 MAC-equivalent로 합산했다. 실제 하드웨어 측정값이 아니다. '
        'Exact aggregate는 원래 무선 설계 비용을 유지한 가상 진단이며 구현 가능한 무잡음 전송 방법이 아니다.</p>',
        '<p>모든 곡선은 실제 저장된 FL+FU 이후 평가에서 가져왔다. KL은 같은 최종 retrain에 대한 양방향 검색 분포 차이다. '
        '초반에는 FL 진행도와 아직 도착하지 않은 요청의 영향도 포함한다. 작은 KL 또는 MIA 차이는 삭제 보증이 아니다. '
        '초기 관측값을 만들지 않아 곡선은 첫 기록인 round 25부터 시작한다.</p>']
    for f in figs:
        n=f["name"];parts.append(f'<article><h2>{n}</h2><img src="{n}.png" alt="{n}"><p><a href="{n}.pdf">PDF</a> · <a href="{n}.svg">SVG</a></p></article>')
    parts.append(references_html());parts.append('</main></html>\n');(DEST/"index.html").write_text(''.join(parts))


def main():
    study.register();v=study.verify();assert v["passed"]
    DEST.mkdir(exist_ok=True);(DEST/"data").mkdir(exist_ok=True)
    for p in [Path(__file__),Path(study.__file__),OUT/"protocol.json",OUT/"verification.json"]:
        SOURCES[str(p.relative_to(ROOT))]=sha(p)
    rows,points=gather();write(DEST/"data/comparison.json",rows);csv_file(DEST/"comparison.csv",rows)
    figs=figures(rows,points);report_html(rows,figs)
    write(DEST/"data/manifest.json",dict(methods=METHODS,figures=figs,source_sha256=SOURCES,seed=0,
        description="All predeclared outcomes; no rank-based curve selection; original final_chart preserved."))
    for p,digest in SOURCES.items():assert sha(ROOT/p)==digest,p
    for f in figs:
        for r in f["files"]:assert sha(DEST/r["file"])==r["sha256"]
    archive()
    result=dict(passed=True,outcomes=v["outcomes"],figures=len(figs),compared_rows=len(rows),
        simultaneous_mR_KL_improvements=[dict(dataset=r["dataset"],method=r["method"],mR_change_pp=r["mR_change_pp"],KL_reduction_percent=r["KL_reduction_percent"]) for r in rows if r.get("mR_not_lower_and_KL_lower") and not r["diagnostic_only"]],
        interpretation="Pilot only. Simultaneous mR/KL improvement does not imply improved MIA, worst-request KL, or cost. See all columns and no-noise diagnostic.")
    write(OUT/"publication.json",result);print(result,flush=True)


if __name__=="__main__":main()
