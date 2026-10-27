"""Report the follow-up component-aware fusion without selecting successful arms."""
import html
import sys
from pathlib import Path

EXISTING="--existing-budget" in sys.argv
if EXISTING:
    import fu_component_existing_budget as study
else:
    import fu_component_fusion as study
ROOT,OUT,VARIANTS,DATASETS,PROPOSED,read,write,sha=(getattr(study,k) for k in ["ROOT","OUT","VARIANTS","DATASETS","PROPOSED","read","write","sha"])
# Stage2 installs its explicit runtime configuration before importing report helpers.
import fu_joint_report as report

DEST=ROOT/"fu_joint_pilot"/("component_existing_budget" if EXISTING else "component_fusion")
report.DEST=DEST


def plot(points):
    plots=report.plots;chart=plots.chart
    chart.configure();plots.DEST=DEST;plots.METHODS=report.METHODS
    series=[];figs=[]
    for name,key,label in [("01_component_mr_1x2","mr","Retained mR (%)"),("02_component_kl_1x2","kl","Forget-set KL")]:
        fig,axes=chart.style.canvas(label);values=[];status_markers=[]
        for ax,ds in zip(axes,DATASETS):
            ys=[]
            for i,(method,title) in enumerate(report.METHODS.items()):
                if (ds,method) not in points:
                    dd=report.directory(ds,method);outcome=report.load(dd/method/"outcome.json")
                    m=report.load(dd/method/"manifest.json");at=m["completed_FL_rounds"]
                    level=1.025+.045*(i-1)
                    ax.plot([at],[level],transform=ax.get_xaxis_transform(),marker="x",ms=7,mew=1.6,
                            color=chart.COLORS[i],linestyle="none",clip_on=False)
                    status_markers.append(dict(dataset=ds,method=method,x=at,status=outcome["status"],axes_y_fraction=level,metric_y=None))
                    series.append(dict(figure=name,dataset=ds,method=method,x=[],y=[],status=outcome["status"],failed_after_FL_round=at))
                    continue
                rr=points[(ds,method)];x=[r["round"] for r in rr];y=[r[key] for r in rr]
                chart.style.curve(ax,x,y,i,title);ys+=y
                series.append(dict(figure=name,dataset=ds,method=method,x=x,y=y))
            chart.regular_axis(ax,"x",[0,50,100,150,200],0);ax.set_xlabel("FL round")
            values.append([0.]+ys if key=="kl" else ys)
        report.finish_axes(axes,values,key)
        if status_markers:
            # Reserve a separate band for stopped runs below the status legend.
            for ax in axes:
                pos=ax.get_position()
                ax.set_position([pos.x0,pos.y0,pos.width,pos.height-.04])
        figs.append(plots.save(fig,axes,name,status_markers))
    write(DEST/"data/plot_series.json",series);return figs


def main():
    v=(study.component.verify() if EXISTING else study.verify());assert v["passed"]
    DEST.mkdir(parents=True,exist_ok=True);(DEST/"data").mkdir(exist_ok=True)
    for p in [Path(__file__),Path(study.__file__),Path(report.__file__),OUT/"protocol.json",OUT/"verification.json"]:
        report.SOURCES[str(p.relative_to(ROOT))]=sha(p)
    rows,points=report.gather()
    for row in rows:
        if row["method"]==PROPOSED or row["status"]!="completed":continue
        arm=ROOT/row["source"]
        work=[report.load(p) for p in sorted((arm/"work_attempts").glob("*.json"))]
        local=[r for w in work for r in w["local"]]
        row["pure_forget_projection_fraction"]=sum(r["component_projected"] for r in local)/len(local)
        row["negative_local_dot_after_count"]=sum(r["retention_forget_dot_after"]<-1e-12 for r in local)
        first=[w["local"][0] for w in work if w["local"]]
        row["global_negative_dot_before_count"]=sum(r["global_retention_forget_dot_before"]<0 for r in first)
        row["global_negative_dot_after_count"]=sum(r["global_retention_forget_dot_after"]<0 for r in first)
        row["global_joint_transactions"]=len(first)
    write(DEST/"data/comparison.json",rows);report.csv_file(DEST/"comparison.csv",rows)
    figs=plot(points)
    guard_note=('무선 입력은 L_eff=R′−P와 U=F+P로 표현해 합계 R′+F를 유지한다. 기존 FU 상대 오차 제약은 복합 U를 대상으로 유지하며, 순수 F에 대한 무선 보호 보장은 하지 않는다.' if EXISTING else 'FU 상대 집계 오차 제약도 기존 복합 FU가 아닌 순수 F에 적용한다. 따라서 학습 방향 처리와 무선 오차 보호의 대상을 일치시키는 후보이다.')
    pieces=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Component-aware FL/FU fusion</title>',
        '<style>body{font:16px system-ui;color:#172438;background:#f4f6f9;margin:0}main{max-width:1250px;margin:auto;padding:28px}p{line-height:1.7}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce1e8;text-align:right}td:first-child,th:first-child{text-align:left}article{background:white;padding:20px;margin:22px 0}img{max-width:100%}a{color:#315c96}.scroll{overflow:auto}</style><main>',
        '<h1>순수 망각 성분을 보호하는 FL/FU 결합</h1><p><a href="../index.html">첫 단계 12개 실험</a> · <a href="comparison.csv">전체 CSV</a> · '
        f'<a href="../../results/{OUT.name}/protocol.json">후속 실험 사전 고정 조건</a></p>',
        '<p>기존 FU 전체에는 유지 학습도 포함돼 있다. 후속 CP는 순수 망각 F와 유지 R을 분리한다. '
        'F는 3FM의 업데이트, R은 FL과 FU의 RM+3CM 업데이트를 합친 것이다. 모두 원래 표본 가중치와 학습률을 포함한다.</p>',
        '<p><b>R′ = R − min(0, R·F)/‖F‖² × F, &nbsp; 전송 = R′ + F.</b> '
        'F가 없으면 기존 FL을 그대로 보낸다. '+guard_note+' 추가 전송 좌표나 encoder 역전파는 없다.</p>',
        '<p>기본 후보는 기존 FU 학습률, +RL 후보는 모든 요청의 FU 학습률을 초기값 0.0035로 유지한다. '
        '둘 다 동일한 200 FL 라운드, 45개 요청×5 FU 단계, seed0, 1MHz를 사용했다. '
        '각 UAV의 방향 보호이며, 서로 다른 UAV의 간섭이나 전체 KL/mR 개선을 보장하지 않는다. '
        '단방향 PCGrad 응용이며 투영 자체를 새로운 발명으로 주장하지 않는다.</p>']
    if EXISTING:
        pieces.append('<p><b>이번 결과에서는 mR 유지와 KL 감소를 동시에 달성하지 못했다.</b> '
            'CP-E는 CapERA에서 KL을 소폭 낮췄지만 mR이 감소했고, AirScape에서는 KL과 mR 모두 악화됐다. '
            '학습률을 높인 CP-E+RL은 두 데이터셋의 KL을 높였다. 원래 제안법을 대체하는 검증된 개선안으로 채택하지 않는다.</p>')
    else:
        pieces.append('<p><b>네 실행 모두 기존 송신 에너지 상한 안에서 유효한 무선 설계를 찾지 못해 중단됐다.</b> '
            '200 라운드를 완료한 성능 수치로 해석하면 안 된다. 그래프 위 ×는 중단 라운드를 표시하며 y축 성능값이 아니다. '
            '동일 라운드의 중단 표시는 세로로 분리했고, 부분 기록과 실패 원인을 모두 보존했다.</p>')
    for ds in DATASETS:
        pieces.append(f'<h2>{ds}</h2><div class="scroll"><table><tr><th>Method</th><th>mR (%) ↑</th><th>ΔmR (pp)</th><th>Forget KL ↓</th><th>KL reduction (%)</th><th>MIA gap ↓</th><th>Energy (kJ)</th><th>Delay (s)</th><th>Projected (%)</th></tr>')
        for row in rows:
            if row["dataset"]!=ds:continue
            if row["status"]!="completed":pieces.append(f'<tr><td>{row["label"]}</td><td colspan="8">{html.escape(row["status"])}</td></tr>');continue
            pieces.append(f'<tr><td>{row["label"]}</td><td>{row["retained_mR_percent"]:.4f}</td><td>{row["mR_change_pp"]:+.4f}</td><td>{row["forget_KL"]:.7f}</td><td>{row["KL_reduction_percent"]:+.2f}</td><td>{row["MIA_AUC_gap"]:.6f}</td><td>{row["energy_kJ"]:.3f}</td><td>{row["delay_s"]:.3f}</td><td>{100*row.get("pure_forget_projection_fraction",0.):.1f}</td></tr>')
        pieces.append('</table></div>')
    pieces += ['<p>앞선 탐색 결과를 바탕으로 등록한 후속 가설 실험이다. 같은 검증셋을 사용했으므로 독립 확인 실험이나 테스트 성능 주장이 아니다. '
        '결과에 따라 설정을 다시 고르거나 원래 final_chart를 교체하지 않았다. retrain은 평가에만 사용했다. '
        '에너지는 통신+연산+호버로, 추가 성분 분리와 투영 연산 비용을 모델에 포함했다. 비용은 하드웨어 실측이 아니다. '
        '원시 기록은 별도 압축파일과 해시 목록으로 보존했다.</p>']
    for f in figs:
        n=f["name"];pieces.append(f'<article><img src="{n}.png" alt="{n}"><p><a href="{n}.pdf">PDF</a> · <a href="{n}.svg">SVG</a></p></article>')
    pieces.append(report.references_html());pieces.append('</main></html>\n');(DEST/"index.html").write_text(''.join(pieces))
    write(DEST/"data/manifest.json",dict(figures=figs,methods=report.METHODS,source_sha256=report.SOURCES,stage=3 if EXISTING else 2,seed=0))
    for name,digest in report.SOURCES.items():assert sha(ROOT/name)==digest,name
    for f in figs:
        for item in f["files"]:assert sha(DEST/item["file"])==item["sha256"]
    report.archive()
    result=dict(passed=True,outcomes=v["outcomes"],figures=len(figs),rows=len(rows),
        simultaneous_mR_KL_improvements=[dict(dataset=r["dataset"],method=r["method"],mR_change_pp=r["mR_change_pp"],KL_reduction_percent=r["KL_reduction_percent"]) for r in rows if r.get("mR_not_lower_and_KL_lower")],
        caveat="Exploratory follow-up; compare all quality, cost, and global-conflict diagnostics. No winner selection.")
    write(OUT/"publication.json",result);print(result,flush=True)


if __name__=="__main__":main()
