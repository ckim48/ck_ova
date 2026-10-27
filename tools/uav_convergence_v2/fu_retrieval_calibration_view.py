"""Show all weight sensitivities; distinguish observed tuning from a future controller."""
from pathlib import Path
import json
import math

import fu_joint_report as report

ROOT = report.ROOT
DEST = ROOT/"fu_dataset_calibration"
read, write, sha = report.read, report.write, report.sha


def main():
    (DEST/"data").mkdir(parents=True, exist_ok=True)
    inputs = [ROOT/"fu_retrieval_pilot/data/comparison.json", ROOT/"fu_retrieval_refinement_pilot/data/comparison.json"]
    combined = {}
    for path in inputs:
        for row in read(path):
            key = (row["dataset"], row["method"])
            if key in combined:
                assert all(combined[key].get(k) == row.get(k) for k in ["status", "retained_mR_percent", "forget_KL", "energy_kJ", "delay_s"])
            else:
                combined[key] = row
    rows = list(combined.values())
    report.csv_file(DEST/"all_outcomes.csv", rows)
    weights = [("0", 0.), ("01", .1), ("025", .25), ("05", .5), ("1", 1.)]
    plots = report.plots; chart = plots.chart
    chart.configure(); plots.DEST = DEST; plots.METHODS = {"embed": "P-E", "rank": "P-R"}
    for key, values in [("COLORS", ["#4C72B0", "#E15759"]), ("MARKERS", ["o", "^"]), ("STYLES", ["-", "--"])]:
        setattr(chart, key, values); setattr(chart.style, key, values)
    series, figs = [], []
    for metric, ylabel in [("retained_mR_percent", "Retained mR (%)"), ("forget_KL", "Forget-set KL")]:
        name = "01_weight_mr_1x2" if metric == "retained_mR_percent" else "02_weight_kl_1x2"
        fig, axes = chart.style.canvas(ylabel); all_values, stopped = [], []
        for ax, dataset in zip(axes, report.DATASETS):
            ys = []
            for index, kind in enumerate(plots.METHODS):
                x, y = [], []
                for tag, weight in weights:
                    row = combined[(dataset, f"fu_proxy_{kind}_rm{tag}")]
                    if row["status"] == "completed":
                        x.append(weight); y.append(row[metric])
                    else:
                        level = 1.025+.045*index
                        ax.plot([weight], [level], marker="x", ms=6, mew=1.5, linestyle="none", color=chart.COLORS[index],
                                transform=ax.get_xaxis_transform(), clip_on=False)
                        stopped.append(dict(dataset=dataset, method=kind, x=weight, metric_y=None, axes_y_fraction=level, status=row["status"]))
                chart.style.curve(ax, x, y, index, plots.METHODS[kind]); ys += y
                series.append(dict(figure=name, dataset=dataset, method=kind, x=x, y=y))
            chart.regular_axis(ax, "x", [0., .25, .5, .75, 1.], 2)
            ax.set_xlabel(r"Retain weight $\lambda_R$")
            all_values.append([0.]+ys if metric == "forget_KL" else ys)
        report.finish_axes(axes, all_values, "kl" if metric == "forget_KL" else "mr")
        for ax in axes:
            pos = ax.get_position(); ax.set_position([pos.x0,pos.y0,pos.width,pos.height-.04])
        figs.append(plots.save(fig, axes, name, stopped))
    write(DEST/"data/plot_series.json", series)
    proposal = dict(status="design_only_not_implemented_or_validated", same_rule_for_both_datasets=True,
        loss="3 KL(clean_proxy || current) on forget pairs + lambda_R * CE on retained pairs + 3 KL(request_teacher || current) on retained retrieval distributions",
        feedback="Relative retained-loss increase on a training-only calibration minibatch against the request-start teacher, normalized by that teacher loss; smooth within the request to reduce minibatch variance.",
        prospective_dual_rule="lambda_next = projection_onto_feasible_Lambda(lambda + rho * (relative_retained_loss_increase - delta))",
        feasibility="Lambda contains weights for which the actual head/resource solver satisfies the unchanged TX cap, power, error and deadline constraints. Empty Lambda must remain an explicit infeasible outcome; a scheduler fallback or early stopping would be a separate method with its own accounting and quality validation.",
        calibration="Choose rho/delta and initialization using separate training-only calibration episodes, with equal tuning budgets for comparison methods. Freeze them before an independent evaluation. Current validation mR/KL and reference200 must not enter the online controller.",
        verification_needed="New code, cost accounting, paired seeds, independent split and matched-objective radio controls are still required. Weight optima below are descriptive development results, not parameters learned by this prospective controller.")
    write(DEST/"data/adaptive_controller_proposal.json", proposal)
    examples = [combined[("CapERA", "fu_proxy_rank_rm025")], combined[("AirScape", "fu_proxy_rank_rm05")]]
    pieces = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dataset-dependent FU calibration</title>'
              '<style>body{font:16px system-ui;background:#f4f6f9;color:#172438;margin:0}main{max-width:1250px;margin:auto;padding:28px}p{line-height:1.7}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}article{background:white;padding:20px;margin:24px 0}img{max-width:100%}a{color:#315c96}</style><main>',
              '<h1>데이터 특성에 따른 FU 가중치</h1><p><a href="all_outcomes.csv">22개 신규 실행과 기존 대조군 전체 CSV</a> · '
              '<a href="../fu_retrieval_pilot/index.html">첫 단계</a> · <a href="../fu_retrieval_refinement_pilot/index.html">중간 가중치 후속 실험</a></p>',
              '<p>동일한 proxy 삭제 + 검색 분포 보존(P-R) 목적함수에서, 관측한 검증 KL이 가장 작은 실행 가능 가중치는 CapERA 0.25, AirScape 0.5였다. '
              '<b>이는 기존 검증셋을 확인한 뒤의 탐색적 해석이며, 자동 적응 알고리즘의 검증 결과가 아니다.</b> '
              '전체 결과와 실패를 아래 민감도 차트 및 CSV에 남겼다. 기존 Proposed보다 mR이 내려간 점도 함께 비교해야 한다.</p>',
              '<table><tr><th>Dataset</th><th>λ_R</th><th>mR (%)</th><th>Forget KL</th><th>Energy (kJ)</th><th>Delay (s)</th></tr>']
    for row, weight in zip(examples, [.25, .5]):
        pieces.append(f'<tr><td>{row["dataset"]}</td><td>{weight}</td><td>{row["retained_mR_percent"]:.4f}</td><td>{row["forget_KL"]:.7f}</td><td>{row["energy_kJ"]:.3f}</td><td>{row["delay_s"]:.3f}</td></tr>')
    pieces += ['</table><p>두 데이터셋에 같은 규칙을 적용하는 향후 후보는, 보존 학습 손실이 요청 시작 teacher보다 허용 범위 이상 커지면 λ_R을 높이고, '
               '보존 여유가 생기면 낮추는 제어다. 무선 제약을 만족하는 가중치 집합 안에서 조정해야 한다. 데이터셋 이름이나 평가용 retrain 결과를 온라인 입력으로 사용하지 않는다.</p>',
               '<p><b>이 자동 제어기는 아직 설계 단계다.</b> <a href="data/adaptive_controller_proposal.json">구체적인 규칙과 검증 조건</a>에 구현 여부를 구분했다. '
               '별도 학습용 calibration 에피소드로 설정을 정하고, 같은 튜닝 예산을 준 대조군 및 새로운 seed·독립 분할에서 검증해야 한다.</p>']
    for fig in figs:
        name = fig["name"]; pieces.append(f'<article><img src="{name}.png" alt="{name}"><p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p></article>')
    pieces += ['<p>×는 전송 설계를 찾지 못한 가중치이며 성능 y값이 아니다. 선은 실제 완료한 가중치만 연결했다. '
               '전역 재학습 일치, 통계적 우월성 또는 삭제 보증을 주장하지 않는다.</p>',
               '<p>제약 기반 보존 제어 참고: <a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/06fc38f5c21ae66ef955e28b7a78ece5-Abstract-Conference.html">EUPMU (NeurIPS 2025)</a>. '
               '향후 무선 실행 가능 집합과의 결합은 이 페이지의 설계 제안이며 원 논문의 보장을 그대로 적용한 것이 아니다.</p></main></html>\n']
    (DEST/"index.html").write_text(''.join(pieces))
    sources = {str(p.relative_to(ROOT)): sha(p) for p in inputs+[Path(__file__),Path(report.__file__)]}
    write(DEST/"data/manifest.json", dict(figures=figs, source_sha256=sources, combined_outcomes=len(rows), proposal_implemented=False))
    for fig in figs:
        for export in fig["files"]: assert sha(DEST/export["file"]) == export["sha256"]
    print(dict(passed=True, combined_outcomes=len(rows), figures=len(figs), proposal_implemented=False))


if __name__ == "__main__": main()
