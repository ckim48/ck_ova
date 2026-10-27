"""Publish every retrieval-space development run and unchanged comparison arm."""
import html
import math
from pathlib import Path
import tarfile
import hashlib
import sys
from collections import Counter

REFINE = "--refine" in sys.argv
if REFINE:
    import fu_retrieval_refinement as study
else:
    import fu_retrieval_alignment as study
import fu_joint_report as report

ROOT, OUT, SOURCE, PROPOSED, DATASETS, VARIANTS = (getattr(study, k) for k in
    ["ROOT", "OUT", "SOURCE", "PROPOSED", "DATASETS", "VARIANTS"])
read, write, sha = study.read, study.write, study.sha
DEST = ROOT / ("fu_retrieval_refinement_pilot" if REFINE else "fu_retrieval_pilot")
report.DEST = DEST
PEERS = {"joint_fm_cm_no_rm": "FM-CM", "separate_aircomp_periodic": "S-AC",
         "aircomp_ofdma_periodic": "AC-OF", "sequential_digital_proxy": "S-OF"}
CONTROLS = {"heuristic_ova": "H-OVA", "sequential_digital_periodic": "S-OF+", "unguarded_ova": "NG-OVA"}


def baseline_rows():
    rows = []
    for dataset in DATASETS:
        dd = SOURCE / "cases/baseline" / (dataset.lower()+"_seed0")
        params = report.load(dd.parent/"settings.json")["energy_model"]
        prefix = report.plots.prefix_energy(dd, params, report.SOURCES)
        for method, label in dict(PEERS, **CONTROLS).items():
            arm = dd/method; outcome = report.load(arm/"outcome.json")
            row = dict(dataset=dataset, method=method, label=label, status=outcome["status"],
                       role="main_baseline" if method in PEERS else "matched_ablation",
                       source=str(arm.relative_to(ROOT)))
            if outcome["status"] == "completed":
                evaluation = report.load(arm/"manifest.json")["evaluation"]
                mia = report.load(arm/"mia.json")
                ledger = report.plots.service_ledger(dd, method, params, prefix, report.SOURCES)
                row.update(retained_mR_percent=100*evaluation["retained"]["mR"], retained_loss=evaluation["retained"]["loss"],
                           forget_KL=evaluation["forget"]["KL_retrain_to_candidate"], worst_request_KL=evaluation["worst_request_KL"],
                           MIA_AUC_gap=mia["auc_gap_vs_retrain"], energy_kJ=ledger["total_energy_j"]/1000, delay_s=ledger["delay_s"])
            rows.append(row)
    return rows


def compare(rows):
    for row in rows:
        if row["method"] not in VARIANTS or row["status"] != "completed":
            continue
        peers = [r for r in rows if r["dataset"] == row["dataset"] and r["method"] in PEERS and r["status"] == "completed"]
        all_old = [r for r in rows if r["dataset"] == row["dataset"] and r["method"] not in VARIANTS and r["status"] == "completed"]
        for title, group in [("main_baselines", peers), ("all_existing_controls", all_old)]:
            for key in ["retained_mR_percent", "forget_KL", "worst_request_KL", "MIA_AUC_gap", "energy_kJ", "delay_s"]:
                better = (row[key] >= max(r[key] for r in group)-1e-8) if key == "retained_mR_percent" else (row[key] <= min(r[key] for r in group)+1e-12)
                row[f"not_worse_{key}_vs_{title}"] = better
            row[f"all_six_metrics_not_worse_vs_{title}"] = all(row[f"not_worse_{key}_vs_{title}"] for key in
                ["retained_mR_percent", "forget_KL", "worst_request_KL", "MIA_AUC_gap", "energy_kJ", "delay_s"])


def figures(points):
    plots = report.plots; chart = plots.chart
    chart.configure(); plots.DEST = DEST
    methods = list(report.METHODS)
    colors = ["#4C72B0", "#8C6D31", "#60AE62", "#2993A8", "#E15759", "#8D6BB8", "#F4A64B", "#C16D9D"]
    markers = ["o", "d", "v", "P", "^", "X", "s", "h"]
    styles = ["-", "--", "-.", ":", (0,(5,2)), (0,(2,2)), (0,(4,1,1,1)), (0,(6,2))]
    groups = [("objective", [PROPOSED, "fu_retrieval_cm", "fu_proxy_embed_rm1", "fu_proxy_rank_rm1"]),
              ("proxy_embed", [PROPOSED, "fu_proxy_embed_rm0", "fu_proxy_embed_rm01", "fu_proxy_embed_rm1"]),
              ("proxy_rank", [PROPOSED, "fu_proxy_rank_rm0", "fu_proxy_rank_rm01", "fu_proxy_rank_rm1"])]
    if REFINE:
        groups = [("proxy_embed", [PROPOSED, "fu_proxy_embed_rm025", "fu_proxy_embed_rm05"]),
                  ("proxy_rank", [PROPOSED, "fu_proxy_rank_rm025", "fu_proxy_rank_rm05"])]
    result, series = [], []
    for group, selected in groups:
        plots.METHODS = {m: report.METHODS[m] for m in selected}
        for name, source in [("COLORS", colors), ("MARKERS", markers), ("STYLES", styles)]:
            values = [source[methods.index(m)] for m in selected]
            setattr(chart, name, values); setattr(chart.style, name, values)
        for metric, ylabel in [("mr", "Retained mR (%)"), ("kl", "Forget-set KL")]:
            name = f"{len(result)+1:02d}_{group}_{metric}_1x2"
            fig, axes = chart.style.canvas(ylabel)
            values, stopped = [], []
            for ax, dataset in zip(axes, DATASETS):
                ys = []
                for i, method in enumerate(selected):
                    if (dataset, method) not in points:
                        arm = OUT/"cases/baseline"/(dataset.lower()+"_seed0")/method
                        at = report.load(arm/"manifest.json")["completed_FL_rounds"]
                        outcome = report.load(arm/"outcome.json")
                        level = 1.025+.04*(i-1)
                        ax.plot([at], [level], marker="x", ms=6, mew=1.5, linestyle="none", color=chart.COLORS[i],
                                transform=ax.get_xaxis_transform(), clip_on=False)
                        stopped.append(dict(dataset=dataset, method=method, x=at, axes_y_fraction=level, metric_y=None, status=outcome["status"]))
                        series.append(dict(figure=name, dataset=dataset, method=method, x=[], y=[], status=outcome["status"]))
                        continue
                    rows = points[(dataset, method)]
                    x, y = [r["round"] for r in rows], [r[metric] for r in rows]
                    chart.style.curve(ax, x, y, i, report.METHODS[method]); ys += y
                    series.append(dict(figure=name, dataset=dataset, method=method, x=x, y=y))
                chart.regular_axis(ax, "x", [0,50,100,150,200], 0); ax.set_xlabel("FL round")
                values.append([0.]+ys if metric == "kl" else ys)
            report.finish_axes(axes, values, metric)
            if stopped:
                for ax in axes:
                    pos = ax.get_position(); ax.set_position([pos.x0,pos.y0,pos.width,pos.height-.06])
            result.append(plots.save(fig, axes, name, stopped))
    write(DEST/"data/plot_series.json", series)
    return result


def gallery(rows, figs):
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>Retrieval-space FU pilot</title><style>body{font:16px system-ui;color:#172438;background:#f4f6f9;margin:0}main{max-width:1350px;margin:auto;padding:26px}p{line-height:1.7}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}article{background:white;padding:18px;margin:22px 0}img{max-width:100%}a{color:#315c96}</style><main>',
             '<h1>검색 분포를 직접 다루는 FU 후보</h1><p>CapERA · AirScape | seed 0 · 1 MHz · 200 FL rounds · 45×5 FU steps. '
             '<a href="comparison.csv">전체 CSV</a> · <a href="../results/'+OUT.name+'/protocol.json">실행 전에 고정한 조건</a> · '
             '<a href="data/verification.json">검증</a></p>',
             '<p>목표함수는 3×삭제 항 + λ<sub>R</sub>×유지 CE + 3×보존 항이다. R-KL은 기존 양성쌍 cosine 삭제를 유지하고 보존 항을 양방향 검색 KL로 바꾼다. '
             'P-E는 삭제 항을 기존 clean proxy20의 검색 KL로 바꾸고 임베딩 보존을 유지한다. P-R은 두 항 모두 검색 KL을 사용한다. '
             '접미사 0/01/025/05/1은 λ<sub>R</sub>=0/0.1/0.25/0.5/1을 뜻한다. 모든 후보를 표시하며 기존 방법은 수정하지 않았다.</p>',
             '<p><a href="../fu_retrieval_pilot/index.html">첫 단계 전체 후보</a> · '
             '<a href="../fu_retrieval_refinement_pilot/index.html">가중치 0.25/0.5 후속 탐색</a></p>',
             '<p><b>이 결과는 기존 검증셋에서의 방법 개발 실험이다.</b> 후보 선택 후 새로운 paired seed와 독립 데이터 분할에서 확인하기 전에는 최종 우월성으로 주장할 수 없다. '
             '전체 기존 방법 및 강화된 대조군과의 비교 판정을 CSV에 함께 기록했다. 기존 Proposed보다 낮은 mR도 숨기지 않았다.</p>',
             '<p>학습 목적함수까지 바꾼 전체 시스템 비교이며, 전송 방식만의 인과적 효과를 입증하는 비교가 아니다. '
             '새 목적함수를 동일하게 적용한 통신 대조군과 동등한 튜닝 예산의 비교가 추가로 필요하다.</p>',
             '<p>삭제 teacher는 기존 비교 방법에도 제공된 retain-only proxy20이며 평가용 reference200을 사용하지 않았다. '
             '이 proxy는 임무 전 삭제 대상 집합을 알고 준비했다는 기존 오프라인 가정을 이어받는다. 알려지지 않은 미래 요청에 대한 인과적 해법으로 주장하지 않는다. '
             '추가 teacher 순전파와 검색 KL 행렬 연산 비용을 포함했고, 동일한 hard 송신 상한을 유지했다.</p>']
    for dataset in DATASETS:
        parts.append(f'<h2>{dataset}</h2><div class="scroll"><table><tr><th>Method</th><th>Role</th><th>mR (%) ↑</th><th>KL ↓</th><th>Worst KL ↓</th><th>MIA gap ↓</th><th>Energy (kJ)</th><th>Delay (s)</th></tr>')
        for row in rows:
            if row["dataset"] != dataset: continue
            if row["status"] != "completed":
                parts.append(f'<tr><td>{row["label"]}</td><td colspan="7">{html.escape(row["status"])}</td></tr>'); continue
            parts.append(f'<tr><td>{row["label"]}</td><td>{row["role"]}</td><td>{row["retained_mR_percent"]:.4f}</td><td>{row["forget_KL"]:.7f}</td><td>{row["worst_request_KL"]:.7f}</td><td>{row["MIA_AUC_gap"]:.6f}</td><td>{row["energy_kJ"]:.3f}</td><td>{row["delay_s"]:.3f}</td></tr>')
        parts.append('</table></div>')
    parts.append('<p>에너지는 FL/FU 처리의 통신+연산+호버이며 하드웨어 실측이 아니다. 모든 곡선은 저장된 평가값을 연결하며 처음 관측 전 값을 만들지 않았다. '
                 'KL은 동일한 최종 reference200과 비교하므로 초반 값에는 학습 진행도 차이도 포함된다. 작은 KL이나 MIA gap은 삭제 보증이 아니다.</p>')
    for fig in figs:
        name = fig["name"]
        parts.append(f'<article><img src="{name}.png" alt="{name}"><p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p></article>')
    parts.append('<p>참고: <a href="https://aclanthology.org/2025.acl-long.1469/">CLIPErase (ACL 2025)</a> · '
                 '<a href="https://arxiv.org/abs/2307.12732">CLIP-KD (CVPR 2024)</a> · '
                 '<a href="https://arxiv.org/abs/2605.10680">Behavioral Guarantees for Proxy-Based Unlearning (2026 preprint)</a>. '
                 '본 실험의 조합은 탐색적 응용이며 해당 논문의 원래 실험이나 이론적 보장을 재현했다는 주장이 아니다.</p></main></html>\n')
    (DEST/"index.html").write_text(''.join(parts))


def main():
    verified = study.verify(); assert verified["passed"]
    (DEST/"data").mkdir(parents=True, exist_ok=True)
    for path in [Path(__file__), Path(study.__file__), Path(report.__file__), OUT/"protocol.json", OUT/"verification.json"]:
        report.SOURCES[str(path.relative_to(ROOT))] = sha(path)
    rows, points = report.gather()
    for row in rows:
        row["role"] = "original_proposed" if row["method"] == PROPOSED else "development_candidate"
    rows += baseline_rows(); compare(rows)
    for row in rows:
        if row["status"] == "completed":
            forgetting = report.load(ROOT/row["source"]/"manifest.json")["evaluation"]["forget"]
            row["forget_retrieval_mR_percent"] = 100*forgetting["mR"]
            row["forget_retrieval_loss"] = forgetting["loss"]
    report.csv_file(DEST/"comparison.csv", rows); write(DEST/"data/comparison.json", rows)
    figs = figures(points); gallery(rows, figs)
    write(DEST/"data/manifest.json", dict(figures=figs, source_sha256=report.SOURCES, methods=report.METHODS, seed=0))
    report.archive()
    archive = read(OUT/"raw_archive_manifest.json")
    assert sha(OUT/archive["file"]) == archive["sha256"]
    with tarfile.open(OUT/archive["file"], "r:gz") as stream:
        members = stream.getmembers()
        assert {m.name for m in members} == set(archive["entries"]) and len(members) == len(archive["entries"])
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
            with stream.extractfile(member) as data:
                assert hashlib.file_digest(data, "sha256").hexdigest() == archive["entries"][member.name]["sha256"]
    for name, digest in report.SOURCES.items(): assert sha(ROOT/name) == digest, name
    for fig in figs:
        for export in fig["files"]: assert sha(DEST/export["file"]) == export["sha256"]
    result = dict(passed=True, outcomes=verified["outcomes"], archived_files=len(members), figures=len(figs), compared_rows=len(rows),
        all_six_metrics_not_worse_than_all_existing=[dict(dataset=r["dataset"], method=r["method"]) for r in rows if r.get("all_six_metrics_not_worse_vs_all_existing_controls")],
        caveat="Development-set comparison only; all six metrics include mean/worst KL, mR, MIA gap, energy and delay. No independent confirmation or deletion guarantee.")
    write(DEST/"data/verification.json", result); write(OUT/"publication.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
