import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    wireless = read_rows(ROOT / "results/wireless/metrics.csv")
    learning = read_rows(ROOT / "results/learning/final_metrics.csv")
    lines = [
        "# ova_CK 파일럿 결과",
        "",
        "합성 데이터/업데이트 검증 결과. CLIP 및 비교 논문 원본 재현 결과가 아니다.",
        "",
        "## 통신",
        "",
        "3 seeds × 2 tolerance × 2 SIC residual × 4 방법 = 48 records.",
        "에너지는 보수적 communication bound이며 계산/비행 에너지를 포함하지 않는다.",
        "방법별 feasible case 구성이 달라 평균 에너지만으로 순위를 매길 수 없다.",
        "",
        "| 방법 | feasible / 전체 | feasible 평균 energy (mJ) | uplink (ms) |",
        "|---|---:|---:|---:|",
    ]
    for method in ("hybrid", "fixed_head", "orthogonal", "mixed_aircomp"):
        records = [row for row in wireless if row["method"] == method]
        feasible = [row for row in records if row["status"] == "feasible"]
        energy = np.mean([float(row["energy_bound_j"]) for row in feasible]) * 1000
        duration = np.mean([float(row["uplink_s"]) for row in feasible]) * 1000
        lines.append(f"| {method} | {len(feasible)}/{len(records)} | {energy:.4f} | {duration:.3f} |")
    lines += [
        "",
        "### 동일 case의 paired 비교",
        "",
    ]
    key = lambda row: (row["seed"], row["relative_tolerance"], row["chi"])
    hybrid = {key(row): row for row in wireless if row["method"] == "hybrid" and row["status"] == "feasible"}
    for comparator in ("fixed_head", "orthogonal", "mixed_aircomp"):
        baseline = {key(row): row for row in wireless
                    if row["method"] == comparator and row["status"] == "feasible"}
        common = hybrid.keys() & baseline.keys()
        ratios = [float(hybrid[case]["energy_bound_j"]) / float(baseline[case]["energy_bound_j"])
                  for case in common]
        lines.append(f"- hybrid / {comparator} 평균 energy ratio: {np.mean(ratios):.3f} "
                     f"(공통 feasible case {len(common)}개).")
    ratios = [float(row["analytic_rmse"]) / float(row["error_bound"])
              for row in wireless if row["status"] == "feasible"]
    lines += [
        "",
        f"분석적 RMSE / conservative bound 최대값: {max(ratios):.4f}.",
        "모든 feasible case에서 분석적 MSE가 bound² 이하이고, Monte Carlo MSE는",
        "분석적 MSE와 6 standard errors 이내로 일치했다.",
        "",
        "해석: head 선택은 고정 head보다 energy bound를 줄였지만, hybrid의 "
        "orthogonal/mixed AirComp 대비 에너지 우위는 확인되지 않았다. "
        "분리 전송보다 짧은 지연과 L/U 분리 제어의 가치를 함께 검증해야 한다.",
        "",
        "## 학습",
        "",
        "33-parameter 합성 two-tower 모델, 10 users, 100 rounds, 3 seeds, ideal channel.",
        "PUF는 single-step adaptation이며 eta_u=20 고정값의 tuning은 하지 않았다.",
        "",
        "| 방법 | test acc. mean ± sample SD (%) | forget–retrain gap mean (pp) | parameter distance |",
        "|---|---:|---:|---:|",
    ]
    for method in ("coupled", "uncoupled", "retain_only", "puf_regular_adapted"):
        records = [row for row in learning if row["method"] == method]
        accuracy = np.array([float(row["test_accuracy"]) for row in records]) * 100
        gap = np.mean([float(row["absolute_forget_accuracy_gap"]) for row in records]) * 100
        distance = np.mean([float(row["parameter_distance_to_retrain"]) for row in records])
        lines.append(f"| {method} | {accuracy.mean():.2f} ± {accuracy.std(ddof=1):.2f} "
                     f"| {gap:.3f} | {distance:.6f} |")
    lines += [
        "",
        "해석: 이 작은 문제에서는 coupled/uncoupled의 최종 정확도가 구분되지 않았다.",
        "이 결과만으로 결합의 효과나 삭제 성공을 결론낼 수 없다. "
        "실제 VLM 학습량, 요청 시점의 memorization, retrain gap 및 MIA 검증이 필요하다.",
        "",
        "## 재현 자료",
        "",
        "- wireless/metrics.csv: 실패 case를 포함한 전체 통신 기록.",
        "- wireless/convergence.json: 후보 head별 AO 목적함수 이력.",
        "- wireless/inputs_seed*.npz, design_*.npz: 채널/업데이트/선택된 설계.",
        "- wireless/wireless_pilot.png: 통신 파일럿 그림.",
        "- learning/round_metrics.csv, final_metrics.csv: 학습 곡선 및 최종 비교.",
        "- learning/schedule_seed*.json: sample ID별 도착·사용·삭제 일정.",
        "- learning/model_seed*.npy: 최종 모델 파라미터.",
        "- 각 결과 폴더의 provenance.json: 환경과 실험 성격.",
        "",
        "본 실험 미완료 사유: 현재 환경 CUDA device 접근 불가, 실제 데이터/CLIP "
        "checkpoint 연결 및 deep-prompt trainer·원본 baseline 구현 필요.",
        "",
    ]
    (ROOT / "results/REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[:36]))


if __name__ == "__main__":
    main()
