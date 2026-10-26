# ova_CK

첨부 원고의 Hybrid NOMA–AirComp 및 교차모달 결합 업데이트를 검증하는 실험 프로젝트.
원고 사본은 `manuscript/original.tex`에 있으며 마지막 개행만 추가했다.

## 최신 작업 — 2026-10-26

- **최신 반투명 군집·개별 이동선 지도**: [PNG](results/figures/mission_satellite_iot_uav_planned_1x2_global_iot_five_sites_reference_square_cluster_motion.png), [PDF](results/figures/mission_satellite_iot_uav_planned_1x2_global_iot_five_sites_reference_square_cluster_motion.pdf). 동일한 UAV 10대의 5개 방문 시점을 반투명 군집으로 표시하고, UAV별 이동선을 이어 그렸다. 숫자 1–5는 방문 순서이며 빨간 별은 계획용 head다. 그림 안의 설명·범례·축 이름은 생략했다. 군집 외곽선은 시각적 묶음이며 통신 커버리지가 아니다.
- **최신 전역 IoT 분포·5구역 지도**: [PNG](results/figures/mission_satellite_iot_uav_planned_1x2_global_iot_five_sites_reference_square.png), [PDF](results/figures/mission_satellite_iot_uav_planned_1x2_global_iot_five_sites_reference_square.pdf). 가상 IoT 200개를 실제 도로망 전반에 먼저 분산하고, 그 수요로부터 5개 수집 구역·UAV 배치·계획용 head를 정했다. 10×10 공간 격자 모두에 IoT가 포함되며 X·Y축 양 끝은 0–2000 m다. 별들은 같은 UAV 10대의 5개 시점 배치다.
- 이전 5지역·광역 동선 지도: [PNG](results/figures/mission_satellite_iot_uav_planned_1x2_five_sites_reference_square.png), [PDF](results/figures/mission_satellite_iot_uav_planned_1x2_five_sites_reference_square.pdf). 이 버전은 방문 지역 주변의 IoT 생성 방식이며 최신 전역 분포 버전으로 대체했다.
- 사용자 요청으로 4지역 실행은 기록을 보존해 중단했다. 5지역 배치·채널은 `results/iot_uav_deployment_v2_five_sites/`에 있으며 새 비교 실험은 별도 경로에서 재실행한다. 이 지도 커밋은 학습 결과 완료를 의미하지 않는다.
- **IoT 기반 UAV 배치**: [1×2 PNG](results/figures/mission_satellite_iot_uav_planned_1x2.png), [PDF](results/figures/mission_satellite_iot_uav_planned_1x2.pdf). 원은 가상 지상 IoT, 별은 최적화된 UAV, 금색 별은 계획 단계의 head다. 범례 외 설명 글씨는 생략했다.
- 참고 디자인을 적용한 큰 버전: [PNG](results/figures/mission_satellite_iot_uav_planned_1x2_reference_large.png), [PDF](results/figures/mission_satellite_iot_uav_planned_1x2_reference_large.pdf). 파란 UAV 별, 주황 IoT 원, 붉은 head, 상단 공통 범례와 미터 좌표축을 사용하며 원래 버전과 좌표·head는 같다.
- **높이를 늘린 정사각형 패널 버전**: [PNG](results/figures/mission_satellite_iot_uav_planned_1x2_reference_square.png), [PDF](results/figures/mission_satellite_iot_uav_planned_1x2_reference_square.pdf). 남북 표시 범위를 넓혔으며 지리 좌표의 가로·세로 비율은 유지한다.
- IoT 수요를 기준으로 위치·연결·계획용 head를 번갈아 최적화하고, 새 위치에서 채널을 다시 생성했다. 공통 배치에서 네 방법을 비교하는 새 실행은 `tools/iot_vlm_v1/`와 `results/iot_uav_remoteclip_v1/`를 사용한다. 실제 FL/FU head는 전송마다 다시 선택한다. [모델·검증·실행 안내](docs/iot_uav_deployment.md).
- 위성사진 / 지역별 UAV 스냅샷 1×2: [PNG](results/figures/mission_satellite_uav_snapshots_1x2.png), [PDF](results/figures/mission_satellite_uav_snapshots_1x2.pdf), [SVG](results/figures/mission_satellite_uav_snapshots_1x2.svg).
- 스냅샷은 완료된 기존 RSITMD·Proposed 실험의 지역별 첫 전송을 사용한다. 개별 UAV는 저장된 시뮬레이션 좌표에 표시하며 당시 선택된 head를 강조한다. U1–U10은 로그의 인덱스 0–9에 대응한다. 네 지역은 동일한 열 대의 서로 다른 시점이며 실제 비행 실측은 아니다.
- 이전 [군집 도식](results/figures/mission_satellite_swarm_cluster_1x2.png)도 보존했다. 해당 도식의 분산된 점 배치는 시각화용이며 시뮬레이션 좌표를 변경한 결과가 아니다.
- 실제 UAV 영상·문장 데이터 CapERA와 RSITMD를 RemoteCLIP으로 비교하는 기본 구현은 `tools/uav_real_vlm_v1/`에 있다. 그 초기 실행은 중단되어 보존했고, 새 IoT 배치 실험은 별도 경로에서 처음부터 실행한다. 이 커밋은 새 비교 결과가 완료됐다는 의미가 아니다.
- 총에너지는 통신 + 연산 + 비행의 명시적 시나리오 모델로 계산한다. 장비 실측 에너지가 아니다. 자세한 입력과 실행 조건은 [새 실험 안내](docs/uav_remoteclip_capera.md)를 참고한다.

아래 내용은 이전 실험의 기록이다.

## 현재 상태

- 실제 실행: CPU 통신 시뮬레이션, 합성 데이터 100라운드 학습/삭제 실험, 수치 검증.
- 추가 실행: 실제 pretrained CLIP ViT-B/16 + CIFAR-100 CPU 학습/무선 연결 검증.
  2-user, 2-round 통합 검사에서 삭제 요청과 결합 업데이트까지 실행했다.
- CUDA 본 실험: 사용자 SSH 터미널의 RTX 3080에서 90라운드 완료 후 91라운드 전송 탐색 실패.
  마지막 체크포인트에서 CPU 재현 후 convex 초기화 보강으로 해당 전송 실패를 해결했다.
- 미완료: CIFAR-100 10-user/100-round 비교, EuroSAT, 비교 논문 원본 재현, MIA.
- `bash run_comparisons.sh`: 수정 solver로 제안 방법과 7개 대조군을 순차 재실행한다.
  결과는 `results/comparisons_seed0_solver_v2/`에 저장하며 GPU가 보이는 서버 터미널이 필요하다.
- 기존 합성 결과의 학습/통신은 독립이다. 새 `run_clip_federated.py`는 무선 복원값을 모델에 적용한다.
- 실험 결과는 `results/REPORT.md`, 논문 선정과 비교 조건은 `docs/comparisons.md`.
- `configs/full_protocol.json`은 초기 계획이다. 새 실행 설정은 `configs/clip_cifar100.json`이다.
- 최신 실제 데이터 실행 안내와 한계는 `docs/real_experiments.md`에 있다.

## 실행

현재 서버에서 검증한 Python:
`/home/wnlab/anaconda3/envs/DRL-Caching/bin/python`

```bash
cd /home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK
export MPLCONFIGDIR=/tmp/ova_ck_mpl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PYTHON=/home/wnlab/anaconda3/envs/DRL-Caching/bin/python
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" run_wireless.py
"$PYTHON" run_learning_pilot.py
"$PYTHON" summarize.py
```

새 환경에서는 Python 3.10 이상과 `requirements.txt` 의존성을 설치한다.
CLIP까지 검증할 때는 프로젝트 `.venv/bin/python`과 `requirements-real.txt`를 사용한다.
작은 통신 검증만 실행하려면 `run_wireless.py --smoke --output results/smoke`를 사용한다.
동일 출력 경로 재실행 시 해당 결과 파일을 갱신한다.

현재 호스트는 `computing-node-8`, GPU 정보는 NVIDIA GeForce RTX 3080이다.
에이전트 실행 환경에는 `/dev/nvidia*`가 없지만 사용자 SSH 터미널에서는 CUDA가 동작했다.
GPU 학습 로그와 CPU 진단 결과는 별도로 저장한다. 드라이버/디바이스 노드 설정은 변경하지 않았다.
기존 환경의 OR-Tools 관련 GLOP/PDLP 경고와 무관하게 CLARABEL로 최적화했다.

## 구현 범위

`wireless.py`:

- 양자화와 32비트 스케일 메타데이터, 0 payload 처리.
- 고정 수신계수 SOCP 초기화 및 아날로그 갱신.
- grid 초기화 실패 시 receiver-scaled convex 초기화와 원래 제약의 후검증.
- SIC 행렬의 spectral radius 검사와 최소 디지털 전력 선형계 풀이.
- weighted median 수신계수 및 후보 head 전수 탐색.
- 전력/에너지/오차/복호 제약의 해 후검증과 목적함수 증가 해 거부.
- Gaussian 잡음·잔류간섭 모델의 해석적 MSE 및 Monte Carlo 교차검증.

`coupling.py`, `data_schedule.py`, `run_learning_pilot.py`:

- 작은 두 인코더 모델에서 정확한 교차 Hessian 및 SPD 조건 검증.
- Dirichlet 분할, 80/20 데이터 공개, **실제로 minibatch에서 사용한** 샘플만 삭제 요청.
- 100라운드, 51/61/71/81/91라운드 요청, 매회 두 사용자.
- coupled/uncoupled/retain-only/PUF one-step adaptation 비교.
- 최종 삭제 샘플을 모든 과거 라운드에서 제외한 초기화부터 재학습 기준.

본 실험으로 확장할 때의 수식 해석, 공정성 조건, 미구현 항목은
`docs/protocol.md`를 참고한다. 원고의 우수성이나 확실한 망각을 주장하는 결과가 아니다.
