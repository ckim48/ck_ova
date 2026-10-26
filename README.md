# ova_CK

첨부 원고의 Hybrid NOMA–AirComp 및 교차모달 결합 업데이트를 검증하는 실험 프로젝트.
원고 사본은 `manuscript/original.tex`에 있으며 마지막 개행만 추가했다.

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
