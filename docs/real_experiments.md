# 실제 CLIP 실험 실행 상태

## 확보한 자산

- pretrained OpenAI CLIP ViT-B/16, 고정 revision
  `57c216476eefef5ab752ec549e440a49ae4ae5f3`.
- `uoft-cs/cifar100` 배포본의 실제 train 50,000/test 10,000 이미지와 100개 class.
- 모델/데이터 파일은 프로젝트 `checkpoints/`, `data/`에 저장했다.
- Toronto 원본 tar 다운로드는 속도가 느려 중단했다. Hugging Face 배포본을 사용하며
  원본 tar의 MD5 검증을 완료했다고 주장하지 않는다. 데이터 revision과 SHA256은
  `data/cifar-100-python/manifest.json`, 모델은 checkpoint의 `manifest.json`에 있다.

출처: [CLIP 모델](https://huggingface.co/openai/clip-vit-base-patch16),
[CIFAR-100 배포본](https://huggingface.co/datasets/uoft-cs/cifar100),
[CIFAR 원저자](https://cave.cs.toronto.edu/kriz/cifar.html).

## 코드와 검증 범위

`clip_model.py`는 CLIP encoder weight를 고정하고, 처음 8개 transformer layer의
visual/text prompt 2개와 logit scale을 학습한다. 학습 파라미터는 20,481개다.
프롬프트는 CLS/SOS 뒤에 삽입하고 각 prompt layer에서 교체한다.
기존 patch/text 위치 임베딩을 유지하며, 삽입된 prompt에는 별도 pretrained 위치
임베딩을 더하지 않는다. 마지막 prompt layer 이후에도 prompt token을 유지한다.
텍스트 causal/padding mask와 EOS pooling 위치를 갱신한다.
프롬프트가 없는 경우 원래 CLIP logits과 일치하고, prompt가 있는 경우 양쪽
modality에 gradient가 생기며 backbone이 frozen 상태임을 작은 CLIP 모델로 검증했다.
구형 checkpoint의 position_ids buffer는 현재 모델과 일치할 때만 제거한다.

작은 모델의 정확한 Hessian과 달리, 실제 모델에서는 random orthonormal bases로
투영한 cross Hessian을 central finite differences로 근사한다.
K_hat = Qv C Qt^T의 norm은 작은 core C의 SVD로 정확히 계산하여
kappa ||K_hat||₂ <= 0.9를 보장한다. 이는 근사 K_hat에 대한 조건이며
원래 전체 Hessian의 norm bound나 성능 보장이 아니다.
curvature는 retained microbatch로 계산하며 rank/finite-difference step 민감도
검증이 아직 필요하다. 기본 rank=4, step=0.001은 파일럿 설계값이다.

`run_clip_federated.py`의 흐름:

1. CIFAR train에서 class당 50개를 validation으로 분리하고 train만 client에 분배한다.
2. 사용자별 retain/forget gradient를 계산한다. microbatch 누적으로 원래 batch 평균을 유지한다.
3. coupling/uncoupled/retain-only/PUF adaptation 중 지정한 방법으로 업데이트를 만든다.
4. 실제 parameter update를 블록화하여 전송 설계, 양자화, 잡음/SIC residual 복원을 수행한다.
5. **복원된 업데이트를 실제 CLIP prompt에 반영한다.**
6. validation loss와 누적 에너지 상한/기대값을 기록한다.
7. 삭제 요청별 평가 ID를 고정해 이후에도 같은 forget cohort를 평가한다.

삭제 일정은 비교 전에 canonical one-step batch schedule에서 생성하여 방법 간 동일하게
유지한다. PUF에서는 요청 사용자 retain step을 생략하므로 모든 target ID가 모든 방법에서
실제로 사용됐다고 보장하는 일정은 아니다. 이 프로토콜 차이는 본 실험에서 별도 검토한다.
retrain_reference는 최종 삭제 ID를 과거 모든 라운드에서 제외한 이상적 통신 기준이다.
미래 삭제 ID를 미리 사용하므로 최종 라운드 비교에만 쓰고 온라인 oracle로 해석하지 않는다.

전송 실패 시 중단하고 마지막 성공 checkpoint를 보존한다.
실패한 삭제 요청을 성공 처리하거나 자동으로 이상적 통신으로 대체하지 않는다.
자동 resume, EuroSAT loader, MIA, 원본 FedAPT/PUF/UAV 계층 알고리즘 재현은 아직 없다.
현재 full runner는 validation을 기록한다. 전체 test set 및 retrain 비교 집계 코드는
논문용 평가 단계에 추가해야 한다.

## 동일 시간 비교

`run_equal_time.py`는 같은 seed별 채널/업데이트와 같은 총 symbol 수를 사용한다.
분리 전송에서 remote digital payload가 있으면 analog/digital에 절반씩 배정한다.
이 경우 analog noise와 digital rate requirement도 각 구간 길이에 맞춰 달라진다.
분리 비율 50/50는 고정이며 최적 시간 배분 OMA baseline을 재현한 것은 아니다.
디지털 사용자끼리는 여전히 NOMA를 사용한다.

결과 `results/equal_time/metrics.csv`: 총 36개 조합.
모든 feasible case의 uplink 시간은 0.512 ms다.
Hybrid feasible 10/12, 분리 전송 12/12, mixed AirComp 12/12.
공통 feasible 10개에서 hybrid/분리 전송 에너지 상한 비율 평균은 1.0805,
독립 zero-mean codeword 가정 기대 에너지 비율 평균은 1.0590이다.
기대 에너지는 실제 RF 측정치가 아니며 finite-blocklength codec을 구현하지 않았다.
두 방법 모두 보수적 에너지 상한으로 최적화한 설계이며, 기대 에너지는 그 설계의 사후 계산값이다.
Mixed AirComp 대비 hybrid 에너지 상한 비율은 1.4917이다.
현재 조건에서는 NOMA 우위를 확인하지 못했다.

## 실행한 실제 데이터 검사

- `results/real_clip_smoke/cpu.json`: CIFAR 이미지 2개, 실제 CLIP forward/backward,
  프롬프트 SGD 1회 및 2-client duplicate-gradient analog 전송 연결 검사.
  삭제 payload는 없으므로 NOMA 복호/망각 성능 결과는 아니다.
- `results/clip_integration/hybrid_seed0/`: 실제 모델과 서로 다른 client minibatch,
  2 users, 2 rounds, round 2의 삭제 요청 및 저랭크 coupling을 포함한 CPU 통합 검사.
  validation 4개이므로 정확도를 본 실험 성능으로 해석할 수 없다.
- 수치/모델/전송 검증 10개가 통과했다.

CPU 검사에서 무선 SGD loss가 ideal SGD보다 높거나 낮을 수 있다.
같은 두 샘플의 training loss 한 번으로 generalization 또는 무선 이득을 추정하지 않는다.

## 실행 명령

프로젝트 루트에서:

```bash
export MPLCONFIGDIR=/tmp/ova_ck_mpl
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python validate_real_clip.py --device cuda
.venv/bin/python run_clip_federated.py --algorithm coupled --transport hybrid --seed 0
.venv/bin/python run_clip_federated.py --algorithm coupled --transport orthogonal_equal_time --seed 0
.venv/bin/python run_clip_federated.py --algorithm coupled --transport mixed_aircomp --seed 0
.venv/bin/python run_clip_federated.py --algorithm coupled --transport ideal --seed 0
.venv/bin/python run_clip_federated.py --algorithm uncoupled --transport ideal --seed 0
.venv/bin/python run_clip_federated.py --algorithm puf_regular_adapted --transport ideal --seed 0
.venv/bin/python run_clip_federated.py --algorithm retrain_reference --transport ideal --seed 0
```

각 명령은 순서대로 실행한다. RTX 3080 한 장에서 여러 CLIP 작업을 동시에 실행하지 않는다.
CUDA가 없으면 기본 실행은 즉시 실패한다. CPU 실험은 `--device cpu`를 명시해야 한다.
기존 출력 디렉터리는 자동 덮어쓰지 않으므로 재실행 시 별도 `--output`을 지정한다.

`bash run_gpu.sh`는 CUDA 연산 점검 → 실제 데이터 smoke → hybrid 100-round 실행을 수행한다.
이 스크립트는 장치 노드 생성, 권한 변경, sudo, 드라이버 재설치를 수행하지 않는다.
현재 채팅은 `/dev/nvidia*`가 노출되지 않고 호스트 실행 요청도 정책상 거절되어 있다.
2026-10-20 사용자가 일반 SSH 터미널에서 실행하여 CUDA smoke가 완료됐고,
`results/clip/coupled_hybrid_seed0/`에 RTX 3080 본 실험 로그가 기록되고 있다.
실행 중 상태는 manifest와 마지막 round를 함께 확인한다.

## 실제 데이터 비교 큐

같은 서버의 별도 SSH 터미널에서 프로젝트 루트로 이동한 뒤 실행한다:

```bash
bash run_comparisons.sh
```

실행기는 기존 seed 0 실험이 종료될 때까지 기다린 뒤, 수정된 solver로 coupled/hybrid부터
새로 실행하고 아래 7개를 순차 실행한다. 모두 1라운드부터 같은 solver 버전으로 실행한다.
기존 실험이 실패하면 실패 기록을 보존하고 비교군을 계속 실행한다.
running 상태로 30분간 로그가 갱신되지 않으면 중단하므로 원래 터미널을 확인한다.
두 터미널을 모두 유지한다. 이 파일을 만들었다고 큐가 자동 시작되는 것은 아니다.

1. coupled + orthogonal_equal_time: 동일 총 시간의 analog/digital 50:50 분리.
2. coupled + mixed_aircomp: 합산 업데이트의 analog 전송.
3. coupled + fixed_head: 고정 헤드 대조군.
4. uncoupled + hybrid: 같은 통신에서 결합 보정 제거.
5. puf_regular_adapted + hybrid: 같은 통신에서 한 스텝 PUF 변형.
6. coupled + ideal: 무선 오차 없는 학습 대조군; 에너지 0은 전송 이득이 아니라 모델 가정.
7. retrain_reference + ideal: 최종 삭제 집합을 처음부터 제외한 기준, 최종 시점 비교 전용.

이는 세 논문의 원본 재현이 아니라 통신/학습 대조 실험이다.
현재 seed 0의 데이터 분할, 삭제 일정, 채널, 초기화, 학습률, 전력/시간 제한을 유지한다.
모델 궤적은 방법별로 달라지므로 이후 전송 업데이트 자체가 동일한 paired radio-only
비교는 아니다. PUF 정규화/삭제 ID 사용 여부의 차이는 앞의 제한 사항이 그대로 적용된다.
한 seed의 결과만으로 통계적 우위를 주장하지 않는다.

결과는 `results/comparisons_seed0_solver_v2/`에 저장한다. 기존 미완료 결과는 덮어쓰거나 재개하지
않고 건너뛰며 큐 종료 코드는 실패로 표시한다. 각 작업의 `console.log`에 오류를 남기고
다른 대조군은 계속 실행한다. 학습 업데이트 식은 그대로이며 wireless solver 초기화만 수정했다.
수정 전 GPU 결과는 원래 디렉터리에 남겨 비교표와 섞지 않는다.
시작 전 현재 solver 버전의 소스 해시를 고정하고, 완료 후 설정과 삭제 일정을 검증한다.

`bash run_comparisons.sh --summary-only`는 GPU 없이 현재까지의 `comparison.csv`를 갱신한다.
평가가 없는 라운드는 loss를 보간하지 않고 빈칸으로 남기며 미실행 방법도 표시한다.
공통 라운드에서 loss/정확도, 누적 에너지 상한/기대값, 전송 시간과 고정 삭제 cohort를
비교한다. 실패 여부와 완료 라운드도 함께 보고하고 유리한 성공 사례만 고르지 않는다.
에너지는 모델 기반 통신량이며 GPU 전력이나 RF 측정치가 아니다.

2026-10-20 첫 GPU 실험은 90라운드까지 완료하고, 91라운드에서 전송 설계기가
feasible design을 찾지 못해 중단됐다. 이는 전역 infeasibility 증명은 아니다.
실패를 삭제하거나 제약을 완화하여 성공으로 바꾸지 않는다.
`diagnose_failed_round.py`는 마지막 GPU checkpoint와 저장된 batch 일정으로
다음 라운드 gradient를 CPU에서 재구성하고, 동일 업데이트에 네 전송 대조군을 적용한다.
출력은 `results/failed_round_diagnostic/`에 저장한다. GPU와 bitwise 동일하지 않으며
이 한 라운드 전송 비교를 100라운드 학습 비교로 해석하지 않는다.

추가 원인 분석: `check_radio_feasibility.py`는 receiver-scaled 변수의 convex relaxation을
풀고, 발견한 해를 원래 변수로 변환해 전력/배터리/오차/시간 제약을 별도로 검증한다.
91라운드 CPU 재구성에서 유효한 hybrid 해를 찾았으므로, 원래 탐색기의 실패가
물리적 infeasibility를 의미하지 않았음을 확인했다.
`wireless.py`는 기존 grid 초기화가 모두 실패한 경우에만 이 convex 초기화를 시도한다.
초기화 단계에서 생략한 배터리 제약도 원래 feasible 검사에서 확인하며,
제약을 완화한 해를 반환하지 않는다. AO는 이후 동일하게 에너지를 줄인다.
`comparison_original_solver.json`에는 수정 전의 단일 라운드 비교를 보존했다.
