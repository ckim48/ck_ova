# 실험 해석과 본 실험 계획

## 현재 통신 실험

3 seeds × 2 상대 오차 허용치 × 2 SIC residual 값 × 4 방법 = 48 records.
각 case는 10개 head 후보를 평가하고, 동일한 랜덤 채널과 업데이트를 재사용한다.
64차원 합성 weighted update를 32차원씩 두 블록으로 나눈다.
학습 gradient나 실제 CLIP update로 생성한 값은 아니다.

원고 설정을 채택한 값: U=10, 500×500 m², 높이 50 m,
reference gain -60 dB at 1 m, noise -90 dBm = 10^-12 W, max power 0.32 W.
모든 UAV 높이가 같으므로 UAV–UAV 거리에 높이 50 m를 별도로 더하지 않는다.
현재는 seed별 정지 배치이며 시간에 따른 UAV 이동을 시뮬레이션하지 않는다.

추가 설계값: W=1 MHz, L=8d, 양자화 8 bit/coordinate+float32 scale,
available energy 0.1 J/user/round, base overhead 10 µJ/user,
head overhead 50 µJ, coherence/deadline 각 10 ms.
이 값들은 원고에서 정해지지 않아 파일럿에서 명시적으로 선택한 값이다.
고정 overhead는 방법 간 동일하며 circuit/flight/학습 계산 에너지는 추정하지 않는다.

epsilon = relative_tolerance × ||sum_u rho_u||₂를 모든 통신 대조군에 동일하게 준다.
각 head에서 tolerance를 다시 조정하지 않는다. payload는 zero update일 때 0이다.
fixed-head는 head 0, hybrid는 탐색된 feasible design 중 최소 에너지다.

### 정규화와 원고 수식 대응

CVXPY 수치 안정성을 위해 다음 변수로 풀이한다:

- Hn = |h| / sigma
- x = rho |b| / sqrt(L), y = sqrt(p)
- t = |c| sigma sqrt(L)
- analog interference / sigma² = (sum Hn x)²
- block error = sum |t Hn x - rho|
  + t [sqrt(d/(2L)) + sum sqrt(chi) Hn y] + sum delta
- hybrid block energy bound = L/W (x+y)²

rho=0인 아날로그 전송은 0으로 고정한다. 원고의 위상 보상을 적용한 진폭 모델이다.
디지털 최소 전력과 SIC spectral radius 조건을 검사한다.
수신계수는 breakpoints rho/(Hn x)와 0의 weighted median이다.
가능한 초기해가 없으면 `no_feasible_design_found`로 기록한다.
원문제 infeasibility 증명이나 global optimum을 주장하지 않는다.
후검증 tolerance는 power 10^-6, global error 10^-5 상대 오차다.
파일럿 AO 종료는 에너지 상대 변화 10^-5 또는 최대 8회다. 원고의 모든 변수별
변화량 기준까지 구현한 것은 아니다. 수신계수 4개 초기화 중 가장 좋은 해를 저장한다.

Monte Carlo는 정규 복소 잡음과 사용자별 독립 정규 SIC residual을 가정한
post-SIC equivalent model이다. 실제 codeword, finite-blocklength 오류, CSI 추정,
동기 오차, clipping, peak power는 구현하지 않았다.
S는 앞 d개의 좌표로 사상하는 semi-unitary embedding으로 해석할 수 있다.
이 residual 모델의 해석적 MSE와 표본 MSE를 비교하고 보수적 bound 이하인지 검증한다.
순간적인 모든 잡음 표본의 오차가 bound 아래라는 주장은 하지 않는다.

### 대조군의 resource 차이

- Hybrid: analog L와 digital U를 동시 전송.
- Fixed-head: 같은 hybrid 최적화에서 head=0.
- Orthogonal: analog L와 digital NOMA U를 별도 시간구간에 전송.
  디지털 사용자 사이 SIC는 유지되나 analog에 대한 residual은 없다.
  파일럿에서 uplink 길이가 hybrid의 2배이며 동일 전체 deadline 내에서 비교한다.
  각 구간의 전력 상한을 독립 적용하고 energy는 두 구간을 합산한다.
- Mixed AirComp: L+U를 미리 합쳐 analog로 전송. 디지털 payload와 SIC가 없다.
  최종 합을 추정할 수 있지만 수신 후 L/U별 재가중은 불가능하다.

고정 에러 허용치만 같다고 동일 지연 실험이 되는 것은 아니다.
본 실험에서는 동일 지연 모드(분리 전송의 구간 길이/요율 변경)와
동일 per-stream symbols 모드를 각각 보고해야 한다.

## 현재 학습 실험

합성 class prototype과 noisy image/text vectors, 4D visual/text linear adapters,
shared logit scale의 33개 파라미터다. CLIP, transformer, deep prompt 결과가 아니다.
10 users, Dirichlet alpha=0.1, 100 rounds, initial 80%, 이후 20% 도착,
51/61/71/81/91 round의 두 사용자 요청, 사용했던 undeleted 샘플의 50% 삭제를 실행한다.
초기 데이터 접근과 실제 SGD 사용을 구분한다.
이번 schedule은 ideal communication에서 모든 round가 실행되는 경우를 나타낸다.

각 request에서만 retained minibatch의 정확한 Hessian block을 구해
kappa ||K||₂ <= 0.9를 만족시킨다. shared gradient는 결합하지 않는다.
미니배치 Hessian은 원고의 full local loss Hessian에 대한 stochastic approximation이다.
원고대로 같은 cosine learning/unlearning rate와 single-step SGD를 사용한다.
eta_u=20의 PUF adaptation은 tuning을 수행하지 않았다.

재학습 기준은 같은 초기화에서 최종 삭제 ID를 전 라운드에서 제외하고,
데이터 arrival 순서와 per-user batch RNG를 재사용한 최종-round 비교 기준이다.
중간 라운드의 retrain 곡선에는 미래 삭제 정보도 반영되므로
request별 온라인 gold-standard 곡선으로 해석하면 안 된다.
현재 summary는 final round만 비교한다.

삭제 정확도 하락 자체를 망각 성공으로 해석하지 않는다.
Forget accuracy의 재학습 대비 차이와 parameter distance를 기록하되,
MIA나 feature-level leakage 검증이 없어 삭제 보장을 주장할 수 없다.

## 원고에서 먼저 확인할 과학적 쟁점

1. **Mixed AirComp 대조군이 필수다.** 최종 업데이트가 L+U의 합일 때,
   선형 채널에서 L/U의 부호 상쇄는 의도된 합의 일부다. 합 자체만 필요하면
   분리 복원이 자동으로 더 좋은 학습을 보장하지 않는다.
   분리 제어가 필요한 요청별 오차/재가중 요구를 정의하거나, 동일 지연에서
   실제 이득을 측정해야 한다.
2. **MSE bound와 망각 성능은 다르다.** 작은 update RMSE만으로 삭제 데이터
   영향 제거가 보장되지 않는다. retain test, forget–retrain gap, MIA를 병행한다.
3. **SPD는 성능 우위가 아니다.** preconditioner 양의 정부호는 방향의 국소
   성질일 뿐 재학습 모델 근접성이나 cross-modal leakage 제거의 증명이 아니다.
4. **step과 epoch를 구분한다.** 100×1 minibatch로 삭제할 정도의 학습이
   있었는지 pre-request train/test 및 memorization 수준부터 확인한다.
5. **큰 Hessian을 그대로 만들지 않는다.** full CLIP prompt에서는 HVP 또는
   명시적 저랭크 K를 구현한다. power iteration norm은 일반적으로 하한 추정이므로,
   그것만으로 SPD를 보증한다고 하지 않는다. 근사 K의 안전한 norm bound가 필요하다.
6. **누락된 정의를 고정한다.** quantizer, L/W, chi, epsilon, kappa, overhead,
   battery 및 coherence 값을 명시하고 validation에서 결정한다.
   원고의 commented quantization 정의, 중복 문장, 정의되지 않은
   eq:analog_update/algorithm reference도 정리할 필요가 있다.

## 본 실험 실행 순서

1. RTX 3080의 CUDA device 접근을 복구하고 공개 pretrained CLIP checkpoint와
   CIFAR-100/EuroSAT-RGB 경로를 지정한다.
2. freeze encoder + deep visual/text prompt의 실제 모델을 구현/검증한다.
   feature caching은 visual encoder 내부 prompt 학습을 대체할 수 없다.
3. 학습과 삭제가 관측되는 initial-round 수를 검증한다. 원고의 50+50 조건은
   우선 유지하고, 학습량 부족 시 별도의 확장 실험으로 보고한다.
4. 이상적 채널에서 proposed/kappa=0/PUF/retain-only/retrain을 실행한다.
5. 동일 로컬 업데이트로 통신법을 비교한 뒤, end-to-end wireless 학습을 연결한다.
6. 검증된 원본 baseline과 이식 baseline을 구분하여 최소 5 seeds로 실행한다.
7. test accuracy, retrain gap, membership attack ROC-AUC, 양방향 cross-modal
   retrieval leakage, energy, latency, feasible round rate, memory/time을 기록한다.
8. public test는 hyperparameter tuning에 쓰지 않는다. EuroSAT의 train/val/test
   분할 ID를 저장하고 중복/근접 장면 누수를 점검한다.

infeasible round 처리(전송 생략/OMA fallback/다음 라운드 재요청)는 아직 학습에
연결하지 않았다. 연결할 때는 실패한 삭제 요청을 성공 처리하지 않아야 한다.
