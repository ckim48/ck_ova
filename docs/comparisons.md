# 비교 논문 3편

2026-10-20에 아래 1차 출처를 확인했다. 세 논문은 서로 다른 기여 축을 다룬다.
하나의 end-to-end 성능표에 원본 논문 수치를 그대로 옮겨 비교하지 않는다.

| 논문 | 비교 목적 | 현 상태 |
|---|---|---|
| Mora et al., **Federated Unlearning Made Practical: Seamless Integration via Negated Pseudo-Gradients (PUF)**, IEEE TBD 2026 | 동시 FL/FUL, sample deletion, retain–forget trade-off | 합성 모델에서 PUF-Regular의 one-step adaptation 구현; 원본 재현 미완료 |
| Zhai et al., **FedAPT: Federated Adversarial Prompt Tuning for Vision-Language Models**, ACM MM 2025 | 연합 VLM prompt adaptation 및 유틸리티 기준 | 원문 설계 확인; 원본 구현/실험 미완료 |
| Zhong et al., **UAV-Enabled Over-the-Air Federated Learning: A Hierarchical Aggregation Approach**, IEEE TWC 2026 | UAV AirComp와 통신 자원 비교 | 출판사 방법 개요 확인; 원본 알고리즘 재현 미완료 |

## 1. PUF — 주 비교

원문은 일반 로컬 학습 업데이트의 부호를 뒤집고 크기를 조정한다.
Regular 모드는 retained/forget 업데이트를 공통 표본수 분모로 가중한다.
sample unlearning에서는 요청 샘플로 forget 업데이트를 생성한다.
원문 기본 설정의 local epoch와 이번 원고의 single minibatch step은 다르다.

현재 `puf_regular_adapted`는 single-step, eta_u=20을 사용하는 작은 모델 검증이다.
요청 사용자는 요청 라운드에 forget 업데이트만 보낸다.
CADENCE의 retain 및 forget별 정규화와 동일한 식으로 바꾸지 않았다.
본 실험에서는 동일 validation 예산으로 eta_u를 재조정하고,
원본 설정 재현과 CLIP 이식 설정을 구분해 기록해야 한다.

- [원문 및 알고리즘](https://arxiv.org/html/2504.05822v2)
- [TBD 출판 DOI](https://doi.org/10.1109/TBDATA.2026.3705404)
- [저자 대학 출판 목록](https://docenti.unisa.it/070141/ricerca/pubblicazioni?anno=2026&tip=262)
- [원문이 제공하는 코드 주소](https://anonymous.4open.science/r/puf_unlearning/) — 저장소 내용은 이번에 검증하지 않음.

## 2. FedAPT — VLM 학습 기준

FedAPT는 adversarial training, global beacon, text-to-visual prompt generator를 사용한다.
독립적인 visual/text prompt와 shared logit scale만 학습하는 이번 원고와 구조가 다르다.
두 prompt token과 여덟 layer 설정을 차용해도 그것만으로 FedAPT 재현은 아니다.
망각 알고리즘이 아니므로 retain 성능/계산량 비교에 사용하고, 삭제 성능은
별도 조합인 “FedAPT + PUF”를 구현할 경우에만 그 이름으로 보고한다.

- [원문](https://arxiv.org/html/2509.06992v1)
- [출판 DOI](https://doi.org/10.1145/3746027.3755387)

## 3. UAV 계층형 AirComp — 통신 기준

원문은 UAV가 parameter server가 되어 계층적 집계를 수행하며,
trajectory, device selection, aggregation coefficients를 공동 설계한다.
현재 원고는 미리 정해진 UAV 궤적과 라운드별 head를 사용한다.
따라서 기존 계층형 알고리즘을 같은 조건에 맞춘 버전은 adaptation으로 표시해야 한다.
`mixed_aircomp`와 `orthogonal`은 자체 통신 대조군이며 Zhong 논문 재현이 아니다.

- [IEEE 출판사 논문](https://ieeexplore.ieee.org/document/11271111/)
- [출판 DOI](https://doi.org/10.1109/TWC.2025.3635287)

2022년 arXiv:2209.11624는 제목과 저자 구성이 다르므로 위 저널 논문의 동일 버전으로
단정하지 않았다. 원고의 `Wei_KDD_2026`은 참고문헌 목록이 없어 정확한 문헌을
확인하지 못했다. 임의의 문헌으로 채우지 않았다.

## 최종 결과표 구성

1. 통신표: 같은 로컬 L/U 업데이트, head 후보, 채널, error tolerance에서 hybrid,
   fixed-head, OMA/AirComp 분리 전송, mixed AirComp. 에너지와 지연을 함께 비교.
2. 학습표: 이상적 통신에서 proposed, kappa=0, retain-only, tuned PUF, retrain.
3. VLM표: CLIP prompt 기준 및 정확히 구현된 FedAPT를 별도 표기.
4. end-to-end표: 검증된 학습법과 통신법을 조합하고 동일 데이터/요청/채널 seed 사용.

추가 필수 ablation: perfect SIC, 고정 head, 비결합 unlearning, 이상적 채널.
기존 논문 이름을 자체 대조군의 별명으로 사용하지 않는다.
