# DESIGN_TRACE — DAY 1 설계와 DAY 2 구현

기준 문서는 `reports/DS-MINI-Design-울산_4반-박연제.md`다. `reports/day1_design.md` 파일은 없고, 같은 원고가 이 파일명으로 있다 (2026-10-02 확인). 아래 장 번호는 그 파일 기준이다. 리스크는 8장, DAY 2 계획은 9장이다.

상태 값: 대기(구현 전) / 구현(코드는 있으나 설계 수치를 다시 출력해 맞추기 전) / 검증(코드 출력이 설계 수치와 같음).

| ID | 설계서 위치 | 설계 내용 | 구현 위치 | 상태 | 비고 |
| --- | --- | --- | --- | --- | --- |
| D01 | 1장, 6장 | 태스크는 회귀만. 분류 모델은 만들지 않는다. ΔQ는 사이클 100이 필요해 초기 5사이클 분류와 연결되지 않는다. | `src/train.py`, `notebooks/03_modeling.ipynb` | 검증 | src/와 notebooks/ 소스에서 Classifier, accuracy, f1은 0건. "분류"는 01_EDA의 "이 컷은 분류 라벨로 쓰지 않는다" 한 문장뿐이다. |
| D02 | 1장 가설과 데이터 정의 | 1행 = 1셀. Y는 `cycle_life`(SOH 80%). X는 초기 100사이클 셀 피처. 척도는 사이클 수. | `src/features.py` `build_feature_table`, `src/train.py` | 검증 | 사이클 단위 분할 금지. |
| D03 | 4장 첫 문장 | 피처는 초기 100사이클만. `cycle_life`나 EOL 이후 정보로 피처를 만들지 않는다. | `src/features.py` | 구현 | 100~300 기울기는 D19. |
| D04 | 2장 표 | 행은 지우지 않고 `use=False`. 최종 셀 수 Batch 1 36(46 중 제외 10), Batch 2 39(47 중 8), Batch 3 44(46 중 2). 사용 합계 119. | `src/preprocess.py` `EXCLUDE_REASON`, `mark_use`, `load_cells` | 검증 | 제외: b1c0~b1c4, b1c8, b1c10, b1c12, b1c13, b1c22, b2c22, b2c23, b2c35~b2c40, b3c23, b3c32. |
| D05 | 2장, 5장 | b1c0~b1c4는 연속 실험이 아니다. 용량이 0.053~0.098Ah 오르고 정책이 달라 병합하지 않고 b1만 제외. b2c7, b2c8, b2c9, b2c15, b2c16은 테스트에 남긴다. | `src/preprocess.py` `EXCLUDE_REASON` | 구현 | |
| D06 | 2장, 5장 | b3c37(수명 1,390)은 유지. 사이클 597~598의 2사이클 스파이크이고 초기 100사이클 밖. | `src/preprocess.py` (제외 목록에 없음) | 구현 | |
| D07 | 2장 | summary 단발 스파이크만 앞뒤 중앙값으로 교체. 용량 0.20Ah(앞뒤 0.05Ah 이하), IR 0.005Ω, 온도 5°C, chargetime 30. 임계값은 학습으로 맞추지 않음. 사용 셀 22점(Batch 1이 10, 2가 12, 3이 0). b1c18 사이클 40 QD 2.884→1.069Ah. pkl은 수정하지 않음. | `src/features.py` `correct_summary_spikes`, `SPIKE_RULES` | 검증 | IR·온도·chargetime의 앞뒤 허용 차이는 2장에 없다. 코드에는 IR 0.001Ω, 온도 2°C, chargetime 1이 있다. |
| D08 | 2장, 4장 정의서 | ΔQ = Qdlin[행 99] − Qdlin[행 9] (사이클 100 − 10). 사용 셀 119개의 두 행은 유한. Vdlin은 3.5V→2.0V, 1,000점, 배치 공통. | `src/features.py` `delta_q` (`CYCLE_10_ROW=9`, `CYCLE_100_ROW=99`) | 구현 | |
| D09 | 4장 정의서, Q3·Q5 | `log10(var(ΔQ))`: 1,000점 분산에 log10. 전처리 없음. 수명과의 예상 방향 음(−). | `src/features.py` `delta_q_features`의 `log10_var` | 검증 | 분산의 ddof는 설계서에 없다. 기존 함수는 `np.var(ddof=1)`. |
| D10 | 4장 정의서, Q5 | `qd_max_minus_2`: 스파이크 보정 후 사이클 2~100 QD 최댓값 − 사이클 2 QD. 전처리 `correct_summary_spikes`. log10(var) 통제 후 예상 방향 음(−). 단순 상관은 Batch 1 +0.26, Batch 2 −0.31로 부호가 바뀐다. | `src/features.py` `summary_features` | 검증 | |
| D11 | 1장, 4장 정의서, 7장 | 학습 타깃 `log10(cycle_life)`. 예측은 10의 거듭제곱으로 되돌려 MAPE를 원래 사이클 수로 계산. pkl의 `cycle_life`는 원본. 로그는 Batch 2 이봉(왜도 1.64→1.36)과 외삽을 없애지 못한다. | `src/train.py` `mape_original`, `MAPE_SCORER` | 구현 | y는 log10. MAPE와 GridSearchCV scorer는 이 함수만 사용. 왜도 수치는 Batch 2·3를 열지 않아 P9에서 재출력하지 않음. |
| D12 | 4장 표, Q5 | `log10(절댓값 min ΔQ)` 배제. var와 r=0.996, 잔차 상관 0.036, 같이 넣으면 VIF 689·711. | 모델 입력에서 제외. 계산은 `delta_q_features`의 `log10_abs_min` | 검증 | `MODEL_FEATURES`에 아직 들어 있다. |
| D13 | 4장 표, Q5 | ΔQ 2V 값, skew, kurtosis 배제. 2V는 var와 r=−0.885(4장 표는 −0.89). skew는 Batch 3에서 부호 반전. | 모델 입력에서 제외 | 구현 | `MODEL_FEATURES`에 아직 들어 있다. |
| D14 | 4장 표, Q2·Q5 | 초기 QD 기울기·절편, 사이클 2 QD 배제. 절편과 사이클 2 QD는 r=0.999. 기울기 부호는 Batch 2에서 반전. | 모델 입력에서 제외 | 구현 | `summary_features`는 계산한다. |
| D15 | 4장 표, 5장, Q2 | knee 배제. 수명의 78%, r=0.99. 단수명 knee 중앙 348사이클. 검토 후 누수로 배제. | 피처 함수 없음 | 구현 | 모델 피처로 만드는 코드는 없다. |
| D16 | 4장 표, Q4, 5장 | C1, 전환 SOC, C2, 평균 C-rate, 초기 chargetime 배제. Batch 1 r=−0.53이 테스트에서 소멸(Batch 2 −0.04, Batch 3 −0.05). 같은 정책 중앙 수명 범위 492~1,640. 미학습 전류식 Batch 2 29/39, Batch 3 38/44. | `COMPARE_FEATURES`. 메인 입력에서 제외 | 구현 | 비교용 계산은 있다. 메인 목록 잠금은 P8. |
| D17 | 4장 표, Q3·Q5, 5장 | 정규화 ΔQ와 배치 번호 배제. 3.0V 깊이 비 1.30→1.29. 배치 효과는 Batch 1만으로 추정 불가. | `normalized_delta_q`는 계산만. 모델 입력 금지 | 구현 | `build_feature_table`이 `log10_var_norm` 열을 만든다. |
| D18 | Q5 시사점 | 온도 적분, IR, 기울기는 넣지 않는다. 온도 두 적분 상관 0.953. 학습 셀 36개에서 세 번째 피처는 공선이거나 부호가 바뀐다. | 모델 입력은 `log10_var`, `qd_max_minus_2`만 | 구현 | D41과 함께 P8에서 목록을 잠근다. |
| D19 | Q4 | 사이클 100~300 방전용량 기울기는 피처로 쓰지 않는다. Pearson Batch 1 −0.74, Batch 2 −0.32, Batch 3 −0.59. 사이클 100 이후라 누수. | EDA 그림만. 모델 피처 아님 | 구현 | `q4_fade_slope.png`. 피처 표에 넣지 말 것. |
| D20 | 4장 표, Q3, Q5 | 재현할 수치: 전체 r=−0.897 (요약은 −0.90), Spearman −0.888. Batch 1 Pearson −0.844, Spearman −0.826, R² 0.71 (Q5 본문 0.712). 잔차 상관 −0.529/−0.631/−0.427 (4장 표는 −0.53/−0.63/−0.43). 둘만의 VIF 1.73. | `notebooks/02_feature_engineering.ipynb` 출력 | 검증 | 어느 반올림을 맞출지는 아래 불일치 목록. 코드로 다시 출력하기 전에는 검증으로 두지 않는다. |
| D21 | 4장, 7장 | `Pipeline(StandardScaler → 모델)`. 스케일러는 학습 폴드에서만 fit. | `src/train.py` `make_pipeline` | 구현 | 메인·비교군 전부 동일 구조. 폴드 계수도 해당 폴드에서만 scaler fit. |
| D22 | 7장 | Batch 1 36셀, 정책 20종. `charging_policy` 그룹 `GroupShuffleSplit` train 80% / hold-out 20%. `RANDOM_STATE=42`. 같은 프로토콜이 양쪽에 동시에 들어가지 않음. | `src/train.py` `split_batch1` | 검증 | 그룹 열은 `policy_readable`. 출력: train 29셀·정책 16, hold-out 7셀·정책 4, 겹침 0. 셀 비율 19.444%는 그룹 20%의 근사. |
| D23 | 7장, 8장 | 튜닝과 Train MAPE는 train 80% 안 `GroupKFold`만. Valid는 hold-out 20%(약 7셀). Valid MAPE 한 장으로 모델을 잠그지 않고 GroupKFold 평균과 함께 본다. | `src/train.py` `fit_candidates`, `select_final` | 구현 | 폴드 5는 구체화 기록. 1위와 0.5%p 이내면 Valid가 낮은 메인. P9에서 세 메인이 해당해 Ridge. |
| D24 | 7장, 9장 | Batch 2는 모델과 하이퍼파라미터를 잠근 뒤 한 번만 평가. P10 전에는 모델 평가에 쓰지 않는다. | `notebooks/03_modeling.ipynb`, `results/model_performance.csv`, `results/predictions_batch2.csv` | 구현 | `load_features(2)`는 예측 셀 한 번. Test MAPE 20.259. 모델·하이퍼파라미터·피처·분할은 유지. |
| D25 | 7장, 9장 | Batch 3는 선택 검증이고 Batch 2 다음이다. P12 전에는 모델 평가에 쓰지 않는다. | `src/train.py`, `results/predictions_batch3.csv` | 구현 | 잠긴 Ridge로 한 번. 44셀. Test 24.945. 모델·피처·분할 유지. |
| D26 | 7장 후보 모델 | 메인: ElasticNet, Ridge, Lasso. L1·L2로 소표본 계수를 줄이고, 계수가 피처 영향이라 해석이 된다. 학습 수명 밖 외삽 가능. | `src/train.py` `_candidate_specs` | 구현 | 실제 학습은 train 29셀. 설계서 7장의 메인 선택 이유 중 외삽 가능, 계수가 피처 영향이라 해석 가능은 유지되고, L1·L2 계수 축소의 실제 효과는 작았다. 정규화 모델을 메인으로 둔 설계는 유지한다. |
| D27 | 7장 후보 모델 | 비교군: 얕은 트리, 부스팅. 학습 때 본 타깃 밖을 예측하지 못한다. 수명 392인 셀도 534 이상으로만 나온다. | `src/train.py` `fit_locked_comparisons` | 검증 | Batch 2 예측 최소 DecisionTree 666.987, XGBoost 588.958. train 최소보다 작은 예측 0. Batch 3 예측 최대 DecisionTree 872.099, XGBoost 947.558. train 최대 1,054보다 큰 예측 0. 실제 최장 1,935. |
| D28 | 7장 | 딥러닝은 쓰지 않는다. 학습 셀 36개 중 80%만 쓰고, 파라미터 수가 셀 수를 넘기 쉽다. | 모델 목록에 넣지 않음 | 구현 | P9 후보 표에 딥러닝 없음. |
| D29 | 8장 외삽 | Batch 2의 30/39는 534보다 짧다. Batch 3의 16셀은 1,074보다 길다. 오차를 534 미만과 1,074 초과로 나눈다. 트리는 이 구간에서 비교만. 학습 범위는 534~1,074. | `results/predictions_batch2.csv`, `results/predictions_batch3.csv`, `notebooks/03_modeling.ipynb` | 검증 | Batch 2: <534 30, 534~1,074 7, >1,074 2. Batch 3: <534 0, 534~1,074 28, >1,074 16. 16셀 일치. DecisionTree 최대 872.099, train 최대 초과 0. |
| D30 | 8장 ΔQ 오프셋 | 같은 수명인데 Batch 1의 ΔQ 골이 더 깊다. 수명 700~1,100의 3.0V 중앙 ΔQ는 Batch 1 −0.031Ah, Batch 2 −0.021Ah, Batch 3 −0.024Ah. 정규화로 줄지 않았다. Gap(Valid-Test)가 양수면 과대예측을 먼저 의심. 배치 번호로 맞추지 않는다. | `notebooks/03_modeling.ipynb` | 검증 | 설계 예상 부분 적중: 534 이상 과대예측 확인, 534 미만은 qd_max_minus_2 기여(평균 −0.078)가 과대예측을 상쇄. 3.0V는 Batch 1 −0.031, Batch 2 −0.021, Batch 3 −0.024로 일치. Batch 3 평균 부호 +22.500. 배치 번호는 넣지 않음. |
| D31 | 8장 hold-out | 정책 20종 그룹이라 Valid가 약 7셀. 순위는 GroupKFold 평균과 함께 본다. | `src/train.py` `select_final` | 구현 | hold-out 7셀. 세 메인이 0.5%p 안이라 Valid로 Ridge. Valid 단독 확정은 아님. |
| D32 | 8장 두 번째 피처 | `qd_max_minus_2`의 수명 상관 부호는 배치마다 다르다. 계수가 불안정하면 var 하나만 둔 모델을 비교군으로 남긴다. | `src/train.py` LinearRegression(log10_var), `fold_coefficients` | 구현 | 5폴드 계수 모두 음. 불안정 조건은 비해당. 단일 피처는 비교군으로 유지, 메인에서 빼지 않음. |
| D33 | 8장 원논문 | 원논문 9.1%는 학습·테스트 구성이 달라 직접 비교에 한계. Gap(Target-Test)가 양수일 가능성이 높고, 원인은 배치 간 분포 차이로 해석한다. | `results/model_performance.csv` | 구현 | Test 20.259, Gap(Target-Test)=11.159. 양수. 직접 비교하지 않고 배치 간 분포로 해석. |
| D34 | 9장 | 피처 2개로 `Pipeline`을 고정한다. | `src/train.py`, `src/features.py` `MODEL_FEATURES` | 검증 | 노트북 출력 `('log10_var', 'qd_max_minus_2')`. |
| D35 | 9장 | train 80%의 GroupKFold로 ElasticNet·Ridge·Lasso의 정규화 강도를 고른다. | `src/train.py` `fit_candidates` | 구현 | 하한 확장 후 Ridge alpha=1e-06, Lasso·ElasticNet alpha=1e-07(l1_ratio=0.1). 작은 alpha 구간은 평탄. 이전 잠금 대비 Train·Valid 변화 0.000196·0.000111%p라 Ridge로 덮어 잠금. |
| D36 | 9장 | hold-out으로 Valid MAPE를 보고 모델을 잠근 뒤 Batch 2를 한 번만 평가한다. | `notebooks/03_modeling.ipynb`, `results/final_model.joblib` | 구현 | 잠금 유지: Ridge alpha=1e-06. Batch 2 1회, Test 20.259. 재학습·재예측 없음. |
| D37 | 9장 | Gap(Train-Valid), Gap(Valid-Test), Gap(Target-Test) = Test MAPE − 9.1을 같이 본다. | `results/model_performance.csv` | 구현 | −1.769 / 15.371 / 11.159. Gap(A-B)=B−A. |
| D38 | 9장, 3장 배치 간 비교 | 오차는 수명 구간과 ΔQ 깊이로 나눈다. | `notebooks/03_modeling.ipynb`, `results/predictions_batch2.csv` | 구현 | 수명 구간은 P10. 3.0V ΔQ 3분위 Ridge MAPE는 깊음 16.285, 중간 15.960, 얕음 28.531. 얕은 구간 평균 부호 +20.228. |
| D39 | 9장 | Batch 3 평가는 Batch 2 다음의 선택이다. | `notebooks/03_modeling.ipynb`, `results/model_performance.csv` | 구현 | Test 24.945. Gap(Batch2-Batch3)=4.686. Gap(Target-Test, Batch 3)=15.845. D25. |
| D40 | 4장 정의서 마지막 줄 | DAY 2 코드는 정의서를 그대로 구현하고, 각 함수 주석에 표의 피처명을 적는다. | `src/features.py`, `src/train.py` 주석. 표기 `[설계 N장]`, `[Q번호]`, `[D번호]` | 구현 | `features.py`와 `train.py` 주석에 피처명 `log10(var(ΔQ))`, `qd_max_minus_2`, `log10(cycle_life)`와 D번호를 넣었다. |
| D41 | Q5 시사점, 핵심 요약 | 메인 피처는 두 개로 확정. 세 개로 늘리지 않는다. | `src/features.py` `MODEL_FEATURES` | 구현 | `("log10_var", "qd_max_minus_2")`로 잠금. |

## 설계서 안에서 맞추지 않은 표현

설계서는 수정하지 않았다. 구현 중 아래 중 하나를 골라야 하면 멈추고 승인을 받은 뒤 변경 기록에 남긴다.

1. 첫 문단은 데이터 셀 수를 124로 적는다. 2장 원본 합은 46+47+46=139이고, 사용 셀은 36+39+44=119다.
2. 같은 통계의 자릿수가 다르다. 전체 상관은 요약 r=−0.90, Q3·4장 −0.897. Batch 1 R²는 Q3·4장 0.71, Q5 본문 0.712. 잔차 상관은 4장 −0.53/−0.63/−0.43, Q5 −0.529/−0.631/−0.427. 2V와 var의 상관은 4장 −0.89, Q5 −0.885. Q1 본문 중앙은 772, 표는 772.5.
3. 8장 외삽 문장은 Batch 2의 534 미만 30/39와 Batch 3의 1,074 초과 16셀만 적는다. Q1과 배치 간 비교에는 Batch 2의 1,074 초과 2셀이 있고, 요약의 32셀은 30+2다.
4. 장수명·단수명이 절마다 다르다. Q1·Q2는 >1,000과 <500이고, 사이클 100 용량 차이 0.031Ah(1.069 대 1.100)는 이 기준이다. Q3 곡선은 배치 안 상위·하위 25%다.
5. Q4의 `4.8C(80%)-4.8C` 중앙 수명 753, 492, 809, 1,640은 값이 네 개인데 어느 집단인지 문장에 없다. DAY 2 구현과 무관하므로 **보류**.
6. 아래 구체화 기록으로 채운 항목: 수치 자릿수 비교, 오류 구간, 장·단수명, ΔQ 분산 ddof, 스파이크 앞뒤 허용 차이, GroupKFold 폴드 수, 비교군 클래스. 1번 124셀은 설계 변경 기록.

## 설계 구체화 기록

설계를 바꾼 것이 아니라, 설계서에 없던 세부값을 DAY 1 코드 기준으로 채운 것이다.

| 항목 | 설계서 상태 | 구체화 내용 | 근거 |
| --- | --- | --- | --- |
| 수치 일치 판정 | 같은 통계의 자릿수가 여럿이다 | 소수 셋째 자리까지 출력하고, 설계서에 적힌 자릿수로 반올림해 비교한다 | Q3·Q5와 4장 표 |
| 오류 분석 구간 | 8장은 534 미만과 1,074 초과만 적고, Q1에는 Batch 2의 1,074 초과 2셀이 있다 | 학습 범위 기준 <534 / 534~1,074 / >1,074 세 구간. Batch 2의 2셀은 상한에 포함한다 | [D29] [Q1] |
| 장·단수명 기준 | Q1·Q2는 >1,000/<500, Q3 곡선은 배치 내 25% | DAY 2는 학습 범위(534~1,074) 기준만 사용한다 | [D29] |
| ΔQ 분산 | ddof가 없다 | ddof=1. DAY 1 수치를 만든 기존 함수를 유지한다 | [D09] |
| 스파이크 앞뒤 허용 차이 | 용량만 앞뒤 0.05Ah가 적혀 있다 | IR 0.001Ω, 온도 2°C, chargetime 1. 기존 코드값을 유지하고 2장 22점으로 검증한다 | [D07] |
| GroupKFold 폴드 수 | 폴드 수가 없다 | 5. P9에서 사용한다 | [D23] [D35] |
| 비교군 클래스 | "얕은 트리, 부스팅"만 있다 | 얕은 트리 = DecisionTreeRegressor(max_depth≤3), 부스팅 = XGBRegressor(max_depth≤2). P9에서 사용한다 | [D27] |
| Q4 중앙 수명 4개 | 집단이 지정되지 않았다 | 보류. DAY 2 구현과 무관하다 | Q4 |
| 격자 하한 확장 | 설계서에 격자 없음 | 최적값이 격자 경계라 평탄 구간 확인 목적. Ridge logspace(−6, 3), Lasso·ElasticNet logspace(−7, 0), l1_ratio 0.1/0.5/0.9. Batch 2 미사용 | [D35] |

## 설계 변경 기록

| 항목 | 설계 | 변경 | 이유 | 근거 |
| --- | --- | --- | --- | --- |
| 데이터 셀 수 | 첫 문단 124셀 | DAY 2 산출물은 2장 기준 사용 119셀(36/39/44)로 통일 | 124는 오기 | 2장 표. 원본 46+47+46 중 사용 36/39/44 |
