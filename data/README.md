# 데이터

## 출처

MIT-Stanford Battery Dataset (Severson et al., *Nature Energy*, 2019). LFP/흑연 셀, 다양한 고속 충전 프로토콜. 배포본은 Kaggle의 MIT-Stanford Dataset이다.

## 두는 곳

원본 `.mat`은 `data/raw/`에 둔다. 이 저장소에는 이미 받아 둔 파일이 있으므로 이동·복사하지 않는다. `data/raw/`와 `data/processed/`는 git에 올리지 않는다.

`src/preprocess.py`로 한 번만 변환하고, 이후 분석은 `data/processed/`만 사용한다. 변환 파일의 `cycle_life`는 원본 사이클 수 그대로 저장한다. log 변환 여부는 EDA Q1 이후에 정한다.

## 파일

| 파일 | Batch | 용도 |
|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 | 학습. `charging_policy` 그룹으로 train 80% / hold-out 20% |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 | 최종 테스트. 모델 확정 후 한 번만 평가 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 | DAY 1 EDA 비교, DAY 2 추가 검증(선택) |
| `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` | extra | 사용하지 않음 |

분석 단위는 배터리 셀 1개(1행)다.
