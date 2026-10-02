"""Batch 1 분할·학습 파이프라인. [설계 7장] [설계 9장]

피처명: `log10(var(ΔQ))`는 열 `log10_var`, `qd_max_minus_2`, 타깃 `log10(cycle_life)`.
[설계 4장] [D11] [D34] [D40]

피처는 초기 100사이클만 쓴 표를 읽는다. 이 모듈은 사이클 단위로 나누지 않는다. [D02] [D03]
P9는 Batch 1만 읽는다. Batch 2 평가는 P10, Batch 3 평가는 P12. [D24] [D25]
"""

from __future__ import annotations

import csv
from io import StringIO

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import DATA_PROCESSED_DIR, HOLDOUT_SIZE, RANDOM_STATE, RESULTS_DIR
from src.features import MODEL_FEATURES

# 설계서는 폴드 수를 적지 않는다. 구체화 기록: GroupKFold 5. [D23] [D35]
N_SPLITS = 5
# 1위와 이 차이(%p) 안이면 Valid MAPE로 메인 모델을 가른다. [D23] [D31]
CV_TIE_MAPE = 0.5
# 비교군 깊이 상한. 구체화 기록. [D27]
TREE_MAX_DEPTH = 3
XGB_MAX_DEPTH = 2

# 설계서에 격자는 없다. 하한을 확장한 것은 최적값이 경계에 있어 평탄 구간을 보기 위함이다. [D35]
# numpy logspace 기본 50점. l1_ratio는 0.1/0.5/0.9.
RIDGE_ALPHAS = np.logspace(-6, 3)
LASSO_ALPHAS = np.logspace(-7, 0)
ENET_ALPHAS = np.logspace(-7, 0)
ENET_L1_RATIOS = (0.1, 0.5, 0.9)


def mape_original(y_true_log, y_pred_log) -> float:
    """log10(cycle_life) 예측을 10의 거듭제곱으로 되돌린 사이클 수 MAPE(%). [설계 7장] [D11]

    지표는 이 함수 하나만 쓴다. 로그 스케일 MAPE는 만들지 않는다.
    """
    y_true = np.power(10.0, np.asarray(y_true_log, dtype=float))
    y_pred = np.power(10.0, np.asarray(y_pred_log, dtype=float))
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


# greater_is_better=False: GridSearchCV는 MAPE가 작은 하이퍼파라미터를 고른다. [D11]
MAPE_SCORER = make_scorer(mape_original, greater_is_better=False)


def make_pipeline(estimator) -> Pipeline:
    """StandardScaler 다음 모델. 스케일러는 학습 폴드에서만 fit된다. [설계 7장] [D21]"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def load_features(batch: int) -> pd.DataFrame:
    """features_batch{N}.csv. X는 MODEL_FEATURES 두 열. y는 log10(cycle_life). [D11] [D34]

    반환 표에 `log_life` 열을 붙인다. 원본 cycle_life는 그대로 둔다.
    P9 호출은 batch=1만. Batch 2는 P10, Batch 3는 P12에서 각각 한 번 연다. [D24] [D25]
    """
    path = DATA_PROCESSED_DIR / f"features_batch{int(batch)}.csv"
    frame = pd.read_csv(path)
    missing = [name for name in MODEL_FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"피처 열이 없다: {missing}")
    frame = frame.copy()
    frame["log_life"] = np.log10(frame["cycle_life"].to_numpy(dtype=float))
    return frame


def split_batch1(frame: pd.DataFrame | None = None):
    """Batch 1을 policy_readable 그룹으로 train 80% / hold-out 20%. [설계 7장] [D22]

    설계서의 charging_policy와 같다. Batch 1에는 newstructure가 없어 문자열이 같다.
    GroupShuffleSplit의 test_size는 셀이 아니라 정책 그룹 비율이다.
    """
    if frame is None:
        frame = load_features(1)
    if not (frame["batch"] == 1).all():
        raise ValueError("split_batch1은 Batch 1만 나눈다.")
    groups = frame["policy_readable"].to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, valid_idx = next(splitter.split(frame, frame["log_life"], groups=groups))
    train = frame.iloc[train_idx].reset_index(drop=True)
    valid = frame.iloc[valid_idx].reset_index(drop=True)
    return train, valid


def _xy(frame: pd.DataFrame, columns=MODEL_FEATURES):
    x = frame.loc[:, list(columns)]
    y = frame["log_life"]
    groups = frame["policy_readable"]
    return x, y, groups


def _search(estimator, param_grid, x, y, groups) -> GridSearchCV:
    """train 80% 안에서만 GridSearchCV + GroupKFold. [D23] [D35]"""
    search = GridSearchCV(
        make_pipeline(estimator),
        param_grid,
        scoring=MAPE_SCORER,
        cv=GroupKFold(n_splits=N_SPLITS),
        refit=True,
        n_jobs=1,
    )
    search.fit(x, y, groups=groups)
    return search


def _cv_stats(search: GridSearchCV) -> tuple[float, float]:
    """선택 하이퍼파라미터의 GroupKFold MAPE 평균과 표준편차.

    GridSearchCV는 점수를 음수로 저장한다. 표준편차는 폴드 점수의 ddof=0이다.
    """
    idx = search.best_index_
    mean_mape = float(-search.cv_results_["mean_test_score"][idx])
    std_mape = float(search.cv_results_["std_test_score"][idx])
    return mean_mape, std_mape


def _params_text(params: dict) -> str:
    parts = []
    for key, value in params.items():
        name = key.replace("model__", "")
        if isinstance(value, float):
            parts.append(f"{name}={value:.4g}")
        else:
            parts.append(f"{name}={value}")
    return ", ".join(parts) if parts else "(없음)"


def _candidate_specs() -> list[dict]:
    """메인 3개, 비교군 3개. 딥러닝은 없다. [D26] [D27] [D28] [D32]"""
    return [
        {
            "model": "Ridge",
            "role": "메인",
            "estimator": Ridge(max_iter=10000, random_state=RANDOM_STATE),
            "param_grid": {"model__alpha": RIDGE_ALPHAS},
            "columns": MODEL_FEATURES,
        },
        {
            "model": "Lasso",
            "role": "메인",
            "estimator": Lasso(max_iter=10000, random_state=RANDOM_STATE),
            "param_grid": {"model__alpha": LASSO_ALPHAS},
            "columns": MODEL_FEATURES,
        },
        {
            "model": "ElasticNet",
            "role": "메인",
            "estimator": ElasticNet(max_iter=10000, random_state=RANDOM_STATE),
            "param_grid": {
                "model__alpha": ENET_ALPHAS,
                "model__l1_ratio": list(ENET_L1_RATIOS),
            },
            "columns": MODEL_FEATURES,
        },
        {
            "model": "DecisionTree",
            "role": "비교군",
            "estimator": DecisionTreeRegressor(
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {"model__max_depth": [1, 2, TREE_MAX_DEPTH]},
            "columns": MODEL_FEATURES,
        },
        {
            "model": "XGBoost",
            "role": "비교군",
            "estimator": XGBRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            "param_grid": {
                "model__max_depth": [1, XGB_MAX_DEPTH],
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.05, 0.1],
            },
            "columns": MODEL_FEATURES,
        },
        {
            "model": "LinearRegression(log10_var)",
            "role": "비교군",
            "estimator": LinearRegression(),
            "param_grid": {},
            "columns": ("log10_var",),
        },
    ]


def fit_candidates(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """후보별 GroupKFold MAPE와, train 80% 재학습 뒤 hold-out MAPE. [D23] [D36]

    비교군은 표를 만들기 위해 같은 방식으로 점수를 낸다. 선택 대상은 메인이 아니다. [D27] [D32]
    """
    rows = []
    fitted = {}
    for spec in _candidate_specs():
        x_train, y_train, groups = _xy(train, spec["columns"])
        x_valid, y_valid, _ = _xy(valid, spec["columns"])
        search = _search(spec["estimator"], spec["param_grid"], x_train, y_train, groups)
        cv_mean, cv_std = _cv_stats(search)
        pred = search.predict(x_valid)
        valid_mape = mape_original(y_valid, pred)
        fitted[spec["model"]] = search
        rows.append({
            "model": spec["model"],
            "role": spec["role"],
            "params": _params_text(search.best_params_),
            "best_params": dict(search.best_params_),
            "cv_mape": cv_mean,
            "cv_std": cv_std,
            "valid_mape": valid_mape,
            "search": search,
        })
    table = pd.DataFrame(rows)
    table.attrs["fitted"] = fitted
    return table


def select_final(table: pd.DataFrame) -> dict:
    """메인 중에서만 고른다. GroupKFold 평균이 1순위, 0.5%p 이내면 Valid가 낮은 쪽. [D23] [D31]

    Valid MAPE만으로 결정하지 않는다. 비교군은 후보에서 뺀다. [D27]
    """
    main = table.loc[table["role"] == "메인"].copy()
    main = main.sort_values(["cv_mape", "valid_mape", "model"]).reset_index(drop=True)
    best_cv = float(main.loc[0, "cv_mape"])
    close = main.loc[main["cv_mape"] - best_cv <= CV_TIE_MAPE].copy()
    if len(close) == 1:
        chosen = close.iloc[0]
        reason = (
            f"GroupKFold 평균 MAPE가 가장 낮은 메인은 {chosen['model']} "
            f"({best_cv:.3f}%). 0.5%p 이내의 다른 메인은 없다."
        )
    else:
        close = close.sort_values(["valid_mape", "cv_mape", "model"]).reset_index(drop=True)
        chosen = close.iloc[0]
        names = ", ".join(
            f"{row.model} CV {row.cv_mape:.3f}% / Valid {row.valid_mape:.4f}%"
            for row in close.itertuples()
        )
        reason = (
            f"GroupKFold 1위와 0.5%p 이내인 메인이 있어 Valid MAPE가 낮은 쪽을 골랐다. "
            f"대상: {names}. 선택: {chosen['model']}."
        )
    return {
        "model": chosen["model"],
        "params": chosen["params"],
        "best_params": chosen["best_params"],
        "cv_mape": float(chosen["cv_mape"]),
        "cv_std": float(chosen["cv_std"]),
        "valid_mape": float(chosen["valid_mape"]),
        "reason": reason,
        "estimator": chosen["search"].best_estimator_,
    }


def main_alpha_curves(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """메인 모델의 alpha별 GroupKFold 평균 MAPE. [D35]

    ElasticNet은 alpha마다 MAPE가 가장 낮은 l1_ratio 한 점만 남긴다.
    train 분할 안에서 이미 fit한 GridSearchCV 결과를 읽는다. Batch 2·3는 쓰지 않는다.
    """
    curves = {}
    for row in table.itertuples(index=False):
        if row.role != "메인":
            continue
        result = row.search.cv_results_
        frame = pd.DataFrame({
            "alpha": [params["model__alpha"] for params in result["params"]],
            "l1_ratio": [params.get("model__l1_ratio") for params in result["params"]],
            "cv_mape": -result["mean_test_score"],
            "cv_std": result["std_test_score"],
        })
        if frame["l1_ratio"].notna().any():
            pick = frame.groupby("alpha", sort=True)["cv_mape"].idxmin()
            frame = frame.loc[pick]
        curves[row.model] = frame.sort_values("alpha").reset_index(drop=True)
    return curves


def holdout_predictions(estimator, valid: pd.DataFrame) -> pd.DataFrame:
    """hold-out 셀별 수명과 오차%. 오차% = (예측−실제)/실제×100. [D36]"""
    x, _, _ = _xy(valid, MODEL_FEATURES)
    pred_life = np.power(10.0, estimator.predict(x))
    actual = valid["cycle_life"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "cell_id": valid["cell_id"].to_numpy(),
        "policy": valid["policy_readable"].to_numpy(),
        "실제_수명": actual,
        "예측_수명": pred_life,
        "오차%": (pred_life - actual) / actual * 100.0,
    })
    return out.sort_values("cell_id").reset_index(drop=True)


def model_coefficients(estimator) -> pd.Series:
    """스케일링 뒤 선형 계수. 절편은 intercept. [D09] [D10]"""
    coef = np.asarray(estimator.named_steps["model"].coef_, dtype=float).ravel()
    names = list(MODEL_FEATURES)
    if len(coef) != len(names):
        raise ValueError("최종 모델 계수 수가 피처 2개와 다르다.")
    series = pd.Series(coef, index=names)
    series.loc["intercept"] = float(estimator.named_steps["model"].intercept_)
    return series


def fold_coefficients(estimator, train: pd.DataFrame) -> pd.DataFrame:
    """잠근 하이퍼파라미터로 GroupKFold 5폴드 계수. [D23] [D32]

    폴드마다 다시 튜닝하지 않는다. 스케일러는 그 폴드의 학습 셀에서만 fit한다. [D21]
    """
    x, y, groups = _xy(train, MODEL_FEATURES)
    rows = []
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, te) in enumerate(splitter.split(x, y, groups), start=1):
        pipe = clone(estimator)
        pipe.fit(x.iloc[tr], y.iloc[tr])
        coef = np.asarray(pipe.named_steps["model"].coef_, dtype=float).ravel()
        rows.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_valid_fold": int(len(te)),
            "log10_var": float(coef[0]),
            "qd_max_minus_2": float(coef[1]),
        })
    return pd.DataFrame(rows)


def save_final_model(estimator, path=None) -> str:
    """P10은 이 파일만 불러 평가한다. train 80%에 재학습된 파이프라인이다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULTS_DIR / "final_model.joblib" if path is None else path
    dump(estimator, target)
    return str(target)


# P9에서 잠근 비교군 하이퍼파라미터. P10에서 다시 고르지 않는다. [D27] [D32]
LOCKED_TRAIN_MAPE = 6.657
LOCKED_VALID_MAPE = 4.888
PAPER_TARGET_MAPE = 9.1
LIFE_SHORT = 534
LIFE_LONG = 1074
LIFE_BANDS = ("<534", "534~1,074", ">1,074")


def load_final_model():
    """results/final_model.joblib. 평가 전에 불러 두고 Batch 2로 다시 맞추지 않는다. [D24] [D36]"""
    path = RESULTS_DIR / "final_model.joblib"
    model = load(path)
    name = type(model.named_steps["model"]).__name__
    if name != "Ridge":
        raise RuntimeError(f"잠긴 모델이 Ridge가 아니다: {name}")
    return model


def fit_locked_comparisons(train: pd.DataFrame) -> dict:
    """P9 하이퍼파라미터로 train 셀에만 비교군을 다시 맞춘다. [D27] [D29] [D32]

    DecisionTree max_depth=1, min_samples_leaf=3.
    XGBRegressor max_depth=2, n_estimators=50, learning_rate=0.05.
    단일 피처는 log10(var(ΔQ))만 쓴 LinearRegression.
    """
    specs = {
        "DecisionTree": (
            DecisionTreeRegressor(max_depth=1, min_samples_leaf=3, random_state=RANDOM_STATE),
            MODEL_FEATURES,
            "max_depth=1, min_samples_leaf=3",
        ),
        "XGBoost": (
            XGBRegressor(
                max_depth=2,
                n_estimators=50,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            MODEL_FEATURES,
            "max_depth=2, n_estimators=50, learning_rate=0.05",
        ),
        "LinearRegression(log10_var)": (
            LinearRegression(),
            ("log10_var",),
            "피처 log10_var 하나",
        ),
    }
    fitted = {}
    for name, (estimator, columns, text) in specs.items():
        pipe = make_pipeline(estimator)
        x, y, _ = _xy(train, columns)
        pipe.fit(x, y)
        fitted[name] = {"pipe": pipe, "columns": columns, "params": text}
    return fitted


def life_band(cycle_life: float) -> str:
    """학습 범위 기준 세 구간. [D29]"""
    if cycle_life < LIFE_SHORT:
        return "<534"
    if cycle_life > LIFE_LONG:
        return ">1,074"
    return "534~1,074"


def range_side(value: float, low: float, high: float) -> str:
    """train 범위 기준 위/아래/안. 경계는 안. [설계 8장]"""
    if value < low:
        return "아래"
    if value > high:
        return "위"
    return "안"


def _round3(value: float) -> float:
    return float(f"{value:.3f}")


def performance_table(test_mape: float) -> pd.DataFrame:
    """가이드 리포팅 표. Train·Valid는 잠긴 소수 셋째 자리. [D33] [D37]

    Gap은 표에 적는 소수 셋째 자리 MAPE로 계산한다. Gap(A-B)=B−A.
    """
    train_mape = LOCKED_TRAIN_MAPE
    valid_mape = LOCKED_VALID_MAPE
    test_mape = _round3(test_mape)
    counts = "29 / 7 / 39"
    rows = [
        ("Train (Batch 1 CV)", train_mape, f"{counts}. GroupKFold 평균. 학습 셀 29"),
        ("Valid (Batch 1 Hold-out)", valid_mape, f"{counts}. hold-out 셀 7"),
        ("Test (Batch 2)", test_mape, f"{counts}. 테스트 셀 39"),
        ("Gap (Train-Valid)", _round3(valid_mape - train_mape), f"{counts}. (+) : 과적합 의심"),
        ("Gap (Valid-Test)", _round3(test_mape - valid_mape), f"{counts}. (+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", _round3(test_mape - PAPER_TARGET_MAPE), f"{counts}. Target : 원논문 9.1%"),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"])


def build_eval_predictions(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    ridge,
    comparisons: dict,
    batch: int,
) -> pd.DataFrame:
    """이미 읽어 둔 배치 표에 네 모델 예측을 한 번씩 붙인다. [D24] [D25]

    이 함수는 파일을 열지 않는다. 오차% = (예측−실제)/실제×100.
    """
    if not (frame["batch"] == int(batch)).all():
        raise ValueError(f"Batch {batch}만 평가한다.")
    actual = frame["cycle_life"].to_numpy(dtype=float)
    ridge_life = np.power(10.0, ridge.predict(frame.loc[:, list(MODEL_FEATURES)]))
    signed = (ridge_life - actual) / actual * 100.0
    out = pd.DataFrame({
        "cell_id": frame["cell_id"].to_numpy(),
        "policy_readable": frame["policy_readable"].to_numpy(),
        "cycle_life": actual,
        "수명_구간": [life_band(value) for value in actual],
        "log10_var": frame["log10_var"].to_numpy(dtype=float),
        "qd_max_minus_2": frame["qd_max_minus_2"].to_numpy(dtype=float),
    })
    for column in MODEL_FEATURES:
        low = float(train[column].min())
        high = float(train[column].max())
        out[f"{column}_범위"] = [range_side(value, low, high) for value in out[column]]
    out["Ridge_예측"] = ridge_life
    out["오차%"] = signed
    out["|오차%|"] = np.abs(signed)
    pred_names = {
        "DecisionTree": "DecisionTree_예측",
        "XGBoost": "XGBoost_예측",
        "LinearRegression(log10_var)": "log10_var_단일_예측",
    }
    for key, column in pred_names.items():
        pipe = comparisons[key]["pipe"]
        cols = comparisons[key]["columns"]
        out[column] = np.power(10.0, pipe.predict(frame.loc[:, list(cols)]))
    return out.sort_values("cell_id").reset_index(drop=True)


def build_batch2_predictions(batch2: pd.DataFrame, train: pd.DataFrame, ridge, comparisons: dict) -> pd.DataFrame:
    """이미 읽어 둔 Batch 2 표에 네 모델 예측을 한 번씩 붙인다. [D24]"""
    return build_eval_predictions(batch2, train, ridge, comparisons, batch=2)


def append_batch3_performance(test_mape: float, n_cells: int, path=None) -> pd.DataFrame:
    """Batch 2 리포팅 6행은 글자 그대로 두고 Batch 3 행만 뒤에 붙인다. [D25] [D39]

    Gap(Batch2-Batch3) = Batch 3 Test − Batch 2 Test. Gap(A-B)=B−A.
    """
    target = RESULTS_DIR / "model_performance.csv" if path is None else path
    lines = target.read_text().splitlines()
    if len(lines) < 7:
        raise RuntimeError("Batch 2 리포팅 6행이 없다.")
    header, body = lines[0], lines[1:7]
    batch2_line = next(line for line in body if line.startswith("Test (Batch 2),"))
    batch2_mape = float(batch2_line.split(",")[1])
    batch3_mape = _round3(test_mape)
    rows = [
        ["Test (Batch 3)", f"{batch3_mape:.3f}", f"테스트 셀 {int(n_cells)}"],
        ["Gap (Batch2-Batch3)", f"{_round3(batch3_mape - batch2_mape):.3f}", "테스트 배치 간 비교"],
        ["Gap (Target-Test, Batch 3)", f"{_round3(batch3_mape - PAPER_TARGET_MAPE):.3f}", "Target : 원논문 9.1%"],
    ]
    extra = []
    for row in rows:
        buffer = StringIO()
        csv.writer(buffer, lineterminator="").writerow(row)
        extra.append(buffer.getvalue())
    target.write_text("\n".join([header, *body, *extra]) + "\n")
    return pd.read_csv(target)


def _mape_from_prediction(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(predicted.to_numpy(float) - actual.to_numpy(float)) / actual.to_numpy(float)) * 100.0)


def life_band_table(pred: pd.DataFrame) -> pd.DataFrame:
    """수명 구간별 Ridge와 비교군 MAPE. 비교군은 참고용이다. [D29] [D30]"""
    rows = []
    compare = (
        ("DecisionTree_예측", "DecisionTree MAPE"),
        ("XGBoost_예측", "XGBoost MAPE"),
        ("log10_var_단일_예측", "log10_var 단일 MAPE"),
    )
    for band in LIFE_BANDS:
        part = pred.loc[pred["수명_구간"] == band]
        row = {
            "수명_구간": band,
            "셀_수": int(len(part)),
            "Ridge_MAPE": float(part["|오차%|"].mean()) if len(part) else float("nan"),
            "Ridge_평균_부호_오차%": float(part["오차%"].mean()) if len(part) else float("nan"),
            "과대예측_셀_수": int((part["오차%"] > 0).sum()),
        }
        for column, label in compare:
            row[label] = _mape_from_prediction(part["cycle_life"], part[column]) if len(part) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def feature_range_table(pred: pd.DataFrame) -> pd.DataFrame:
    """train 29셀 범위 밖 셀 수와 그 셀의 Ridge MAPE. [설계 8장]"""
    rows = []
    for column in MODEL_FEATURES:
        side = pred[f"{column}_범위"]
        for label in ("아래", "위", "안"):
            part = pred.loc[side == label]
            rows.append({
                "피처": column,
                "위치": label,
                "셀_수": int(len(part)),
                "Ridge_MAPE": float(part["|오차%|"].mean()) if len(part) else float("nan"),
            })
    return pd.DataFrame(rows)
