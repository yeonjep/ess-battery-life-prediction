"""셀 단위 피처.

요약값 스파이크 보정은 셀마다 앞뒤 사이클만 보는 고정 규칙이다.
학습 집합에서 임계값을 추정하지 않으므로 배치를 나눠도 누수가 없다.

Qdlin 인덱스: 저장 배열의 행 k = 사이클 번호 k+1.
사이클 10 = 행 9, 사이클 100 = 행 99.
ΔQ(V) = Qdlin[행 99] − Qdlin[행 9].
"""

from __future__ import annotations

import pickle
import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.config import DATA_PROCESSED_DIR

# 5.4C(50%)-3.6C 또는 끝에 -newstructure. C-rate·SOC는 소수 가능.
_POLICY_PATTERN = re.compile(
    r"^(?P<c1>\d+(?:\.\d+)?)C\((?P<soc>\d+(?:\.\d+)?)%\)-(?P<c2>\d+(?:\.\d+)?)C"
    r"(?:-newstructure)?$"
)

# 보정 대상은 피처에 쓰는 초기 100사이클. 사이클 1의 QD=0은 Batch 1의 빈 사이클이라
# 앞뒤가 서로 멀어 이 규칙에 걸리지 않는다.
FEATURE_CYCLE_MIN = 2
FEATURE_CYCLE_MAX = 100
SUMMARY_VALUE_COLS = ("QD", "QC", "IR", "Tmax", "Tavg", "Tmin", "chargetime")

# (가운데가 앞뒤 중앙값에서 벗어나는 최소 크기, 앞뒤가 서로 가깝다고 볼 최대 차이)
# 단위는 각 열의 원래 단위. 데이터로 맞추지 않은 고정값이다.
SPIKE_RULES: dict[str, tuple[float, float]] = {
    "QD": (0.20, 0.05),
    "QC": (0.20, 0.05),
    "IR": (0.005, 0.001),
    "Tmax": (5.0, 2.0),
    "Tavg": (5.0, 2.0),
    "Tmin": (5.0, 2.0),
    "chargetime": (30.0, 1.0),
}

# 행 k = 사이클 k+1
CYCLE_10_ROW = 9
CYCLE_100_ROW = 99
QDLIN_LEN = 1000


def correct_summary_spikes(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`qd_max_minus_2` 전처리. [설계 2장] [D07] [D10]

    초기 100사이클만 사용. cycle_life나 EOL 이후 정보는 쓰지 않는다. [D03]
    단발 스파이크만 앞뒤 중앙값으로 바꾼다. 앞뒤 허용 차이는 DAY 1 코드값
    (IR 0.001Ω, 온도 2°C, chargetime 1)을 유지한다.
    두 사이클 연속 스파이크는 앞뒤가 서로 멀어 바꾸지 않는다. pkl은 수정하지 않는다.
    """
    corrected = summary.copy()
    logs: list[dict] = []
    for cell_id, index in corrected.groupby("cell_id", sort=False).groups.items():
        block = corrected.loc[index].sort_values("cycle")
        positions = block.index.to_numpy()
        cycles = block["cycle"].to_numpy(dtype=float)
        for column, (min_dev, max_neighbor_gap) in SPIKE_RULES.items():
            values = block[column].to_numpy(dtype=float).copy()
            for i in range(1, len(values) - 1):
                cycle = cycles[i]
                if cycle < FEATURE_CYCLE_MIN or cycle > FEATURE_CYCLE_MAX:
                    continue
                if cycles[i - 1] != cycle - 1 or cycles[i + 1] != cycle + 1:
                    continue
                left, mid, right = values[i - 1], values[i], values[i + 1]
                if not np.isfinite([left, mid, right]).all():
                    continue
                if abs(left - right) > max_neighbor_gap:
                    continue
                neighbor_median = float(np.median([left, right]))
                if abs(mid - neighbor_median) <= min_dev:
                    continue
                values[i] = neighbor_median
                logs.append(
                    {
                        "cell_id": cell_id,
                        "batch": int(block["batch"].iloc[0]),
                        "cycle": int(cycle),
                        "column": column,
                        "old": float(mid),
                        "new": neighbor_median,
                    }
                )
            corrected.loc[positions, column] = values
    return corrected, pd.DataFrame(logs)


def delta_q(qdlin: np.ndarray) -> np.ndarray:
    """피처명 `log10(var(ΔQ))`의 ΔQ. [설계 2장] [설계 4장] [D08] [D09]

    ΔQ = Qdlin[행 99] − Qdlin[행 9] (사이클 100 − 10).
    초기 100사이클만 사용. cycle_life로 피처를 만들지 않는다. [D03]
    """
    if qdlin.shape != (100, QDLIN_LEN):
        raise ValueError(f"Qdlin shape는 (100, 1000)이어야 한다. 받음: {qdlin.shape}")
    return np.asarray(qdlin[CYCLE_100_ROW] - qdlin[CYCLE_10_ROW], dtype=float)


def delta_q_features(dq: np.ndarray, vdlin: np.ndarray) -> dict[str, float]:
    """피처명 `log10(var(ΔQ))`. [설계 4장] [D09] [D40]

    1,000점 분산은 `np.var(ddof=1)` 뒤 log10. 이 피처의 전처리는 없다.
    초기 100사이클만 사용. [D03]
    `log10_abs_min`, 2V, skew, kurtosis는 계산만 하고 모델 입력에서 뺀다. [D12] [D13]
    """
    voltage_index = int(np.argmin(np.abs(np.asarray(vdlin, dtype=float) - 2.0)))
    variance = float(np.var(dq, ddof=1))
    minimum = float(np.min(dq))
    return {
        "dq_min": minimum,
        "dq_mean": float(np.mean(dq)),
        "dq_var": variance,
        "dq_skew": float(skew(dq, bias=False)),
        "dq_kurtosis": float(kurtosis(dq, bias=False, fisher=True)),
        "dq_at_2v": float(dq[voltage_index]),
        "log10_abs_min": float(np.log10(abs(minimum))) if minimum != 0 else np.nan,
        "log10_var": float(np.log10(variance)) if variance > 0 else np.nan,
    }


def parse_policy(policy: str) -> dict[str, float | bool | str]:
    """충전 정책 문자열에서 C1, 전환 SOC(%), C2, 평균 C-rate를 뽑는다.

    평균 C-rate는 같은 용량을 두 단계로 채울 때의 등가 속도다.
    s = SOC/100 일 때 1 / (s/C1 + (1-s)/C2). 용량 가중 산술평균이 아니다.
    newstructure 접미사는 전류식과 분리한다. 파싱 실패 시 policy_ok가 False다.
    """
    text = str(policy).strip()
    newstructure = text.endswith("-newstructure")
    match = _POLICY_PATTERN.fullmatch(text)
    if match is None:
        return {
            "policy_ok": False,
            "c1": np.nan,
            "soc_pct": np.nan,
            "c2": np.nan,
            "c_avg": np.nan,
            "newstructure": newstructure,
            "policy_base": text,
        }
    c1 = float(match.group("c1"))
    soc_pct = float(match.group("soc"))
    c2 = float(match.group("c2"))
    soc_fraction = soc_pct / 100.0
    c_avg = 1.0 / (soc_fraction / c1 + (1.0 - soc_fraction) / c2)
    base = text[: -len("-newstructure")] if newstructure else text
    return {
        "policy_ok": True,
        "c1": c1,
        "soc_pct": soc_pct,
        "c2": c2,
        "c_avg": c_avg,
        "newstructure": newstructure,
        "policy_base": base,
    }


def parse_policies(policies: pd.Series) -> pd.DataFrame:
    """정책 시리즈를 셀 순서 그대로 피처 표로 만든다."""
    return pd.DataFrame([parse_policy(value) for value in policies], index=policies.index)


def early_chargetime_mean(summary: pd.DataFrame, n_cycles: int = 5) -> pd.Series:
    """셀별 실제 충전 처음 n회의 chargetime 평균.

    Batch 1 사이클 1은 chargetime이 0인 빈 사이클이라 빼기 위해, chargetime > 0인
    사이클만 시간순으로 n개 평균한다. summary는 스파이크 보정 후를 넘긴다.
    """
    usable = summary.loc[summary["chargetime"] > 0, ["cell_id", "cycle", "chargetime"]]
    usable = usable.sort_values(["cell_id", "cycle"])
    first = usable.groupby("cell_id", sort=False).head(n_cycles)
    return first.groupby("cell_id", sort=False)["chargetime"].mean()


def _linear_fit(block: pd.DataFrame, value_column: str) -> tuple[float, float]:
    """사이클에 대한 직선 기울기와 절편. 점이 둘 미만이면 NaN."""
    x = block["cycle"].to_numpy(dtype=float)
    y = block[value_column].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return float(slope), float(intercept)


def qd_window_slope(
    summary: pd.DataFrame,
    cycle_min: int = 2,
    cycle_max: int = 100,
) -> pd.Series:
    """셀별 QD의 사이클 직선 기울기(Ah/사이클). summary는 스파이크 보정 후를 넘긴다."""
    window = summary.loc[summary["cycle"].between(cycle_min, cycle_max), ["cell_id", "cycle", "QD"]]
    slopes = {}
    for cell_id, block in window.groupby("cell_id", sort=False):
        slopes[cell_id] = _linear_fit(block, "QD")[0]
    return pd.Series(slopes, name="qd_slope")


def _value_at_cycle(block: pd.DataFrame, cycle: int, column: str) -> float:
    chosen = block.loc[block["cycle"] == cycle, column]
    if chosen.empty or not np.isfinite(chosen.iloc[0]):
        return np.nan
    return float(chosen.iloc[0])


def summary_features(summary: pd.DataFrame) -> pd.DataFrame:
    """피처명 `qd_max_minus_2`. [설계 4장] [D10] [D40]

    스파이크 보정 후 사이클 2~100 QD 최댓값 − 사이클 2 QD.
    초기 100사이클만 사용. cycle_life로 피처를 만들지 않는다. [D03]
    기울기·절편·사이클 2 QD·온도·IR은 계산만 하고 모델 입력에서 뺀다. [D14] [D18]
    """
    rows = []
    for cell_id, block in summary.groupby("cell_id", sort=False):
        block = block.sort_values("cycle")
        early = block.loc[block["cycle"].between(2, 100)]
        late = block.loc[block["cycle"].between(91, 100)]
        slope_early, intercept_early = _linear_fit(early, "QD")
        slope_late, intercept_late = _linear_fit(late, "QD")
        qd_cycle2 = _value_at_cycle(block, 2, "QD")
        qd_cycle10 = _value_at_cycle(block, 10, "QD")
        finite_qd = early.loc[np.isfinite(early["QD"]), "QD"]
        rows.append(
            {
                "cell_id": cell_id,
                "slope_2_100": slope_early,
                "intercept_2_100": intercept_early,
                "slope_91_100": slope_late,
                "intercept_91_100": intercept_late,
                "qd_cycle2": qd_cycle2,
                "qd_cycle10": qd_cycle10,
                "qd_max_minus_2": float(finite_qd.max() - qd_cycle2) if len(finite_qd) else np.nan,
                "early_ct": np.nan,
                "tmax_integral": float(early["Tmax"].sum()),
                "tavg_integral": float(early["Tavg"].sum()),
                "ir_min": float(early["IR"].min()),
                "ir_diff_100_2": _value_at_cycle(block, 100, "IR") - _value_at_cycle(block, 2, "IR"),
            }
        )
    table = pd.DataFrame(rows).set_index("cell_id")
    table["early_ct"] = early_chargetime_mean(summary)
    return table


# 모델 입력은 두 열만. [설계 4장] [D34] [D41]
# 아래는 계산 함수에 남기고 모델 입력에서만 뺀다.
# log10_abs_min: var와 r=0.996, 잔차 0.036, VIF 689·711 [D12]
# dq_at_2v, dq_skew, dq_kurtosis: 2V는 var와 겹치고 skew는 Batch 3에서 부호 반전 [D13]
# slope·intercept·qd_cycle2: 절편과 사이클 2 QD는 r=0.999, 기울기 부호는 Batch 2에서 반전 [D14]
# knee: 수명 78% 지점이라 누수. 피처 함수 없음 [D15]
# c1, soc_pct, c2, c_avg, early_ct: 테스트 배치에서 상관이 사라짐 [D16]
# log10_var_norm, 배치 번호: 깊이 비 1.30→1.29, 배치 효과는 Batch 1만으로 추정 불가 [D17]
# tmax_integral, tavg_integral, ir_min, ir_diff_100_2: 추가 신호가 아님 [D18]
MODEL_FEATURES = ("log10_var", "qd_max_minus_2")
# 비교용 계산만. 메인 모델 입력 아님. [D16]
COMPARE_FEATURES = ("c1", "soc_pct", "c2", "c_avg", "early_ct")


def variance_inflation_factors(frame: pd.DataFrame) -> pd.Series:
    """각 열을 나머지 열로 회귀한 VIF. 절편을 포함한다."""
    values = frame.to_numpy(dtype=float)
    scores = {}
    for index, name in enumerate(frame.columns):
        target = values[:, index]
        others = np.delete(values, index, axis=1)
        design = np.column_stack([np.ones(len(target)), others])
        coefficient, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = design @ coefficient
        residual_ss = float(np.sum((target - fitted) ** 2))
        total_ss = float(np.sum((target - target.mean()) ** 2))
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan
        if not np.isfinite(r_squared) or r_squared >= 1.0 - 1e-12:
            scores[name] = np.inf
        else:
            scores[name] = 1.0 / (1.0 - r_squared)
    return pd.Series(scores, name="vif").sort_values(ascending=False)


def normalized_delta_q(dq: np.ndarray, qd_cycle10: float) -> np.ndarray:
    """ΔQ(V)를 사이클 10 방전용량(Ah)으로 나눈다. 용량이 0이면 NaN."""
    if not np.isfinite(qd_cycle10) or qd_cycle10 == 0:
        return np.full_like(dq, np.nan, dtype=float)
    return np.asarray(dq, dtype=float) / qd_cycle10


def build_feature_table(batch: int) -> pd.DataFrame:
    """배치 하나의 모델 입력 표. 피처명 `log10(var(ΔQ))`, `qd_max_minus_2`. [설계 4장] [D02] [D34] [D40] [D41]

    1행 = 1셀, `use=True`만. 초기 100사이클만 사용. [D03] [D04]
    열은 cell_id, batch, policy_readable, cycle_life, log10_var, qd_max_minus_2.
    정규화 ΔQ 등 배제 피처 열은 넣지 않는다. [D17]
    pkl 원본은 바꾸지 않는다.
    """
    path = DATA_PROCESSED_DIR / f"batch{int(batch)}.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    cells = payload["cells"]
    cells = cells.loc[cells["use"]].reset_index(drop=True)
    cell_ids = set(cells["cell_id"])
    summary = payload["summary"]
    summary = summary.loc[
        summary["cell_id"].isin(cell_ids) & summary["cycle"].between(1, 100)
    ].copy()
    corrected, _logs = correct_summary_spikes(summary)
    base = build_delta_q_table(cells, payload["qdlin"], payload["vdlin"])
    extra = summary_features(corrected)[["qd_max_minus_2"]]
    base = base.join(extra, on="cell_id")
    columns = ["cell_id", "batch", "policy_readable", "cycle_life", "log10_var", "qd_max_minus_2"]
    return base[columns].reset_index(drop=True)


def build_delta_q_table(
    cells: pd.DataFrame,
    qdlin_by_cell: dict[str, np.ndarray],
    vdlin: np.ndarray,
) -> pd.DataFrame:
    """use 여부와 무관하게, 넘긴 셀 표의 ΔQ 피처를 만든다."""
    rows = []
    for record in cells.itertuples(index=False):
        grid = qdlin_by_cell[record.cell_id]
        dq = delta_q(grid)
        row = {
            "cell_id": record.cell_id,
            "batch": int(record.batch),
            "cycle_life": float(record.cycle_life),
            "policy_readable": record.policy_readable,
            "newstructure": "newstructure" in str(record.policy_readable),
            "qdlin_10_ok": bool(np.isfinite(grid[CYCLE_10_ROW]).all()),
            "qdlin_100_ok": bool(np.isfinite(grid[CYCLE_100_ROW]).all()),
        }
        row.update(delta_q_features(dq, vdlin))
        rows.append(row)
    return pd.DataFrame(rows)
