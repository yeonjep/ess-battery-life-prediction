"""원본 .mat에서 필요한 필드만 h5py로 읽어 data/processed/batch{N}.pkl로 저장한다.

mat73은 파일 전체를 메모리에 올리므로 사용하지 않는다.
varcharge 파일은 BATCH_MAT_FILES에 없으며 읽지 않는다.

Qdlin 인덱스 규칙 (Batch 1 셀 b1c0으로 확인, Batch 2/3도 같은 스키마):
- cycles/Qdlin 은 (n, 1) object reference 이고, 행 순서는 summary와 같다.
- summary.cycle 은 1부터 시작한다. 행 k 의 사이클 번호는 summary.cycle[k] 이다.
- Batch 1의 사이클 1(행 0)은 빈 참조(uint64 길이 2)이고 summary QD도 0이다.
- Batch 2/3의 사이클 1(행 0)은 길이 1000인 실제 Qdlin 이다.
- 저장 배열의 r행 = 사이클 번호 r+1. 빈 참조·길이 불일치는 NaN 행으로 둔다.
- Vdlin 은 셀마다 같고, 이 파일에서는 3.5V → 2.0V, 1000점이다.
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BATCH_MAT_FILES, DATA_PROCESSED_DIR

# 공칭 1.1Ah의 80%는 0.88Ah다.
# 수명이 기록된 셀의 최저 QD는 0.880~0.885에 몰리고, 미도달 셀은 0.913 이상이다.
# 0.88 이하만 도달로 보면 정상 종료 셀까지 미도달로 잡히므로, 공백이 있는 0.90을 기준으로 본다.
NOT_REACHED_AH = 0.90
SUMMARY_FIELDS = (
    ("QDischarge", "QD"),
    ("QCharge", "QC"),
    ("IR", "IR"),
    ("Tmax", "Tmax"),
    ("Tavg", "Tavg"),
    ("Tmin", "Tmin"),
    ("chargetime", "chargetime"),
    ("cycle", "cycle"),
)
N_QDLIN_CYCLES = 100
QDLIN_LEN = 1000

# 셀은 삭제하지 않는다. use=False 인 셀만 이후 분석에서 뺀다.
EXCLUDE_REASON = {
    **{
        f"b1c{i}": "80% 미도달. 저장된 cycle_life가 EOL이 아님 (연속 실험 아님)"
        for i in range(5)
    },
    **{cell_id: "80% 미도달" for cell_id in ("b1c8", "b1c10", "b1c12", "b1c13", "b1c22")},
    **{
        cell_id: "cycle_life 결측"
        for cell_id in (
            "b2c22",
            "b2c23",
            "b2c35",
            "b2c36",
            "b2c37",
            "b2c38",
            "b2c39",
            "b2c40",
            "b3c23",
            "b3c32",
        )
    },
}


def mark_use(cells: pd.DataFrame) -> pd.DataFrame:
    """use와 exclude_reason을 붙인다. 제외 셀의 행은 남긴다."""
    out = cells.copy()
    out["exclude_reason"] = out["cell_id"].map(EXCLUDE_REASON).fillna("")
    out["use"] = out["exclude_reason"].eq("")
    return out


def _decode_matlab_char(arr: np.ndarray) -> str:
    """MATLAB char 열벡터(uint16)를 문자열로 디코딩한다."""
    raw = np.asarray(arr).ravel()
    return "".join(chr(int(c)) for c in raw if int(c) != 0)


def _as_1d_float(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).ravel()


def _scalar(arr: np.ndarray) -> float:
    values = _as_1d_float(arr)
    if values.size == 0 or not np.isfinite(values[0]):
        return float("nan")
    return float(values[0])


def _read_qdlin_vector(h5: h5py.File, ref: h5py.Reference) -> np.ndarray | None:
    """길이 1000인 Qdlin만 반환한다. 빈 참조는 None."""
    arr = np.asarray(h5[ref][()])
    if arr.dtype.kind != "f" or arr.size != QDLIN_LEN:
        return None
    return arr.reshape(-1)


def load_batch(batch_id: int) -> dict:
    """배치 하나의 cells, summary, qdlin, vdlin을 만든다. cycle_life는 원본 그대로 둔다."""
    path = BATCH_MAT_FILES[batch_id]
    cell_rows: list[dict] = []
    summary_parts: list[pd.DataFrame] = []
    qdlin: dict[str, np.ndarray] = {}
    vdlin: np.ndarray | None = None
    n_vdlin_mismatch = 0

    with h5py.File(path, "r") as h5:
        batch = h5["batch"]
        n_cells = int(batch["cycle_life"].shape[0])
        for i in range(n_cells):
            cell_id = f"b{batch_id}c{i}"
            cycle_life = _scalar(h5[batch["cycle_life"][i, 0]][()])
            policy = _decode_matlab_char(h5[batch["policy_readable"][i, 0]][()])
            summary_group = h5[batch["summary"][i, 0]]
            columns = {
                name: _as_1d_float(summary_group[src][()])
                for src, name in SUMMARY_FIELDS
            }
            n_cycles = int(columns["QD"].size)
            cell_rows.append(
                {
                    "cell_id": cell_id,
                    "batch": batch_id,
                    "cycle_life": cycle_life,
                    "policy_readable": policy,
                    "n_cycles": n_cycles,
                }
            )
            part = pd.DataFrame(columns)
            part.insert(0, "cell_id", cell_id)
            part.insert(1, "batch", batch_id)
            summary_parts.append(part)

            cycle_numbers = columns["cycle"]
            qrefs = h5[batch["cycles"][i, 0]]["Qdlin"]
            grid = np.full((N_QDLIN_CYCLES, QDLIN_LEN), np.nan, dtype=np.float64)
            n_q = int(qrefs.shape[0])
            for cycle_no in range(1, N_QDLIN_CYCLES + 1):
                matches = np.flatnonzero(cycle_numbers == cycle_no)
                if matches.size == 0:
                    continue
                row = int(matches[0])
                if row >= n_q:
                    continue
                vector = _read_qdlin_vector(h5, qrefs[row, 0])
                if vector is not None:
                    grid[cycle_no - 1] = vector
            qdlin[cell_id] = grid

            voltage = _as_1d_float(h5[batch["Vdlin"][i, 0]][()])
            if vdlin is None:
                vdlin = voltage
            elif voltage.shape != vdlin.shape or not np.allclose(vdlin, voltage):
                n_vdlin_mismatch += 1

    if vdlin is None:
        raise RuntimeError(f"Batch {batch_id}: Vdlin을 읽지 못했다.")
    if n_vdlin_mismatch:
        print(f"Batch {batch_id}: Vdlin이 셀마다 다른 경우 {n_vdlin_mismatch}건. 첫 셀 값을 저장한다.")

    return {
        "cells": mark_use(pd.DataFrame(cell_rows)),
        "summary": pd.concat(summary_parts, ignore_index=True),
        "qdlin": qdlin,
        "vdlin": vdlin,
    }


def apply_use_flags(batch_ids: tuple[int, ...] = (1, 2, 3)) -> None:
    """이미 저장된 pkl의 cells에 use, exclude_reason을 다시 붙인다."""
    for batch_id in batch_ids:
        path = DATA_PROCESSED_DIR / f"batch{batch_id}.pkl"
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        payload["cells"] = mark_use(payload["cells"].drop(columns=["use", "exclude_reason"], errors="ignore"))
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        used = int(payload["cells"]["use"].sum())
        print(f"batch {batch_id}: use {used} / {len(payload['cells'])}")
        del payload
        gc.collect()


def load_cells(usable_only: bool = True) -> pd.DataFrame:
    """배치 1·2·3 셀 표를 이어 붙인다. 기본은 use=True만."""
    frames = []
    for batch_id in (1, 2, 3):
        path = DATA_PROCESSED_DIR / f"batch{batch_id}.pkl"
        with path.open("rb") as handle:
            cells = pickle.load(handle)["cells"]
        frames.append(cells)
    out = pd.concat(frames, ignore_index=True)
    if usable_only:
        out = out.loc[out["use"]].reset_index(drop=True)
    return out


def convert_batch(batch_id: int) -> tuple[Path, float, int]:
    """배치 하나를 변환해 저장하고, 메모리에서 내려놓는다."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload = load_batch(batch_id)
    out = DATA_PROCESSED_DIR / f"batch{batch_id}.pkl"
    with out.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - started
    size = out.stat().st_size
    del payload
    gc.collect()
    return out, elapsed, size


def _positive_qd(qd: np.ndarray) -> np.ndarray:
    values = qd[np.isfinite(qd) & (qd > 0)]
    return values


def _load_processed(batch_id: int) -> dict:
    path = DATA_PROCESSED_DIR / f"batch{batch_id}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def audit_batches(batch_ids: tuple[int, ...] = (1, 2, 3)) -> None:
    """변환 결과를 배치별 짧은 표로 출력한다. 원본 .mat는 다시 읽지 않는다."""
    overview = []
    censored_rows = []
    qdlin_rows = []
    known_rows = []
    policies: dict[int, pd.Series] = {}

    for batch_id in batch_ids:
        data = _load_processed(batch_id)
        cells = data["cells"]
        summary = data["summary"]
        qdlin = data["qdlin"]
        life = cells["cycle_life"].to_numpy(dtype=float)
        n_cycles = cells["n_cycles"].to_numpy(dtype=int)
        diff = n_cycles - life
        nan_life = int(np.isnan(life).sum())
        summary_nan = int(summary[["QD", "QC", "IR", "Tmax", "Tavg", "Tmin", "chargetime", "cycle"]].isna().sum().sum())

        close = int(np.sum(np.isfinite(life) & (np.abs(diff) <= 1)))
        off_by_one = int(np.sum(np.isfinite(life) & (diff == -1)))

        first_qd_zero = 0
        not_eol = []
        for row in cells.itertuples(index=False):
            qd = summary.loc[summary["cell_id"] == row.cell_id, "QD"].to_numpy()
            if qd.size and qd[0] == 0:
                first_qd_zero += 1
            positive = _positive_qd(qd)
            min_qd = float(positive.min()) if positive.size else float("nan")
            if (not positive.size) or min_qd > NOT_REACHED_AH:
                not_eol.append(row.cell_id)
                censored_rows.append(
                    {
                        "cell_id": row.cell_id,
                        "cycle_life": row.cycle_life,
                        "n_cycles": row.n_cycles,
                        "min_QD": round(min_qd, 4),
                        "policy": row.policy_readable,
                    }
                )

        missing_cycle_counts = np.zeros(N_QDLIN_CYCLES, dtype=int)
        cells_complete = 0
        for grid in qdlin.values():
            present = np.isfinite(grid).all(axis=1)
            missing_cycle_counts += (~present).astype(int)
            if bool(present.all()):
                cells_complete += 1
        missing_cycles = (np.flatnonzero(missing_cycle_counts) + 1).tolist()

        overview.append(
            {
                "batch": batch_id,
                "n_cells": len(cells),
                "life_min": np.nanmin(life),
                "life_median": float(np.nanmedian(life)),
                "life_max": np.nanmax(life),
                "life_nan": nan_life,
                "summary_nan": summary_nan,
                "n_approx_life": close,
                "n_eq_life_minus_1": off_by_one,
                "first_QD_0": first_qd_zero,
                "not_eol": len(not_eol),
                "qdlin_100_complete": cells_complete,
            }
        )
        qdlin_rows.append(
            {
                "batch": batch_id,
                "cells_missing_any": int(len(cells) - cells_complete),
                "missing_cycle_numbers": missing_cycles if len(missing_cycles) <= 12 else missing_cycles[:12],
                "cells_missing_per_listed_cycle": {
                    int(c): int(missing_cycle_counts[c - 1]) for c in missing_cycles[:12]
                },
            }
        )
        policies[batch_id] = cells.set_index("cell_id")["policy_readable"]

        watched = {
            1: ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4", "b1c8", "b1c10", "b1c12", "b1c13", "b1c22"],
            3: [],
        }.get(batch_id, [])
        for cell_id in watched:
            if cell_id not in set(cells["cell_id"]):
                known_rows.append({"cell_id": cell_id, "present": False})
                continue
            info = cells.loc[cells["cell_id"] == cell_id].iloc[0]
            qd = summary.loc[summary["cell_id"] == cell_id, "QD"].to_numpy()
            positive = _positive_qd(qd)
            known_rows.append(
                {
                    "cell_id": cell_id,
                    "present": True,
                    "cycle_life": info.cycle_life,
                    "n_cycles": int(info.n_cycles),
                    "min_QD": round(float(positive.min()), 4) if positive.size else None,
                    "not_reached": bool((not positive.size) or positive.min() > NOT_REACHED_AH),
                    "policy": info.policy_readable,
                }
            )

        if batch_id == 3:
            jump_rows = []
            for cell_id, group in summary.groupby("cell_id", sort=False):
                qd = _positive_qd(group["QD"].to_numpy())
                ir = group["IR"].to_numpy()
                max_jump = float(np.max(np.abs(np.diff(qd)))) if qd.size > 1 else float("nan")
                jump_rows.append(
                    {
                        "cell_id": cell_id,
                        "max_abs_dQD": max_jump,
                        "max_IR": float(np.nanmax(ir)) if ir.size else float("nan"),
                    }
                )
            jumps = pd.DataFrame(jump_rows)
            limit = jumps["max_abs_dQD"].quantile(0.75) + 1.5 * (
                jumps["max_abs_dQD"].quantile(0.75) - jumps["max_abs_dQD"].quantile(0.25)
            )
            outliers = jumps.loc[jumps["max_abs_dQD"] > limit].sort_values("max_abs_dQD", ascending=False)
            print("\n[Batch 3] QD 연속 차이 IQR 상한 밖 셀 (노이즈 후보, 제외하지 않음)")
            print(f"상한={limit:.4f} Ah, 후보 {len(outliers)}개")
            if outliers.empty:
                print("(없음)")
            else:
                print(outliers.head(15).to_string(index=False))

        del data
        gc.collect()

    print("\n[개요]")
    print(pd.DataFrame(overview).to_string(index=False))
    print("\n[Qdlin 사이클 1~100 결측]")
    for row in qdlin_rows:
        print(row)
    print("\n[최저 QD > 0.90Ah: 80% 미도달]")
    if censored_rows:
        print(pd.DataFrame(censored_rows).to_string(index=False))
    else:
        print("(없음)")
    print("\n[BRIEF에 적힌 Batch 1 셀]")
    if known_rows:
        print(pd.DataFrame(known_rows).to_string(index=False))

    early = [f"b1c{i}" for i in range(5)]
    early_policies = policies[1].reindex(early).dropna()
    print("\n[b1c0~b1c4 정책이 Batch 2에 있는지]")
    b2 = policies[2]
    for cell_id, policy in early_policies.items():
        matched = b2[b2 == policy]
        print(f"{cell_id} policy={policy} -> batch2 {len(matched)}셀 {matched.index.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="배치 .mat을 pkl로 변환하거나 점검한다.")
    parser.add_argument("--batches", nargs="+", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.batches:
        for batch_id in args.batches:
            out, elapsed, size = convert_batch(batch_id)
            print(f"batch {batch_id}: {elapsed:.1f}s, {size / 1e6:.1f} MB, {out}")
            gc.collect()
    if args.report:
        audit_batches()


if __name__ == "__main__":
    main()
