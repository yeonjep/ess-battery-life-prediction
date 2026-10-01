"""프로젝트 공통 상수, 경로, 그래프 폰트."""

from pathlib import Path

import matplotlib.pyplot as plt

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.2
# 학습 타깃은 log10(cycle_life). 저장값과 MAPE는 원래 사이클 수.
USE_LOG_TARGET = True

# 프로젝트 루트에서 실행하는 것을 전제로 한 상대경로
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("results")

BATCH_MAT_FILES = {
    1: DATA_RAW_DIR / "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: DATA_RAW_DIR / "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    3: DATA_RAW_DIR / "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}


def set_korean_font() -> None:
    """그래프 한글을 AppleGothic으로 표시하고, 마이너스 기호가 깨지지 않게 한다."""
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
