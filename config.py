from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"
LOG_DIR = OUTPUT_DIR / "logs"

for d in [MODEL_DIR, RESULT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -- Data --
RAW_PRICE_CSV = DATA_DIR / "yfinance_sp500.csv"
MERGED_CSV = DATA_DIR / "merged_sp500_dataset.csv"
FINAL_FEATURES_CSV = DATA_DIR / "final_dataset_for_modeling.csv"

DENOISED_CSV = OUTPUT_DIR / "denoised_sp500.csv"
FEATURES_CSV = OUTPUT_DIR / "features_directional.csv"

DATE_COL = "Date"
PRICE_COLS = ["Open", "High", "Low", "Close"]
VOLUME_COL = "Volume"

# -- Wavelet Denoising --
WAVELET_FAMILY = "db8"
WAVELET_LEVEL = 4
WAVELET_MODE = "soft"
THRESHOLD_METHOD = "universal"    # or "sure"
MAD_SCALE = 0.6745                # std of normal for MAD->sigma conversion
WAVELET_CAUSAL_LOOKBACK = 256     # causal window: only use past data for denoising

# -- Feature Engineering --
MA_WINDOWS = [7, 14, 30, 60]
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_STD = 2
ATR_WINDOW = 14
VOLATILITY_WINDOW = 14
STOCH_WINDOW = 14
RETURN_LAG_WINDOWS = [1, 2, 3, 5, 7, 14, 21]

# -- Modelling --
SEQUENCE_LENGTH = 60               # lookback days
PREDICTION_HORIZON = 1
TARGET_COL = "target_direction"

# Walk-forward chronological split
TRAIN_START_YEAR = 2008
VAL_WINDOW_YEARS = 1
TEST_START_YEAR = 2022

# -- xLSTM-TS Model --
D_MODEL = 48
N_BLOCKS = 2                       # alternating mLSTM / sLSTM blocks
NUM_HEADS = 4
EXPAND_FACTOR = 2                  # inner MLP ratio in mLSTM
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10                      # early stopping
GRAD_CLIP = 1.0

# -- Random Seed --
SEED = 42
