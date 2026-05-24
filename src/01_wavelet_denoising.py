import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    RAW_PRICE_CSV, DENOISED_CSV, RESULT_DIR,
    PRICE_COLS, VOLUME_COL, DATE_COL,
    WAVELET_FAMILY, WAVELET_LEVEL, WAVELET_MODE,
    MAD_SCALE, SEED, WAVELET_CAUSAL_LOOKBACK,
)

np.random.seed(SEED)


def universal_threshold(coeffs):
    sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / MAD_SCALE
    n = len(np.concatenate(coeffs))
    return sigma * np.sqrt(2 * np.log(n))


def causal_wavelet_denoise_series(series, wavelet=WAVELET_FAMILY, level=WAVELET_LEVEL,
                                   mode=WAVELET_MODE, lookback=WAVELET_CAUSAL_LOOKBACK):
    min_window = 2 ** (level + 1)
    result = np.empty(len(series))
    result[:] = np.nan

    for i in range(len(series)):
        start = max(0, i - lookback + 1)
        window = series[start:i + 1]
        if len(window) < min_window:
            result[i] = series[i]
            continue
        pad_len = (2 ** level) - (len(window) % (2 ** level)) if len(window) % (2 ** level) != 0 else 0
        padded_window = np.pad(window, (0, pad_len), mode='constant', constant_values=0)
        coeffs = pywt.wavedec(padded_window, wavelet, level=level)
        threshold = universal_threshold(coeffs)
        coeffs_thresholded = [coeffs[0]]
        for detail in coeffs[1:]:
            coeffs_thresholded.append(pywt.threshold(detail, threshold, mode=mode))
        reconstructed = pywt.waverec(coeffs_thresholded, wavelet)
        result[i] = reconstructed[len(window) - 1] if len(window) - 1 < len(reconstructed) else reconstructed[-1]

    return result


def denoise_dataframe(df):
    denoised = df[[DATE_COL, VOLUME_COL]].copy()
    for col in PRICE_COLS:
        if col in df.columns:
            original = df[col].values.astype(np.float64)
            denoised_col = causal_wavelet_denoise_series(original)
            denoised[col] = denoised_col
            print(f"  Causal denoised {col}  (wavelet={WAVELET_FAMILY}, level={WAVELET_LEVEL}, lookback={WAVELET_CAUSAL_LOOKBACK})")
    return denoised


def plot_comparison(df_orig, df_denoised, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flat, PRICE_COLS):
        ax.plot(df_orig[DATE_COL], df_orig[col], alpha=0.4, label="Original", linewidth=0.5)
        ax.plot(df_denoised[DATE_COL], df_denoised[col],
                label="Denoised", linewidth=0.8)
        ax.set_title(col)
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle(f"Causal Wavelet Denoising  |  {WAVELET_FAMILY}  |  Level {WAVELET_LEVEL}  |  Lookback {WAVELET_CAUSAL_LOOKBACK}",
                 fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot saved to {save_path}")


def main():
    print("=" * 60)
    print(" Step 1 — Causal Wavelet Denoising (No Lookahead)")
    print("=" * 60)

    df = pd.read_csv(RAW_PRICE_CSV, parse_dates=[DATE_COL])
    print(f"  Loaded {RAW_PRICE_CSV.name}  ({len(df)} rows)")

    df_denoised = denoise_dataframe(df)

    df_denoised.to_csv(DENOISED_CSV, index=False)
    print(f"  Saved denoised data to {DENOISED_CSV}  ({len(df_denoised)} rows)")

    plot_comparison(df, df_denoised,
                    save_path=RESULT_DIR / "wavelet_comparison.png")

    print("Causal wavelet denoising complete.\n")


if __name__ == "__main__":
    main()
