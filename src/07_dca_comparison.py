import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import RESULT_DIR, FEATURES_CSV, DATE_COL, RAW_PRICE_CSV

DCA_MONTHLY_AMOUNT = 10_000


def load_model_predictions():
    with open(RESULT_DIR / "fold_results.json") as f:
        results = json.load(f)

    df_features = pd.read_csv(FEATURES_CSV, parse_dates=[DATE_COL])
    df_features = df_features.sort_values(DATE_COL).reset_index(drop=True)

    def normalize_date(d):
        return pd.Timestamp(d).strftime('%Y-%m-%d')

    date_to_raw_return = {}
    date_to_raw_close = {}
    for i in range(len(df_features)):
        d = normalize_date(df_features.loc[i, DATE_COL])
        if 'raw_daily_return' in df_features.columns:
            date_to_raw_return[d] = df_features.loc[i, 'raw_daily_return']
            date_to_raw_close[d] = df_features.loc[i, 'raw_close']
        else:
            date_to_raw_return[d] = df_features.loc[i, 'daily_return']
            date_to_raw_close[d] = df_features.loc[i, 'Close']

    all_probs, all_labels, all_dates, all_returns, all_closes = [], [], [], [], []
    seen_dates = set()

    for fold_name in sorted(results.keys()):
        fold_data = results[fold_name]
        fold_probs = fold_data['preds']
        fold_labels = fold_data['labels']
        fold_dates = [normalize_date(d) for d in fold_data['dates']]

        for i, d in enumerate(fold_dates):
            if d in seen_dates:
                continue
            seen_dates.add(d)
            all_probs.append(fold_probs[i])
            all_labels.append(fold_labels[i])
            all_dates.append(pd.Timestamp(d))
            all_returns.append(date_to_raw_return.get(d, 0.0))
            all_closes.append(date_to_raw_close.get(d, 0.0))

    return (np.array(all_probs), np.array(all_labels).astype(int),
            np.array(all_dates), np.array(all_returns), np.array(all_closes))


def simulate_long_short(dates, probs, daily_returns, closes):
    """Long if pred > 0.5, short otherwise. Uses RAW returns for realistic backtest."""
    preds = (probs > 0.5).astype(int)
    position = np.where(preds == 1, 1.0, -1.0)
    strategy_returns = position * daily_returns
    cumulative = np.cumprod(1 + strategy_returns)
    return cumulative - 1.0, strategy_returns


def simulate_long_flat(dates, probs, daily_returns, closes):
    """Long if pred > 0.5, flat (cash) otherwise. Uses RAW returns."""
    preds = (probs > 0.5).astype(int)
    position = np.where(preds == 1, 1.0, 0.0)
    strategy_returns = position * daily_returns
    cumulative = np.cumprod(1 + strategy_returns)
    return cumulative - 1.0, strategy_returns


def simulate_dca(dates, daily_returns, closes):
    """Monthly DCA: invest fixed amount on first trading day of each month.
    Uses RAW close prices (actual market prices)."""
    df = pd.DataFrame({'date': dates, 'close': closes, 'daily_return': daily_returns})
    df = df.sort_values('date').reset_index(drop=True)
    df['year_month'] = df['date'].dt.to_period('M')

    shares_owned = 0.0
    total_invested = 0.0
    portfolio_values = []
    invested_values = []

    investment_dates = set()
    for idx, row in df.iterrows():
        ym = row['year_month']
        if ym not in investment_dates:
            investment_dates.add(ym)
            shares_bought = DCA_MONTHLY_AMOUNT / row['close']
            shares_owned += shares_bought
            total_invested += DCA_MONTHLY_AMOUNT

        portfolio_value = shares_owned * row['close']
        portfolio_values.append(portfolio_value)
        invested_values.append(total_invested)

    portfolio_values = np.array(portfolio_values)
    invested_values = np.array(invested_values)
    cumulative_return = (portfolio_values - invested_values) / invested_values

    return cumulative_return, portfolio_values, invested_values


def compute_metrics(returns, dates):
    n = len(returns)
    if n < 2:
        return {}

    total_return = (np.prod(1 + returns) - 1)
    annual_factor = 252 / n
    cagr = (1 + total_return) ** annual_factor - 1

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-10
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(252)

    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()

    win_rate = (returns > 0).mean()

    n_years = n / 252

    return {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'n_periods': n,
        'n_years': round(n_years, 2),
        'avg_daily_return': float(mean_ret),
    }


def main():
    print("=" * 60)
    print(" Step 7 — DCA vs Model Comparison (Raw Prices)")
    print("=" * 60)

    probs, labels, dates, daily_returns, closes = load_model_predictions()

    print(f"\n  Test period: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    print(f"  Total trading days: {len(dates)}")
    print(f"  DCA monthly investment: ${DCA_MONTHLY_AMOUNT:,}")
    print(f"  Using RAW (actual market) prices and returns")

    # --- Strategy simulations ---
    ls_cum_ret, ls_returns = simulate_long_short(dates, probs, daily_returns, closes)
    lf_cum_ret, lf_returns = simulate_long_flat(dates, probs, daily_returns, closes)
    dca_cum_ret, dca_portfolio, dca_invested = simulate_dca(dates, daily_returns, closes)

    # Also compute buy-and-hold for reference
    bh_returns = daily_returns.copy()
    bh_cum = np.cumprod(1 + bh_returns) - 1.0

    # --- Metrics ---
    ls_metrics = compute_metrics(ls_returns, dates)
    lf_metrics = compute_metrics(lf_returns, dates)
    dca_metrics = compute_metrics(daily_returns, dates)
    bh_metrics = compute_metrics(bh_returns, dates)

    # DCA-specific metrics
    final_portfolio = dca_portfolio[-1]
    total_invested = dca_invested[-1]
    dca_total_gain = final_portfolio - total_invested
    dca_total_return_pct = (final_portfolio - total_invested) / total_invested * 100

    print(f"\n  {'Metric':<25} {'Long/Short':>12} {'Long/Flat':>12} {'DCA ($10k/mo)':>15} {'Buy & Hold':>12}")
    print("  " + "-" * 78)

    for metric_key in ['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'n_periods']:
        ls_val = ls_metrics.get(metric_key, 0)
        lf_val = lf_metrics.get(metric_key, 0)
        dca_val = dca_metrics.get(metric_key, 0)
        bh_val = bh_metrics.get(metric_key, 0)

        if metric_key in ('total_return', 'cagr', 'max_drawdown', 'win_rate'):
            print(f"  {metric_key:<25} {ls_val:>12.2%} {lf_val:>12.2%} {dca_val:>15.2%} {bh_val:>12.2%}")
        elif metric_key == 'sharpe_ratio':
            print(f"  {metric_key:<25} {ls_val:>12.4f} {lf_val:>12.4f} {dca_val:>15.4f} {bh_val:>12.4f}")
        elif metric_key == 'n_periods':
            print(f"  {metric_key:<25} {int(ls_val):>12} {int(lf_val):>12} {int(dca_val):>15} {int(bh_val):>12}")

    print(f"\n  DCA Specifics:")
    print(f"    Total invested:       ${total_invested:>12,.0f}")
    print(f"    Final portfolio value: ${final_portfolio:>12,.2f}")
    print(f"    Total gain:            ${dca_total_gain:>12,.2f}")
    print(f"    Total return:          {dca_total_return_pct:>12.2f}%")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Strategies vs DCA vs Buy & Hold (Raw Prices)', fontsize=14, fontweight='bold')

    # Equity curves
    ax1 = axes[0, 0]
    ax1.plot(dates, ls_cum_ret * 100, label='Model Long/Short', linewidth=1.0, alpha=0.9)
    ax1.plot(dates, lf_cum_ret * 100, label='Model Long/Flat', linewidth=1.0, alpha=0.9)
    ax1.plot(dates, bh_cum * 100, label='Buy & Hold', linewidth=1.0, alpha=0.9)
    ax1.plot(dates, dca_cum_ret * 100, label=f'DCA (${DCA_MONTHLY_AMOUNT/1000:.0f}k/mo)', linewidth=1.0, alpha=0.9)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Cumulative Returns (%)')
    ax1.set_ylabel('Return (%)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # DCA portfolio value vs invested
    ax2 = axes[0, 1]
    ax2.plot(dates, dca_portfolio, label='Portfolio Value', linewidth=1.0, color='green')
    ax2.plot(dates, dca_invested, label='Total Invested', linewidth=1.0, color='gray', linestyle='--')
    ax2.fill_between(dates, dca_invested, dca_portfolio, alpha=0.2, color='green')
    ax2.set_title(f'DCA Portfolio (Monthly ${DCA_MONTHLY_AMOUNT:,})')
    ax2.set_ylabel('Value ($)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Drawdown comparison
    ax3 = axes[1, 0]
    ls_cum = np.cumprod(1 + ls_returns)
    ls_dd = (ls_cum - np.maximum.accumulate(ls_cum)) / np.maximum.accumulate(ls_cum)
    lf_cum_arr = np.cumprod(1 + lf_returns)
    lf_dd = (lf_cum_arr - np.maximum.accumulate(lf_cum_arr)) / np.maximum.accumulate(lf_cum_arr)
    bh_cum_arr = np.cumprod(1 + bh_returns)
    bh_dd = (bh_cum_arr - np.maximum.accumulate(bh_cum_arr)) / np.maximum.accumulate(bh_cum_arr)

    ax3.fill_between(dates, ls_dd * 100, 0, alpha=0.3, color='blue', label='Long/Short')
    ax3.fill_between(dates, lf_dd * 100, 0, alpha=0.3, color='orange', label='Long/Flat')
    ax3.fill_between(dates, bh_dd * 100, 0, alpha=0.3, color='green', label='Buy & Hold')
    ax3.plot(dates, ls_dd * 100, linewidth=0.5, alpha=0.7, color='blue')
    ax3.plot(dates, lf_dd * 100, linewidth=0.5, alpha=0.7, color='orange')
    ax3.plot(dates, bh_dd * 100, linewidth=0.5, alpha=0.7, color='green')
    ax3.set_title('Drawdown (%)')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # Rolling 60-day Sharpe
    window = 60
    ax4 = axes[1, 1]
    if len(ls_returns) >= window:
        ls_rolling_sharpe = pd.Series(ls_returns).rolling(window).apply(
            lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(252), raw=True)
        lf_rolling_sharpe = pd.Series(lf_returns).rolling(window).apply(
            lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(252), raw=True)
        bh_rolling_sharpe = pd.Series(bh_returns).rolling(window).apply(
            lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(252), raw=True)

        ax4.plot(dates, ls_rolling_sharpe, label='Long/Short', linewidth=0.8, alpha=0.8)
        ax4.plot(dates, lf_rolling_sharpe, label='Long/Flat', linewidth=0.8, alpha=0.8)
        ax4.plot(dates, bh_rolling_sharpe, label='Buy & Hold', linewidth=0.8, alpha=0.8)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_title(f'Rolling {window}-day Sharpe Ratio')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "dca_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to {RESULT_DIR / 'dca_comparison.png'}")

    # --- Year-by-year breakdown ---
    print(f"\n  Year-by-Year Performance:")
    year_borders = []
    current_year = dates[0].year
    year_borders.append(0)
    for i, d in enumerate(dates):
        if d.year != current_year:
            year_borders.append(i)
            current_year = d.year
    year_borders.append(len(dates))

    years = []
    for k in range(len(year_borders) - 1):
        start_idx = year_borders[k]
        end_idx = year_borders[k + 1]
        year_label = str(dates[start_idx].year)
        years.append((year_label, start_idx, end_idx))

    print(f"\n  {'Year':<8} {'LS Ret%':>10} {'LF Ret%':>10} {'B&H Ret%':>10} {'LS Sharpe':>10} {'LF Sharpe':>10} {'BH Sharpe':>10}")
    print("  " + "-" * 68)

    yearly_stats = {}
    for year_label, si, ei in years:
        yr_ls_ret = ls_returns[si:ei]
        yr_lf_ret = lf_returns[si:ei]
        yr_bh_ret = bh_returns[si:ei]

        ls_yr = np.prod(1 + yr_ls_ret) - 1
        lf_yr = np.prod(1 + yr_lf_ret) - 1
        bh_yr = np.prod(1 + yr_bh_ret) - 1

        ls_sh = np.mean(yr_ls_ret) / (np.std(yr_ls_ret, ddof=1) + 1e-10) * np.sqrt(252) if len(yr_ls_ret) > 1 else 0
        lf_sh = np.mean(yr_lf_ret) / (np.std(yr_lf_ret, ddof=1) + 1e-10) * np.sqrt(252) if len(yr_lf_ret) > 1 else 0
        bh_sh = np.mean(yr_bh_ret) / (np.std(yr_bh_ret, ddof=1) + 1e-10) * np.sqrt(252) if len(yr_bh_ret) > 1 else 0

        print(f"  {year_label:<8} {ls_yr:>10.2%} {lf_yr:>10.2%} {bh_yr:>10.2%} {ls_sh:>10.4f} {lf_sh:>10.4f} {bh_sh:>10.4f}")

        yearly_stats[year_label] = {
            'long_short_return': float(ls_yr),
            'long_flat_return': float(lf_yr),
            'buy_hold_return': float(bh_yr),
            'long_short_sharpe': float(ls_sh),
            'long_flat_sharpe': float(lf_sh),
            'buy_hold_sharpe': float(bh_sh),
        }

    # --- Save results ---
    results = {
        'dca_monthly_amount': DCA_MONTHLY_AMOUNT,
        'test_period': f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}",
        'n_trading_days': int(len(dates)),
        'using_raw_prices': True,
        'long_short': ls_metrics,
        'long_flat': lf_metrics,
        'dca': {
            **dca_metrics,
            'total_invested': float(total_invested),
            'final_portfolio_value': float(final_portfolio),
            'total_gain': float(dca_total_gain),
            'total_return_pct': float(dca_total_return_pct),
        },
        'buy_hold': bh_metrics,
        'yearly_breakdown': yearly_stats,
    }

    with open(RESULT_DIR / "dca_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULT_DIR / 'dca_comparison.json'}")

    print("\nComparison complete.\n")


if __name__ == "__main__":
    main()