import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
        if 'forward_1d_return' in df_features.columns:
            ret = df_features.loc[i, 'forward_1d_return']
        elif 'raw_daily_return' in df_features.columns:
            ret = df_features.loc[i, 'raw_daily_return']
        else:
            ret = df_features.loc[i, 'daily_return']
        if pd.notna(ret):
            date_to_raw_return[d] = ret
        if 'raw_close' in df_features.columns:
            date_to_raw_close[d] = df_features.loc[i, 'raw_close']
        else:
            date_to_raw_close[d] = df_features.loc[i, 'Close']

    all_probs, all_labels, all_dates, all_returns, all_closes = [], [], [], [], []
    seen_dates = set()

    def fold_sort_key(name):
        metadata = results[name].get('metadata', {})
        return metadata.get('retrain_date', name)

    for fold_name in sorted(results.keys(), key=fold_sort_key):
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
    """Long if pred > 0.5, short otherwise. Returns strategy daily returns."""
    preds = (probs > 0.5).astype(int)
    position = np.where(preds == 1, 1.0, -1.0)
    strategy_returns = position * daily_returns
    return strategy_returns


def simulate_long_flat(dates, probs, daily_returns, closes):
    """Long if pred > 0.5, flat (cash) otherwise. Returns strategy daily returns."""
    preds = (probs > 0.5).astype(int)
    position = np.where(preds == 1, 1.0, 0.0)
    strategy_returns = position * daily_returns
    return strategy_returns


def simulate_dca(dates, daily_returns, closes):
    """Monthly DCA: invest fixed amount on first trading day of each month.

    Returns:
        dca_market_returns: daily return on existing holdings, excluding new inflows.
            This is the pure market return earned by DCA's deployed capital,
            making it directly comparable to lump-sum strategy returns:
                r[t] = (portfolio[t] - portfolio[t-1] - inflow[t]) / portfolio[t-1]
        portfolio_values:  dollar portfolio value at each date.
        invested_values:   cumulative dollars invested at each date.
    """
    df = pd.DataFrame({'date': dates, 'close': closes, 'daily_return': daily_returns})
    df = df.sort_values('date').reset_index(drop=True)
    df['year_month'] = df['date'].dt.to_period('M')

    shares_owned = 0.0
    total_invested = 0.0
    portfolio_values = []
    invested_values = []
    market_returns = []

    investment_dates_set = set()
    prev_portfolio = 0.0
    for idx, row in df.iterrows():
        ym = row['year_month']
        new_investment = 0.0
        if ym not in investment_dates_set:
            investment_dates_set.add(ym)
            new_investment = DCA_MONTHLY_AMOUNT
            shares_bought = DCA_MONTHLY_AMOUNT / row['close']
            shares_owned += shares_bought
            total_invested += DCA_MONTHLY_AMOUNT

        portfolio_value = shares_owned * row['close']
        portfolio_values.append(portfolio_value)
        invested_values.append(total_invested)

        market_gain = portfolio_value - prev_portfolio - new_investment
        capital_at_risk = prev_portfolio if prev_portfolio > 0 else new_investment
        if capital_at_risk > 0:
            market_returns.append(market_gain / capital_at_risk)
        else:
            market_returns.append(0.0)

        prev_portfolio = portfolio_value

    return (np.array(market_returns), np.array(portfolio_values),
            np.array(invested_values))


def compute_metrics(returns):
    """Compute performance metrics from a daily return series.

    All strategies should provide returns on the same basis:
    - Model strategies: weighted market returns (position * market_return)
    - DCA: return on existing holdings (market_gain / prev_portfolio)
    - Buy & Hold: raw market returns
    """
    n = len(returns)
    if n < 2:
        return {}

    total_return = np.prod(1 + returns) - 1
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
    print(f"  All return series on same basis: daily return on deployed capital")

    # --- Simulate strategies ---
    ls_returns = simulate_long_short(dates, probs, daily_returns, closes)
    lf_returns = simulate_long_flat(dates, probs, daily_returns, closes)
    dca_market_returns, dca_portfolio, dca_invested = simulate_dca(dates, daily_returns, closes)
    bh_returns = daily_returns.copy()

    dca_returns = dca_market_returns

    # --- Compute metrics ---
    ls_metrics = compute_metrics(ls_returns)
    lf_metrics = compute_metrics(lf_returns)
    dca_metrics = compute_metrics(dca_returns)
    bh_metrics = compute_metrics(bh_returns)

    # --- Cumulative returns for plotting (all from compounding $1) ---
    ls_cum = np.cumprod(1 + ls_returns) - 1.0
    lf_cum = np.cumprod(1 + lf_returns) - 1.0
    bh_cum = np.cumprod(1 + bh_returns) - 1.0
    dca_cum = np.cumprod(1 + dca_returns) - 1.0

    # DCA-specific dollar metrics
    final_portfolio = dca_portfolio[-1]
    total_invested = dca_invested[-1]
    dca_total_gain = final_portfolio - total_invested
    dca_roi = (final_portfolio - total_invested) / total_invested * 100
    n_years = len(dates) / 252
    dca_cagr_invested = (1 + dca_total_gain / total_invested) ** (1 / n_years) - 1 if n_years > 0 else 0

    # --- Print comparison table ---
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

    print(f"\n  DCA Dollar Metrics (return on invested capital):")
    print(f"    Total invested:       ${total_invested:>12,.0f}")
    print(f"    Final portfolio value: ${final_portfolio:>12,.2f}")
    print(f"    Total gain:            ${dca_total_gain:>12,.2f}")
    print(f"    Return on invested:    {dca_roi:>12.2f}%")
    print(f"    CAGR on invested:      {dca_cagr_invested:>12.2%}")

    # =========================================================
    # FIGURE 1: Original 2×2 comparison
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Strategies vs DCA vs Buy & Hold (Raw Prices)', fontsize=14, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.plot(dates, ls_cum * 100, label='Model Long/Short', linewidth=1.0, alpha=0.9)
    ax1.plot(dates, lf_cum * 100, label='Model Long/Flat', linewidth=1.0, alpha=0.9)
    ax1.plot(dates, bh_cum * 100, label='Buy & Hold', linewidth=1.0, alpha=0.9)
    ax1.plot(dates, dca_cum * 100, label=f'DCA (${DCA_MONTHLY_AMOUNT/1000:.0f}k/mo)', linewidth=1.0, alpha=0.9)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Cumulative Returns (%)')
    ax1.set_ylabel('Return (%)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

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

    ax3 = axes[1, 0]
    ls_cumprod = np.cumprod(1 + ls_returns)
    lf_cumprod = np.cumprod(1 + lf_returns)
    bh_cumprod = np.cumprod(1 + bh_returns)
    dca_cumprod = np.cumprod(1 + dca_returns)

    ls_dd = (ls_cumprod - np.maximum.accumulate(ls_cumprod)) / np.maximum.accumulate(ls_cumprod)
    lf_dd = (lf_cumprod - np.maximum.accumulate(lf_cumprod)) / np.maximum.accumulate(lf_cumprod)
    bh_dd = (bh_cumprod - np.maximum.accumulate(bh_cumprod)) / np.maximum.accumulate(bh_cumprod)
    dca_dd = (dca_cumprod - np.maximum.accumulate(dca_cumprod)) / np.maximum.accumulate(dca_cumprod)

    ax3.fill_between(dates, ls_dd * 100, 0, alpha=0.3, color='blue', label='Long/Short')
    ax3.fill_between(dates, lf_dd * 100, 0, alpha=0.3, color='orange', label='Long/Flat')
    ax3.fill_between(dates, bh_dd * 100, 0, alpha=0.3, color='green', label='Buy & Hold')
    ax3.fill_between(dates, dca_dd * 100, 0, alpha=0.3, color='purple', label='DCA')
    ax3.plot(dates, ls_dd * 100, linewidth=0.5, alpha=0.7, color='blue')
    ax3.plot(dates, lf_dd * 100, linewidth=0.5, alpha=0.7, color='orange')
    ax3.plot(dates, bh_dd * 100, linewidth=0.5, alpha=0.7, color='green')
    ax3.plot(dates, dca_dd * 100, linewidth=0.5, alpha=0.7, color='purple')
    ax3.set_title('Drawdown (%)')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    window = 60
    ax4 = axes[1, 1]
    if len(ls_returns) >= window:
        def rolling_sharpe(returns, w):
            return pd.Series(returns).rolling(w).apply(
                lambda x: x.mean() / (x.std(ddof=1) + 1e-10) * np.sqrt(252), raw=True)

        ls_rs = rolling_sharpe(ls_returns, window)
        lf_rs = rolling_sharpe(lf_returns, window)
        bh_rs = rolling_sharpe(bh_returns, window)
        dca_rs = rolling_sharpe(dca_returns, window)

        ax4.plot(dates, ls_rs, label='Long/Short', linewidth=0.8, alpha=0.8)
        ax4.plot(dates, lf_rs, label='Long/Flat', linewidth=0.8, alpha=0.8)
        ax4.plot(dates, bh_rs, label='Buy & Hold', linewidth=0.8, alpha=0.8)
        ax4.plot(dates, dca_rs.values, label='DCA', linewidth=0.8, alpha=0.8, color='purple')
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
    print(f"  Plot saved to {RESULT_DIR / 'dca_comparison.png'}")

    # =========================================================
    # FIGURE 2: Portfolio Growth ($)
    # =========================================================
    initial_capital = 100_000
    ls_nav = initial_capital * np.cumprod(1 + ls_returns)
    lf_nav = initial_capital * np.cumprod(1 + lf_returns)
    bh_nav = initial_capital * np.cumprod(1 + bh_returns)
    bh_nav = np.maximum(bh_nav, 0)
    dca_nav_grow = initial_capital * np.cumprod(1 + dca_returns)

    fig2, ax_pg = plt.subplots(figsize=(14, 7))
    ax_pg.plot(dates, ls_nav, label=f'Long/Short (final: ${ls_nav[-1]:,.0f})', linewidth=1.0, color='blue')
    ax_pg.plot(dates, lf_nav, label=f'Long/Flat (final: ${lf_nav[-1]:,.0f})', linewidth=1.0, color='orange')
    ax_pg.plot(dates, bh_nav, label=f'B&H (final: ${bh_nav[-1]:,.0f})', linewidth=1.0, color='green')
    ax_pg.plot(dates, dca_nav_grow, label=f'DCA (final: ${dca_nav_grow[-1]:,.0f})', linewidth=1.0, color='purple')
    ax_pg.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax_pg.set_title(f'Portfolio Growth — Starting Capital ${initial_capital:,}', fontsize=13, fontweight='bold')
    ax_pg.set_ylabel('Portfolio Value ($)')
    ax_pg.legend(fontsize=9)
    ax_pg.grid(True, alpha=0.3)
    ax_pg.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_pg.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_pg.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_pg.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "portfolio_growth.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {RESULT_DIR / 'portfolio_growth.png'}")

    # =========================================================
    # FIGURE 3: S&P 500 Price with Model Trade Arrows (LF & LS)
    # =========================================================
    ls_preds = (probs > 0.5).astype(int)
    lf_positions = np.where(ls_preds == 1, 1.0, 0.0)
    ls_positions = np.where(ls_preds == 1, 1.0, -1.0)

    ls_trades = np.diff(ls_positions)
    lf_trades = np.diff(lf_positions)

    ls_enters = np.where(ls_trades != 0)[0]
    lf_enters = np.where(lf_trades != 0)[0]

    fig3, (ax_ls, ax_lf) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig3.suptitle('S&P 500 Price with Model Trade Signals', fontsize=14, fontweight='bold')

    ax_ls.plot(dates, closes, color='black', linewidth=0.6, alpha=0.7, label='S&P 500 Close')
    buy_mask_ls = ls_trades > 0
    sell_mask_ls = ls_trades < 0
    ax_ls.scatter(dates[1:][buy_mask_ls], closes[1:][buy_mask_ls],
                  marker='^', c='limegreen', s=18, zorder=5, label=f'Go Long (n={buy_mask_ls.sum()})', edgecolors='darkgreen', linewidths=0.3)
    ax_ls.scatter(dates[1:][sell_mask_ls], closes[1:][sell_mask_ls],
                  marker='v', c='red', s=18, zorder=5, label=f'Go Short (n={sell_mask_ls.sum()})', edgecolors='darkred', linewidths=0.3)
    ax_ls.set_title('Long/Short Strategy Trades', fontsize=11)
    ax_ls.set_ylabel('S&P 500 Close')
    ax_ls.legend(fontsize=8, loc='upper left')
    ax_ls.grid(True, alpha=0.3)
    ax_ls.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_ls.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    ax_lf.plot(dates, closes, color='black', linewidth=0.6, alpha=0.7, label='S&P 500 Close')
    buy_mask_lf = lf_trades > 0
    sell_mask_lf = lf_trades < 0
    ax_lf.scatter(dates[1:][buy_mask_lf], closes[1:][buy_mask_lf],
                  marker='^', c='limegreen', s=18, zorder=5, label=f'Enter Long (n={buy_mask_lf.sum()})', edgecolors='darkgreen', linewidths=0.3)
    ax_lf.scatter(dates[1:][sell_mask_lf], closes[1:][sell_mask_lf],
                  marker='v', c='red', s=18, zorder=5, label=f'Exit to Cash (n={sell_mask_lf.sum()})', edgecolors='darkred', linewidths=0.3)
    ax_lf.set_title('Long/Flat Strategy Trades', fontsize=11)
    ax_lf.set_ylabel('S&P 500 Close')
    ax_lf.legend(fontsize=8, loc='upper left')
    ax_lf.grid(True, alpha=0.3)
    ax_lf.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_lf.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_lf.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "sp500_trade_signals.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {RESULT_DIR / 'sp500_trade_signals.png'}")

    # =========================================================
    # FIGURE 4: S&P 500 Price with Model Prediction Confidence
    # =========================================================
    fig4, ax_price = plt.subplots(figsize=(16, 8))
    ax_conf = ax_price.twinx()

    ax_price.plot(dates, closes, color='black', linewidth=0.8, alpha=0.7, label='S&P 500 Close')
    ax_conf.fill_between(dates, 0.5, probs, where=(probs >= 0.5),
                         alpha=0.25, color='green', label='Bullish confidence')
    ax_conf.fill_between(dates, 0.5, probs, where=(probs < 0.5),
                         alpha=0.25, color='red', label='Bearish confidence')
    ax_conf.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_conf.plot(dates, probs, color='steelblue', linewidth=0.4, alpha=0.6)
    ax_conf.set_ylim(0, 1)
    ax_conf.set_ylabel('Model Probability (Up)', color='steelblue', fontsize=10)

    ax_price.set_title('S&P 500 Price & Model Prediction Confidence', fontsize=13, fontweight='bold')
    ax_price.set_ylabel('S&P 500 Close', fontsize=10)
    ax_price.grid(True, alpha=0.3)
    lines_price, labels_price = ax_price.get_legend_handles_labels()
    lines_conf, labels_conf = ax_conf.get_legend_handles_labels()
    ax_price.legend(lines_price + lines_conf, labels_price + labels_conf,
                    fontsize=8, loc='upper left')
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "sp500_prediction_confidence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {RESULT_DIR / 'sp500_prediction_confidence.png'}")

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

    print(f"\n  {'Year':<8} {'LS Ret%':>10} {'LF Ret%':>10} {'DCA Ret%':>10} {'B&H Ret%':>10} {'LS Sh':>8} {'LF Sh':>8} {'DCA Sh':>8} {'BH Sh':>8}")
    print("  " + "-" * 88)

    yearly_stats = {}
    for year_label, si, ei in years:
        yr_ls = ls_returns[si:ei]
        yr_lf = lf_returns[si:ei]
        yr_bh = bh_returns[si:ei]
        yr_dca = dca_returns[si:ei]

        ls_yr = np.prod(1 + yr_ls) - 1
        lf_yr = np.prod(1 + yr_lf) - 1
        bh_yr = np.prod(1 + yr_bh) - 1
        dca_yr = np.prod(1 + yr_dca) - 1

        ls_sh = np.mean(yr_ls) / (np.std(yr_ls, ddof=1) + 1e-10) * np.sqrt(252) if len(yr_ls) > 1 else 0
        lf_sh = np.mean(yr_lf) / (np.std(yr_lf, ddof=1) + 1e-10) * np.sqrt(252) if len(yr_lf) > 1 else 0
        bh_sh = np.mean(yr_bh) / (np.std(yr_bh, ddof=1) + 1e-10) * np.sqrt(252) if len(yr_bh) > 1 else 0
        dca_sh = np.mean(yr_dca) / (np.std(yr_dca, ddof=1) + 1e-10) * np.sqrt(252) if len(yr_dca) > 1 else 0

        print(f"  {year_label:<8} {ls_yr:>10.2%} {lf_yr:>10.2%} {dca_yr:>10.2%} {bh_yr:>10.2%} {ls_sh:>8.4f} {lf_sh:>8.4f} {dca_sh:>8.4f} {bh_sh:>8.4f}")

        yearly_stats[year_label] = {
            'long_short_return': float(ls_yr),
            'long_flat_return': float(lf_yr),
            'dca_return': float(dca_yr),
            'buy_hold_return': float(bh_yr),
            'long_short_sharpe': float(ls_sh),
            'long_flat_sharpe': float(lf_sh),
            'dca_sharpe': float(dca_sh),
            'buy_hold_sharpe': float(bh_sh),
        }

    # --- Save results ---
    results = {
        'dca_monthly_amount': DCA_MONTHLY_AMOUNT,
        'test_period': f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}",
        'n_trading_days': int(len(dates)),
        'using_raw_prices': True,
        'methodology': (
            'All metrics computed from daily return-on-deployed-capital series. '
            'Model returns = position * market_return. '
            'DCA returns = market gain on existing holdings / prev portfolio value '
            '(excludes cash inflows). '
            'B&H returns = raw market returns. '
            'DCA dollar metrics (ROI on invested capital) reported separately.'
        ),
        'long_short': ls_metrics,
        'long_flat': lf_metrics,
        'dca': {
            **dca_metrics,
            'total_invested': float(total_invested),
            'final_portfolio_value': float(final_portfolio),
            'total_gain': float(dca_total_gain),
            'return_on_invested_pct': float(dca_roi),
            'cagr_on_invested': float(dca_cagr_invested),
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
