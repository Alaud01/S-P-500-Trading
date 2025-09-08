import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------
# Utility computations
# ----------------------

def compute_cagr(series: pd.Series) -> float:
    if series.empty:
        return float('nan')
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    if start_value <= 0:
        return float('nan')
    num_days = (series.index[-1] - series.index[0]).days
    if num_days <= 0:
        return float('nan')
    years = num_days / 365.25
    return (end_value / start_value) ** (1 / years) - 1


def compute_max_drawdown(series: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    if series.empty:
        return float('nan'), pd.NaT, pd.NaT
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    min_dd_idx = drawdown.idxmin()
    peak_idx = series.loc[:min_dd_idx].idxmax() if min_dd_idx is not pd.NaT else pd.NaT
    return drawdown.min(), peak_idx, min_dd_idx


def compute_sharpe(returns: pd.Series, risk_free_rate_annual: float = 0.0) -> float:
    if returns.empty:
        return float('nan')
    # Convert annual risk-free to daily approximation
    rf_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return float('nan')
    return np.sqrt(252) * excess.mean() / std


# ----------------------
# Data Loading
# ----------------------

def load_prices(prices_csv_path: str, start_date: str) -> pd.DataFrame:
    df = pd.read_csv(prices_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    df = df.loc[df.index >= pd.to_datetime(start_date)]
    # Keep only business days intersection as present in data
    return df[['Close']].copy()


def load_meta_predictions(preds_csv_path: str, start_date: str) -> pd.DataFrame:
    df = pd.read_csv(preds_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.loc[df['Date'] >= pd.to_datetime(start_date)]
    # Keep essential columns if present
    cols = [c for c in ['Date', 'prob_meta', 'pred_meta', 'target'] if c in df.columns]
    return df[cols].copy()


# ----------------------
# Strategy Simulations
# ----------------------

@dataclass
class SimulationConfig:
    start_date: str = '2022-01-01'
    initial_capital: float = 10000.0
    monthly_contribution: float = 500.0
    neutral_band: float = 0.3
    trade_threshold_shares: float = 1e-9  # avoid counting dust as trades
    risk_free_rate_annual: float = 0.0


def first_business_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.empty:
        return index
    # Group by year-month and get the first date of each month
    df_temp = pd.DataFrame({'date': index})
    df_temp['year_month'] = df_temp['date'].dt.to_period('M')
    first_days = df_temp.groupby('year_month')['date'].first()
    return pd.DatetimeIndex(first_days.values)


def simulate_dca(prices: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    df = prices.copy()
    df['cash'] = 0.0
    df['shares'] = 0.0
    df['portfolio_value'] = 0.0
    df['trade_shares'] = 0.0

    cash = cfg.initial_capital
    shares = 0.0

    # Invest all initial capital on first day
    if not df.empty:
        first_price = df['Close'].iloc[0]
        buy_shares = cash / first_price
        shares += buy_shares
        cash -= buy_shares * first_price
        initial_trade_shares = buy_shares
    else:
        initial_trade_shares = 0.0

    # Determine contribution days (first business day of each month)
    contrib_days = set(first_business_days(df.index))

    for date, row in df.iterrows():
        price = row['Close']
        trade_today = 0.0
        # Record initial investment as a trade on the first day
        if date == df.index[0] and initial_trade_shares > 0.0:
            trade_today += initial_trade_shares
            initial_trade_shares = 0.0
        if date in contrib_days:
            cash += cfg.monthly_contribution
            # Immediately invest contribution
            add_shares = cash / price
            shares += add_shares
            cash -= add_shares * price
            trade_today += add_shares

        df.at[date, 'cash'] = cash
        df.at[date, 'shares'] = shares
        df.at[date, 'portfolio_value'] = cash + shares * price
        df.at[date, 'trade_shares'] = trade_today

    df['daily_return'] = df['portfolio_value'].pct_change().fillna(0.0)
    return df


def allocation_from_probability(prob: float, prev_weight: float, band: float) -> float:
    # Decision boundaries:
    # - Buy when prob > 0.8 (allocate 100%)
    # - Hold when 0.7 <= prob <= 0.8 (keep previous allocation)
    # - Sell when prob < 0.7 (allocate 0%)
    if prob > 0.8:
        return 1.0
    elif prob < 0.7:
        return 0.0
    else:
        return prev_weight


def simulate_meta_strategy(prices: pd.DataFrame, preds: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    # Merge on trading days, forward-fill probabilities to trading days
    df = prices.copy()
    preds = preds.copy()

    # Align predictions to trading day index
    preds = preds.set_index('Date').sort_index()
    # Keep only prob_meta if present; otherwise fallback to pred_meta as 0/1 prob
    if 'prob_meta' in preds.columns:
        prob_series = preds['prob_meta']
    elif 'pred_meta' in preds.columns:
        prob_series = preds['pred_meta'].astype(float)
    else:
        raise ValueError("Predictions CSV must contain 'prob_meta' or 'pred_meta'.")

    # Reindex to trading days and forward-fill last signal; default 0.5 before first
    prob_on_trading = prob_series.reindex(df.index)
    first_valid = prob_on_trading.first_valid_index()
    if first_valid is not None:
        prob_on_trading.loc[:first_valid] = prob_on_trading.loc[first_valid]
    prob_on_trading = prob_on_trading.ffill().fillna(0.5)

    df['prob_meta'] = prob_on_trading.values

    df['cash'] = 0.0
    df['shares'] = 0.0
    df['portfolio_value'] = 0.0
    df['target_weight'] = 0.0
    df['trade_shares'] = 0.0
    df['avg_purchase_price'] = 0.0  # Track average purchase price

    cash = cfg.initial_capital
    shares = 0.0
    avg_purchase_price = 0.0  # Track average purchase price for loss prevention

    # Determine contribution days
    contrib_days = set(first_business_days(df.index))

    prev_weight = 0.0

    for date, row in df.iterrows():
        price = row['Close']
        prob = row['prob_meta']

        # Monthly contribution
        if date in contrib_days:
            cash += cfg.monthly_contribution

        # Compute target allocation weight based on probability and band
        weight = allocation_from_probability(prob, prev_weight, cfg.neutral_band)

        # Current portfolio value before rebalancing
        portfolio_value = cash + shares * price
        desired_equity_value = weight * portfolio_value
        current_equity_value = shares * price

        # Compute trade to reach target
        delta_value = desired_equity_value - current_equity_value
        delta_shares = delta_value / price if price > 0 else 0.0

        # Only allow sells when probability is below 0.7
        if delta_shares < 0 and not (prob < 0.7):
            # Suppress sells outside sell zone
            delta_shares = 0.0
            weight = prev_weight

        if abs(delta_shares) > cfg.trade_threshold_shares:
            # Update average purchase price for buys
            if delta_shares > 0:
                # Buying: update average purchase price
                total_cost = shares * avg_purchase_price + delta_shares * price
                total_shares = shares + delta_shares
                avg_purchase_price = total_cost / total_shares if total_shares > 0 else price
            
            shares += delta_shares
            cash -= delta_shares * price
            df.at[date, 'trade_shares'] = delta_shares
        else:
            df.at[date, 'trade_shares'] = 0.0

        portfolio_value = cash + shares * price

        df.at[date, 'cash'] = cash
        df.at[date, 'shares'] = shares
        df.at[date, 'portfolio_value'] = portfolio_value
        df.at[date, 'target_weight'] = weight
        df.at[date, 'avg_purchase_price'] = avg_purchase_price

        prev_weight = weight

    df['daily_return'] = df['portfolio_value'].pct_change().fillna(0.0)
    return df


# ----------------------
# Reporting and Plotting
# ----------------------

def summarize_performance(df: pd.DataFrame, cfg: SimulationConfig) -> Dict[str, float]:
    series = df['portfolio_value']
    cagr = compute_cagr(series)
    mdd, peak_date, trough_date = compute_max_drawdown(series)
    sharpe = compute_sharpe(df['daily_return'], cfg.risk_free_rate_annual)

    num_trades = (df['trade_shares'] != 0).sum() if 'trade_shares' in df.columns else 0

    # Contributions
    n_months = len(first_business_days(df.index))
    total_contrib = cfg.initial_capital + cfg.monthly_contribution * n_months

    return {
        'start_date': df.index[0].strftime('%Y-%m-%d') if not df.empty else '',
        'end_date': df.index[-1].strftime('%Y-%m-%d') if not df.empty else '',
        'final_value': float(series.iloc[-1]) if not df.empty else float('nan'),
        'total_contributed': float(total_contrib),
        'absolute_return': float(series.iloc[-1] / total_contrib - 1) if not df.empty and total_contrib > 0 else float('nan'),
        'CAGR': float(cagr),
        'Sharpe': float(sharpe),
        'MaxDrawdown': float(mdd),
        'num_trades': int(num_trades),
        'peak_date': str(peak_date) if peak_date is not pd.NaT else '',
        'trough_date': str(trough_date) if trough_date is not pd.NaT else '',
    }


def plot_price_with_trades(prices: pd.DataFrame, model_df: pd.DataFrame, out_path: str, neutral_band: float = 0.15) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sns.set_style('whitegrid')

    # Create subplots: price chart on top, predictions on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, height_ratios=[2, 1])
    
    # Top subplot: S&P 500 Price with trades
    ax1.plot(prices.index, prices['Close'], label='S&P 500 Close', color='black', linewidth=1.5)

    # Mark buys and sells where trade_shares != 0
    trades = model_df.loc[model_df['trade_shares'] != 0]
    buys = trades.loc[trades['trade_shares'] > 0]
    sells = trades.loc[trades['trade_shares'] < 0]
    # Filter sells to only those where prob is below 0.7
    if 'prob_meta' in model_df.columns:
        sells = sells.loc[model_df.loc[sells.index, 'prob_meta'] < 0.7]

    ax1.scatter(buys.index, prices.loc[buys.index, 'Close'], marker='^', color='green', s=60, label='Buy')
    ax1.scatter(sells.index, prices.loc[sells.index, 'Close'], marker='v', color='red', s=60, label='Sell')

    ax1.set_title('S&P 500 Price with Model Strategy Buy/Sell Signals', fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: Model predictions and decision boundaries
    if 'prob_meta' in model_df.columns:
        # Plot model predictions
        ax2.plot(model_df.index, model_df['prob_meta'], label='Model Prediction', color='blue', linewidth=1.5, alpha=0.8)
        
        # Decision thresholds for plotting
        buy_threshold = 0.8
        sell_threshold = 0.7

        ax2.axhline(y=buy_threshold, color='green', linestyle='--', alpha=0.7, label=f'Buy Threshold ({buy_threshold:.1f})')
        ax2.axhline(y=sell_threshold, color='red', linestyle='--', alpha=0.7, label=f'Sell Threshold ({sell_threshold:.1f})')

        # Shade the hold zone (0.7 to 0.8)
        ax2.axhspan(sell_threshold, buy_threshold, alpha=0.2, color='yellow', label='Hold Zone')
        
        # Mark regions
        ax2.fill_between(model_df.index, 0, sell_threshold, alpha=0.1, color='red', label='Sell Zone')
        ax2.fill_between(model_df.index, buy_threshold, 1, alpha=0.1, color='green', label='Buy Zone')
        
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Prediction Probability')
        ax2.set_xlabel('Date')
        ax2.set_title('Meta-Model Predictions and Decision Boundaries', fontweight='bold')
        ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax2.grid(True, alpha=0.3)
    else:
        # Fallback if no probability data
        ax2.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Meta-Model Predictions (No Data)', fontweight='bold')
        ax2.set_ylabel('Prediction Probability')
        ax2.set_xlabel('Date')
    
    # Format x-axis dates
    if len(model_df.index) > 0 and isinstance(model_df.index[0], (np.datetime64, pd.Timestamp)):
        ax2.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_equity_curves(dca_df: pd.DataFrame, model_df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dca_df.index, dca_df['portfolio_value'], label='DCA Portfolio', color='blue', linewidth=1.5)
    ax.plot(model_df.index, model_df['portfolio_value'], label='Meta Strategy Portfolio', color='purple', linewidth=1.5)

    ax.set_title('Portfolio Value Comparison (DCA vs Meta Strategy)', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_allocation(model_df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.plot(model_df.index, model_df['target_weight'], label='Target Equity Allocation', color='orange', linewidth=1.2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Meta Strategy Target Allocation Over Time', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_report(name: str, stats: Dict[str, float]) -> None:
    print(f"\n{name} Strategy Report")
    print("-" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            if k in {'final_value', 'total_contributed'}:
                print(f"{k:>20}: ${v:,.2f}")
            elif k in {'absolute_return', 'CAGR', 'Sharpe', 'MaxDrawdown'}:
                if k in {'Sharpe', 'MaxDrawdown'}:
                    print(f"{k:>20}: {v:,.3f}")
                else:
                    print(f"{k:>20}: {v*100:,.2f}%")
            else:
                print(f"{k:>20}: {v}")
        else:
            print(f"{k:>20}: {v}")


# ----------------------
# Main entrypoint
# ----------------------

def main():
    parser = argparse.ArgumentParser(description='Compare DCA vs Meta-Model Strategy (2022+).')
    parser.add_argument('--prices-csv', type=str, default='data/yfinance_sp500.csv')
    parser.add_argument('--preds-csv', type=str, default='models/meta/meta_holdout_predictions.csv')
    parser.add_argument('--start-date', type=str, default='2022-01-01')
    parser.add_argument('--initial-capital', type=float, default=10000.0)
    parser.add_argument('--monthly-contribution', type=float, default=500.0)
    parser.add_argument('--neutral-band', type=float, default=0.15)
    parser.add_argument('--risk-free', type=float, default=0.0, help='Annualized risk-free rate for Sharpe.')
    parser.add_argument('--outdir', type=str, default='plots/comparison')

    args = parser.parse_args()

    cfg = SimulationConfig(
        start_date=args.start_date,
        initial_capital=args.initial_capital,
        monthly_contribution=args.monthly_contribution,
        neutral_band=args.neutral_band,
        risk_free_rate_annual=args.risk_free,
    )

    # Load data
    prices = load_prices(args.prices_csv, cfg.start_date)
    preds = load_meta_predictions(args.preds_csv, cfg.start_date)

    if prices.empty:
        raise ValueError('No price data available for the specified start date.')

    # Align horizon to shared period with predictions
    # For fairness, use the intersection of dates
    if not preds.empty:
        min_common = max(prices.index.min(), preds['Date'].min())
        max_common = min(prices.index.max(), preds['Date'].max())
        prices = prices.loc[(prices.index >= min_common) & (prices.index <= max_common)]
    else:
        # If no predictions for the period, we cannot simulate the model strategy
        raise ValueError('No prediction data available for the specified start date.')

    # Simulations
    dca_df = simulate_dca(prices, cfg)
    model_df = simulate_meta_strategy(prices, preds, cfg)

    # Reports
    dca_stats = summarize_performance(dca_df, cfg)
    model_stats = summarize_performance(model_df, cfg)

    print_report('DCA', dca_stats)
    print_report('Meta-Model', model_stats)

    # Plots
    os.makedirs(args.outdir, exist_ok=True)
    plot_price_with_trades(prices, model_df, os.path.join(args.outdir, 'price_with_trades.png'), cfg.neutral_band)
    plot_equity_curves(dca_df, model_df, os.path.join(args.outdir, 'equity_curves.png'))
    plot_allocation(model_df, os.path.join(args.outdir, 'allocation.png'))


if __name__ == '__main__':
    main()


