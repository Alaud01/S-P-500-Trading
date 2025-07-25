import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

class StrategyComparer:
    def __init__(self, data_file='sp500_features_for_prediction.csv', start_date='2022-01-01'):
        """Initialize the strategy comparison"""
        self.data_file = data_file
        self.start_date = pd.to_datetime(start_date)
        self.models = {}
        self.scalers = {}
        self.df = None
        self.results = {}
        
        # Load models and data
        self.load_trained_models()
        self.prepare_data()

    def load_trained_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        self.models['lstm'] = load_model('models/lstm_model.h5')
        self.scalers['lstm'] = joblib.load('models/lstm_scaler.pkl')
        self.models['xgboost'] = joblib.load('models/xgboost_model.pkl')
        self.scalers['xgboost'] = joblib.load('models/xgboost_scaler.pkl')
        self.models['linear'] = joblib.load('models/linear_model.pkl')
        self.scalers['linear'] = joblib.load('models/linear_scaler.pkl')
        self.models['meta'] = joblib.load('models/meta_model.pkl')
        print("All models loaded successfully!")

    def prepare_data(self):
        """Prepare data for backtesting"""
        df = pd.read_csv(self.data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        self.df = df[df['Date'] >= self.start_date].reset_index(drop=True)
        print(f"Backtesting from {self.df['Date'].min().strftime('%Y-%m-%d')} to {self.df['Date'].max().strftime('%Y-%m-%d')}")

    def get_model_prediction(self, date):
        """Get the meta-model's prediction for a given date"""
        # Find the row for the given date
        data_for_date = self.df[self.df['Date'] == date]
        if data_for_date.empty:
            return 0 # Default to no signal if date not found
        
        # Get full historical data up to the prediction date
        historical_df = pd.read_csv(self.data_file)
        historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        historical_df = historical_df[historical_df['Date'] <= date]
        
        # Prepare features
        technical_features = [
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate', 'MA_5', 'MA_20', 'MA_50', 
            'RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'BB_Width', 'Volatility_20', 
            'Volume_Ratio', 'Price_Change_5d', 'Price_Change_20d', 'Sentiment_MA_5'
        ]
        lag_features = [f'Price_Lag_{lag}' for lag in [1, 5, 10, 20]] + [f'Returns_Lag_{lag}' for lag in [1, 5, 10, 20]]
        all_features = technical_features + lag_features
        
        lstm_features = [
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate', 'RSI', 'MACD', 'Volatility_20', 
            'BB_Position', 'Sentiment_MA_5'
        ]
        
        xgb_linear_features = data_for_date[all_features]
        
        # LSTM sequence
        sequence_length = 60
        lstm_sequence = historical_df[lstm_features].iloc[-sequence_length:].values
        
        # Make base predictions
        xgb_scaled = self.scalers['xgboost'].transform(xgb_linear_features)
        xgb_pred_proba = self.models['xgboost'].predict_proba(xgb_scaled)[0][1]
        
        linear_scaled = self.scalers['linear'].transform(xgb_linear_features)
        linear_pred_proba = self.models['linear'].predict_proba(linear_scaled)[0][1]
        
        lstm_scaled = self.scalers['lstm'].transform(lstm_sequence.reshape(-1, lstm_sequence.shape[-1]))
        lstm_scaled = lstm_scaled.reshape(1, sequence_length, len(lstm_features))
        lstm_pred_proba = self.models['lstm'].predict(lstm_scaled, verbose=0)[0][0]
        
        # Meta-model prediction
        base_model_preds = np.array([[lstm_pred_proba, xgb_pred_proba, linear_pred_proba]])
        additional_features_list = ['CP', 'Volatility_20', 'RSI', 'Interest_Rate', 'Inflation_Rate', 'Sentiment_MA_5']
        additional_values = data_for_date[additional_features_list].values
        
        meta_features = np.column_stack([base_model_preds, additional_values])
        meta_pred_proba = self.models['meta'].predict_proba(meta_features)[0]
        
        return meta_pred_proba[1] # Return probability of "Up"

    def run_dca_strategy(self, monthly_investment=1000):
        """Run the Dollar Cost Averaging strategy"""
        dca_portfolio = pd.DataFrame(index=self.df['Date'])
        dca_portfolio['cash'] = 0
        dca_portfolio['shares'] = 0
        dca_portfolio['portfolio_value'] = 0
        
        cash = 0
        shares = 0
        
        # Get first business day of each month
        investment_dates = self.df.groupby(self.df['Date'].dt.to_period('M')).first()['Date']
        
        for date in self.df['Date']:
            if date in investment_dates.values:
                cash += monthly_investment
                price = self.df[self.df['Date'] == date]['CP'].iloc[0]
                shares_bought = cash / price
                shares += shares_bought
                cash = 0 # All cash invested
            
            current_price = self.df[self.df['Date'] == date]['CP'].iloc[0]
            dca_portfolio.loc[date, 'shares'] = shares
            dca_portfolio.loc[date, 'portfolio_value'] = shares * current_price
            
        self.results['DCA'] = dca_portfolio
        return dca_portfolio

    def run_model_strategy(self, monthly_investment=1000):
        """Run the strategy guided by the meta-model with daily, confidence-scaled decisions and fund accumulation."""
        model_portfolio = pd.DataFrame(index=self.df['Date'])
        model_portfolio['cash'] = 0
        model_portfolio['shares'] = 0
        model_portfolio['portfolio_value'] = 0
        model_portfolio['buy_signal'] = 0  # Will now store the amount invested

        shares = 0
        accumulated_funds = 0
        last_cash_injection_period = None
        trades_this_month = 0

        # --- Strategy Parameters ---
        trade_decision_threshold = 0.985  # Increased from 0.65 to be more selective
        max_trades_per_month = 15         # Limit number of trades per month
        min_investment_ratio = 0.2       # At least 20% of available funds
        # --- End of Parameters ---

        for index, row in self.df.iterrows():
            date = row['Date']
            current_period = date.to_period('M')

            # If it's the start of a new month-year period, add monthly investment and reset trade counter
            if current_period != last_cash_injection_period:
                accumulated_funds += monthly_investment
                last_cash_injection_period = current_period
                trades_this_month = 0

            # Get model prediction for the day
            prediction = self.get_model_prediction(date)

            # Make investment decision based on prediction and monthly trade limit
            if prediction > trade_decision_threshold and trades_this_month < max_trades_per_month:
                # Calculate confidence-based investment amount
                # Scales from 0 to 1 based on how far prediction is from threshold to 1.0
                confidence_scaler = (prediction - trade_decision_threshold) / (1.0 - trade_decision_threshold)
                
                # Scale investment from min_investment_ratio to 1.0
                investment_scaler = min_investment_ratio + (confidence_scaler * (1.0 - min_investment_ratio))

                amount_to_invest = accumulated_funds * investment_scaler

                # Execute the investment only if it's a meaningful amount
                if amount_to_invest > 1.0:
                    shares += amount_to_invest / row['CP']
                    accumulated_funds -= amount_to_invest
                    
                    model_portfolio.loc[date, 'buy_signal'] = amount_to_invest
                    trades_this_month += 1
                else:
                    model_portfolio.loc[date, 'buy_signal'] = 0
            else:
                model_portfolio.loc[date, 'buy_signal'] = 0

            # Update portfolio value for the day - ALWAYS calculate this
            current_price = row['CP']
            portfolio_value = (shares * current_price) + accumulated_funds

            # Store values in DataFrame - ensure proper indexing
            model_portfolio.at[date, 'cash'] = accumulated_funds
            model_portfolio.at[date, 'shares'] = shares
            model_portfolio.at[date, 'portfolio_value'] = portfolio_value

        self.results['Model'] = model_portfolio
        return model_portfolio
        
    def calculate_max_drawdown(self, portfolio_values):
        """Calculate the maximum drawdown for a portfolio"""
        # Handle any NaN or zero values and ensure minimum value
        portfolio_values = portfolio_values.fillna(method='ffill').fillna(0.01)
        portfolio_values = portfolio_values.clip(lower=0.01)  # Minimum value of 0.01 to avoid division by zero
        
        # We use a rolling maximum to track the running peak
        running_max = portfolio_values.expanding().max()
        # Calculate the drawdown from the running peak
        drawdown = (portfolio_values - running_max) / running_max
        
        # Cap the maximum drawdown at -99% to avoid -100% issues
        max_drawdown = max(drawdown.min(), -0.99)
        return max_drawdown

    def display_strategy_metrics(self):
        """Calculate and display key performance metrics for both strategies"""
        print("\n" + "="*80)
        print("STRATEGY PERFORMANCE METRICS (2022-01-01 onwards)")
        print("="*80)

        metrics = {}
        for strategy_name in ['DCA', 'Model']:
            portfolio = self.results[strategy_name]
            final_value = portfolio['portfolio_value'].iloc[-1]
            
            if strategy_name == 'DCA':
                investment_dates = self.df.groupby(self.df['Date'].dt.to_period('M')).first()['Date']
                total_invested = len(investment_dates) * 1000
                num_investments = len(investment_dates)
            else:
                num_investments = (portfolio['buy_signal'] > 0).sum()
                total_invested = portfolio['buy_signal'].sum()

            net_profit = final_value - total_invested
            total_return_pct = (net_profit / total_invested) if total_invested > 0 else 0
            

            
            max_drawdown = self.calculate_max_drawdown(portfolio['portfolio_value'])

            metrics[strategy_name] = {
                "Final Portfolio Value": f"${final_value:,.2f}",
                "Total Amount Invested": f"${total_invested:,.2f}",
                "Net Profit": f"${net_profit:,.2f}",
                "Total Return on Investment": f"{total_return_pct:.2f}%",
                "Number of Investments": num_investments,
                "Maximum Drawdown": f"{max_drawdown:.2%}"
            }
        
        # Format and print the results
        df_metrics = pd.DataFrame(metrics).T
        print(df_metrics)
        print("="*80)

        # Print insights
        model_return = float(metrics['Model']['Total Return on Investment'].strip('%'))
        dca_return = float(metrics['DCA']['Total Return on Investment'].strip('%'))
        model_drawdown = float(metrics['Model']['Maximum Drawdown'].strip('%'))
        dca_drawdown = float(metrics['DCA']['Maximum Drawdown'].strip('%'))
        
        print("\nKey Differences:")
        if model_return > dca_return:
            print(f"• OUTPERFORMANCE: The Meta-Model strategy outperformed DCA by {model_return - dca_return:.2f}%.")
        else:
            print(f"• UNDERPERFORMANCE: The Meta-Model strategy underperformed DCA by {dca_return - model_return:.2f}%.")

        if abs(model_drawdown) < abs(dca_drawdown):
            print(f"• RISK REDUCTION: The model strategy was less risky, with a max drawdown of {model_drawdown:.2f}% vs DCA's {dca_drawdown:.2f}%.")
        else:
            print(f"• RISK INCREASE: The model strategy was riskier, with a max drawdown of {model_drawdown:.2f}% vs DCA's {dca_drawdown:.2f}%.")
            
        print(f"• SELECTIVITY: The model chose to invest in only {metrics['Model']['Number of Investments']} out of a possible {metrics['DCA']['Number of Investments']} months.")


    def plot_performance_comparison(self):
        """Plot portfolio performance of both strategies"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        dca_final_value = self.results['DCA']['portfolio_value'].iloc[-1]
        model_final_value = self.results['Model']['portfolio_value'].iloc[-1]

        ax.plot(self.results['DCA'].index, self.results['DCA']['portfolio_value'], label=f'DCA Strategy (Final: ${dca_final_value:,.2f})', color='cyan')
        ax.plot(self.results['Model'].index, self.results['Model']['portfolio_value'], label=f'Meta-Model Strategy (Final: ${model_final_value:,.2f})', color='magenta')
        
        ax.set_title('Strategy Comparison: DCA vs. Meta-Model', fontsize=18)
        ax.set_ylabel('Portfolio Value ($)', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_roi_comparison(self):
        """Plot ROI comparison over time"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Calculate ROI for both strategies
        dca_roi = []
        model_roi = []
        dates = []
        
        # Calculate total invested at each point
        dca_investment_dates = self.df.groupby(self.df['Date'].dt.to_period('M')).first()['Date']
        model_investments = self.results['Model']['buy_signal'].cumsum()
        
        for i, date in enumerate(self.df['Date']):
            # DCA ROI calculation
            months_passed = len(dca_investment_dates[dca_investment_dates <= date])
            dca_total_invested = months_passed * 1000
            dca_current_value = self.results['DCA'].loc[date, 'portfolio_value']
            dca_roi_pct = ((dca_current_value - dca_total_invested) / dca_total_invested * 100) if dca_total_invested > 0 else 0
            
            # Model ROI calculation
            model_total_invested = model_investments.loc[date]
            model_current_value = self.results['Model'].loc[date, 'portfolio_value']
            model_roi_pct = ((model_current_value - model_total_invested) / model_total_invested * 100) if model_total_invested > 0 else 0
            
            dca_roi.append(dca_roi_pct)
            model_roi.append(model_roi_pct)
            dates.append(date)
        
        # Plot ROI over time
        ax1.plot(dates, dca_roi, label='DCA Strategy', color='cyan', linewidth=2)
        ax1.plot(dates, model_roi, label='Meta-Model Strategy', color='magenta', linewidth=2)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax1.set_title('ROI Comparison Over Time', fontsize=16)
        ax1.set_ylabel('ROI (%)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative returns
        dca_cumulative_return = [(roi/100 + 1) for roi in dca_roi]
        model_cumulative_return = [(roi/100 + 1) for roi in model_roi]
        
        ax2.plot(dates, dca_cumulative_return, label='DCA Strategy', color='cyan', linewidth=2)
        ax2.plot(dates, model_cumulative_return, label='Meta-Model Strategy', color='magenta', linewidth=2)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Initial Investment')
        ax2.set_title('Cumulative Returns (1 = Initial Investment)', fontsize=16)
        ax2.set_ylabel('Cumulative Return', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_drawdown_analysis(self):
        """Plot drawdown analysis for both strategies"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Calculate drawdown for both strategies
        def calculate_drawdown(portfolio_values):
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max * 100
            return drawdown
        
        dca_drawdown = calculate_drawdown(self.results['DCA']['portfolio_value'])
        model_drawdown = calculate_drawdown(self.results['Model']['portfolio_value'])
        
        # Plot drawdown
        ax1.fill_between(self.df['Date'], dca_drawdown, 0, alpha=0.3, color='cyan', label='DCA Strategy')
        ax1.fill_between(self.df['Date'], model_drawdown, 0, alpha=0.3, color='magenta', label='Meta-Model Strategy')
        ax1.plot(self.df['Date'], dca_drawdown, color='cyan', linewidth=1, alpha=0.8)
        ax1.plot(self.df['Date'], model_drawdown, color='magenta', linewidth=1, alpha=0.8)
        ax1.set_title('Drawdown Analysis', fontsize=16)
        ax1.set_ylabel('Drawdown (%)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Invert y-axis so drawdowns appear as negative values
        
        # Plot rolling volatility (30-day)
        def calculate_rolling_volatility(returns, window=30):
            return returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized
        
        # Calculate daily returns
        dca_returns = self.results['DCA']['portfolio_value'].pct_change()
        model_returns = self.results['Model']['portfolio_value'].pct_change()
        
        dca_vol = calculate_rolling_volatility(dca_returns)
        model_vol = calculate_rolling_volatility(model_returns)
        
        ax2.plot(self.df['Date'], dca_vol, label='DCA Strategy', color='cyan', linewidth=2)
        ax2.plot(self.df['Date'], model_vol, label='Meta-Model Strategy', color='magenta', linewidth=2)
        ax2.set_title('30-Day Rolling Volatility (Annualized)', fontsize=16)
        ax2.set_ylabel('Volatility (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_risk_return_analysis(self):
        """Plot risk-return scatter and monthly returns distribution"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate monthly returns for both strategies
        dca_monthly = self.results['DCA']['portfolio_value'].resample('M').last().pct_change().dropna()
        model_monthly = self.results['Model']['portfolio_value'].resample('M').last().pct_change().dropna()
        
        # Risk-return scatter
        dca_avg_return = dca_monthly.mean() * 12 * 100  # Annualized
        dca_volatility = dca_monthly.std() * np.sqrt(12) * 100  # Annualized
        model_avg_return = model_monthly.mean() * 12 * 100
        model_volatility = model_monthly.std() * np.sqrt(12) * 100
        
        ax1.scatter(dca_volatility, dca_avg_return, s=200, color='cyan', alpha=0.7, label='DCA Strategy')
        ax1.scatter(model_volatility, model_avg_return, s=200, color='magenta', alpha=0.7, label='Meta-Model Strategy')
        ax1.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax1.set_ylabel('Annualized Return (%)', fontsize=12)
        ax1.set_title('Risk-Return Profile', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add text annotations
        ax1.annotate(f'DCA: {dca_avg_return:.1f}% return\n{dca_volatility:.1f}% volatility', 
                    (dca_volatility, dca_avg_return), xytext=(5, 5), textcoords='offset points')
        ax1.annotate(f'Model: {model_avg_return:.1f}% return\n{model_volatility:.1f}% volatility', 
                    (model_volatility, model_avg_return), xytext=(5, 5), textcoords='offset points')
        
        # Monthly returns distribution
        ax2.hist(dca_monthly * 100, bins=15, alpha=0.6, color='cyan', label='DCA Strategy', density=True)
        ax2.hist(model_monthly * 100, bins=15, alpha=0.6, color='magenta', label='Meta-Model Strategy', density=True)
        ax2.set_xlabel('Monthly Return (%)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Monthly Returns Distribution', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Monthly returns comparison
        months = range(len(dca_monthly))
        ax3.bar([m-0.2 for m in months], dca_monthly * 100, width=0.4, alpha=0.7, color='cyan', label='DCA Strategy')
        ax3.bar([m+0.2 for m in months], model_monthly * 100, width=0.4, alpha=0.7, color='magenta', label='Meta-Model Strategy')
        ax3.set_xlabel('Month', fontsize=12)
        ax3.set_ylabel('Monthly Return (%)', fontsize=12)
        ax3.set_title('Monthly Returns Comparison', fontsize=14)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Sharpe ratio comparison (assuming risk-free rate of 2%)
        risk_free_rate = 0.02 / 12  # Monthly risk-free rate
        dca_sharpe = (dca_monthly.mean() - risk_free_rate) / dca_monthly.std() * np.sqrt(12)
        model_sharpe = (model_monthly.mean() - risk_free_rate) / model_monthly.std() * np.sqrt(12)
        
        strategies = ['DCA Strategy', 'Meta-Model Strategy']
        sharpe_ratios = [dca_sharpe, model_sharpe]
        colors = ['cyan', 'magenta']
        
        bars = ax4.bar(strategies, sharpe_ratios, color=colors, alpha=0.7)
        ax4.set_ylabel('Sharpe Ratio', fontsize=12)
        ax4.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def plot_trading_activity(self):
        """Plot trading activity and cash management"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Trading frequency over time
        model_buy_signals = self.results['Model']['buy_signal']
        monthly_trades = model_buy_signals.resample('M').apply(lambda x: (x > 0).sum())
        
        ax1.bar(monthly_trades.index, monthly_trades.values, alpha=0.7, color='magenta')
        ax1.set_title('Monthly Trading Frequency (Model Strategy)', fontsize=14)
        ax1.set_ylabel('Number of Trades', fontsize=12)
        ax1.set_xlabel('Month', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Cash management
        ax2.plot(self.df['Date'], self.results['DCA']['cash'], label='DCA Strategy', color='cyan', linewidth=2)
        ax2.plot(self.df['Date'], self.results['Model']['cash'], label='Meta-Model Strategy', color='magenta', linewidth=2)
        ax2.set_title('Cash Holdings Over Time', fontsize=14)
        ax2.set_ylabel('Cash ($)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Investment amount distribution
        investment_amounts = model_buy_signals[model_buy_signals > 0]
        ax3.hist(investment_amounts, bins=20, alpha=0.7, color='magenta', edgecolor='black')
        ax3.set_title('Distribution of Investment Amounts (Model Strategy)', fontsize=14)
        ax3.set_xlabel('Investment Amount ($)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Portfolio composition over time
        dca_shares = self.results['DCA']['shares']
        model_shares = self.results['Model']['shares']
        
        ax4.plot(self.df['Date'], dca_shares, label='DCA Strategy', color='cyan', linewidth=2)
        ax4.plot(self.df['Date'], model_shares, label='Meta-Model Strategy', color='magenta', linewidth=2)
        ax4.set_title('Number of Shares Owned Over Time', fontsize=14)
        ax4.set_ylabel('Number of Shares', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_model_confidence_vs_price(self):
        """Plot model prediction confidence compared to S&P 500 price"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Get model predictions for all dates
        predictions = []
        dates = []
        
        for date in self.df['Date']:
            prediction = self.get_model_prediction(date)
            predictions.append(prediction)
            dates.append(date)
        
        # Create the main plot with dual y-axes
        ax1_twin = ax1.twinx()
        
        # Plot S&P 500 price on primary y-axis
        line1 = ax1.plot(dates, self.df['CP'], color='blue', linewidth=2, label='S&P 500 Price')
        ax1.set_ylabel('S&P 500 Price ($)', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot model confidence on secondary y-axis
        line2 = ax1_twin.plot(dates, predictions, color='red', linewidth=2, alpha=0.8, label='Model Confidence')
        ax1_twin.set_ylabel('Model Prediction Confidence', fontsize=12, color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1_twin.set_ylim(0, 1)
        
        # Add threshold line
        threshold = 0.985  # Current threshold
        ax1_twin.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                        label=f'Buy Threshold ({threshold})')
        
        # Add buy signals
        model_buy_signals = self.results['Model'][self.results['Model']['buy_signal'] > 0]
        if not model_buy_signals.empty:
            buy_dates = model_buy_signals.index
            buy_prices = self.df[self.df['Date'].isin(buy_dates)]['CP'].values
            buy_confidences = [predictions[dates.index(date)] for date in buy_dates]
            
            ax1.scatter(buy_dates, buy_prices, color='green', s=100, marker='^', 
                       label='Buy Signal', zorder=5, edgecolors='black')
            ax1_twin.scatter(buy_dates, buy_confidences, color='green', s=100, marker='^', 
                           zorder=5, edgecolors='black')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        ax1_twin.legend(loc='upper right', fontsize=10)
        
        ax1.set_title('Model Confidence vs S&P 500 Price', fontsize=16)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Second subplot: Confidence distribution and buy signal analysis
        # Plot confidence distribution over time
        ax2.plot(dates, predictions, color='purple', linewidth=1, alpha=0.6, label='Daily Confidence')
        
        # Add moving average of confidence
        confidence_series = pd.Series(predictions, index=dates)
        confidence_ma = confidence_series.rolling(window=20).mean()
        ax2.plot(dates, confidence_ma, color='purple', linewidth=2, label='20-Day Moving Average')
        
        # Add threshold line
        ax2.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Buy Threshold ({threshold})')
        
        # Highlight periods above threshold
        above_threshold = confidence_series > threshold
        ax2.fill_between(dates, 0, 1, where=above_threshold, alpha=0.2, color='green', 
                        label='Above Threshold')
        
        # Add buy signals
        if not model_buy_signals.empty:
            ax2.scatter(buy_dates, buy_confidences, color='green', s=100, marker='^', 
                       label='Buy Signal', zorder=5, edgecolors='black')
        
        ax2.set_title('Model Confidence Analysis', fontsize=16)
        ax2.set_ylabel('Confidence Level', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_buy_signals(self):
        """Plot the S&P 500 price with buy signals for both strategies"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.plot(self.df['Date'], self.df['CP'], label='S&P 500 Price', color='lightgray', zorder=1)
        
        # Get buy signals for the model strategy
        model_buy_signals = self.results['Model'][self.results['Model']['buy_signal'] > 0]
        # Get the S&P 500 prices for the buy signal dates
        model_buy_dates = model_buy_signals.index
        model_buy_prices = self.df[self.df['Date'].isin(model_buy_dates)]['CP'].values
        
        # Scale marker size by investment amount for better visualization
        marker_sizes = model_buy_signals['buy_signal'] / 5 
        ax.scatter(model_buy_dates, model_buy_prices, 
                   label='Model Buy Signal', marker='^', color='green', s=marker_sizes, zorder=3, edgecolors='black', alpha=0.8)

        # Get buy signals for the DCA strategy
        dca_investment_dates = self.df.groupby(self.df['Date'].dt.to_period('M')).first()['Date']
        dca_buy_signals = self.df[self.df['Date'].isin(dca_investment_dates)]
        ax.scatter(dca_buy_signals['Date'], dca_buy_signals['CP'],
                   label='DCA Buy Signal', marker='o', color='cyan', s=100, zorder=2, edgecolors='black', alpha=0.7)

        ax.set_title('Strategy Buy Signals: DCA vs. Meta-Model', fontsize=18)
        ax.set_ylabel('S&P 500 Price ($)', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()

    def run_comparison(self):
        """Run the full comparison and generate plots"""
        print("\nRunning Dollar Cost Averaging Strategy...")
        self.run_dca_strategy()
        
        print("\nRunning Meta-Model Guided Strategy...")
        self.run_model_strategy()

        # Display key metrics before plotting
        self.display_strategy_metrics()
        
        print("\nGenerating comprehensive comparison plots...")
        self.plot_performance_comparison()
        self.plot_roi_comparison()
        self.plot_drawdown_analysis()
        self.plot_risk_return_analysis()
        self.plot_trading_activity()
        self.plot_model_confidence_vs_price()
        self.plot_buy_signals()
        
        print("\nComparison complete!")

def main():
    """Main function to run the strategy comparison"""
    comparer = StrategyComparer(start_date='2022-01-01')
    comparer.run_comparison()

if __name__ == "__main__":
    main() 