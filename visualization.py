import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SP500DataAnalyzer:
    def __init__(self, data_file='enhanced_sp500_dataset.csv'):
        """Initialize the analyzer with S&P 500 data"""
        print("Loading S&P 500 enhanced dataset...")
        self.df = pd.read_csv(data_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Remove duplicate dates by taking the last entry per date
        self.df = self.df.drop_duplicates(subset=['Date'], keep='last')
        
        print(f"Dataset loaded: {len(self.df)} records from {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        # Clean and prepare data
        self._clean_data()
        self._create_technical_indicators()
        
    def _clean_data(self):
        """Clean and preprocess the data"""
        # Handle missing values
        self.df['Volume'] = pd.to_numeric(self.df['Volume'], errors='coerce')
        self.df['Interest_Rate'] = pd.to_numeric(self.df['Interest_Rate'], errors='coerce')
        self.df['Inflation_Rate'] = pd.to_numeric(self.df['Inflation_Rate'], errors='coerce')
        
        # Forward fill missing values for rates (they change monthly)
        self.df['Interest_Rate'] = self.df['Interest_Rate'].fillna(method='ffill')
        self.df['Inflation_Rate'] = self.df['Inflation_Rate'].fillna(method='ffill')
        
        # Handle missing volume data
        self.df['Volume'] = self.df['Volume'].fillna(self.df['Volume'].median())
        
        print("Data cleaning completed.")
        
    def _create_technical_indicators(self):
        """Create technical indicators for analysis and prediction"""
        # Price-based indicators
        self.df['Returns'] = self.df['CP'].pct_change()
        self.df['Log_Returns'] = np.log(self.df['CP'] / self.df['CP'].shift(1))
        
        # Moving averages
        self.df['MA_5'] = self.df['CP'].rolling(window=5).mean()
        self.df['MA_20'] = self.df['CP'].rolling(window=20).mean()
        self.df['MA_50'] = self.df['CP'].rolling(window=50).mean()
        self.df['MA_200'] = self.df['CP'].rolling(window=200).mean()
        
        # Volatility indicators
        self.df['Volatility_20'] = self.df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # RSI (Relative Strength Index)
        delta = self.df['CP'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        self.df['BB_Middle'] = self.df['CP'].rolling(window=bb_period).mean()
        bb_std_dev = self.df['CP'].rolling(window=bb_period).std()
        self.df['BB_Upper'] = self.df['BB_Middle'] + (bb_std_dev * bb_std)
        self.df['BB_Lower'] = self.df['BB_Middle'] - (bb_std_dev * bb_std)
        self.df['BB_Width'] = self.df['BB_Upper'] - self.df['BB_Lower']
        self.df['BB_Position'] = (self.df['CP'] - self.df['BB_Lower']) / self.df['BB_Width']
        
        # MACD
        exp1 = self.df['CP'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['CP'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Histogram'] = self.df['MACD'] - self.df['MACD_Signal']
        
        # Price momentum
        self.df['Price_Change_5d'] = self.df['CP'].pct_change(5)
        self.df['Price_Change_20d'] = self.df['CP'].pct_change(20)
        
        # Volume indicators
        self.df['Volume_MA_20'] = self.df['Volume'].rolling(window=20).mean()
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_MA_20']
        
        print("Technical indicators created.")
        
    def basic_statistics(self):
        """Display basic statistics about the dataset"""
        print("\n" + "="*60)
        print("BASIC DATASET STATISTICS")
        print("="*60)
        
        # Dataset overview
        print(f"Date Range: {self.df['Date'].min().strftime('%Y-%m-%d')} to {self.df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Total Trading Days: {len(self.df)}")
        print(f"Years Covered: {(self.df['Date'].max() - self.df['Date'].min()).days / 365.25:.1f}")
        
        # Price statistics
        price_stats = self.df['CP'].describe()
        print(f"\nS&P 500 Price Statistics:")
        print(f"  Minimum: ${price_stats['min']:,.2f}")
        print(f"  Maximum: ${price_stats['max']:,.2f}")
        print(f"  Mean: ${price_stats['mean']:,.2f}")
        print(f"  Median: ${price_stats['50%']:,.2f}")
        print(f"  Standard Deviation: ${price_stats['std']:,.2f}")
        
        # Returns statistics
        returns_stats = self.df['Returns'].describe()
        print(f"\nDaily Returns Statistics:")
        print(f"  Mean: {returns_stats['mean']*100:.4f}%")
        print(f"  Standard Deviation: {returns_stats['std']*100:.4f}%")
        print(f"  Minimum: {returns_stats['min']*100:.2f}%")
        print(f"  Maximum: {returns_stats['max']*100:.2f}%")
        
        # Volatility
        annual_vol = self.df['Returns'].std() * np.sqrt(252) * 100
        print(f"  Annualized Volatility: {annual_vol:.2f}%")
        
        # Economic indicators
        print(f"\nEconomic Indicators:")
        print(f"  Interest Rate Range: {self.df['Interest_Rate'].min():.2f}% - {self.df['Interest_Rate'].max():.2f}%")
        print(f"  Inflation Rate Range: {self.df['Inflation_Rate'].min():.2f}% - {self.df['Inflation_Rate'].max():.2f}%")
        
    def create_time_series_plots(self):
        """Create comprehensive time series visualizations"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('S&P 500 Price with Moving Averages', 'Trading Volume', 
                            'Interest Rate', 'Inflation Rate'),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}]]
        )
        
        # S&P 500 Price with Moving Averages
        fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df['CP'], 
                                name='S&P 500 Price', line=dict(color='blue', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df['MA_20'], 
                                name='MA 20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df['MA_50'], 
                                name='MA 50', line=dict(color='red', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df['MA_200'], 
                                name='MA 200', line=dict(color='green', width=1)), row=1, col=1)
        
        # Trading Volume
        fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df['Volume']/1e9, 
                                name='Volume (B)', line=dict(color='purple')), row=2, col=1)
        
        # Interest Rate
        fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df['Interest_Rate'], 
                                name='Interest Rate %', line=dict(color='darkred')), row=3, col=1)
        
        # Inflation Rate
        fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df['Inflation_Rate'], 
                                name='Inflation Rate %', line=dict(color='darkgreen')), row=4, col=1)
        
        fig.update_layout(height=1200, title_text="S&P 500 Data Time Series Analysis", showlegend=True)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (Billions)", row=2, col=1)
        fig.update_yaxes(title_text="Rate (%)", row=3, col=1)
        fig.update_yaxes(title_text="Rate (%)", row=4, col=1)
        
        fig.show()
        
    def create_correlation_analysis(self):
        """Analyze correlations between variables"""
        # Select numeric columns for correlation
        corr_columns = ['CP', 'Volume', 'Interest_Rate', 'Inflation_Rate', 
                        'Returns', 'Volatility_20', 'RSI', 'MACD']
        
        corr_data = self.df[corr_columns].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix: S&P 500 Metrics and Economic Indicators', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
        # Print strongest correlations
        print("\n" + "="*60)
        print("STRONGEST CORRELATIONS WITH S&P 500 PRICE")
        print("="*60)
        correlations = corr_data['CP'].drop('CP').sort_values(key=abs, ascending=False)
        for var, corr in correlations.head(5).items():
            print(f"{var:20}: {corr:6.3f}")
            
    def create_technical_analysis_plots(self):
        """Create technical analysis visualizations"""
        # RSI and Price
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Price with Bollinger Bands
        recent_data = self.df.tail(252)  # Last year
        ax1.plot(recent_data['Date'], recent_data['CP'], label='S&P 500', color='blue', linewidth=2)
        ax1.fill_between(recent_data['Date'], recent_data['BB_Lower'], recent_data['BB_Upper'], 
                        alpha=0.2, color='gray', label='Bollinger Bands')
        ax1.plot(recent_data['Date'], recent_data['BB_Upper'], color='red', alpha=0.7)
        ax1.plot(recent_data['Date'], recent_data['BB_Lower'], color='red', alpha=0.7)
        ax1.set_title('S&P 500 with Bollinger Bands (Last Year)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI
        ax2.plot(recent_data['Date'], recent_data['RSI'], color='purple', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.fill_between(recent_data['Date'], 30, 70, alpha=0.1, color='yellow')
        ax2.set_title('RSI (Relative Strength Index)', fontsize=14)
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MACD
        ax3.plot(recent_data['Date'], recent_data['MACD'], label='MACD', color='blue')
        ax3.plot(recent_data['Date'], recent_data['MACD_Signal'], label='Signal', color='red')
        ax3.bar(recent_data['Date'], recent_data['MACD_Histogram'], label='Histogram', 
                alpha=0.3, color='green')
        ax3.set_title('MACD Analysis', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Volatility
        ax4.plot(recent_data['Date'], recent_data['Volatility_20']*100, color='orange', linewidth=2)
        ax4.set_title('20-Day Rolling Volatility (%)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_returns_distribution(self):
        """Analyze the distribution of returns"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Daily returns histogram
        returns = self.df['Returns'].dropna() * 100
        ax1.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.3f}%')
        ax1.set_title('Distribution of Daily Returns', fontsize=14)
        ax1.set_xlabel('Daily Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot for normality
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot: Returns vs Normal Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax3.plot(self.df['Date'], self.df['Volatility_20']*100, color='orange')
        ax3.set_title('20-Day Rolling Volatility (%)', fontsize=14)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volatility (%)')
        ax3.grid(True, alpha=0.3)
        
        # Monthly returns boxplot
        monthly_returns = self.df.set_index('Date')['Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_data = []
        months = []
        for date, ret in monthly_returns.items():
            monthly_data.append(ret)
            months.append(date.strftime('%b'))
        
        # Group by month name
        month_dict = {}
        for i, month in enumerate(months):
            if month not in month_dict:
                month_dict[month] = []
            month_dict[month].append(monthly_data[i])
            
        months_ordered = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        box_data = [month_dict.get(month, []) for month in months_ordered]
        
        ax4.boxplot(box_data, labels=months_ordered)
        ax4.set_title('Monthly Returns Distribution by Month', fontsize=14)
        ax4.set_ylabel('Monthly Return (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\n" + "="*60)
        print("RETURNS DISTRIBUTION ANALYSIS")
        print("="*60)
        print(f"Skewness: {stats.skew(returns):.4f}")
        print(f"Kurtosis: {stats.kurtosis(returns):.4f}")
        print(f"Sharpe Ratio (daily): {returns.mean() / returns.std():.4f}")
        print(f"Max Drawdown: {((self.df['CP'] / self.df['CP'].cummax()) - 1).min()*100:.2f}%")
        
    def economic_indicators_analysis(self):
        """Analyze relationship between economic indicators and S&P 500"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Interest rates vs S&P 500
        ax1.scatter(self.df['Interest_Rate'], self.df['CP'], alpha=0.6, color='blue')
        z = np.polyfit(self.df['Interest_Rate'].dropna(), 
                        self.df.loc[self.df['Interest_Rate'].notna(), 'CP'], 1)
        p = np.poly1d(z)
        ax1.plot(self.df['Interest_Rate'], p(self.df['Interest_Rate']), "r--", alpha=0.8)
        ax1.set_xlabel('Interest Rate (%)')
        ax1.set_ylabel('S&P 500 Price')
        ax1.set_title('Interest Rate vs S&P 500 Price')
        ax1.grid(True, alpha=0.3)
        
        # Inflation vs S&P 500
        ax2.scatter(self.df['Inflation_Rate'], self.df['CP'], alpha=0.6, color='green')
        z2 = np.polyfit(self.df['Inflation_Rate'].dropna(), 
                        self.df.loc[self.df['Inflation_Rate'].notna(), 'CP'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(self.df['Inflation_Rate'], p2(self.df['Inflation_Rate']), "r--", alpha=0.8)
        ax2.set_xlabel('Inflation Rate (%)')
        ax2.set_ylabel('S&P 500 Price')
        ax2.set_title('Inflation Rate vs S&P 500 Price')
        ax2.grid(True, alpha=0.3)
        
        # Time series of interest rates and S&P 500
        ax3_twin = ax3.twinx()
        ax3.plot(self.df['Date'], self.df['Interest_Rate'], color='red', label='Interest Rate')
        ax3_twin.plot(self.df['Date'], self.df['CP'], color='blue', alpha=0.7, label='S&P 500')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Interest Rate (%)', color='red')
        ax3_twin.set_ylabel('S&P 500 Price', color='blue')
        ax3.set_title('Interest Rate and S&P 500 Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Time series of inflation and S&P 500
        ax4_twin = ax4.twinx()
        ax4.plot(self.df['Date'], self.df['Inflation_Rate'], color='green', label='Inflation Rate')
        ax4_twin.plot(self.df['Date'], self.df['CP'], color='blue', alpha=0.7, label='S&P 500')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Inflation Rate (%)', color='green')
        ax4_twin.set_ylabel('S&P 500 Price', color='blue')
        ax4.set_title('Inflation Rate and S&P 500 Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def prepare_prediction_features(self):
        """Prepare features for future price prediction"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING FOR PREDICTION")
        print("="*60)
        
        # Create lagged features
        lags = [1, 5, 10, 20]
        for lag in lags:
            self.df[f'Price_Lag_{lag}'] = self.df['CP'].shift(lag)
            self.df[f'Returns_Lag_{lag}'] = self.df['Returns'].shift(lag)
            self.df[f'Volume_Lag_{lag}'] = self.df['Volume'].shift(lag)
        
        # Create future targets (1 week and 1 month ahead)
        self.df['Price_1Week_Future'] = self.df['CP'].shift(-5)  # 5 trading days
        self.df['Price_1Month_Future'] = self.df['CP'].shift(-20)  # 20 trading days
        self.df['Return_1Week_Future'] = (self.df['Price_1Week_Future'] / self.df['CP']) - 1
        self.df['Return_1Month_Future'] = (self.df['Price_1Month_Future'] / self.df['CP']) - 1
        
        # Feature list for prediction
        feature_columns = [
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate',
            'MA_5', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Position', 'BB_Width', 'Volatility_20', 'Volume_Ratio',
            'Price_Change_5d', 'Price_Change_20d'
        ] + [f'Price_Lag_{lag}' for lag in lags] + [f'Returns_Lag_{lag}' for lag in lags]
        
        # Remove rows with NaN values for feature analysis
        feature_df = self.df[feature_columns + ['Return_1Week_Future', 'Return_1Month_Future']].dropna()
        
        print(f"Available complete records for modeling: {len(feature_df)}")
        print(f"Features available: {len(feature_columns)}")
        
        # Feature importance analysis using correlation with future returns
        correlations_1week = feature_df[feature_columns].corrwith(feature_df['Return_1Week_Future'])
        correlations_1month = feature_df[feature_columns].corrwith(feature_df['Return_1Month_Future'])
        
        print(f"\nTop 10 Features Correlated with 1-Week Future Returns:")
        print("-" * 55)
        for feature, corr in correlations_1week.abs().sort_values(ascending=False).head(10).items():
            print(f"{feature:25}: {correlations_1week[feature]:6.3f}")
            
        print(f"\nTop 10 Features Correlated with 1-Month Future Returns:")
        print("-" * 55)
        for feature, corr in correlations_1month.abs().sort_values(ascending=False).head(10).items():
            print(f"{feature:25}: {correlations_1month[feature]:6.3f}")
        
        # Save processed data for modeling
        self.df.to_csv('sp500_features_for_prediction.csv', index=False)
        print(f"\nFeature-engineered dataset saved as 'sp500_features_for_prediction.csv'")
        
        return feature_columns, feature_df
        
    def create_comprehensive_dashboard(self):
        """Create a comprehensive analysis dashboard"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE S&P 500 ANALYSIS")
        print("="*80)
        
        # Run all analyses
        self.basic_statistics()
        self.create_time_series_plots()
        self.create_correlation_analysis()
        self.create_technical_analysis_plots()
        self.analyze_returns_distribution()
        self.economic_indicators_analysis()
        feature_columns, feature_df = self.prepare_prediction_features()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - READY FOR PREDICTIVE MODELING")
        print("="*80)
        print("Next steps for prediction:")
        print("1. Use 'sp500_features_for_prediction.csv' for model training")
        print("2. Consider ensemble methods (Random Forest, XGBoost, Neural Networks)")
        print("3. Implement proper train/validation/test splits with time series")
        print("4. Evaluate models using appropriate metrics (MAE, RMSE, directional accuracy)")
        print("5. Consider regime-based models due to economic cycles")

def main():
    """Main function to run the complete analysis"""
    # Initialize analyzer
    analyzer = SP500DataAnalyzer('enhanced_sp500_dataset.csv')
    
    # Run comprehensive analysis
    analyzer.create_comprehensive_dashboard()

if __name__ == "__main__":
    main()
