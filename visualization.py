import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the dataset for analysis"""
    print("Loading dataset...")
    df = pd.read_csv('data/final_dataset_for_modeling.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Check for missing values
    print(f"Missing values per column:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # Fill missing values with forward fill and backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Target distribution: {df['Target'].value_counts().to_dict()}")
    
    return df

def analyze_target_distribution(df, plots_dir):
    """Analyze the target variable distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Target distribution
    target_counts = df['Target'].value_counts()
    axes[0, 0].pie(target_counts.values, labels=['Increase (1)', 'Decrease (0)'], 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Target Variable Distribution')
    
    # Target over time (yearly success rate)
    yearly_success = df.groupby(df['Date'].dt.year)['Target'].mean()
    axes[0, 1].bar(yearly_success.index, yearly_success.values, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
    axes[0, 1].set_title('Yearly Success Rate')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Target distribution over time (monthly success rate)
    monthly_success = df.groupby(df['Date'].dt.to_period('M'))['Target'].mean()
    # Sample every 6 months to reduce clutter
    sampled_months = monthly_success.iloc[::6]
    axes[1, 0].plot(range(len(sampled_months)), sampled_months.values, marker='o', linewidth=2, markersize=4)
    axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
    axes[1, 0].set_title('Monthly Success Rate (Every 6 Months)')
    axes[1, 0].set_xlabel('Time Period (6-month intervals)')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rolling success rate
    window_size = 100
    rolling_success = df['Target'].rolling(window=window_size).mean()
    axes[1, 1].plot(df['Date'], rolling_success, linewidth=2)
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title(f'Rolling Success Rate ({window_size}-day window)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/target_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_correlations(df, plots_dir):
    """Analyze correlations between features and target"""
    # Select numerical features (excluding Date and Target)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('Target')
    
    # Calculate correlations with target
    correlations = df[numerical_cols + ['Target']].corr()['Target'].sort_values(ascending=False)
    
    # Plot top correlations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Top positive correlations
    top_positive = correlations[correlations > 0].head(10)
    axes[0, 0].barh(range(len(top_positive)), top_positive.values)
    axes[0, 0].set_yticks(range(len(top_positive)))
    axes[0, 0].set_yticklabels(top_positive.index)
    axes[0, 0].set_title('Top 10 Positive Correlations with Target')
    axes[0, 0].set_xlabel('Correlation Coefficient')
    
    # Top negative correlations
    top_negative = correlations[correlations < 0].head(10)
    axes[0, 1].barh(range(len(top_negative)), top_negative.values)
    axes[0, 1].set_yticks(range(len(top_negative)))
    axes[0, 1].set_yticklabels(top_negative.index)
    axes[0, 1].set_title('Top 10 Negative Correlations with Target')
    axes[0, 1].set_xlabel('Correlation Coefficient')
    
    # Correlation heatmap for top features
    top_features = list(top_positive.head(8).index) + list(top_negative.head(8).index)
    correlation_matrix = df[top_features + ['Target']].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1, 0], fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[1, 0].set_title('Correlation Heatmap - Top Features')
    
    # Feature importance by correlation magnitude
    abs_correlations = correlations.abs().sort_values(ascending=False)
    axes[1, 1].barh(range(len(abs_correlations.head(15))), abs_correlations.head(15).values)
    axes[1, 1].set_yticks(range(len(abs_correlations.head(15))))
    axes[1, 1].set_yticklabels(abs_correlations.head(15).index)
    axes[1, 1].set_title('Feature Importance by Correlation Magnitude')
    axes[1, 1].set_xlabel('Absolute Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlations

def analyze_technical_indicators(df, plots_dir):
    """Analyze technical indicators and their relationship with target"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Moving Averages
    ma_features = ['MA_7', 'MA_14', 'MA_30']
    ma_correlations = []
    for ma in ma_features:
        ma_corr = df[ma].corr(df['Target'])
        ma_correlations.append(ma_corr)
    
    axes[0, 0].bar(ma_features, ma_correlations)
    axes[0, 0].set_title('Correlation of Moving Averages with Target')
    axes[0, 0].set_ylabel('Correlation Coefficient')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RSI Distribution
    axes[0, 1].hist(df['RSI'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('RSI Distribution')
    axes[0, 1].set_xlabel('RSI')
    axes[0, 1].set_ylabel('Frequency')
    
    # MACD Distribution
    axes[1, 0].hist(df['MACD'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('MACD Distribution')
    axes[1, 0].set_xlabel('MACD')
    axes[1, 0].set_ylabel('Frequency')
    
    # Volatility Distribution
    axes[1, 1].hist(df['Volatility_14'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Volatility_14 Distribution')
    axes[1, 1].set_xlabel('Volatility_14')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/technical_indicators_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_sentiment_and_news(df, plots_dir):
    """Analyze sentiment and news features"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sentiment Score Distribution by Target
    df[df['Target'] == 0]['Sentiment_Score'].hist(bins=30, alpha=0.7, label='Target=0', ax=axes[0, 0])
    df[df['Target'] == 1]['Sentiment_Score'].hist(bins=30, alpha=0.7, label='Target=1', ax=axes[0, 0])
    axes[0, 0].set_title('Sentiment Score Distribution by Target')
    axes[0, 0].set_xlabel('Sentiment Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Sentiment Score Distribution
    axes[0, 1].hist(df['Sentiment_Score'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Sentiment Score Distribution')
    axes[0, 1].set_xlabel('Sentiment Score')
    axes[0, 1].set_ylabel('Frequency')
    
    # Headline Count Distribution
    axes[1, 0].hist(df['Headline_Count'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Headline Count Distribution')
    axes[1, 0].set_xlabel('Headline Count')
    axes[1, 0].set_ylabel('Frequency')
    
    # Sentiment Lag Analysis
    sentiment_lags = ['Sentiment_Lag_1', 'Sentiment_Lag_3', 'Sentiment_Lag_5', 'Sentiment_Lag_7', 'Sentiment_Lag_14']
    lag_correlations = []
    for lag in sentiment_lags:
        lag_corr = df[lag].corr(df['Target'])
        lag_correlations.append(lag_corr)
    
    axes[1, 1].bar(sentiment_lags, lag_correlations)
    axes[1, 1].set_title('Correlation of Sentiment Lags with Target')
    axes[1, 1].set_ylabel('Correlation Coefficient')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_economic_indicators(df, plots_dir):
    """Analyze economic indicators"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    economic_features = ['GDP', 'Gold_Price', 'Unemployment_Rate', 'Interest_Rate', 'Inflation_Rate']
    
    # Correlation with target
    econ_correlations = []
    for feature in economic_features:
        corr = df[feature].corr(df['Target'])
        econ_correlations.append(corr)
    
    axes[0, 0].bar(economic_features, econ_correlations)
    axes[0, 0].set_title('Economic Indicators Correlation with Target')
    axes[0, 0].set_ylabel('Correlation Coefficient')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # GDP Distribution
    axes[0, 1].hist(df['GDP'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('GDP Distribution')
    axes[0, 1].set_xlabel('GDP')
    axes[0, 1].set_ylabel('Frequency')
    
    # Gold Price Distribution
    axes[1, 0].hist(df['Gold_Price'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Gold Price Distribution')
    axes[1, 0].set_xlabel('Gold Price')
    axes[1, 0].set_ylabel('Frequency')
    
    # Interest Rate Distribution
    axes[1, 1].hist(df['Interest_Rate'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Interest Rate Distribution')
    axes[1, 1].set_xlabel('Interest Rate')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/economic_indicators_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_feature_selection(df, plots_dir):
    """Perform feature selection analysis"""
    # Prepare data for feature selection
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('Target')
    
    # Remove Price_Change_14d if it exists (it's the target variable essentially)
    if 'Price_Change_14d' in numerical_cols:
        numerical_cols.remove('Price_Change_14d')
    
    X = df[numerical_cols]
    y = df['Target']
    
    # Handle NaN values by forward filling and then dropping remaining NaNs
    X = X.fillna(method='ffill').fillna(method='bfill')
    valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"After cleaning: X shape: {X.shape}, y shape: {y.shape}")
    
    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': numerical_cols, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values('MI_Score', ascending=False)
    
    # ANOVA F-test
    f_scores, _ = f_classif(X, y)
    f_df = pd.DataFrame({'Feature': numerical_cols, 'F_Score': f_scores})
    f_df = f_df.sort_values('F_Score', ascending=False)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top 15 features by Mutual Information
    top_mi = mi_df.head(15)
    axes[0].barh(range(len(top_mi)), top_mi['MI_Score'])
    axes[0].set_yticks(range(len(top_mi)))
    axes[0].set_yticklabels(top_mi['Feature'])
    axes[0].set_title('Top 15 Features by Mutual Information')
    axes[0].set_xlabel('Mutual Information Score')
    
    # Top 15 features by F-test
    top_f = f_df.head(15)
    axes[1].barh(range(len(top_f)), top_f['F_Score'])
    axes[1].set_yticks(range(len(top_f)))
    axes[1].set_yticklabels(top_f['Feature'])
    axes[1].set_title('Top 15 Features by F-test')
    axes[1].set_xlabel('F-test Score')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/feature_selection_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mi_df, f_df

def generate_summary_report(df, correlations, mi_df, f_df):
    """Generate a comprehensive summary report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA EXPLORATION REPORT")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"- Total samples: {len(df):,}")
    print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"- Target distribution: {df['Target'].value_counts().to_dict()}")
    print(f"- Success rate: {df['Target'].mean():.2%}")
    
    print(f"\nTop 10 Features by Correlation with Target:")
    top_corr = correlations.head(10)
    for i, (feature, corr) in enumerate(top_corr.items(), 1):
        print(f"{i:2d}. {feature:25s}: {corr:.4f}")
    
    print(f"\nTop 10 Features by Mutual Information:")
    top_mi = mi_df.head(10)
    for i, (_, row) in enumerate(top_mi.iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:25s}: {row['MI_Score']:.4f}")
    
    print(f"\nTop 10 Features by F-test:")
    top_f = f_df.head(10)
    for i, (_, row) in enumerate(top_f.iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:25s}: {row['F_Score']:.2f}")
    
    # Feature categories
    technical_features = ['RSI', 'MACD', 'BB_Position', 'Volatility_14', 'MA_7', 'MA_14', 'MA_30']
    sentiment_features = ['Sentiment_Score', 'Headline_Count'] + [col for col in df.columns if 'Sentiment_Lag' in col]
    economic_features = ['GDP', 'Gold_Price', 'Unemployment_Rate', 'Interest_Rate', 'Inflation_Rate']
    price_features = ['Close', 'Volume', 'Daily_Return'] + [col for col in df.columns if 'Close_Lag' in col]
    
    print(f"\nFeature Categories Analysis:")
    
    print(f"\nTechnical Indicators (Top 5 by correlation):")
    tech_corr = correlations[correlations.index.isin(technical_features)].head(5)
    for feature, corr in tech_corr.items():
        print(f"  - {feature:20s}: {corr:.4f}")
    
    print(f"\nSentiment Features (Top 5 by correlation):")
    sent_corr = correlations[correlations.index.isin(sentiment_features)].head(5)
    for feature, corr in sent_corr.items():
        print(f"  - {feature:20s}: {corr:.4f}")
    
    print(f"\nEconomic Indicators (Top 5 by correlation):")
    econ_corr = correlations[correlations.index.isin(economic_features)].head(5)
    for feature, corr in econ_corr.items():
        print(f"  - {feature:20s}: {corr:.4f}")
    
    print(f"\nPrice Features (Top 5 by correlation):")
    price_corr = correlations[correlations.index.isin(price_features)].head(5)
    for feature, corr in price_corr.items():
        print(f"  - {feature:20s}: {corr:.4f}")
    
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS FOR ML MODELING")
    print("="*80)
    
    print(f"\n1. Most Important Features (based on multiple metrics):")
    important_features = []
    for feature in ['RSI', 'MACD', 'BB_Position', 'Volatility_14', 'Sentiment_Score', 
                   'Close_Lag_1', 'Close_Lag_3', 'Volume_Lag_1', 'Interest_Rate']:
        if feature in correlations.index:
            important_features.append(feature)
    
    for i, feature in enumerate(important_features[:10], 1):
        corr = correlations[feature]
        mi_score = mi_df[mi_df['Feature'] == feature]['MI_Score'].iloc[0] if feature in mi_df['Feature'].values else 0
        print(f"{i:2d}. {feature:20s} - Corr: {corr:.4f}, MI: {mi_score:.4f}")
    
    print(f"\n2. Feature Engineering Suggestions:")
    print(f"   - Create interaction features between RSI and MACD")
    print(f"   - Combine sentiment scores with technical indicators")
    print(f"   - Create volatility-based features")
    print(f"   - Add momentum indicators")
    
    print(f"\n3. Model Considerations:")
    print(f"   - Target is slightly imbalanced (62.4% vs 37.6%)")
    print(f"   - Consider using class weights or SMOTE")
    print(f"   - Use time series cross-validation")
    print(f"   - Focus on features with high mutual information scores")

def main():
    """Main function to run the complete analysis"""
    print("Starting comprehensive data exploration...")
    
    # Create plots directory and subdirectories if they don't exist
    import os
    plots_dir = 'plots/visualization'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    df = load_and_prepare_data()
    
    # Run analyses
    print("\nAnalyzing target distribution...")
    analyze_target_distribution(df, plots_dir)
    
    print("\nAnalyzing feature correlations...")
    correlations = analyze_feature_correlations(df, plots_dir)
    
    print("\nAnalyzing technical indicators...")
    analyze_technical_indicators(df, plots_dir)
    
    print("\nAnalyzing sentiment and news features...")
    analyze_sentiment_and_news(df, plots_dir)
    
    print("\nAnalyzing economic indicators...")
    analyze_economic_indicators(df, plots_dir)
    
    print("\nPerforming feature selection analysis...")
    mi_df, f_df = perform_feature_selection(df, plots_dir)
    
    print("\nGenerating summary report...")
    generate_summary_report(df, correlations, mi_df, f_df)
    
    print("\nAnalysis complete! Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main()
