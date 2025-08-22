import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('data/final_dataset_for_modeling.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def analyze_feature_importance_with_rf(df):
    """Analyze feature importance using Random Forest"""
    # Prepare data
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('Target')
    
    # Remove Price_Change_14d if it exists (it's the target variable essentially)
    if 'Price_Change_14d' in numerical_cols:
        numerical_cols.remove('Price_Change_14d')
    
    X = df[numerical_cols]
    y = df['Target']
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': numerical_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

def analyze_feature_interactions(df):
    """Analyze potential feature interactions"""
    # Select top features for interaction analysis
    top_features = ['RSI', 'MACD', 'BB_Position', 'Volatility_14', 'Sentiment_Score', 
                   'Close_Lag_1', 'Volume_Lag_1', 'Interest_Rate']
    
    interaction_results = []
    
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            if feat1 in df.columns and feat2 in df.columns:
                # Create interaction feature
                interaction = df[feat1] * df[feat2]
                corr = interaction.corr(df['Target'])
                interaction_results.append({
                    'Feature1': feat1,
                    'Feature2': feat2,
                    'Interaction_Correlation': corr
                })
    
    interaction_df = pd.DataFrame(interaction_results)
    interaction_df = interaction_df.sort_values('Interaction_Correlation', key=abs, ascending=False)
    
    return interaction_df

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data"""
    # Add time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    
    # Analyze success rate by time periods
    temporal_analysis = {}
    
    for period in ['Year', 'Month', 'DayOfWeek', 'Quarter']:
        period_success = df.groupby(period)['Target'].agg(['mean', 'count'])
        temporal_analysis[period] = period_success
    
    return temporal_analysis

def create_comprehensive_report(df, rf_importance, interactions, temporal_analysis):
    """Create a comprehensive analysis report"""
    
    print("="*100)
    print("COMPREHENSIVE ML FEATURE SELECTION REPORT")
    print("="*100)
    
    print(f"\n1. DATASET OVERVIEW")
    print(f"   - Total samples: {len(df):,}")
    print(f"   - Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   - Target distribution: {dict(df['Target'].value_counts())}")
    print(f"   - Success rate: {df['Target'].mean():.2%}")
    
    print(f"\n2. TOP 20 FEATURES BY RANDOM FOREST IMPORTANCE")
    top_20_rf = rf_importance.head(20)
    for i, (_, row) in enumerate(top_20_rf.iterrows(), 1):
        print(f"   {i:2d}. {row['Feature']:25s}: {row['Importance']:.4f}")
    
    print(f"\n3. TOP 10 FEATURE INTERACTIONS")
    top_10_interactions = interactions.head(10)
    for i, (_, row) in enumerate(top_10_interactions.iterrows(), 1):
        print(f"   {i:2d}. {row['Feature1']:15s} × {row['Feature2']:15s}: {row['Interaction_Correlation']:.4f}")
    
    print(f"\n4. TEMPORAL PATTERNS ANALYSIS")
    
    # Yearly patterns
    yearly = temporal_analysis['Year']
    print(f"   Success Rate by Year (Top 5):")
    yearly_success = yearly.sort_values('mean', ascending=False).head(5)
    for year, row in yearly_success.iterrows():
        print(f"      {year}: {row['mean']:.3f} ({row['count']} samples)")
    
    # Monthly patterns
    monthly = temporal_analysis['Month']
    print(f"   Success Rate by Month:")
    for month in range(1, 13):
        if month in monthly.index:
            success_rate = monthly.loc[month, 'mean']
            count = monthly.loc[month, 'count']
            print(f"      {month:2d}: {success_rate:.3f} ({count} samples)")
    
    # Day of week patterns
    dow = temporal_analysis['DayOfWeek']
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"   Success Rate by Day of Week:")
    for i, day_name in enumerate(dow_names):
        if i in dow.index:
            success_rate = dow.loc[i, 'mean']
            count = dow.loc[i, 'count']
            print(f"      {day_name:10s}: {success_rate:.3f} ({count} samples)")
    
    print(f"\n5. FEATURE CATEGORIES ANALYSIS")
    
    # Technical indicators
    tech_features = ['RSI', 'MACD', 'BB_Position', 'Volatility_14', 'MA_7', 'MA_14', 'MA_30']
    tech_importance = rf_importance[rf_importance['Feature'].isin(tech_features)]
    print(f"   Technical Indicators (Top 5 by RF importance):")
    for _, row in tech_importance.head(5).iterrows():
        print(f"      {row['Feature']:20s}: {row['Importance']:.4f}")
    
    # Sentiment features
    sent_features = [col for col in df.columns if 'Sentiment' in col]
    sent_importance = rf_importance[rf_importance['Feature'].isin(sent_features)]
    print(f"   Sentiment Features (Top 5 by RF importance):")
    for _, row in sent_importance.head(5).iterrows():
        print(f"      {row['Feature']:20s}: {row['Importance']:.4f}")
    
    # Economic indicators
    econ_features = ['GDP', 'Gold_Price', 'Unemployment_Rate', 'Interest_Rate', 'Inflation_Rate']
    econ_importance = rf_importance[rf_importance['Feature'].isin(econ_features)]
    print(f"   Economic Indicators (Top 5 by RF importance):")
    for _, row in econ_importance.head(5).iterrows():
        print(f"      {row['Feature']:20s}: {row['Importance']:.4f}")
    
    # Price/Volume features
    price_features = [col for col in df.columns if any(x in col for x in ['Close', 'Volume', 'Return'])]
    price_importance = rf_importance[rf_importance['Feature'].isin(price_features)]
    print(f"   Price/Volume Features (Top 5 by RF importance):")
    for _, row in price_importance.head(5).iterrows():
        print(f"      {row['Feature']:20s}: {row['Importance']:.4f}")
    
    print(f"\n6. ML MODELING RECOMMENDATIONS")
    print(f"   A. Feature Selection Strategy:")
    print(f"      - Use top 15-20 features based on Random Forest importance")
    print(f"      - Include key interaction features (RSI×MACD, Sentiment×Volatility)")
    print(f"      - Consider temporal features (month, day of week)")
    
    print(f"   B. Model Architecture:")
    print(f"      - Use ensemble methods (Random Forest, XGBoost, LightGBM)")
    print(f"      - Implement time series cross-validation")
    print(f"      - Handle class imbalance with class weights or SMOTE")
    
    print(f"   C. Feature Engineering:")
    print(f"      - Create RSI×MACD interaction feature")
    print(f"      - Add sentiment×volatility interaction")
    print(f"      - Create momentum indicators (price change over different periods)")
    print(f"      - Add volatility-based features")
    
    print(f"   D. Validation Strategy:")
    print(f"      - Use time-based train/test splits")
    print(f"      - Implement walk-forward analysis")
    print(f"      - Focus on out-of-sample performance")
    
    print(f"\n7. MOST PROMISING FEATURES FOR ML MODELING")
    print(f"   Based on Random Forest importance, the most promising features are:")
    
    top_features = rf_importance.head(15)['Feature'].tolist()
    for i, feature in enumerate(top_features, 1):
        importance = rf_importance[rf_importance['Feature'] == feature]['Importance'].iloc[0]
        print(f"   {i:2d}. {feature:25s} (Importance: {importance:.4f})")
    
    print(f"\n8. POTENTIAL FEATURE INTERACTIONS TO ENGINEER")
    print(f"   Based on correlation analysis, consider these interactions:")
    
    top_interactions = interactions.head(8)
    for i, (_, row) in enumerate(top_interactions.iterrows(), 1):
        print(f"   {i:2d}. {row['Feature1']:15s} × {row['Feature2']:15s} (Corr: {row['Interaction_Correlation']:.4f})")
    
    print(f"\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"The analysis reveals that the most important features for predicting S&P 500")
    print(f"price movements are a combination of technical indicators (RSI, MACD, BB_Position),")
    print(f"sentiment analysis (with lagged values), economic indicators (Gold Price,")
    print(f"Unemployment Rate), and price/volume features. The dataset shows temporal")
    print(f"patterns that should be incorporated into the model design.")
    
    return {
        'top_features': top_features,
        'top_interactions': top_interactions,
        'temporal_patterns': temporal_analysis
    }

def create_feature_importance_visualization(rf_importance, interactions):
    """Create comprehensive feature importance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    
    # Top 20 features by Random Forest importance
    top_20 = rf_importance.head(20)
    axes[0, 0].barh(range(len(top_20)), top_20['Importance'])
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20['Feature'])
    axes[0, 0].set_title('Top 20 Features by Random Forest Importance', fontsize=16, fontweight='bold')
    axes[0, 0].set_xlabel('Importance Score', fontsize=12)
    
    # Feature categories importance
    categories = {
        'Technical': ['RSI', 'MACD', 'BB_Position', 'Volatility_14', 'MA_7', 'MA_14', 'MA_30'],
        'Sentiment': [col for col in rf_importance['Feature'] if 'Sentiment' in col],
        'Economic': ['GDP', 'Gold_Price', 'Unemployment_Rate', 'Interest_Rate', 'Inflation_Rate'],
        'Price/Volume': [col for col in rf_importance['Feature'] if any(x in col for x in ['Close', 'Volume', 'Return'])]
    }
    
    category_importance = {}
    for category, features in categories.items():
        category_features = rf_importance[rf_importance['Feature'].isin(features)]
        category_importance[category] = category_features['Importance'].sum()
    
    axes[0, 1].pie(category_importance.values(), labels=category_importance.keys(), autopct='%1.1f%%')
    axes[0, 1].set_title('Feature Importance by Category', fontsize=16, fontweight='bold')
    
    # Top interactions
    top_interactions = interactions.head(10)
    interaction_labels = [f"{row['Feature1']}×{row['Feature2']}" for _, row in top_interactions.iterrows()]
    axes[1, 0].barh(range(len(top_interactions)), top_interactions['Interaction_Correlation'])
    axes[1, 0].set_yticks(range(len(top_interactions)))
    axes[1, 0].set_yticklabels(interaction_labels)
    axes[1, 0].set_title('Top 10 Feature Interactions', fontsize=16, fontweight='bold')
    axes[1, 0].set_xlabel('Correlation with Target', fontsize=12)
    
    # Feature importance distribution
    axes[1, 1].hist(rf_importance['Importance'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Feature Importance Scores', fontsize=16, fontweight='bold')
    axes[1, 1].set_xlabel('Importance Score', fontsize=12)
    axes[1, 1].set_ylabel('Number of Features', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/visualization/comprehensive_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the detailed analysis"""
    print("Starting detailed feature analysis...")
    
    # Load data
    df = load_data()
    
    # Perform analyses
    print("Analyzing feature importance with Random Forest...")
    rf_importance = analyze_feature_importance_with_rf(df)
    
    print("Analyzing feature interactions...")
    interactions = analyze_feature_interactions(df)
    
    print("Analyzing temporal patterns...")
    temporal_analysis = analyze_temporal_patterns(df)
    
    print("Creating comprehensive report...")
    results = create_comprehensive_report(df, rf_importance, interactions, temporal_analysis)
    
    print("Creating visualizations...")
    create_feature_importance_visualization(rf_importance, interactions)
    
    print("\nDetailed analysis complete!")
    print("Check 'plots/visualization/comprehensive_feature_analysis.png' for visualizations.")

if __name__ == "__main__":
    main()
