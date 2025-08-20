import pandas as pd
import numpy as np
from datetime import datetime

# Optional sentiment imports (robust fallback if unavailable)
try:
    from transformers import pipeline
    _has_transformers = True
except Exception:
    _has_transformers = False

print("Loading yfinance S&P 500 dataset (cleaned)...")
yf_path = 'data/yfinance_sp500_cleaned.csv'
yf_df = pd.read_csv(yf_path)
yf_df['Date'] = pd.to_datetime(yf_df['Date'])

# Standardize columns and types
rename_map = {
    'Adj_Close': 'Adj_Close',
    'Adj Close': 'Adj_Close',
    'Close': 'Close',
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Volume': 'Volume'
}
yf_df = yf_df.rename(columns=rename_map)

# Use Close as CP for consistency with downstream code
yf_df['CP'] = pd.to_numeric(yf_df.get('Close', yf_df.get('Adj_Close')), errors='coerce')
yf_df['Volume'] = pd.to_numeric(yf_df['Volume'], errors='coerce')

# Keep only necessary columns but retain OHLC for potential future use
base_cols = ['Date', 'CP', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
available_cols = [c for c in base_cols if c in yf_df.columns]
df_base = yf_df[available_cols].copy()

print("Loading S&P 500 headlines dataset for sentiment and counts...")
news_df = pd.read_csv('data/sp500 headlines 2008 to 2024.csv')
news_df['Date'] = pd.to_datetime(news_df['Date'])
news_df = news_df[news_df['Date'] >= '2008-01-02'].reset_index(drop=True)

print("Aggregating daily headline counts and computing daily sentiment...")

# Headline counts
headline_counts = news_df.groupby('Date').size().reset_index(name='Headline_Count')

# Sentiment computation
def _compute_daily_sentiment(df_group: pd.DataFrame) -> float:
    titles = df_group['Title'].dropna().astype(str).tolist() if 'Title' in df_group.columns else []
    if not titles:
        return np.nan
    # Try transformers pipeline first
    if _has_transformers:
        try:
            clf = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
            preds = clf(titles, truncation=True)
            # Map to [-1, 1] via score, POS -> +score, NEG -> -score
            scores = [p['score'] if p['label'].upper().startswith('POS') else -p['score'] for p in preds]
            return float(np.mean(scores))
        except Exception:
            pass
    # Fallback lexicon-based sentiment
    positive_words = set([
        'gain','gains','up','rise','rises','surge','surges','bull','bullish','beat','beats','strong','positive','jump','jumps','rally','rallies','record','optimism','better','improve','improves','improved'
    ])
    negative_words = set([
        'loss','losses','down','fall','falls','drop','drops','bear','bearish','miss','misses','weak','negative','plunge','plunges','selloff','fear','worse','decline','declines','declined','crash','crashes'
    ])
    scores = []
    for t in titles:
        tokens = [tok.strip(".,!?;:'\"()[]{} ").lower() for tok in t.split()]
        pos = sum(tok in positive_words for tok in tokens)
        neg = sum(tok in negative_words for tok in tokens)
        score = 0.0
        if pos + neg > 0:
            score = (pos - neg) / (pos + neg)
        scores.append(score)
    return float(np.mean(scores)) if scores else np.nan

daily_sentiment = news_df.groupby('Date').apply(_compute_daily_sentiment).reset_index(name='Sentiment')

# Merge counts and sentiment onto the yfinance base (left join on trading days)
df = df_base.merge(headline_counts, on='Date', how='left')
df = df.merge(daily_sentiment, on='Date', how='left')

# Fill missing headline metrics sensibly
df['Headline_Count'] = df['Headline_Count'].fillna(0).astype(int)
# Keep Sentiment as NaN where no headlines; replace NaN with 0 to preserve prior behavior if desired
df['Sentiment'] = df['Sentiment'].fillna(0.0)

# Load interest rate data
print("Loading interest rate data...")
interest_df = pd.read_csv('data/interest rate.csv')
interest_df['observation_date'] = pd.to_datetime(interest_df['observation_date'])
interest_df['Year_Month'] = interest_df['observation_date'].dt.to_period('M')

# Load inflation rate data
print("Loading inflation rate data...")
inflation_df = pd.read_csv('data/inflation rate.csv')
# Convert Date format (e.g., "Apr-24" to proper datetime)
inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y')
inflation_df['Year_Month'] = inflation_df['Date'].dt.to_period('M')

# Load GDP data
print("Loading GDP data...")
gdp_df = pd.read_csv('data/GDP.csv')
gdp_df['observation_date'] = pd.to_datetime(gdp_df['observation_date'])
gdp_df['Year_Month'] = gdp_df['observation_date'].dt.to_period('M')

# Load gold price data
print("Loading gold price data...")
gold_df = pd.read_csv('data/gold_price.csv')
gold_df['Date'] = pd.to_datetime(gold_df['Date'], format='%m/%d/%Y')
gold_df['Year_Month'] = gold_df['Date'].dt.to_period('M')

# Load unemployment data
print("Loading unemployment data...")
unemployment_df = pd.read_csv('data/Unemployment.csv')
# Convert Month format (e.g., "Jan-08" to proper datetime)
unemployment_df['Date'] = pd.to_datetime(unemployment_df['Month'], format='%b-%y')
unemployment_df['Year_Month'] = unemployment_df['Date'].dt.to_period('M')
# Rename the Total column to Unemployment_Rate
unemployment_df.rename(columns={'Total': 'Unemployment_Rate'}, inplace=True)

# Add Year_Month column to main dataframe for merging
df['Year_Month'] = df['Date'].dt.to_period('M')

# Merge interest rate data
print("Merging interest rate data...")
df = df.merge(interest_df[['Year_Month', 'FEDFUNDS']], on='Year_Month', how='left')
df.rename(columns={'FEDFUNDS': 'Interest_Rate'}, inplace=True)

# Merge inflation rate data
print("Merging inflation rate data...")
df = df.merge(inflation_df[['Year_Month', 'Inflation']], on='Year_Month', how='left')
df.rename(columns={'Inflation': 'Inflation_Rate'}, inplace=True)

# Merge GDP data
print("Merging GDP data...")
df = df.merge(gdp_df[['Year_Month', 'GDP']], on='Year_Month', how='left')

# Forward fill GDP data to handle missing values
print("Forward filling GDP data...")
df['GDP'] = df['GDP'].ffill()

# Merge gold price data
print("Merging gold price data...")
df = df.merge(gold_df[['Year_Month', 'Value']], on='Year_Month', how='left')
df.rename(columns={'Value': 'Gold_Price'}, inplace=True)

# Merge unemployment data
print("Merging unemployment data...")
df = df.merge(unemployment_df[['Year_Month', 'Unemployment_Rate']], on='Year_Month', how='left')

# Remove the temporary Year_Month column
df.drop('Year_Month', axis=1, inplace=True)

# Display results
print(f"\nFinal enhanced dataset shape: {df.shape}")
print("\nSample of enhanced dataset with all features:")
sample_cols = [c for c in ['Date','CP','Volume','Open','High','Low','Close','Interest_Rate','Inflation_Rate','GDP','Gold_Price','Unemployment_Rate','Headline_Count','Sentiment'] if c in df.columns]
print(df[sample_cols].head(10))

# Display data info
print(f"\nData summary:")
print(f"- Total records: {len(df)}")
print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"- Volume data availability: {df['Volume'].notna().sum()}/{len(df)} records")
print(f"- Interest rate data availability: {df['Interest_Rate'].notna().sum()}/{len(df)} records")
print(f"- Inflation rate data availability: {df['Inflation_Rate'].notna().sum()}/{len(df)} records")
print(f"- GDP data availability: {df['GDP'].notna().sum()}/{len(df)} records")
print(f"- Gold price data availability: {df['Gold_Price'].notna().sum()}/{len(df)} records")
print(f"- Unemployment rate data availability: {df['Unemployment_Rate'].notna().sum()}/{len(df)} records")
print(f"- Headline count data availability: {df['Headline_Count'].notna().sum()}/{len(df)} records")

# Save the enhanced dataset
output_filename = 'data/enhanced_sp500_dataset.csv'
df.to_csv(output_filename, index=False)
print(f"\nEnhanced dataset saved as '{output_filename}'")

# Display some statistics
print(f"\nVolume statistics:")
print(df['Volume'].describe())
print(f"\nInterest Rate statistics:")
print(df['Interest_Rate'].describe())
print(f"\nInflation Rate statistics:")
print(df['Inflation_Rate'].describe())
print(f"\nGDP statistics:")
print(df['GDP'].describe())
print(f"\nGold Price statistics:")
print(df['Gold_Price'].describe())
print(f"\nUnemployment Rate statistics:")
print(df['Unemployment_Rate'].describe())
print(f"\nHeadline Count statistics:")
print(df['Headline_Count'].describe())

# Optionally regenerate features_for_prediction via the analyzer
try:
    from visualization import SP500DataAnalyzer
    print("\nRegenerating feature-engineered dataset for modeling...")
    analyzer = SP500DataAnalyzer(output_filename)
    analyzer.prepare_prediction_features()
    print("Feature-engineered dataset saved as 'data/sp500_features_for_prediction.csv'")
except Exception as e:
    print(f"Skipping feature regeneration due to error: {e}")
