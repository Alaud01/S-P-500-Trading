import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Sentiment Analysis with FinBERT ---
print("Loading FinBERT model for sentiment analysis...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment(text):
    """Calculates sentiment score for a given text using FinBERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Score is positive - negative sentiment
    sentiment_score = scores[:, 0].item() - scores[:, 1].item()
    return sentiment_score

# --- Data Loading and Processing ---
# Read the main S&P 500 headlines dataset
print("Loading S&P 500 headlines data...")
df = pd.read_csv('sp500 headlines 2008 to 2024.csv')
print(f"Original dataset shape: {df.shape}")
print(df.head())

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Calculate sentiment for each headline
print("Calculating sentiment for headlines (this may take a while)...")
# Limit to a subset for faster processing if needed, e.g., df.head(1000)
df['Sentiment'] = df['Title'].apply(get_sentiment)

# Aggregate sentiment per day
print("Aggregating daily sentiment scores...")
daily_sentiment = df.groupby('Date')['Sentiment'].mean().reset_index()

# Drop the original headlines to keep one row per day
df_daily = df.drop(['Title', 'Sentiment'], axis=1).drop_duplicates(subset=['Date']).reset_index(drop=True)

# Merge the aggregated sentiment back into the daily data
df = pd.merge(df_daily, daily_sentiment, on='Date', how='left')

# Get unique dates for volume data
unique_dates = df['Date'].unique()
print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Number of unique dates: {len(unique_dates)}")

# Download S&P 500 volume data using yfinance
print("\nDownloading S&P 500 volume data from yfinance...")
ticker = "^GSPC"  # S&P 500 ticker symbol
start_date = df['Date'].min()
end_date = df['Date'].max()

# Download data
sp500_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
sp500_data.reset_index(inplace=True)

# Handle MultiIndex columns if present
if isinstance(sp500_data.columns, pd.MultiIndex):
    # Flatten the MultiIndex columns
    sp500_data.columns = ['_'.join(col).strip() for col in sp500_data.columns.values]
    # Fix the Date column name
    sp500_data.rename(columns={'Date_': 'Date'}, inplace=True)
    volume_col = [col for col in sp500_data.columns if 'Volume' in col][0]
    sp500_data.rename(columns={volume_col: 'Volume'}, inplace=True)

print(f"Available columns in sp500_data: {sp500_data.columns.tolist()}")

# Merge volume data with the main dataframe
print("Merging volume data...")
df = df.merge(sp500_data[['Date', 'Volume']], on='Date', how='left')

# Load interest rate data
print("Loading interest rate data...")
interest_df = pd.read_csv('interest rate.csv')
interest_df['observation_date'] = pd.to_datetime(interest_df['observation_date'])
interest_df['Year_Month'] = interest_df['observation_date'].dt.to_period('M')

# Load inflation rate data
print("Loading inflation rate data...")
inflation_df = pd.read_csv('inflation rate.csv')
# Convert Date format (e.g., "Apr-24" to proper datetime)
inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y')
inflation_df['Year_Month'] = inflation_df['Date'].dt.to_period('M')

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

# Remove the temporary Year_Month column
df.drop('Year_Month', axis=1, inplace=True)

# Display results
print(f"\nFinal dataset shape: {df.shape}")
print("\nSample of enhanced dataset with Sentiment:")
print(df[['Date', 'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate', 'Sentiment']].head(10))

# Display data info
print(f"\nData summary:")
print(f"- Total records: {len(df)}")
print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"- Volume data availability: {df['Volume'].notna().sum()}/{len(df)} records")
print(f"- Interest rate data availability: {df['Interest_Rate'].notna().sum()}/{len(df)} records")
print(f"- Inflation rate data availability: {df['Inflation_Rate'].notna().sum()}/{len(df)} records")
print(f"- Sentiment data availability: {df['Sentiment'].notna().sum()}/{len(df)} records")

# Save the enhanced dataset
output_filename = 'enhanced_sp500_dataset.csv'
df.to_csv(output_filename, index=False)
print(f"\nEnhanced dataset saved as '{output_filename}'")

# Display some statistics
print(f"\nVolume statistics:")
print(df['Volume'].describe())
print(f"\nInterest Rate statistics:")
print(df['Interest_Rate'].describe())
print(f"\nInflation Rate statistics:")
print(df['Inflation_Rate'].describe())
print(f"\nSentiment statistics:")
print(df['Sentiment'].describe())
