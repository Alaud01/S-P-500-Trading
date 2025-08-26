import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_process_datasets():
    """
    Load and process all datasets, handling different frequencies and date ranges.
    """
    print("Loading datasets...")
    
    # Load S&P 500 data (daily)
    sp500_df = pd.read_csv('data/yfinance_sp500.csv')
    sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
    # Filter to only include dates up to 2024-03-04
    cutoff_date = pd.to_datetime('2024-03-04')
    sp500_df = sp500_df[sp500_df['Date'] <= cutoff_date]
    sp500_df = sp500_df.sort_values('Date').reset_index(drop=True)
    
    # Load GDP data (quarterly)
    gdp_df = pd.read_csv('data/GDP.csv')
    gdp_df['observation_date'] = pd.to_datetime(gdp_df['observation_date'])
    gdp_df = gdp_df.rename(columns={'observation_date': 'Date', 'GDP': 'GDP'})
    # Filter to only include dates up to 2024-03-04
    cutoff_date = pd.to_datetime('2024-03-04')
    gdp_df = gdp_df[gdp_df['Date'] <= cutoff_date]
    gdp_df = gdp_df.sort_values('Date').reset_index(drop=True)
    
    # Load Gold price data (monthly)
    gold_df = pd.read_csv('data/gold_price.csv')
    gold_df['Date'] = pd.to_datetime(gold_df['Date'], format='%m/%d/%Y')
    gold_df = gold_df.rename(columns={'Value': 'Gold_Price'})
    # Filter to only include dates up to 2024-03-04
    cutoff_date = pd.to_datetime('2024-03-04')
    gold_df = gold_df[gold_df['Date'] <= cutoff_date]
    gold_df = gold_df.sort_values('Date').reset_index(drop=True)
    
    # Load Unemployment data (monthly)
    unemployment_df = pd.read_csv('data/Unemployment.csv')
    # Convert Month column to datetime (format: Jul-05)
    unemployment_df['Date'] = pd.to_datetime(unemployment_df['Month'], format='%b-%y')
    unemployment_df = unemployment_df.rename(columns={'Total': 'Unemployment_Rate'})
    unemployment_df = unemployment_df[['Date', 'Unemployment_Rate']].dropna()
    # Filter to only include dates up to 2024-03-04
    cutoff_date = pd.to_datetime('2024-03-04')
    unemployment_df = unemployment_df[unemployment_df['Date'] <= cutoff_date]
    unemployment_df = unemployment_df.sort_values('Date').reset_index(drop=True)
    
    # Load Interest rate data (monthly)
    interest_df = pd.read_csv('data/interest rate.csv')
    interest_df['observation_date'] = pd.to_datetime(interest_df['observation_date'])
    interest_df = interest_df.rename(columns={'observation_date': 'Date', 'FEDFUNDS': 'Interest_Rate'})
    # Filter to only include dates up to 2024-03-04
    cutoff_date = pd.to_datetime('2024-03-04')
    interest_df = interest_df[interest_df['Date'] <= cutoff_date]
    interest_df = interest_df.sort_values('Date').reset_index(drop=True)
    
    # Load Inflation rate data (monthly)
    inflation_df = pd.read_csv('data/inflation rate.csv')
    # Convert Date column to datetime (format: Apr-24)
    inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y')
    inflation_df = inflation_df.rename(columns={'Inflation': 'Inflation_Rate'})
    # Filter to only include dates up to 2024-03-04
    cutoff_date = pd.to_datetime('2024-03-04')
    inflation_df = inflation_df[inflation_df['Date'] <= cutoff_date]
    inflation_df = inflation_df.sort_values('Date').reset_index(drop=True)
    
    # Load headlines data for sentiment analysis
    headlines_df = pd.read_csv('data/sp500 headlines 2008 to 2024.csv')
    headlines_df['Date'] = pd.to_datetime(headlines_df['Date'])
    # Filter to only include dates up to 2024-03-04
    cutoff_date = pd.to_datetime('2024-03-04')
    headlines_df = headlines_df[headlines_df['Date'] <= cutoff_date]
    headlines_df = headlines_df.sort_values('Date').reset_index(drop=True)
    
    # Load recent headlines data
    recent_headlines_df = pd.read_csv('data/recent_headlines.csv')
    # Drop the original Date column and use published_date instead
    if 'Date' in recent_headlines_df.columns:
        recent_headlines_df = recent_headlines_df.drop(columns=['Date'])
    # Convert published_date to datetime and rename to Date
    recent_headlines_df['published_date'] = pd.to_datetime(recent_headlines_df['published_date']).dt.tz_localize(None)
    recent_headlines_df = recent_headlines_df.rename(columns={'published_date': 'Date'})
    # Filter to only include dates up to 2024-03-04
    cutoff_date = pd.to_datetime('2024-03-04')
    recent_headlines_df = recent_headlines_df[recent_headlines_df['Date'] <= cutoff_date]
    recent_headlines_df = recent_headlines_df.sort_values('Date').reset_index(drop=True)
    
    # Load sentiment analysis results
    with open('data/sentiment_analysis_results.json', 'r') as f:
        sentiment_data = json.load(f)
    
    return {
        'sp500': sp500_df,
        'gdp': gdp_df,
        'gold': gold_df,
        'unemployment': unemployment_df,
        'interest': interest_df,
        'inflation': inflation_df,
        'headlines': headlines_df,
        'recent_headlines': recent_headlines_df,
        'sentiment_data': sentiment_data
    }

def process_sentiment_data(headlines_df, recent_headlines_df, sentiment_data):
    """
    Create daily-varying sentiment features from per-headline results if available.
    Falls back to yearly distribution if detailed results are unavailable.
    """
    print("Processing sentiment data...")

    cutoff_date = pd.to_datetime('2024-03-04')

    # Preferred path: use detailed per-headline results to build daily sentiment
    try:
        detailed_sentiment = pd.read_csv('data/sentiment_analysis_results.csv')

        # Normalize/parse date column
        if 'Date' in detailed_sentiment.columns:
            detailed_sentiment['Date'] = pd.to_datetime(detailed_sentiment['Date'])
        elif 'published_date' in detailed_sentiment.columns:
            detailed_sentiment['Date'] = pd.to_datetime(detailed_sentiment['published_date']).dt.tz_localize(None)
        else:
            raise ValueError('No recognizable date column in detailed sentiment results')

        # Clip to cutoff
        detailed_sentiment = detailed_sentiment[detailed_sentiment['Date'] <= cutoff_date]

        # Build a continuous sentiment score per headline
        if {'positive_prob', 'negative_prob'}.issubset(detailed_sentiment.columns):
            detailed_sentiment['Sentiment_Score'] = (
                detailed_sentiment['positive_prob'] - detailed_sentiment['negative_prob']
            )
        elif 'sentiment' in detailed_sentiment.columns:
            label_to_score = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
            detailed_sentiment['Sentiment_Score'] = detailed_sentiment['sentiment'].map(label_to_score).fillna(0.0)
        else:
            raise ValueError('No sentiment probabilities or labels found in detailed sentiment results')

        # Aggregate to daily level
        daily_sentiment = detailed_sentiment.groupby('Date').agg(
            Sentiment_Score=('Sentiment_Score', 'mean'),
            Headline_Count=('Sentiment_Score', 'size')
        ).reset_index()

        return daily_sentiment.sort_values('Date').reset_index(drop=True)

    except Exception as e:
        print(f"Detailed per-headline sentiment not available or failed to load. Falling back. Reason: {e}")

        # Fallback path: approximate using yearly distribution from JSON
        all_headlines = pd.concat([
            headlines_df[['Date', 'Title']],
            recent_headlines_df[['Date', 'headline']].rename(columns={'headline': 'Title'})
        ], ignore_index=True)

        all_headlines = all_headlines.drop_duplicates().sort_values('Date').reset_index(drop=True)

        yearly_sentiment = sentiment_data['yearly_sentiment']

        sentiment_scores = {}
        for year in yearly_sentiment['positive'].keys():
            positive_count = yearly_sentiment['positive'][year]
            neutral_count = yearly_sentiment['neutral'][year]
            negative_count = yearly_sentiment['negative'][year]
            total_count = positive_count + neutral_count + negative_count

            if total_count > 0:
                avg_sentiment = (positive_count - negative_count) / total_count
                sentiment_scores[int(year)] = avg_sentiment
            else:
                sentiment_scores[int(year)] = 0

        all_headlines['Year'] = all_headlines['Date'].dt.year
        all_headlines['Sentiment_Score'] = all_headlines['Year'].map(sentiment_scores)

        daily_sentiment = all_headlines.groupby('Date').agg({
            'Sentiment_Score': 'mean',
            'Title': 'count'
        }).reset_index()
        daily_sentiment = daily_sentiment.rename(columns={'Title': 'Headline_Count'})

        return daily_sentiment.sort_values('Date').reset_index(drop=True)

def create_merged_dataset(datasets):
    """
    Create a merged dataset with all variables aligned to daily frequency.
    """
    print("Creating merged dataset...")
    
    # Get the date range from sentiment analysis (headlines data)
    headlines_df = datasets['headlines']
    recent_headlines_df = datasets['recent_headlines']
    
    # Process sentiment data
    sentiment_df = process_sentiment_data(headlines_df, recent_headlines_df, datasets['sentiment_data'])
    
    # Get the date range from sentiment data
    start_date = sentiment_df['Date'].min()
    end_date = sentiment_df['Date'].max()
    
    # Filter to only include dates up to 2024-03-04
    cutoff_date = pd.to_datetime('2024-03-04')
    end_date = min(end_date, cutoff_date)
    
    print(f"Date range: {start_date} to {end_date}")
    
    # Create a complete date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    merged_df = pd.DataFrame({'Date': date_range})
    
    # Add S&P 500 data (daily)
    sp500_df = datasets['sp500']
    merged_df = merged_df.merge(sp500_df, on='Date', how='left')
    
    # Add sentiment data (daily)
    merged_df = merged_df.merge(sentiment_df, on='Date', how='left')
    
    # Add GDP data (quarterly) - forward fill
    gdp_df = datasets['gdp']
    merged_df = merged_df.merge(gdp_df, on='Date', how='left')
    merged_df['GDP'] = merged_df['GDP'].fillna(method='ffill')
    
    # Add Gold price data (monthly) - forward fill
    gold_df = datasets['gold']
    merged_df = merged_df.merge(gold_df, on='Date', how='left')
    merged_df['Gold_Price'] = merged_df['Gold_Price'].fillna(method='ffill')
    
    # Add Unemployment data (monthly) - forward fill
    unemployment_df = datasets['unemployment']
    merged_df = merged_df.merge(unemployment_df, on='Date', how='left')
    merged_df['Unemployment_Rate'] = merged_df['Unemployment_Rate'].fillna(method='ffill')
    
    # Add Interest rate data (monthly) - forward fill
    interest_df = datasets['interest']
    merged_df = merged_df.merge(interest_df, on='Date', how='left')
    merged_df['Interest_Rate'] = merged_df['Interest_Rate'].fillna(method='ffill')
    
    # Add Inflation rate data (monthly) - forward fill
    inflation_df = datasets['inflation']
    merged_df = merged_df.merge(inflation_df, on='Date', how='left')
    merged_df['Inflation_Rate'] = merged_df['Inflation_Rate'].fillna(method='ffill')
    
    # Backward fill any remaining NaN values at the beginning
    merged_df = merged_df.fillna(method='bfill')
    
    # Sort by date
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
    return merged_df

def feature_engineering(df):
    """
    Create features for predicting S&P 500 direction.
    """
    print("Performing feature engineering...")
    df.set_index('Date', inplace=True)
    
    # Target variable: 1 if S&P 500 goes up in next 14 days, 0 otherwise
    df['Future_Close'] = df['Close'].shift(-14)
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
    
    # Technical indicators
    # Moving Averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_14'] = df['Close'].rolling(window=14).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    
    # Volatility (14-day rolling standard deviation of daily returns)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_14'] = df['Daily_Return'].rolling(window=14).std()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands (20-day SMA with 2 standard deviations)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Macro change features (exclude gold)
    # Note: These will typically be zero on non-release days due to forward fill.
    df['GDP_QoQ_Pct_Change'] = df['GDP'].pct_change()
    df['Unemployment_MoM_Change'] = df['Unemployment_Rate'].diff()
    df['Interest_MoM_Change'] = df['Interest_Rate'].diff()
    df['Inflation_MoM_Change'] = df['Inflation_Rate'].diff()
    
    # Binary/interaction features
    df['Price_vs_MA30'] = (df['Close'] > df['MA_30']).astype(int)
    df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
    df['MACD_Crossover'] = (df['MACD'] > df['Signal_Line']).astype(int)
    
    # Lagged features
    for lag in [1, 3, 5, 7, 14]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Sentiment_Lag_{lag}'] = df['Sentiment_Score'].shift(lag)

    # Drop intermediate and NaN-containing rows
    df.drop(columns=['Future_Close'], inplace=True)
    df.dropna(inplace=True)
    
    return df

def main():
    """
    Main function to merge all datasets.
    """
    print("Starting dataset merge process...")
    
    # Load all datasets
    datasets = load_and_process_datasets()
    
    # Create merged dataset
    merged_df = create_merged_dataset(datasets)
    
    # Perform feature engineering
    final_df = feature_engineering(merged_df.copy())
    
    # Display dataset info
    print("\nFinal dataset info:")
    print(f"Shape: {final_df.shape}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")
    print(f"Columns: {list(final_df.columns)}")
    
    # Display sample data
    print("\nSample data (first 10 rows):")
    print(final_df.head(10))
    
    # Display data types and missing values
    print("\nData types and missing values:")
    print(final_df.info())
    
    print("\nMissing values summary:")
    print(final_df.isnull().sum())
    
    # Save final dataset
    output_file = 'data/final_dataset_for_modeling.csv'
    final_df.to_csv(output_file)
    print(f"\nFinal dataset saved to: {output_file}")
    
    # Display summary statistics
    print("\nSummary statistics:")
    print(final_df.describe())
    
    return final_df

if __name__ == "__main__":
    final_data = main()
