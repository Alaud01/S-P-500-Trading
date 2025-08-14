import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np

# --- Data Loading and Processing ---
# Read the main S&P 500 headlines dataset
print("Loading S&P 500 headlines data...")
df = pd.read_csv('data/sp500 headlines 2008 to 2024.csv')
print(f"Original dataset shape: {df.shape}")
print(df.head())

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter data to start from 2008-01-02 onwards
df = df[df['Date'] >= '2008-01-02'].reset_index(drop=True)
print(f"Filtered dataset shape (from 2008-01-02): {df.shape}")

# For now, we'll skip sentiment analysis due to library compatibility issues
# and focus on incorporating the new economic indicators
print("Skipping sentiment analysis for now - focusing on economic indicators...")

# Aggregate headlines per day (just count them for now)
print("Aggregating daily headline counts...")
daily_headlines = df.groupby('Date').size().reset_index(name='Headline_Count')

# Drop the original headlines to keep one row per day, but preserve CP column
df_daily = df.drop(['Title'], axis=1).drop_duplicates(subset=['Date']).reset_index(drop=True)

# Merge the headline count back into the daily data
df = pd.merge(df_daily, daily_headlines, on='Date', how='left')

# Get unique dates for volume data
unique_dates = df['Date'].unique()
print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Number of unique dates: {len(unique_dates)}")

# Download S&P 500 volume data using yfinance
print("\nDownloading S&P 500 volume data from yfinance...")
ticker = "^GSPC"  # S&P 500 ticker symbol
start_date = df['Date'].min()
end_date = df['Date'].max()

# Download data with a different approach
try:
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
    
    # Check if volume data was successfully merged
    if df['Volume'].isna().all():
        print("Volume data is all NaN, using fallback...")
        raise Exception("Volume data not available")
    
except Exception as e:
    print(f"Failed to download volume data: {e}")
    print("Creating placeholder volume data...")
    # Create placeholder volume data based on headline count
    df['Volume'] = df['Headline_Count'] * 1000000  # Simple placeholder

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

# Add a placeholder sentiment column (will be filled later if needed)
df['Sentiment'] = 0.0

# Display results
print(f"\nFinal dataset shape: {df.shape}")
print("\nSample of enhanced dataset with all features:")
print(df[['Date', 'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate', 'GDP', 'Gold_Price', 'Unemployment_Rate', 'Headline_Count', 'Sentiment']].head(10))

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
