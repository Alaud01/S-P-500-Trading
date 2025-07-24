import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np

# Read the main S&P 500 headlines dataset
print("Loading S&P 500 headlines data...")
df = pd.read_csv('sp500 headlines 2008 to 2024.csv')
print(f"Original dataset shape: {df.shape}")
print(df.head())

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

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
print("\nSample of enhanced dataset:")
print(df[['Title', 'Date', 'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate']].head(10))

# Display data info
print(f"\nData summary:")
print(f"- Total records: {len(df)}")
print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"- Volume data availability: {df['Volume'].notna().sum()}/{len(df)} records")
print(f"- Interest rate data availability: {df['Interest_Rate'].notna().sum()}/{len(df)} records")
print(f"- Inflation rate data availability: {df['Inflation_Rate'].notna().sum()}/{len(df)} records")

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
