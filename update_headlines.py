import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Import the API classes
from gnews_api import GNewsAPI
from newsapi_script import NewsAPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeadlinesUpdater:
    """
    A class to find the latest date in existing datasets and fetch new headlines from both APIs.
    """
    
    def __init__(self):
        """
        Initialize the updater with both API clients.
        """
        load_dotenv()
        
        # Initialize API clients
        try:
            self.gnews_api = GNewsAPI()
            logger.info("GNews API client initialized successfully")
        except ValueError as e:
            logger.warning(f"GNews API not available: {e}")
            self.gnews_api = None
        
        try:
            self.newsapi_client = NewsAPI()
            logger.info("NewsAPI client initialized successfully")
        except ValueError as e:
            logger.warning(f"NewsAPI not available: {e}")
            self.newsapi_client = None
    
    def find_latest_date_in_datasets(self, data_dir: str = "data") -> Optional[datetime]:
        """
        Find the latest published date across all headline datasets.
        
        Args:
            data_dir (str): Directory containing the CSV files
            
        Returns:
            Optional[datetime]: Latest date or None if no data
        """
        latest_date = None
        
        # Look for existing CSV files
        csv_files = []
        for file in os.listdir(data_dir):
            if file.endswith('.csv') and 'headline' in file.lower():
                csv_files.append(os.path.join(data_dir, file))
        
        logger.info(f"Found {len(csv_files)} headline CSV files: {csv_files}")
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                # Check if published_date column exists
                if 'published_date' in df.columns:
                    df['published_date'] = pd.to_datetime(df['published_date'])
                    file_latest = df['published_date'].max()
                    
                    if latest_date is None or file_latest > latest_date:
                        latest_date = file_latest
                    
                    logger.info(f"Latest date in {os.path.basename(file_path)}: {file_latest}")
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        if latest_date:
            logger.info(f"Overall latest date found: {latest_date}")
        else:
            logger.info("No existing data found")
        
        return latest_date
    
    def fetch_new_headlines(self, start_date: datetime, end_date: datetime = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch new headlines from both APIs after the latest date.
        
        Args:
            start_date (datetime): Start date for fetching new headlines
            end_date (datetime): End date (defaults to today)
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with new headlines from each API
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching new headlines from {start_date_str} to {end_date_str}")
        
        new_headlines = {}
        
        # Fetch from GNews API
        if self.gnews_api:
            try:
                logger.info("Fetching new headlines from GNews API...")
                gnews_results = self.gnews_api.fetch_market_news(start_date_str, end_date_str)
                
                if 'sp500' in gnews_results and not gnews_results['sp500'].empty:
                    new_headlines['gnews'] = gnews_results['sp500']
                    logger.info(f"Retrieved {len(gnews_results['sp500'])} new articles from GNews")
                else:
                    logger.info("No new articles found from GNews API")
                    
            except Exception as e:
                logger.error(f"Error fetching from GNews API: {e}")
        
        # Fetch from NewsAPI
        if self.newsapi_client:
            try:
                logger.info("Fetching new headlines from NewsAPI...")
                newsapi_df = self.newsapi_client.fetch_sp500_news(start_date_str, end_date_str)
                
                if not newsapi_df.empty:
                    new_headlines['newsapi'] = newsapi_df
                    logger.info(f"Retrieved {len(newsapi_df)} new articles from NewsAPI")
                else:
                    logger.info("No new articles found from NewsAPI")
                    
            except Exception as e:
                logger.error(f"Error fetching from NewsAPI: {e}")
        
        return new_headlines
    
    def save_new_headlines(self, new_headlines: Dict[str, pd.DataFrame], 
                          output_dir: str = "data") -> str:
        """
        Save new headlines to separate CSV files for each API source.
        
        Args:
            new_headlines (Dict[str, pd.DataFrame]): New headlines from APIs
            output_dir (str): Output directory
            
        Returns:
            str: Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        for source, df in new_headlines.items():
            if not df.empty:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recent_headlines_{timestamp}.csv"
                filepath = os.path.join(output_dir, filename)
                
                # Save to CSV
                df.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"Saved {len(df)} new articles from {source} to {filepath}")
                saved_files.append(filepath)
        
        if saved_files:
            return saved_files[0]  # Return first file path
        else:
            logger.warning("No new headlines to save")
            return None
    
    def run_update(self) -> str:
        """
        Main method to run the complete update process.
        
        Returns:
            str: Path to the saved file
        """
        logger.info("Starting headline update process...")
        
        # Find latest date in existing datasets
        latest_date = self.find_latest_date_in_datasets()
        
        if latest_date is None:
            # No existing data, fetch last 30 days
            start_date = datetime.now() - timedelta(days=30)
            logger.info("No existing data found. Fetching last 30 days of headlines.")
        else:
            # Add 1 day to avoid duplicates
            start_date = latest_date + timedelta(days=1)
            logger.info(f"Fetching headlines after {start_date}")
        
        # Fetch new headlines
        new_headlines = self.fetch_new_headlines(start_date)
        
        # Save new headlines
        output_file = self.save_new_headlines(new_headlines)
        
        # Print summary
        total_new = sum(len(df) for df in new_headlines.values())
        logger.info(f"Update complete! Added {total_new} new articles.")
        
        return output_file

def main():
    """
    Main function to run the headline update script.
    """
    try:
        updater = HeadlinesUpdater()
        output_file = updater.run_update()
        
        print("\n" + "="*50)
        print("HEADLINE UPDATE COMPLETE")
        print("="*50)
        if output_file:
            print(f"New headlines saved to: {output_file}")
        else:
            print("No new headlines found")
        print("="*50)
        
    except Exception as e:
        logger.error(f"An error occurred during the update process: {e}")

if __name__ == "__main__":
    main()
