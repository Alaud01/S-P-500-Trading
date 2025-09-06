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
        # Resolve absolute data directory based on this file's location
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        
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
    
    def find_latest_date_in_datasets(self, data_dir: Optional[str] = None) -> Optional[datetime]:
        """
        Find the latest published date from recent_headlines.csv specifically.
        
        Args:
            data_dir (str): Directory containing the CSV files. Defaults to absolute data dir.
            
        Returns:
            Optional[datetime]: Latest date or None if no data
        """
        if data_dir is None:
            data_dir = self.data_dir

        latest_date = None

        recent_path = os.path.join(data_dir, "recent_headlines.csv")
        if not os.path.exists(recent_path):
            logger.info("recent_headlines.csv not found; no existing data")
            return None

        try:
            df = pd.read_csv(recent_path)
            if 'published_date' not in df.columns:
                logger.warning("'published_date' column not found in recent_headlines.csv")
                return None

            # Parse dates robustly (handles timezone like +00:00)
            df['published_date'] = pd.to_datetime(df['published_date'], utc=True, errors='coerce')
            latest_date = df['published_date'].max()

            if pd.isna(latest_date):
                logger.info("No valid dates found in recent_headlines.csv")
                return None

            logger.info(f"Latest date in recent_headlines.csv: {latest_date}")
            return latest_date
        except Exception as e:
            logger.error(f"Error reading recent_headlines.csv: {e}")
            return None
    
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
                        output_dir: Optional[str] = None) -> str:
        """
        Append new headlines to the existing recent_headlines.csv file.
        
        Args:
            new_headlines (Dict[str, pd.DataFrame]): New headlines from APIs
            output_dir (str): Output directory (defaults to absolute data dir)
            
        Returns:
            str: Path to the saved file
        """
        # Resolve and create output directory
        if output_dir is None:
            output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        
        existing_file = os.path.join(output_dir, "recent_headlines.csv")
        
        # Define the standard columns to match existing file
        standard_columns = [
            'headline', 'summary', 'content', 'url', 'image_url', 
            'published_date', 'source_name', 'market', 'data_source',
            'Title', 'Date', 'CP', 'author'
        ]
        
        all_new_data = []
        
        for source, df in new_headlines.items():
            if not df.empty:
                # Map the new data to match existing columns
                mapped_df = pd.DataFrame()
                
                # Map columns from new data to standard format
                mapped_df['headline'] = df.get('title', '')
                mapped_df['summary'] = df.get('description', '')
                mapped_df['content'] = df.get('content', '')
                mapped_df['url'] = df.get('url', '')
                mapped_df['image_url'] = ''  # Not available from APIs
                # Preserve published_date as provided (string yyyy-mm-dd) to align with existing
                mapped_df['published_date'] = df.get('published_date', '')
                mapped_df['source_name'] = df.get('source', '')
                mapped_df['market'] = 'SP500'  # Default market
                mapped_df['data_source'] = source
                mapped_df['Title'] = df.get('title', '')  # Duplicate for compatibility
                mapped_df['Date'] = df.get('published_date', '')
                mapped_df['CP'] = ''  # Not available
                mapped_df['author'] = ''  # Not available
                
                all_new_data.append(mapped_df)
        
        if all_new_data:
            # Combine all new data
            combined_df = pd.concat(all_new_data, ignore_index=True)
            
            # Remove duplicates based on title and url
            combined_df = combined_df.drop_duplicates(subset=['headline', 'url'])
            
            # Check if existing file exists and read it
            if os.path.exists(existing_file):
                try:
                    existing_df = pd.read_csv(existing_file)
                    logger.info(f"Found existing file with {len(existing_df)} articles")
                    
                    # Remove duplicates that already exist in the file
                    combined_df = combined_df[~combined_df['headline'].isin(existing_df['headline'])]
                    combined_df = combined_df[~combined_df['url'].isin(existing_df['url'])]
                    
                    if not combined_df.empty:
                        # Append new data to existing file
                        final_df = pd.concat([existing_df, combined_df], ignore_index=True)
                        final_df.to_csv(existing_file, index=False, encoding='utf-8')
                        logger.info(f"Appended {len(combined_df)} new articles to {existing_file}")
                        logger.info(f"Total articles in file: {len(final_df)}")
                    else:
                        logger.info("No new unique articles to add")
                        return existing_file
                        
                except Exception as e:
                    logger.error(f"Error reading existing file: {e}")
                    # If error reading existing file, just save new data
                    combined_df.to_csv(existing_file, index=False, encoding='utf-8')
                    logger.info(f"Created new file with {len(combined_df)} articles")
            else:
                # Create new file if it doesn't exist
                combined_df.to_csv(existing_file, index=False, encoding='utf-8')
                logger.info(f"Created new file with {len(combined_df)} articles")
            
            return existing_file
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
        
        # Find latest date in recent_headlines.csv
        latest_date = self.find_latest_date_in_datasets()
        
        if latest_date is None:
            # No existing data, fetch last 30 days
            start_date = datetime.now() - timedelta(days=30)
            logger.info("No existing data found. Fetching last 30 days of headlines.")
        else:
            # Inclusive: start from the latest date present in recent_headlines.csv
            if latest_date.tzinfo is not None:
                latest_date = latest_date.replace(tzinfo=None)
            start_date = latest_date
            now = datetime.now()
            if start_date > now:
                # Edge case: if file has a future timestamp, step back one day
                start_date = now - timedelta(days=1)
            logger.info(f"Fetching headlines from {start_date} (inclusive)")
        
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
