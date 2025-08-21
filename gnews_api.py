import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GNewsAPI:
    """
    A class to interact with the GNews API to fetch news articles about S&P 500 and NASDAQ.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the GNews API client.
        
        Args:
            api_key (str, optional): Your GNews API key. If not provided, will try to load from environment.
        """
        if api_key is None:
            # Load environment variables from .env file
            load_dotenv()
            api_key = os.getenv('GNEWS_API_KEY')
            
            if not api_key:
                raise ValueError("GNEWS_API_KEY not found in environment variables or .env file. "
                               "Please set it in your .env file or pass it as a parameter.")
        
        self.api_key = api_key
        self.base_url = "https://gnews.io/api/v4"
        self.max_articles_per_request = 100  # GNews API limit
        
    def search_articles(self, query: str, start_date: str, end_date: str, 
                       max_articles: int = 1000) -> List[Dict[str, Any]]:
        """
        Search for articles using the GNews API.
        
        Args:
            query (str): Search query
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            max_articles (int): Maximum number of articles to fetch
            
        Returns:
            List[Dict[str, Any]]: List of article dictionaries
        """
        all_articles = []
        page = 1
        
        while len(all_articles) < max_articles:
            try:
                # Construct the API URL
                url = f"{self.base_url}/search"
                requested_per_page = min(self.max_articles_per_request, max_articles - len(all_articles))
                params = {
                    'q': query,
                    'from': start_date,
                    'to': end_date,
                    'sortby': 'publishedAt',
                    'apikey': self.api_key,
                    'max': requested_per_page,
                    'page': page
                }
                
                logger.info(f"Fetching page {page} for query: {query}")
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if 'articles' not in data or not data['articles']:
                    logger.info(f"No more articles found for query: {query}")
                    break
                
                articles = data['articles']
                all_articles.extend(articles)
                
                logger.info(f"Retrieved {len(articles)} articles from page {page}")

                # Proceed to next page; stop only when API returns no articles
                page += 1
                
                # Rate limiting - GNews API has rate limits
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching articles for query '{query}': {e}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response for query '{query}': {e}")
                break
        
        logger.info(f"Total articles retrieved for '{query}': {len(all_articles)}")
        return all_articles
    
    def fetch_market_news(self, start_date: str = "2022-01-01", 
                         end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch news articles about S&P 500 and NASDAQ.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with 'sp500' and 'nasdaq' DataFrames
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Define search queries for each market
        # queries = {
        #     'sp500': ['"S&P 500"', '"Standard & Poor\'s 500"', 'SPX', 'SPY'],
        #     'nasdaq': ['NASDAQ', 'NDX', 'QQQ', '"Nasdaq Composite"']
        # }

        queries = {
            'sp500': ['"S&P 500"'],
            'nasdaq': ['NASDAQ']
        }
        
        results = {}
        
        for market, search_terms in queries.items():
            logger.info(f"Fetching news for {market.upper()}")
            all_articles = []
            
            for query in search_terms:
                articles = self.search_articles(query, start_date, end_date, max_articles=1000)
                print(articles)
                all_articles.extend(articles)
                time.sleep(2)  # Rate limiting between different queries
            
            # Remove duplicates based on URL
            unique_articles = {}
            for article in all_articles:
                url = article.get('url', '')
                if url and url not in unique_articles:
                    unique_articles[url] = article
            
            # Convert to DataFrame
            if unique_articles:
                df = pd.DataFrame(list(unique_articles.values()))
                
                # Clean and standardize the data
                df = self._clean_articles_dataframe(df)
                results[market] = df
                
                logger.info(f"Retrieved {len(df)} unique articles for {market.upper()}")
            else:
                logger.warning(f"No articles found for {market.upper()}")
                results[market] = pd.DataFrame()
        
        return results
    
    def _clean_articles_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the articles DataFrame.
        
        Args:
            df (pd.DataFrame): Raw articles DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Select and rename relevant columns
        columns_mapping = {
            'title': 'headline',
            'description': 'summary',
            'content': 'content',
            'url': 'url',
            'image': 'image_url',
            'publishedAt': 'published_date',
            'source': 'source'
        }
        
        # Create a new DataFrame with only the columns we want
        cleaned_df = pd.DataFrame()
        
        for old_col, new_col in columns_mapping.items():
            if old_col in df.columns:
                cleaned_df[new_col] = df[old_col]
        
        # Extract source name if source is a dictionary
        if 'source' in cleaned_df.columns:
            cleaned_df['source_name'] = cleaned_df['source'].apply(
                lambda x: x.get('name', '') if isinstance(x, dict) else str(x)
            )
            cleaned_df = cleaned_df.drop('source', axis=1)
        
        # Convert published_date to datetime
        if 'published_date' in cleaned_df.columns:
            cleaned_df['published_date'] = pd.to_datetime(cleaned_df['published_date'])
        
        # Add a market identifier column
        cleaned_df['market'] = 'unknown'  # Will be set by the calling function
        
        # Remove rows with missing headlines
        cleaned_df = cleaned_df.dropna(subset=['headline'])
        
        # Sort by published date
        if 'published_date' in cleaned_df.columns:
            cleaned_df = cleaned_df.sort_values('published_date', ascending=False)
        
        return cleaned_df.reset_index(drop=True)

def save_to_csv(dataframes: Dict[str, pd.DataFrame], output_dir: str = "data"):
    """
    Save the DataFrames to CSV files.
    
    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary of DataFrames
        output_dir (str): Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for market, df in dataframes.items():
        if not df.empty:
            # Set the market identifier
            df['market'] = market.upper()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{market}_headlines_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Save to CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Saved {len(df)} articles to {filepath}")
        else:
            logger.warning(f"No data to save for {market}")

def main():
    """
    Main function to run the news fetching script.
    """
    try:
        # Initialize the API client (will automatically load from .env)
        gnews = GNewsAPI()
    except ValueError as e:
        logger.error(str(e))
        logger.error("Please create a .env file with your API key:")
        logger.error("GNEWS_API_KEY=your_api_key_here")
        logger.error("Or get your API key from https://gnews.io/")
        return
    
    # Get date range from environment or use defaults
    start_date = os.getenv('START_DATE', "2022-01-01")
    end_date = os.getenv('END_DATE', datetime.now().strftime("%Y-%m-%d"))
    
    logger.info(f"Fetching news from {start_date} to {end_date}")
    
    try:
        # Fetch the news data
        market_news = gnews.fetch_market_news(start_date, end_date)
        
        # Save to CSV files
        save_to_csv(market_news)
        
        # Print summary
        print("\n" + "="*50)
        print("FETCHING COMPLETE")
        print("="*50)
        for market, df in market_news.items():
            print(f"{market.upper()}: {len(df)} articles")
        print("="*50)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
