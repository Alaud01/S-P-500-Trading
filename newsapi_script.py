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

class NewsAPI:
    """
    A class to interact with the NewsAPI to fetch news articles about S&P 500.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the NewsAPI client.
        
        Args:
            api_key (str, optional): Your NewsAPI key. If not provided, will try to load from environment.
        """
        if api_key is None:
            # Load environment variables from .env file
            load_dotenv()
            api_key = os.getenv('NEWSAPI_KEY')
            
            if not api_key:
                raise ValueError("NEWSAPI_KEY not found in environment variables or .env file. "
                               "Please set it in your .env file or pass it as a parameter.")
        
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.max_articles_per_request = 100  # NewsAPI limit
        
    def search_articles(self, query: str, start_date: str, end_date: str, 
                       max_articles: int = 1000) -> List[Dict[str, Any]]:
        """
        Search for articles using the NewsAPI.
        
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
                url = f"{self.base_url}/everything"
                requested_per_page = min(self.max_articles_per_request, max_articles - len(all_articles))
                params = {
                    'q': query,
                    'from': start_date,
                    'to': end_date,
                    'sortBy': 'publishedAt',
                    'apiKey': self.api_key,
                    'pageSize': requested_per_page,
                    'page': page,
                    'language': 'en'
                }
                
                logger.info(f"Fetching page {page} for query: {query}")
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('status') != 'ok':
                    logger.error(f"API Error: {data.get('message', 'Unknown error')}")
                    break
                
                if 'articles' not in data or not data['articles']:
                    logger.info(f"No more articles found for query: {query}")
                    break
                
                articles = data['articles']
                all_articles.extend(articles)
                
                logger.info(f"Retrieved {len(articles)} articles from page {page}")
                
                # If we got fewer articles than requested for this page, we've reached the end
                if len(articles) < requested_per_page:
                    logger.info("Reached last available page for this query")
                    break
                
                page += 1
                
                # Rate limiting - NewsAPI has rate limits
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching articles for query '{query}': {e}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response for query '{query}': {e}")
                break
        
        logger.info(f"Total articles retrieved for '{query}': {len(all_articles)}")
        return all_articles
    
    def fetch_sp500_news(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch news articles about S&P 500 for the last month.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format (defaults to 30 days ago)
            end_date (str): End date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            pd.DataFrame: DataFrame with S&P 500 articles
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if start_date is None:
            # Default to 30 days ago
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Define search queries for S&P 500
        queries = [
            '"S&P 500"',
            '"Standard & Poor\'s 500"',
            'SPX',
            'SPY',
            'S&P500',
            'S&P 500 index'
        ]
        
        logger.info(f"Fetching S&P 500 news from {start_date} to {end_date}")
        all_articles = []
        
        for query in queries:
            logger.info(f"Searching for: {query}")
            articles = self.search_articles(query, start_date, end_date, max_articles=1000)
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
            
            logger.info(f"Retrieved {len(df)} unique S&P 500 articles")
            return df
        else:
            logger.warning("No S&P 500 articles found")
            return pd.DataFrame()
    
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
            'urlToImage': 'image_url',
            'publishedAt': 'published_date',
            'source': 'source',
            'author': 'author'
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
        
        # Remove rows with missing headlines
        cleaned_df = cleaned_df.dropna(subset=['headline'])
        
        # Sort by published date
        if 'published_date' in cleaned_df.columns:
            cleaned_df = cleaned_df.sort_values('published_date', ascending=False)
        
        return cleaned_df.reset_index(drop=True)

def save_to_csv(df: pd.DataFrame, output_dir: str = "data", filename: str = "newsAPI_headlines.csv"):
    """
    Save the DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_dir (str): Output directory
        filename (str): Output filename
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not df.empty:
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Saved {len(df)} articles to {filepath}")
        return filepath
    else:
        logger.warning("No data to save")
        return None

def main():
    """
    Main function to run the NewsAPI fetching script.
    """
    try:
        # Initialize the API client (will automatically load from .env)
        newsapi = NewsAPI()
    except ValueError as e:
        logger.error(str(e))
        logger.error("Please create a .env file with your API key:")
        logger.error("NEWSAPI_KEY=your_api_key_here")
        logger.error("Or get your API key from https://newsapi.org/")
        return
    
    # Calculate date range for last month
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Fetching S&P 500 news from {start_date_str} to {end_date_str}")
    
    try:
        # Fetch the news data
        sp500_news = newsapi.fetch_sp500_news(start_date_str, end_date_str)
        
        # Save to CSV file
        filepath = save_to_csv(sp500_news)
        
        # Print summary
        print("\n" + "="*50)
        print("FETCHING COMPLETE")
        print("="*50)
        print(f"S&P 500 Articles: {len(sp500_news)}")
        if filepath:
            print(f"Saved to: {filepath}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
