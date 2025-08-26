import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from newsapi import NewsApiClient
import logging

logger = logging.getLogger(__name__)

class NewsAPI:
    """
    Wrapper class for the NewsAPI to fetch S&P 500 related news.
    """
    
    def __init__(self):
        """
        Initialize the NewsAPI client.
        """
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            raise ValueError("NEWS_API_KEY environment variable not found")
        
        self.newsapi = NewsApiClient(api_key=api_key)
        
    def fetch_sp500_news(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch S&P 500 related news articles from NewsAPI.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame containing articles
        """
        try:
            # Search for S&P 500 related terms
            search_terms = [
                "S&P 500",
                "SP500", 
                "S&P500",
                "Standard & Poor's 500",
                "stock market",
                "market index"
            ]
            
            all_articles = []
            
            for term in search_terms:
                try:
                    # Get articles from NewsAPI
                    articles = self.newsapi.get_everything(
                        q=term,
                        from_param=start_date,
                        to=end_date,
                        language='en',
                        sort_by='publishedAt',
                        page_size=100
                    )
                    
                    for article in articles.get('articles', []):
                        # Parse the published date
                        if article.get('publishedAt'):
                            try:
                                pub_date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                            except:
                                pub_date = datetime.now()
                        else:
                            pub_date = datetime.now()
                        
                        article_data = {
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'published_date': pub_date.strftime('%Y-%m-%d'),
                            'url': article.get('url', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'search_term': term,
                            'content': article.get('content', '')
                        }
                        all_articles.append(article_data)
                        
                except Exception as e:
                    logger.warning(f"Error fetching articles for term '{term}': {e}")
                    continue
            
            # Convert to DataFrame and remove duplicates
            if all_articles:
                df = pd.DataFrame(all_articles)
                df = df.drop_duplicates(subset=['title', 'url'])
                df = df.sort_values('published_date', ascending=False)
                
                logger.info(f"Retrieved {len(df)} unique articles from NewsAPI")
                return df
            else:
                logger.info("No articles found from NewsAPI")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in NewsAPI: {e}")
            return pd.DataFrame()
