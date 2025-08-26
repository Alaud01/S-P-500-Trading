import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from gnews import GNews
import logging

logger = logging.getLogger(__name__)

class GNewsAPI:
    """
    Wrapper class for the GNews API to fetch S&P 500 related news.
    """
    
    def __init__(self):
        """
        Initialize the GNews API client.
        """
        self.gnews = GNews()
        self.gnews.language = 'en'
        self.gnews.country = 'US'
        self.gnews.max_results = 100
        
    def fetch_market_news(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch S&P 500 related news articles.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing 'sp500' key with DataFrame of articles
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
                    self.gnews.search = term
                    articles = self.gnews.get_news(term)
                    
                    for article in articles:
                        # Parse the published date
                        if 'published date' in article:
                            try:
                                pub_date = datetime.strptime(article['published date'], '%a, %d %b %Y %H:%M:%S %Z')
                                # Make timezone naive for comparison
                                pub_date = pub_date.replace(tzinfo=None)
                            except:
                                pub_date = datetime.now()
                        else:
                            pub_date = datetime.now()
                        
                        # Filter by date range
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                        
                        if start_dt <= pub_date <= end_dt:
                            article_data = {
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'published_date': pub_date.strftime('%Y-%m-%d'),
                                'url': article.get('url', ''),
                                'source': article.get('publisher', {}).get('title', ''),
                                'search_term': term
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
                
                logger.info(f"Retrieved {len(df)} unique articles from GNews API")
                return {'sp500': df}
            else:
                logger.info("No articles found from GNews API")
                return {'sp500': pd.DataFrame()}
                
        except Exception as e:
            logger.error(f"Error in GNews API: {e}")
            return {'sp500': pd.DataFrame()}
