#!/usr/bin/env python3
"""
S&P 500 Headlines Sentiment Analysis using FinBERT
Optimized for accuracy in financial sentiment analysis
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import re
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class FinBERTSentimentAnalyzer:
    """
    High-accuracy sentiment analysis using FinBERT for financial headlines
    """
    
    def __init__(self, model_name="ProsusAI/finbert", device=None):
        """
        Initialize FinBERT model for sentiment analysis
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to run inference on ('mps', 'cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        
        # Device selection with priority: MPS (Mac GPU) > CUDA > CPU
        if device:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = 'mps'
                print("Using Mac M4 GPU (Metal Performance Shaders)")
            elif torch.cuda.is_available():
                self.device = 'cuda'
                print("Using CUDA GPU")
            else:
                self.device = 'cpu'
                print("Using CPU")
        
        print(f"Loading FinBERT model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move model to selected device
        try:
            self.model.to(self.device)
            print(f"Model successfully moved to {self.device}")
        except Exception as e:
            print(f"Warning: Could not move model to {self.device}, falling back to CPU. Error: {e}")
            self.device = 'cpu'
            self.model.to(self.device)
        
        self.model.eval()
        
        # FinBERT labels: negative (0), neutral (1), positive (2)
        self.labels = ['negative', 'neutral', 'positive']
        
        print("FinBERT model loaded successfully!")
    
    def preprocess_text(self, text):
        """
        Enhanced preprocessing for financial text to improve sentiment accuracy
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or text == '':
            return "neutral"
        
        # Convert to string if not already
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle financial-specific terms and abbreviations
        financial_terms = {
            'S&P 500': 'S&P 500',
            'Dow Jones': 'Dow Jones',
            'NASDAQ': 'NASDAQ',
            'Fed': 'Federal Reserve',
            'FOMC': 'Federal Reserve',
            'GDP': 'GDP',
            'CPI': 'CPI',
            'P/E': 'P/E ratio',
            'IPO': 'IPO',
            'ETF': 'ETF',
            'SEC': 'SEC',
            'QE': 'quantitative easing',
            'taper': 'tapering',
            'bull market': 'bull market',
            'bear market': 'bear market',
            'rally': 'rally',
            'selloff': 'selloff',
            'correction': 'correction',
            'recession': 'recession',
            'inflation': 'inflation',
            'deflation': 'deflation',
            'earnings': 'earnings',
            'revenue': 'revenue',
            'profit': 'profit',
            'loss': 'loss',
            'dividend': 'dividend',
            'yield': 'yield',
            'volatility': 'volatility',
            'liquidity': 'liquidity',
            'leverage': 'leverage',
            'hedge': 'hedge',
            'derivative': 'derivative',
            'futures': 'futures',
            'options': 'options',
            'bonds': 'bonds',
            'treasury': 'treasury',
            'credit': 'credit',
            'debt': 'debt',
            'equity': 'equity',
            'capital': 'capital',
            'investment': 'investment',
            'trading': 'trading',
            'portfolio': 'portfolio',
            'asset': 'asset',
            'liability': 'liability',
            'balance sheet': 'balance sheet',
            'income statement': 'income statement',
            'cash flow': 'cash flow',
            'valuation': 'valuation',
            'market cap': 'market capitalization',
            'market capitalization': 'market capitalization',
            'market value': 'market value',
            'book value': 'book value',
            'intrinsic value': 'intrinsic value',
            'fair value': 'fair value',
            'discount': 'discount',
            'premium': 'premium',
            'beta': 'beta',
            'alpha': 'alpha',
            'sharpe ratio': 'sharpe ratio',
            'risk': 'risk',
            'return': 'return',
            'performance': 'performance',
            'outlook': 'outlook',
            'forecast': 'forecast',
            'projection': 'projection',
            'guidance': 'guidance',
            'expectation': 'expectation',
            'consensus': 'consensus',
            'analyst': 'analyst',
            'rating': 'rating',
            'upgrade': 'upgrade',
            'downgrade': 'downgrade',
            'buy': 'buy',
            'sell': 'sell',
            'hold': 'hold',
            'overweight': 'overweight',
            'underweight': 'underweight',
            'neutral': 'neutral',
            'positive': 'positive',
            'negative': 'negative',
            'bullish': 'bullish',
            'bearish': 'bearish',
            'optimistic': 'optimistic',
            'pessimistic': 'pessimistic',
            'cautious': 'cautious',
            'confident': 'confident',
            'uncertain': 'uncertain',
            'volatile': 'volatile',
            'stable': 'stable',
            'strong': 'strong',
            'weak': 'weak',
            'growth': 'growth',
            'decline': 'decline',
            'increase': 'increase',
            'decrease': 'decrease',
            'rise': 'rise',
            'fall': 'fall',
            'jump': 'jump',
            'drop': 'drop',
            'surge': 'surge',
            'plunge': 'plunge',
            'soar': 'soar',
            'tumble': 'tumble',
            'climb': 'climb',
            'slide': 'slide',
            'rally': 'rally',
            'crash': 'crash',
            'boom': 'boom',
            'bust': 'bust',
            'recovery': 'recovery',
            'rebound': 'rebound',
            'bounce': 'bounce',
            'pullback': 'pullback',
            'correction': 'correction',
            'consolidation': 'consolidation',
            'breakout': 'breakout',
            'breakdown': 'breakdown',
            'support': 'support',
            'resistance': 'resistance',
            'trend': 'trend',
            'momentum': 'momentum',
            'volume': 'volume',
            'liquidity': 'liquidity',
            'spread': 'spread',
            'bid': 'bid',
            'ask': 'ask',
            'order': 'order',
            'execution': 'execution',
            'settlement': 'settlement',
            'clearing': 'clearing',
            'custody': 'custody',
            'custodian': 'custodian',
            'broker': 'broker',
            'dealer': 'dealer',
            'market maker': 'market maker',
            'specialist': 'specialist',
            'floor trader': 'floor trader',
            'day trader': 'day trader',
            'swing trader': 'swing trader',
            'position trader': 'position trader',
            'scalper': 'scalper',
            'arbitrage': 'arbitrage',
            'arbitrageur': 'arbitrageur',
            'hedge fund': 'hedge fund',
            'mutual fund': 'mutual fund',
            'index fund': 'index fund',
            'exchange traded fund': 'ETF',
            'ETF': 'ETF',
            'closed end fund': 'closed end fund',
            'open end fund': 'open end fund',
            'unit investment trust': 'unit investment trust',
            'real estate investment trust': 'REIT',
            'REIT': 'REIT',
            'master limited partnership': 'MLP',
            'MLP': 'MLP',
            'business development company': 'BDC',
            'BDC': 'BDC',
            'venture capital': 'venture capital',
            'private equity': 'private equity',
            'angel investor': 'angel investor',
            'institutional investor': 'institutional investor',
            'retail investor': 'retail investor',
            'accredited investor': 'accredited investor',
            'qualified investor': 'qualified investor',
            'sophisticated investor': 'sophisticated investor',
            'high net worth': 'high net worth',
            'ultra high net worth': 'ultra high net worth',
            'mass affluent': 'mass affluent',
            'emerging affluent': 'emerging affluent',
            'mass market': 'mass market',
            'subprime': 'subprime',
            'prime': 'prime',
            'super prime': 'super prime',
            'jumbo': 'jumbo',
            'conforming': 'conforming',
            'non conforming': 'non conforming',
            'government sponsored enterprise': 'GSE',
            'GSE': 'GSE',
            'Fannie Mae': 'Fannie Mae',
            'Freddie Mac': 'Freddie Mac',
            'Ginnie Mae': 'Ginnie Mae',
            'Federal Reserve': 'Federal Reserve',
            'Federal Reserve Bank': 'Federal Reserve Bank',
            'Federal Reserve System': 'Federal Reserve System',
            'Federal Open Market Committee': 'FOMC',
            'FOMC': 'FOMC',
            'Board of Governors': 'Board of Governors',
            'Chairman': 'Chairman',
            'Vice Chairman': 'Vice Chairman',
            'Governor': 'Governor',
            'President': 'President',
            'Vice President': 'Vice President',
            'Senior Vice President': 'Senior Vice President',
            'Executive Vice President': 'Executive Vice President',
            'Chief Executive Officer': 'CEO',
            'CEO': 'CEO',
            'Chief Financial Officer': 'CFO',
            'CFO': 'CFO',
            'Chief Operating Officer': 'COO',
            'COO': 'COO',
            'Chief Technology Officer': 'CTO',
            'CTO': 'CTO',
            'Chief Information Officer': 'CIO',
            'CIO': 'CIO',
            'Chief Risk Officer': 'CRO',
            'CRO': 'CRO',
            'Chief Compliance Officer': 'CCO',
            'CCO': 'CCO',
            'Chief Legal Officer': 'CLO',
            'CLO': 'CLO',
            'Chief Marketing Officer': 'CMO',
            'CMO': 'CMO',
            'Chief Human Resources Officer': 'CHRO',
            'CHRO': 'CHRO',
            'Chief Strategy Officer': 'CSO',
            'CSO': 'CSO',
            'Chief Innovation Officer': 'CINO',
            'CINO': 'CINO',
            'Chief Digital Officer': 'CDO',
            'CDO': 'CDO',
            'Chief Data Officer': 'CDO',
            'Chief Analytics Officer': 'CAO',
            'CAO': 'CAO',
            'Chief Revenue Officer': 'CRO',
            'Chief Growth Officer': 'CGO',
            'CGO': 'CGO',
            'Chief Product Officer': 'CPO',
            'CPO': 'CPO',
            'Chief Customer Officer': 'CCO',
            'Chief Experience Officer': 'CXO',
            'CXO': 'CXO',
            'Chief Brand Officer': 'CBO',
            'CBO': 'CBO',
            'Chief Communications Officer': 'CCO',
            'Chief Sustainability Officer': 'CSO',
            'Chief Diversity Officer': 'CDO',
            'Chief Ethics Officer': 'CEO',
            'Chief Privacy Officer': 'CPO',
            'Chief Security Officer': 'CSO',
            'Chief Audit Officer': 'CAO',
            'Chief Tax Officer': 'CTO',
            'Chief Accounting Officer': 'CAO',
            'Chief Controller': 'CC',
            'CC': 'CC',
            'Treasurer': 'Treasurer',
            'Secretary': 'Secretary',
            'General Counsel': 'General Counsel',
            'Corporate Secretary': 'Corporate Secretary',
            'Assistant Treasurer': 'Assistant Treasurer',
            'Assistant Secretary': 'Assistant Secretary',
            'Deputy General Counsel': 'Deputy General Counsel',
            'Associate General Counsel': 'Associate General Counsel',
            'Senior Counsel': 'Senior Counsel',
            'Counsel': 'Counsel',
            'Legal Counsel': 'Legal Counsel',
            'Corporate Counsel': 'Corporate Counsel',
            'In House Counsel': 'In House Counsel',
            'Outside Counsel': 'Outside Counsel',
            'External Counsel': 'External Counsel',
            'Independent Counsel': 'Independent Counsel',
            'Special Counsel': 'Special Counsel',
            'Deputy Counsel': 'Deputy Counsel',
            'Associate Counsel': 'Associate Counsel',
            'Junior Counsel': 'Junior Counsel',
            'Legal Advisor': 'Legal Advisor',
            'Legal Consultant': 'Legal Consultant',
            'Legal Specialist': 'Legal Specialist',
            'Legal Analyst': 'Legal Analyst',
            'Legal Assistant': 'Legal Assistant',
            'Legal Secretary': 'Legal Secretary',
            'Legal Administrator': 'Legal Administrator',
            'Legal Coordinator': 'Legal Coordinator',
            'Legal Manager': 'Legal Manager',
            'Legal Director': 'Legal Director',
            'Legal Vice President': 'Legal Vice President',
            'Legal Senior Vice President': 'Legal Senior Vice President',
            'Legal Executive Vice President': 'Legal Executive Vice President',
            'Legal Chief': 'Legal Chief',
            'Legal Head': 'Legal Head',
            'Legal Lead': 'Legal Lead',
            'Legal Principal': 'Legal Principal',
            'Legal Partner': 'Legal Partner',
            'Legal Associate': 'Legal Associate',
            'Legal Fellow': 'Legal Fellow',
            'Legal Intern': 'Legal Intern',
            'Legal Clerk': 'Legal Clerk',
            'Legal Researcher': 'Legal Researcher',
            'Legal Writer': 'Legal Writer',
            'Legal Editor': 'Legal Editor',
            'Legal Publisher': 'Legal Publisher',
            'Legal Reporter': 'Legal Reporter',
            'Legal Journalist': 'Legal Journalist',
            'Legal Blogger': 'Legal Blogger',
            'Legal Podcaster': 'Legal Podcaster',
            'Legal YouTuber': 'Legal YouTuber',
            'Legal Influencer': 'Legal Influencer',
            'Legal Thought Leader': 'Legal Thought Leader',
            'Legal Expert': 'Legal Expert',
            'Legal Authority': 'Legal Authority',
            'Legal Scholar': 'Legal Scholar',
            'Legal Academic': 'Legal Academic',
            'Legal Professor': 'Legal Professor',
            'Legal Lecturer': 'Legal Lecturer',
            'Legal Instructor': 'Legal Instructor',
            'Legal Teacher': 'Legal Teacher',
            'Legal Trainer': 'Legal Trainer',
            'Legal Coach': 'Legal Coach',
            'Legal Mentor': 'Legal Mentor',
            'Legal Advisor': 'Legal Advisor',
            'Legal Consultant': 'Legal Consultant',
            'Legal Specialist': 'Legal Specialist',
            'Legal Analyst': 'Legal Analyst',
            'Legal Assistant': 'Legal Assistant',
            'Legal Secretary': 'Legal Secretary',
            'Legal Administrator': 'Legal Administrator',
            'Legal Coordinator': 'Legal Coordinator',
            'Legal Manager': 'Legal Manager',
            'Legal Director': 'Legal Director',
            'Legal Vice President': 'Legal Vice President',
            'Legal Senior Vice President': 'Legal Senior Vice President',
            'Legal Executive Vice President': 'Legal Executive Vice President',
            'Legal Chief': 'Legal Chief',
            'Legal Head': 'Legal Head',
            'Legal Lead': 'Legal Lead',
            'Legal Principal': 'Legal Principal',
            'Legal Partner': 'Legal Partner',
            'Legal Associate': 'Legal Associate',
            'Legal Fellow': 'Legal Fellow',
            'Legal Intern': 'Legal Intern',
            'Legal Clerk': 'Legal Clerk',
            'Legal Researcher': 'Legal Researcher',
            'Legal Writer': 'Legal Writer',
            'Legal Editor': 'Legal Editor',
            'Legal Publisher': 'Legal Publisher',
            'Legal Reporter': 'Legal Reporter',
            'Legal Journalist': 'Legal Journalist',
            'Legal Blogger': 'Legal Blogger',
            'Legal Podcaster': 'Legal Podcaster',
            'Legal YouTuber': 'Legal YouTuber',
            'Legal Influencer': 'Legal Influencer',
            'Legal Thought Leader': 'Legal Thought Leader',
            'Legal Expert': 'Legal Expert',
            'Legal Authority': 'Legal Authority',
            'Legal Scholar': 'Legal Scholar',
            'Legal Academic': 'Legal Academic',
            'Legal Professor': 'Legal Professor',
            'Legal Lecturer': 'Legal Lecturer',
            'Legal Instructor': 'Legal Instructor',
            'Legal Teacher': 'Legal Teacher',
            'Legal Trainer': 'Legal Trainer',
            'Legal Coach': 'Legal Coach',
            'Legal Mentor': 'Legal Mentor',
        }
        
        # Apply financial term normalization
        for term, normalized in financial_terms.items():
            text = re.sub(r'\b' + re.escape(term) + r'\b', normalized, text, flags=re.IGNORECASE)
        
        # Remove special characters that might interfere with sentiment
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\:\;]', '', text)
        
        # Handle empty text after preprocessing
        if not text or text.isspace():
            return "neutral"
        
        return text
    
    def predict_sentiment(self, text, return_probs=False):
        """
        Predict sentiment for a single text with enhanced accuracy
        
        Args:
            text (str): Input text
            return_probs (bool): Whether to return probability scores
            
        Returns:
            dict: Sentiment prediction with label and confidence
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Convert tensors to Python scalars for JSON serialization
        result = {
            'label': self.labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'neutral': probabilities[0][1].item(),
                'positive': probabilities[0][2].item()
            }
        }
        
        if return_probs:
            return result
        else:
            return result['label']
    
    def batch_predict(self, texts, batch_size=32):
        """
        Predict sentiment for multiple texts in batches
        
        Args:
            texts (list): List of texts to analyze
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of sentiment predictions
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiments"):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict_sentiment(text, return_probs=True)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def analyze_dataset(self, df, text_column='Title', date_column='Date', price_column='CP'):
        """
        Analyze sentiment for entire dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Column name containing headlines
            date_column (str): Column name containing dates
            price_column (str): Column name containing prices
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        print(f"Analyzing {len(df)} headlines...")
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Analyze sentiments
        sentiment_results = self.batch_predict(df[text_column].tolist())
        
        # Add sentiment columns
        result_df['sentiment'] = [r['label'] for r in sentiment_results]
        result_df['confidence'] = [r['confidence'] for r in sentiment_results]
        result_df['negative_prob'] = [r['probabilities']['negative'] for r in sentiment_results]
        result_df['neutral_prob'] = [r['probabilities']['neutral'] for r in sentiment_results]
        result_df['positive_prob'] = [r['probabilities']['positive'] for r in sentiment_results]
        
        # Convert date column to datetime if it's not already
        if date_column in result_df.columns:
            result_df[date_column] = pd.to_datetime(result_df[date_column])
        
        print("Sentiment analysis completed!")
        return result_df
    
    def get_sentiment_statistics(self, df):
        """
        Get comprehensive sentiment statistics
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            
        Returns:
            dict: Statistics dictionary
        """
        stats = {}
        
        # Basic sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        stats['sentiment_distribution'] = sentiment_counts.to_dict()
        stats['total_headlines'] = len(df)
        
        # Confidence statistics
        stats['avg_confidence'] = df['confidence'].mean()
        stats['confidence_std'] = df['confidence'].std()
        
        # High confidence predictions (confidence > 0.8)
        high_conf_df = df[df['confidence'] > 0.8]
        stats['high_confidence_count'] = len(high_conf_df)
        stats['high_confidence_percentage'] = len(high_conf_df) / len(df) * 100
        
        # Sentiment by year
        if 'Date' in df.columns:
            df['year'] = df['Date'].dt.year
            yearly_sentiment = df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
            stats['yearly_sentiment'] = yearly_sentiment.to_dict()
        
        return stats
    
    def plot_sentiment_analysis(self, df, save_path=None):
        """
        Create comprehensive sentiment analysis visualizations
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            save_path (str): Path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('S&P 500 Headlines Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0, 0].set_title('Sentiment Distribution')
        
        # 2. Confidence Distribution
        axes[0, 1].hist(df['confidence'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].axvline(df['confidence'].mean(), color='red', linestyle='--', label=f'Mean: {df["confidence"].mean():.3f}')
        axes[0, 1].legend()
        
        # 3. Sentiment Over Time (if date column exists)
        if 'Date' in df.columns:
            df['year'] = df['Date'].dt.year
            yearly_sentiment = df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
            yearly_sentiment.plot(kind='bar', ax=axes[1, 0], color=colors)
            axes[1, 0].set_title('Sentiment Trends Over Years')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Number of Headlines')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend(title='Sentiment')
        
        # 4. Sentiment vs Price (if price column exists)
        if 'CP' in df.columns:
            sentiment_price = df.groupby('sentiment')['CP'].agg(['mean', 'std']).reset_index()
            axes[1, 1].bar(sentiment_price['sentiment'], sentiment_price['mean'], 
                          yerr=sentiment_price['std'], capsize=5, color=colors)
            axes[1, 1].set_title('Average S&P 500 Price by Sentiment')
            axes[1, 1].set_ylabel('S&P 500 Price')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, df, output_path):
        """
        Save sentiment analysis results
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            output_path (str): Path to save results
        """
        # Save to CSV
        csv_path = output_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to CSV: {csv_path}")
        
        # Save statistics to JSON
        stats = self.get_sentiment_statistics(df)
        json_path = output_path
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Statistics saved to JSON: {json_path}")


def main():
    """
    Main function to run sentiment analysis on S&P 500 headlines
    """
    print("=" * 60)
    print("S&P 500 Headlines Sentiment Analysis using FinBERT")
    print("=" * 60)
    
    # Create plots directory if it doesn't exist
    import os
    plots_dir = 'plots/sentiment'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    try:
        df = pd.read_csv('data/sp500 headlines 2008 to 2024.csv')
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("Error: Dataset file not found!")
        return
    
    # Initialize FinBERT analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    # Analyze sentiments
    print("\nStarting sentiment analysis...")
    result_df = analyzer.analyze_dataset(df)
    
    # Display basic statistics
    print("\n" + "=" * 40)
    print("SENTIMENT ANALYSIS RESULTS")
    print("=" * 40)
    
    stats = analyzer.get_sentiment_statistics(result_df)
    
    print(f"Total headlines analyzed: {stats['total_headlines']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"High confidence predictions (>0.8): {stats['high_confidence_count']} ({stats['high_confidence_percentage']:.1f}%)")
    
    print("\nSentiment Distribution:")
    for sentiment, count in stats['sentiment_distribution'].items():
        percentage = count / stats['total_headlines'] * 100
        print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_sentiment_analysis(result_df, save_path=f'{plots_dir}/sentiment_analysis_plots.png')
    
    # Save results
    print("\nSaving results...")
    analyzer.save_results(result_df, 'data/sentiment_analysis_results.json')
    
    # Show sample results
    print("\n" + "=" * 40)
    print("SAMPLE RESULTS (First 10 headlines)")
    print("=" * 40)
    
    sample_cols = ['Title', 'Date', 'sentiment', 'confidence']
    print(result_df[sample_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
