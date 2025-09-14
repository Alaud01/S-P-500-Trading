# S&P 500 Trading Strategy with Sentiment Analysis & Machine Learning

This project implements a comprehensive trading strategy that combines **sentiment analysis using FinBERT** with economic indicators to predict S&P 500 movements using multiple machine learning models and a meta-learning approach.

## Overview

The project follows a complete pipeline from data collection to trading strategy implementation:

1. **Sentiment Analysis**: FinBERT-based sentiment analysis of financial headlines (2008-2024)
2. **Economic Data Integration**: Combines sentiment with GDP, inflation, interest rates, unemployment, and gold prices
3. **Dataset Construction**: Builds comprehensive feature set with technical indicators
4. **Pattern Visualization**: Analyzes correlations and trends across all features
5. **Multi-Model Training**: Trains LSTM, Linear Regression, and XGBoost models
6. **Meta-Learning**: Combines individual models using a meta-classifier
7. **Strategy Implementation**: Backtests trading strategy on 2008-2024 data

## Key Features

- **FinBERT Integration**: Uses ProsusAI/finbert for financial sentiment analysis with Mac M4 GPU acceleration
- **Multi-Feature Dataset**: Combines sentiment, economic indicators, and technical analysis
- **Ensemble Learning**: LSTM + Linear + XGBoost models with meta-classifier
- **Comprehensive Visualization**: Pattern analysis and model performance plots
- **Trading Strategy**: Implements buy/sell signals with backtesting results
- **16-Year Dataset**: Complete analysis from 2008 to 2024

## Project Structure

### Core Scripts
- `sentiment.py` - FinBERT sentiment analysis with GPU acceleration
- `built_dataset.py` - Combines sentiment with economic data and builds final dataset
- `visualization.py` - Creates comprehensive analysis plots and correlations
- `train_lstm.py` - LSTM model training and evaluation
- `train_linear.py` - Linear regression model training
- `train_xgboost.py` - XGBoost model training and optimization
- `train_meta.py` - Meta-classifier combining all models
- `comparison.py` - Model comparison and trading strategy backtesting

### Data Files
- `data/sp500 headlines 2008 to 2024.csv` - Original S&P 500 headlines dataset
- `data/sentiment_analysis_results.csv` - FinBERT sentiment analysis results
- `data/final_dataset_for_modeling.csv` - Complete dataset with all features
- `data/GDP.csv`, `data/inflation rate.csv`, `data/interest rate.csv` - Economic indicators
- `data/Unemployment.csv`, `data/gold_price.csv` - Additional economic data

### Model Outputs
- `models/lstm/` - LSTM model files and reports
- `models/linear/` - Linear regression model files
- `models/xgboost/` - XGBoost model files and feature importance
- `models/meta/` - Meta-classifier model and predictions

### Visualizations
- `plots/sentiment/` - Sentiment analysis visualizations
- `plots/visualization/` - Feature correlation and pattern analysis
- `plots/lstm/`, `plots/linear/`, `plots/xgboost/` - Individual model performance
- `plots/meta/` - Meta-model performance and calibration
- `plots/comparison/` - Trading strategy backtesting results

## Installation

1. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Complete Workflow

### Step 1: Sentiment Analysis with FinBERT
Run sentiment analysis on S&P 500 headlines using FinBERT with GPU acceleration:

```bash
python sentiment.py
```

**What it does:**
- Processes 19,127 headlines from 2008-2024
- Uses ProsusAI/finbert model with Mac M4 GPU acceleration
- Generates sentiment scores (positive/neutral/negative) with confidence levels
- Outputs: `data/sentiment_analysis_results.csv` and `data/sentiment_analysis_results.json`

### Step 2: Dataset Construction
Combine sentiment data with economic indicators and build comprehensive dataset:

```bash
python built_dataset.py
```

**What it does:**
- Merges sentiment analysis with economic data (GDP, inflation, interest rates, unemployment, gold prices)
- Adds technical indicators (RSI, MACD, moving averages, volatility)
- Creates target variables for price movement prediction
- Outputs: `data/final_dataset_for_modeling.csv`

### Step 3: Pattern Visualization
Analyze correlations and patterns across all features:

```bash
python visualization.py
```

**What it does:**
- Creates correlation matrices between all features
- Visualizes economic indicators over time
- Analyzes sentiment trends and market relationships
- Generates feature importance and target analysis plots
- Outputs: Various plots in `plots/visualization/` and `plots/sentiment/`

### Step 4: Model Training

#### LSTM Model
```bash
python train_lstm.py
```
- Trains LSTM neural network for sequence-based prediction
- Uses time series features and sentiment data
- Outputs: Model files in `models/lstm/` and performance plots

#### Linear Regression
```bash
python train_linear.py
```
- Trains logistic regression for binary classification
- Uses feature engineering and regularization
- Outputs: Model files in `models/linear/` and performance plots

#### XGBoost
```bash
python train_xgboost.py
```
- Trains gradient boosting model with hyperparameter optimization
- Provides feature importance analysis
- Outputs: Model files in `models/xgboost/` and feature importance plots

### Step 5: Meta-Learning
Combine all models using a meta-classifier:

```bash
python train_meta.py
```

**What it does:**
- Uses predictions from LSTM, Linear, and XGBoost as features
- Trains a meta-classifier to combine model outputs
- Provides model calibration and threshold optimization
- Outputs: Meta-model in `models/meta/` and calibration plots

### Step 6: Trading Strategy & Backtesting
Compare models and implement trading strategy:

```bash
python comparison.py
```

**What it does:**
- Backtests trading strategies using all models
- Generates equity curves and performance metrics
- Shows allocation strategies and trade signals
- Outputs: Trading performance plots in `plots/comparison/`

## GPU Acceleration

### Mac M4 GPU Support
This script automatically detects and uses your Mac M4 GPU through Metal Performance Shaders (MPS) for significantly faster processing:

- **Automatic Detection**: Script automatically detects MPS availability
- **Fallback Support**: Gracefully falls back to CPU if GPU unavailable
- **Performance Boost**: ~2-3x faster processing compared to CPU-only

### Test GPU Functionality
```bash
python test_mps.py
```

## Quick Start

### Run Complete Pipeline
Execute all steps in sequence:

```bash
# Step 1: Sentiment Analysis
python sentiment.py

# Step 2: Build Dataset
python built_dataset.py

# Step 3: Visualize Patterns
python visualization.py

# Step 4: Train Models
python train_lstm.py
python train_linear.py
python train_xgboost.py

# Step 5: Meta-Learning
python train_meta.py

# Step 6: Trading Strategy
python comparison.py
```

### Individual Component Testing

```bash
# Test sentiment analysis accuracy
python test_sentiment.py

# Test GPU functionality
python test_mps.py
```

## Key Outputs

### Data Files
- `data/sentiment_analysis_results.csv` - FinBERT sentiment analysis results
- `data/final_dataset_for_modeling.csv` - Complete dataset with all features
- `data/merged_sp500_dataset.csv` - Intermediate merged dataset

### Model Performance Reports
- `models/lstm/lstm_report.json` - LSTM model performance metrics
- `models/linear/linear_report.json` - Linear regression performance
- `models/xgboost/xgb_best_report.json` - XGBoost best model results
- `models/meta/meta_report.json` - Meta-classifier performance

### Trading Strategy Results
- `models/meta/meta_holdout_predictions.csv` - Meta-model predictions
- `models/xgboost/xgb_best_holdout_predictions.csv` - XGBoost predictions
- Trading performance plots in `plots/comparison/`

## Model Architecture

### Sentiment Analysis (FinBERT)
- **Model**: ProsusAI/finbert
- **Purpose**: Financial text sentiment analysis
- **Labels**: Negative, Neutral, Positive
- **Hardware**: Mac M4 GPU acceleration via MPS
- **Features**: Financial term normalization, batch processing

### Machine Learning Models

#### LSTM Neural Network
- **Architecture**: Long Short-Term Memory for sequence modeling
- **Features**: Time series data, sentiment scores, technical indicators
- **Purpose**: Capture temporal dependencies in market data

#### Linear Regression (Logistic)
- **Type**: Regularized logistic regression
- **Features**: Engineered features from all data sources
- **Purpose**: Baseline classification with interpretable coefficients

#### XGBoost
- **Type**: Gradient boosting with hyperparameter optimization
- **Features**: All available features with importance ranking
- **Purpose**: Non-linear pattern detection and feature selection

#### Meta-Classifier
- **Type**: Logistic regression combining model predictions
- **Input**: Predictions from LSTM, Linear, and XGBoost models
- **Purpose**: Ensemble learning for improved accuracy

## Results Interpretation

### Sentiment Analysis Results
- **Positive**: Headlines indicating market optimism, growth, or positive outcomes
- **Neutral**: Factual reporting without clear sentiment bias  
- **Negative**: Headlines indicating market pessimism, decline, or negative outcomes
- **Confidence Scores**: High (>0.8), Medium (0.6-0.8), Low (<0.6)

### Model Performance Metrics
- **Accuracy**: Classification accuracy on holdout test set
- **Precision/Recall**: Model performance for buy/sell signals
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Feature Importance**: XGBoost feature ranking and LSTM attention weights

### Trading Strategy Results
- **Equity Curves**: Portfolio value over time for each model
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Allocation Strategy**: Model-based position sizing

## Performance

### Sentiment Analysis
- **Processing Speed**: ~1.08 headlines/second on Mac M4 GPU (MPS)
- **GPU Acceleration**: 2-3x faster than CPU-only processing
- **Accuracy**: High confidence predictions for 66.5% of headlines

### Model Training
- **LSTM**: Sequence-based learning with attention mechanisms
- **Linear**: Fast training with interpretable results
- **XGBoost**: Hyperparameter optimization with feature importance
- **Meta-Model**: Ensemble learning for improved accuracy

### Trading Strategy
- **Backtesting Period**: 2008-2024 (16 years of data)
- **Data Points**: 19,127 headlines with economic indicators
- **Model Ensemble**: Combines multiple approaches for robust predictions

## Dataset Information

### Primary Data Sources
- **S&P 500 Headlines**: 19,127 headlines from 2008 to 2024
- **Economic Indicators**: GDP, inflation rate, interest rate, unemployment rate
- **Commodity Prices**: Gold price data
- **Market Data**: S&P 500 closing prices and technical indicators

### Feature Engineering
- **Sentiment Features**: FinBERT sentiment scores and confidence levels
- **Technical Indicators**: RSI, MACD, moving averages, volatility measures
- **Economic Features**: Lagged and differenced economic indicators
- **Target Variables**: Price movement classification (up/down)

### Data Quality
- **Time Period**: 16 years of comprehensive financial data (2008-2024)
- **Frequency**: Daily market data with monthly economic indicators
- **Completeness**: High-quality dataset with minimal missing values
- **Validation**: Cross-validation and holdout testing for model evaluation

## Technical Requirements

- **macOS**: 12.3+ (for MPS support)
- **Python**: 3.11+
- **PyTorch**: 2.8.0+ (with MPS support)
- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4) or Intel Mac with Metal support

### Dependencies
- **Deep Learning**: PyTorch 2.8.0+ (with MPS support), Transformers 4.55.3
- **Data Processing**: Pandas 2.3.2, NumPy, Scikit-learn
- **Visualization**: Matplotlib 3.10.5, Seaborn 0.13.2
- **Machine Learning**: XGBoost, LightGBM
- **Financial Data**: yfinance, ta-lib
- **Utilities**: tqdm, joblib, warnings

## Customization

### Force CPU Usage
```python
analyzer = FinBERTSentimentAnalyzer(device='cpu')
```

### Force GPU Usage
```python
analyzer = FinBERTSentimentAnalyzer(device='mps')
```

### Adjust Batch Size
```python
results = analyzer.batch_predict(texts, batch_size=64)
```

### Custom Preprocessing
Modify the `preprocess_text()` method in the `FinBERTSentimentAnalyzer` class.

## Troubleshooting

### GPU Issues

1. **MPS Not Available**: Ensure macOS 12.3+ and PyTorch 2.8.0+
2. **Model Loading Errors**: Script automatically falls back to CPU
3. **Memory Issues**: Reduce batch size for large datasets

### Performance Tips

- **Optimal Batch Size**: 32-64 for most Mac M4 systems
- **Memory Management**: Monitor GPU memory usage
- **Fallback**: Script automatically uses CPU if GPU unavailable

### Common Issues

1. **Memory Issues**: Reduce batch size for large datasets
2. **Model Loading**: Ensure internet connection for first-time model download
3. **GPU Compatibility**: Test with `python test_mps.py`

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!
