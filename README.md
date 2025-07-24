# S&P 500 Trading Prediction System

A comprehensive machine learning system for predicting S&P 500 prices using multiple models and advanced feature engineering.

## 📊 Project Overview

This project implements a sophisticated machine learning pipeline for S&P 500 price prediction, combining:
- **Data Analysis & Visualization**: Comprehensive exploration of S&P 500 data with technical indicators
- **Multiple ML Models**: LSTM, XGBoost, Linear Regression, and Meta-modeling
- **Feature Engineering**: Technical indicators, economic data, and lagged features
- **Time Series Analysis**: Proper train/validation/test splits for financial data

## 🏗️ Architecture

```
S&P 500 Trading/
├── visualization.py          # Data analysis and visualization
├── model.py                  # Machine learning pipeline
├── prediction_demo.py        # Model prediction demonstration
├── model_summary.py          # Comprehensive model analysis
├── dataset.py               # Data processing utilities
├── requirements.txt         # Python dependencies
├── models/                  # Trained model files
│   ├── lstm_model.h5
│   ├── xgboost_model.pkl
│   ├── linear_model.pkl
│   ├── meta_model.pkl
│   └── *_scaler.pkl
└── Data Files/
    ├── enhanced_sp500_dataset.csv
    ├── sp500_features_for_prediction.csv
    ├── inflation rate.csv
    └── interest rate.csv
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Analysis
```bash
python visualization.py
```
This will:
- Load and clean the S&P 500 dataset
- Create technical indicators
- Generate comprehensive visualizations
- Prepare features for modeling

### 3. Train Models
```bash
python model.py
```
This will:
- Train LSTM, XGBoost, Linear Regression, and Meta-models
- Evaluate model performance
- Save trained models to `models/` directory

### 4. View Model Summary
```bash
python model_summary.py
```
This provides:
- Model performance comparison
- Feature importance analysis
- Recommendations for improvement

## 📈 Model Performance

| Model | MAE | RMSE | R² | Description |
|-------|-----|------|----|-------------|
| **XGBoost** | **0.037429** | **0.045940** | **-0.0021** | Best overall performance |
| LSTM | 0.042240 | 0.054318 | -0.2957 | Deep learning with temporal patterns |
| Linear | 0.104036 | 0.118292 | -5.6441 | Baseline regression model |
| Meta-Model | 0.174551 | 0.189604 | -14.7884 | Ensemble of base models |

## 🔍 Key Features

### Technical Indicators
- Moving Averages (5, 20, 50, 200-day)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volatility measures
- Volume indicators

### Economic Data
- Interest rates
- Inflation rates
- Market sentiment indicators

### Feature Engineering
- Lagged price and return features
- Technical indicator combinations
- Economic regime detection
- Time-based features

## 🎯 Top Predictive Features

Based on XGBoost feature importance:

1. **MA_50** (0.1448) - 50-day moving average
2. **Inflation_Rate** (0.1448) - Economic indicator
3. **Price_Lag_10** (0.1169) - Price 10 days ago
4. **Price_Lag_20** (0.0670) - Price 20 days ago
5. **MA_20** (0.0645) - 20-day moving average

## 📊 Data Coverage

- **Time Period**: 2008-2024 (16+ years)
- **Data Points**: 3,438 clean records
- **Features**: 24 engineered features
- **Target**: 1-month future S&P 500 price

## 🔧 Model Details

### LSTM Model
- **Architecture**: 2 LSTM layers (128 → 64 units)
- **Sequence Length**: 60 days
- **Features**: 8 sequential technical indicators
- **Strengths**: Captures temporal dependencies

### XGBoost Model
- **Parameters**: Optimized for financial time series
- **Features**: 24 technical and economic indicators
- **Strengths**: Handles non-linear relationships, feature importance

### Linear Regression
- **Type**: Ridge Regression
- **Features**: Same as XGBoost
- **Strengths**: Interpretable, handles multicollinearity

### Meta-Model
- **Type**: XGBoost ensemble
- **Features**: Predictions from base models + additional features
- **Purpose**: Combine different modeling approaches

## 📋 Usage Examples

### Making Predictions
```python
from model import SP500Predictor

# Initialize predictor
predictor = SP500Predictor('sp500_features_for_prediction.csv')

# Train all models
results = predictor.train_all_models()

# Access individual models
lstm_model = predictor.models['lstm']['model']
xgboost_model = predictor.models['xgboost']['model']
```

### Data Analysis
```python
from visualization import SP500DataAnalyzer

# Initialize analyzer
analyzer = SP500DataAnalyzer('enhanced_sp500_dataset.csv')

# Run comprehensive analysis
analyzer.create_comprehensive_dashboard()
```

## ⚠️ Important Notes

### Risk Disclaimer
- These models are for **educational and research purposes only**
- Financial markets are inherently unpredictable
- Past performance does not guarantee future results
- Always conduct thorough research before making investment decisions

### Model Limitations
- Models may not capture regime changes or black swan events
- Economic conditions can change rapidly
- Models require regular retraining with new data
- No model can predict market crashes with certainty

## 🔄 Model Maintenance

### Regular Updates
- Retrain models monthly with new data
- Monitor model drift and performance degradation
- Update feature engineering as market conditions change
- Validate predictions against actual outcomes

### Performance Monitoring
- Track prediction accuracy over time
- Monitor feature importance changes
- Assess model stability across different market conditions
- Implement proper backtesting procedures

## 📚 Dependencies

### Core Libraries
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `tensorflow` - Deep learning
- `plotly` - Interactive visualizations

### Data Sources
- S&P 500 historical data
- Federal Reserve economic data
- Technical indicators calculated from price data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with any applicable regulations when using financial data and models.

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review the code comments
3. Open an issue on GitHub

---

**Remember**: This is a research tool. Always use proper risk management and never invest more than you can afford to lose. 