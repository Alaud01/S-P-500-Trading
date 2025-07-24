import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import from our modules
from visualization import SP500DataAnalyzer
from model import SP500Predictor

class SP500PredictionDemo:
    def __init__(self):
        """Initialize the prediction demo"""
        self.predictor = None
        self.models = {}
        self.scalers = {}
        
    def load_trained_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        
        # Load LSTM model
        self.models['lstm'] = load_model('models/lstm_model.h5')
        self.scalers['lstm'] = joblib.load('models/lstm_scaler.pkl')
        
        # Load XGBoost model
        self.models['xgboost'] = joblib.load('models/xgboost_model.pkl')
        self.scalers['xgboost'] = joblib.load('models/xgboost_scaler.pkl')
        
        # Load Linear model
        self.models['linear'] = joblib.load('models/linear_model.pkl')
        self.scalers['linear'] = joblib.load('models/linear_scaler.pkl')
        
        # Load Meta model
        self.models['meta'] = joblib.load('models/meta_model.pkl')
        
        print("All models loaded successfully!")
        
    def prepare_latest_data(self, data_file='sp500_features_for_prediction.csv'):
        """Prepare the latest data for prediction"""
        print("Preparing latest data for prediction...")
        
        # Load data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Get the latest data point
        latest_data = df.iloc[-1:].copy()
        
        # Features for different models
        technical_features = [
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate',
            'MA_5', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Position', 'BB_Width', 'Volatility_20', 'Volume_Ratio',
            'Price_Change_5d', 'Price_Change_20d'
        ]
        
        lag_features = [f'Price_Lag_{lag}' for lag in [1, 5, 10, 20]] + \
                      [f'Returns_Lag_{lag}' for lag in [1, 5, 10, 20]]
        
        all_features = technical_features + lag_features
        
        lstm_features = [
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate',
            'RSI', 'MACD', 'Volatility_20', 'BB_Position'
        ]
        
        # Get the last 60 days for LSTM sequence
        sequence_length = 60
        if len(df) >= sequence_length:
            lstm_sequence = df[lstm_features].iloc[-sequence_length:].values
        else:
            print("Warning: Not enough data for LSTM sequence")
            return None, None, None
            
        return latest_data[all_features], lstm_sequence, latest_data
        
    def make_predictions(self, xgb_features, lstm_sequence, latest_data):
        """Make predictions using all models"""
        print("\n" + "="*60)
        print("MAKING PREDICTIONS")
        print("="*60)
        
        predictions = {}
        
        # XGBoost prediction
        xgb_scaled = self.scalers['xgboost'].transform(xgb_features)
        xgb_pred = self.models['xgboost'].predict(xgb_scaled)[0]
        predictions['XGBoost'] = xgb_pred
        
        # Linear prediction
        linear_scaled = self.scalers['linear'].transform(xgb_features)
        linear_pred = self.models['linear'].predict(linear_scaled)[0]
        predictions['Linear'] = linear_pred
        
        # LSTM prediction
        lstm_scaled = self.scalers['lstm'].transform(lstm_sequence.reshape(-1, lstm_sequence.shape[-1]))
        lstm_scaled = lstm_scaled.reshape(1, lstm_sequence.shape[0], lstm_sequence.shape[1])
        lstm_pred = self.models['lstm'].predict(lstm_scaled)[0][0]
        predictions['LSTM'] = lstm_pred
        
        # Meta-model prediction
        meta_features = np.array([[lstm_pred, xgb_pred, linear_pred]])
        # Add additional features
        additional_features = ['CP', 'Volatility_20', 'RSI', 'Interest_Rate', 'Inflation_Rate']
        additional_values = latest_data[additional_features].values
        meta_features = np.column_stack([meta_features, additional_values])
        
        meta_pred = self.models['meta'].predict(meta_features)[0]
        predictions['Meta-Model'] = meta_pred
        
        return predictions
        
    def display_predictions(self, predictions, latest_data):
        """Display predictions in a formatted way"""
        print(f"\nLatest S&P 500 Price: ${latest_data['CP'].iloc[0]:,.2f}")
        print(f"Date: {latest_data['Date'].iloc[0].strftime('%Y-%m-%d')}")
        
        print(f"\nPredicted 1-Month Returns:")
        print("-" * 40)
        for model, pred in predictions.items():
            print(f"{model:12}: {pred*100:8.3f}%")
            
        print(f"\nPredicted 1-Month Price:")
        print("-" * 40)
        current_price = latest_data['CP'].iloc[0]
        for model, pred in predictions.items():
            predicted_price = current_price * (1 + pred)
            print(f"{model:12}: ${predicted_price:10,.2f}")
            
        # Calculate ensemble prediction (simple average)
        ensemble_pred = np.mean(list(predictions.values()))
        ensemble_price = current_price * (1 + ensemble_pred)
        print(f"\nEnsemble Average:")
        print(f"Return: {ensemble_pred*100:8.3f}%")
        print(f"Price:  ${ensemble_price:10,.2f}")
        
    def plot_prediction_comparison(self, predictions, latest_data):
        """Plot comparison of different model predictions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of returns
        models = list(predictions.keys())
        returns = [pred * 100 for pred in predictions.values()]
        
        bars = ax1.bar(models, returns, color=['blue', 'green', 'red', 'purple', 'orange'])
        ax1.set_title('Predicted 1-Month Returns by Model')
        ax1.set_ylabel('Return (%)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:.2f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Bar plot of prices
        current_price = latest_data['CP'].iloc[0]
        prices = [current_price * (1 + pred) for pred in predictions.values()]
        
        bars2 = ax2.bar(models, prices, color=['blue', 'green', 'red', 'purple', 'orange'])
        ax2.set_title('Predicted 1-Month S&P 500 Price by Model')
        ax2.set_ylabel('Price ($)')
        ax2.axhline(y=current_price, color='black', linestyle='--', alpha=0.7, label='Current Price')
        
        # Add value labels on bars
        for bar, price in zip(bars2, prices):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${price:,.0f}', ha='center', va='bottom' if height > current_price else 'top')
        
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    def analyze_prediction_confidence(self, predictions):
        """Analyze confidence in predictions based on model agreement"""
        print(f"\n" + "="*60)
        print("PREDICTION CONFIDENCE ANALYSIS")
        print("="*60)
        
        returns = list(predictions.values())
        
        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        min_return = np.min(returns)
        max_return = np.max(returns)
        range_return = max_return - min_return
        
        print(f"Mean Prediction: {mean_return*100:.3f}%")
        print(f"Standard Deviation: {std_return*100:.3f}%")
        print(f"Range: {range_return*100:.3f}% ({min_return*100:.3f}% to {max_return*100:.3f}%)")
        
        # Confidence assessment
        if range_return < 0.02:  # Less than 2% range
            confidence = "HIGH"
            reason = "Models are in strong agreement"
        elif range_return < 0.05:  # Less than 5% range
            confidence = "MEDIUM"
            reason = "Models show moderate agreement"
        else:
            confidence = "LOW"
            reason = "Models show significant disagreement"
            
        print(f"\nConfidence Level: {confidence}")
        print(f"Reason: {reason}")
        
        # Direction agreement
        positive_predictions = sum(1 for r in returns if r > 0)
        negative_predictions = len(returns) - positive_predictions
        
        print(f"\nDirection Agreement:")
        print(f"Models predicting UP: {positive_predictions}")
        print(f"Models predicting DOWN: {negative_predictions}")
        
        if positive_predictions > negative_predictions:
            direction = "BULLISH"
        elif negative_predictions > positive_predictions:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
            
        print(f"Overall Direction: {direction}")
        
    def run_demo(self):
        """Run the complete prediction demo"""
        print("="*80)
        print("S&P 500 PREDICTION DEMO")
        print("="*80)
        
        # Load models
        self.load_trained_models()
        
        # Prepare data
        xgb_features, lstm_sequence, latest_data = self.prepare_latest_data()
        
        if xgb_features is None:
            print("Error: Could not prepare data for prediction")
            return
            
        # Make predictions
        predictions = self.make_predictions(xgb_features, lstm_sequence, latest_data)
        
        # Display results
        self.display_predictions(predictions, latest_data)
        
        # Plot comparison
        self.plot_prediction_comparison(predictions, latest_data)
        
        # Analyze confidence
        self.analyze_prediction_confidence(predictions)
        
        print(f"\n" + "="*80)
        print("DEMO COMPLETE")
        print("="*80)
        print("Note: These predictions are for educational purposes only.")
        print("Always do your own research before making investment decisions.")

def main():
    """Main function to run the prediction demo"""
    demo = SP500PredictionDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 