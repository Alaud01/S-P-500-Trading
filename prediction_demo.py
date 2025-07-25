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
            'Price_Change_5d', 'Price_Change_20d', 'Sentiment_MA_5'
        ]
        
        lag_features = [f'Price_Lag_{lag}' for lag in [1, 5, 10, 20]] + \
                      [f'Returns_Lag_{lag}' for lag in [1, 5, 10, 20]]
        
        all_features = technical_features + lag_features
        
        lstm_features = [
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate',
            'RSI', 'MACD', 'Volatility_20', 'BB_Position', 'Sentiment_MA_5'
        ]
        
        # Get the last 60 days for LSTM sequence
        sequence_length = 60
        if len(df) >= sequence_length:
            lstm_sequence_df = df[lstm_features].iloc[-sequence_length:]
            lstm_sequence = lstm_sequence_df.values
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
        xgb_pred_proba = self.models['xgboost'].predict_proba(xgb_scaled)[0]
        predictions['XGBoost'] = xgb_pred_proba[1]
        
        # Linear prediction
        linear_scaled = self.scalers['linear'].transform(xgb_features)
        linear_pred_proba = self.models['linear'].predict_proba(linear_scaled)[0]
        predictions['Linear'] = linear_pred_proba[1]
        
        # LSTM prediction
        lstm_scaled = self.scalers['lstm'].transform(lstm_sequence.reshape(-1, lstm_sequence.shape[-1]))
        lstm_scaled = lstm_scaled.reshape(1, lstm_sequence.shape[0], lstm_sequence.shape[1])
        lstm_pred_proba = self.models['lstm'].predict(lstm_scaled)[0][0]
        predictions['LSTM'] = lstm_pred_proba
        
        # Meta-model prediction
        base_model_preds = np.array([[
            predictions['LSTM'],
            predictions['XGBoost'],
            predictions['Linear']
        ]])
        
        additional_features_list = ['CP', 'Volatility_20', 'RSI', 'Interest_Rate', 'Inflation_Rate', 'Sentiment_MA_5']
        additional_values = latest_data[additional_features_list].values
        
        meta_features = np.column_stack([base_model_preds, additional_values])
        
        meta_pred_proba = self.models['meta'].predict_proba(meta_features)[0]
        predictions['Meta-Model'] = meta_pred_proba[1]
        
        return predictions
        
    def display_predictions(self, predictions, latest_data):
        """Display predictions in a formatted way"""
        print(f"\nLatest S&P 500 Price: ${latest_data['CP'].iloc[0]:,.2f}")
        print(f"Date: {latest_data['Date'].iloc[0].strftime('%Y-%m-%d')}")
        
        print(f"\nPredicted 1-Month Direction & Confidence:")
        print("-" * 50)
        for model, proba in predictions.items():
            direction = "Up" if proba > 0.5 else "Down"
            confidence = proba if direction == "Up" else 1 - proba
            print(f"{model:12}: {direction:5} (Confidence: {confidence:.2%})")
            
        print("-" * 50)
        final_prediction = predictions['Meta-Model']
        final_direction = "Up" if final_prediction > 0.5 else "Down"
        final_confidence = final_prediction if final_direction == "Up" else 1 - final_prediction
        print(f"Final Verdict (Meta-Model): {final_direction} with {final_confidence:.2%} confidence.")
        
    def plot_prediction_comparison(self, predictions):
        """Plot comparison of different model predictions"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        models = list(predictions.keys())
        probabilities = list(predictions.values())
        
        colors = ['#2ECC71' if p > 0.5 else '#E74C3C' for p in probabilities]
        
        bars = ax.bar(models, probabilities, color=colors)
        ax.set_title('Model Prediction Probabilities for S&P 500 Going Up')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{prob:.2%}', ha='center', va='bottom')
        
        ax.legend()
        plt.tight_layout()
        plt.show()
        
    def analyze_prediction_confidence(self, predictions):
        """Analyze confidence in predictions based on model agreement"""
        print(f"\n" + "="*60)
        print("PREDICTION CONFIDENCE ANALYSIS")
        print("="*60)
        
        directions = {model: "Up" if prob > 0.5 else "Down" for model, prob in predictions.items()}
        
        up_votes = sum(1 for d in directions.values() if d == "Up")
        down_votes = len(directions) - up_votes
        
        print(f"Model Agreement:")
        print(f"  Models predicting UP:   {up_votes}")
        print(f"  Models predicting DOWN: {down_votes}")
        
        if up_votes == len(directions) or down_votes == len(directions):
            confidence = "VERY HIGH"
            reason = "All models are in perfect agreement."
        elif up_votes > down_votes and down_votes <= 1:
            confidence = "HIGH"
            reason = "Strong majority agreement for an UP trend."
        elif down_votes > up_votes and up_votes <= 1:
            confidence = "HIGH"
            reason = "Strong majority agreement for a DOWN trend."
        else:
            confidence = "MEDIUM"
            reason = "Models show some disagreement, final verdict should be treated with caution."
            
        print(f"\nOverall Conviction: {confidence}")
        print(f"Reason: {reason}")
        
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
        self.plot_prediction_comparison(predictions)
        
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