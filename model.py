import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Additional utilities
from scipy import stats
import joblib
import os

# Import from visualization
from visualization import SP500DataAnalyzer

class SP500Predictor:
    def __init__(self, data_file='sp500_features_for_prediction.csv'):
        """Initialize the S&P 500 predictor with data"""
        print("Loading S&P 500 prediction dataset...")
        self.df = pd.read_csv(data_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        print(f"Dataset loaded: {len(self.df)} records from {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        # Target variable
        self.target = 'Direction_1Month_Future'
        
        # Initialize scalers
        self.scaler = StandardScaler()
        self.lstm_scaler = MinMaxScaler()
        
        # Model storage
        self.models = {}
        self.meta_model = None
        self.class_weights = None
        self.scale_pos_weight = None
        
    def prepare_features(self):
        """Prepare features for different models"""
        print("\n" + "="*60)
        print("FEATURE PREPARATION")
        print("="*60)
        
        # Technical indicators (for XGBoost and Linear models)
        technical_features = [
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate',
            'MA_5', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Position', 'BB_Width', 'Volatility_20', 'Volume_Ratio',
            'Price_Change_5d', 'Price_Change_20d', 'Sentiment_MA_5'
        ]
        
        # Lag features
        lag_features = [f'Price_Lag_{lag}' for lag in [1, 5, 10, 20]] + \
                      [f'Returns_Lag_{lag}' for lag in [1, 5, 10, 20]]
        
        # All features for XGBoost and Linear models
        self.all_features = technical_features + lag_features
        
        # Sequential features for LSTM (last 60 days)
        self.sequence_length = 60
        self.lstm_features = [
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate',
            'RSI', 'MACD', 'Volatility_20', 'BB_Position', 'Sentiment_MA_5'
        ]
        
        # Remove rows with NaN values
        self.df_clean = self.df[self.all_features + [self.target, 'Date']].dropna()
        
        print(f"Clean dataset: {len(self.df_clean)} records")
        print(f"Features for XGBoost/Linear: {len(self.all_features)}")
        print(f"Features for LSTM: {len(self.lstm_features)}")
        
        return self.df_clean
        
    def create_sequences(self, data, features, target, sequence_length):
        """Create sequences for LSTM model"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[features].iloc[i-sequence_length:i].values)
            y.append(data[target].iloc[i])
            
        return np.array(X), np.array(y)
        
    def split_data_time_series(self, data, test_size=0.2, validation_size=0.1):
        """Split data using time series split"""
        total_size = len(data)
        test_split = int(total_size * (1 - test_size))
        val_split = int(test_split * (1 - validation_size))
        
        train_data = data.iloc[:val_split]
        val_data = data.iloc[val_split:test_split]
        test_data = data.iloc[test_split:]
        
        print(f"Train: {len(train_data)} records ({train_data['Date'].min()} to {train_data['Date'].max()})")
        print(f"Validation: {len(val_data)} records ({val_data['Date'].min()} to {val_data['Date'].max()})")
        print(f"Test: {len(test_data)} records ({test_data['Date'].min()} to {test_data['Date'].max()})")
        
        return train_data, val_data, test_data
        
    def train_lstm_model(self, train_data, val_data, test_data):
        """Train LSTM model on sequential data"""
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL")
        print("="*60)
        
        # Prepare sequences
        X_train, y_train = self.create_sequences(train_data, self.lstm_features, self.target, self.sequence_length)
        X_val, y_val = self.create_sequences(val_data, self.lstm_features, self.target, self.sequence_length)
        X_test, y_test = self.create_sequences(test_data, self.lstm_features, self.target, self.sequence_length)
        
        # Scale the data
        X_train_scaled = self.lstm_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = self.lstm_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = self.lstm_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Build LSTM model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, len(self.lstm_features))),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Make predictions
        train_pred_proba = model.predict(X_train_scaled).flatten()
        val_pred_proba = model.predict(X_val_scaled).flatten()
        test_pred_proba = model.predict(X_test_scaled).flatten()
        
        # Convert probabilities to binary predictions
        train_pred = (train_pred_proba > 0.5).astype(int)
        val_pred = (val_pred_proba > 0.5).astype(int)
        test_pred = (test_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"LSTM Results:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Store model and predictions
        self.models['lstm'] = {
            'model': model,
            'scaler': self.lstm_scaler,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'test_pred': test_pred,
            'train_pred_proba': train_pred_proba,
            'val_pred_proba': val_pred_proba,
            'test_pred_proba': test_pred_proba,
            'train_actual': y_train,
            'val_actual': y_val,
            'test_actual': y_test
        }
        
        return model, train_pred_proba, val_pred_proba, test_pred_proba
        
    def train_xgboost_model(self, train_data, val_data, test_data):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        # Prepare data
        X_train = train_data[self.all_features]
        y_train = train_data[self.target]
        X_val = val_data[self.all_features]
        y_val = val_data[self.target]
        X_test = test_data[self.all_features]
        y_test = test_data[self.target]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # XGBoost parameters optimized for financial time series
        xgb_params = {
            'objective': 'binary:logistic',
            'n_estimators': 1000,
            'max_depth': 4,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'early_stopping_rounds': 50,
            'eval_metric': 'logloss',
            'scale_pos_weight': self.scale_pos_weight
        }
        
        # Train model
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=100
        )
        
        # Make predictions
        train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
        val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        train_pred = (train_pred_proba > 0.5).astype(int)
        val_pred = (val_pred_proba > 0.5).astype(int)
        test_pred = (test_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"XGBoost Results:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.all_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Store model and predictions
        self.models['xgboost'] = {
            'model': model,
            'scaler': self.scaler,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'test_pred': test_pred,
            'train_pred_proba': train_pred_proba,
            'val_pred_proba': val_pred_proba,
            'test_pred_proba': test_pred_proba,
            'train_actual': y_train,
            'val_actual': y_val,
            'test_actual': y_test,
            'feature_importance': feature_importance
        }
        
        return model, train_pred_proba, val_pred_proba, test_pred_proba
        
    def train_linear_model(self, train_data, val_data, test_data):
        """Train Logistic Regression model"""
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION MODEL")
        print("="*60)
        
        # Prepare data
        X_train = train_data[self.all_features]
        y_train = train_data[self.target]
        X_val = val_data[self.all_features]
        y_val = val_data[self.target]
        X_test = test_data[self.all_features]
        y_test = test_data[self.target]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Logistic Regression (with L2 regularization)
        model = LogisticRegression(penalty='l2', C=1.0, random_state=42, solver='liblinear', class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
        val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        train_pred = (train_pred_proba > 0.5).astype(int)
        val_pred = (val_pred_proba > 0.5).astype(int)
        test_pred = (test_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"Logistic Regression Results:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Store model and predictions
        self.models['linear'] = {
            'model': model,
            'scaler': self.scaler,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'test_pred': test_pred,
            'train_pred_proba': train_pred_proba,
            'val_pred_proba': val_pred_proba,
            'test_pred_proba': test_pred_proba,
            'train_actual': y_train,
            'val_actual': y_val,
            'test_actual': y_test
        }
        
        return model, train_pred_proba, val_pred_proba, test_pred_proba
        
    def train_meta_model(self, train_data, val_data, test_data):
        """Train meta-model using predictions from base models"""
        print("\n" + "="*60)
        print("TRAINING META-MODEL")
        print("="*60)
        
        # Get the number of LSTM predictions (smallest set due to sequence requirements)
        lstm_train_size = len(self.models['lstm']['train_pred'])
        lstm_val_size = len(self.models['lstm']['val_pred'])
        lstm_test_size = len(self.models['lstm']['test_pred'])
        
        # Align all predictions to LSTM size (trim from the end)
        xgb_train_pred = self.models['xgboost']['train_pred'][-lstm_train_size:]
        xgb_val_pred = self.models['xgboost']['val_pred'][-lstm_val_size:]
        xgb_test_pred = self.models['xgboost']['test_pred'][-lstm_test_size:]
        
        linear_train_pred = self.models['linear']['train_pred'][-lstm_train_size:]
        linear_val_pred = self.models['linear']['val_pred'][-lstm_val_size:]
        linear_test_pred = self.models['linear']['test_pred'][-lstm_test_size:]
        
        # Get predictions from base models (aligned)
        meta_features_train = np.column_stack([
            self.models['lstm']['train_pred_proba'],
            xgb_train_pred,
            linear_train_pred
        ])
        
        meta_features_val = np.column_stack([
            self.models['lstm']['val_pred_proba'],
            xgb_val_pred,
            linear_val_pred
        ])
        
        meta_features_test = np.column_stack([
            self.models['lstm']['test_pred_proba'],
            xgb_test_pred,
            linear_test_pred
        ])
        
        # Add some original features to meta-model
        additional_features = ['CP', 'Volatility_20', 'RSI', 'Interest_Rate', 'Inflation_Rate', 'Sentiment_MA_5']
        
        # Get additional features for each split (aligned with LSTM)
        train_additional = train_data[additional_features].iloc[-lstm_train_size:].values
        val_additional = val_data[additional_features].iloc[-lstm_val_size:].values
        test_additional = test_data[additional_features].iloc[-lstm_test_size:].values
        
        # Combine predictions with additional features
        meta_features_train = np.column_stack([meta_features_train, train_additional])
        meta_features_val = np.column_stack([meta_features_val, val_additional])
        meta_features_test = np.column_stack([meta_features_test, test_additional])
        
        # Get actual values (already aligned with LSTM)
        y_train = self.models['lstm']['train_actual']
        y_val = self.models['lstm']['val_actual']
        y_test = self.models['lstm']['test_actual']
        
        # Calculate class weights for meta-model to handle its own training imbalance
        meta_neg, meta_pos = np.bincount(y_train)
        if meta_pos > 0:
            meta_scale_pos_weight = meta_neg / meta_pos
        else:
            meta_scale_pos_weight = 1
        print(f"Meta-model scale_pos_weight: {meta_scale_pos_weight:.2f}")
        
        # Train meta-model (XGBoost for meta-learning)
        meta_model = XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=meta_scale_pos_weight,
            random_state=42
        )
        
        meta_model.fit(meta_features_train, y_train)
        
        # Make meta-predictions
        train_meta_pred_proba = meta_model.predict_proba(meta_features_train)[:, 1]
        val_meta_pred_proba = meta_model.predict_proba(meta_features_val)[:, 1]
        test_meta_pred_proba = meta_model.predict_proba(meta_features_test)[:, 1]
        
        train_meta_pred = (train_meta_pred_proba > 0.5).astype(int)
        val_meta_pred = (val_meta_pred_proba > 0.5).astype(int)
        test_meta_pred = (test_meta_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_meta_pred)
        val_accuracy = accuracy_score(y_val, val_meta_pred)
        test_accuracy = accuracy_score(y_test, test_meta_pred)
        
        print(f"Meta-Model Results:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Store meta-model
        self.meta_model = {
            'model': meta_model,
            'train_pred': train_meta_pred,
            'val_pred': val_meta_pred,
            'test_pred': test_meta_pred,
            'train_actual': y_train,
            'val_actual': y_val,
            'test_actual': y_test
        }
        
        return meta_model, train_meta_pred, val_meta_pred, test_meta_pred
        
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON AND EVALUATION")
        print("="*80)
        
        results = {}
        
        # Evaluate each model
        for model_name, model_data in self.models.items():
            accuracy = accuracy_score(model_data['test_actual'], model_data['test_pred'])
            precision = precision_score(model_data['test_actual'], model_data['test_pred'])
            recall = recall_score(model_data['test_actual'], model_data['test_pred'])
            f1 = f1_score(model_data['test_actual'], model_data['test_pred'])
            roc_auc = roc_auc_score(model_data['test_actual'], model_data['test_pred_proba'])
            
            results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            }
            
            print(f"{model_name.upper()} Model:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  ROC AUC:   {roc_auc:.4f}")
            print()
        
        # Evaluate meta-model
        if self.meta_model:
            accuracy = accuracy_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            precision = precision_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            recall = recall_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            f1 = f1_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            
            # For meta-model, we need its prediction probabilities for ROC AUC
            test_pred_proba_meta = self.meta_model['model'].predict_proba(
                np.column_stack([
                    self.models['lstm']['test_pred_proba'],
                    self.models['xgboost']['test_pred_proba'][-len(self.models['lstm']['test_pred_proba']):],
                    self.models['linear']['test_pred_proba'][-len(self.models['lstm']['test_pred_proba']):],
                    self.df_clean[self.all_features].iloc[-len(self.models['lstm']['test_pred_proba']):][['CP', 'Volatility_20', 'RSI', 'Interest_Rate', 'Inflation_Rate', 'Sentiment_MA_5']].values
                ])
            )[:,1]
            roc_auc = roc_auc_score(self.meta_model['test_actual'], test_pred_proba_meta)

            results['meta'] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            }
            
            print(f"META-MODEL:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  ROC AUC:   {roc_auc:.4f}")
            print()
        
        return results
        
    def plot_predictions(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot for each model
        models_to_plot = ['lstm', 'xgboost', 'linear', 'meta']
        
        for i, model_name in enumerate(models_to_plot):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if model_name == 'meta' and self.meta_model:
                actual = self.meta_model['test_actual']
                pred = self.meta_model['test_pred']
            elif model_name in self.models:
                actual = self.models[model_name]['test_actual']
                pred = self.models[model_name]['test_pred']
            else:
                continue
            
            cm = confusion_matrix(actual, pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'{model_name.upper()} Confusion Matrix')
        
        plt.tight_layout()
        plt.show()
        
    def save_models(self, directory='models'):
        """Save all trained models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save base models
        for model_name, model_data in self.models.items():
            if model_name == 'lstm':
                model_data['model'].save(f'{directory}/lstm_model.h5')
            else:
                joblib.dump(model_data['model'], f'{directory}/{model_name}_model.pkl')
            joblib.dump(model_data['scaler'], f'{directory}/{model_name}_scaler.pkl')
            
        # Save meta-model
        if self.meta_model:
            joblib.dump(self.meta_model['model'], f'{directory}/meta_model.pkl')
            
        print(f"Models saved to {directory}/ directory")
        
    def train_all_models(self):
        """Train all models in sequence"""
        print("="*80)
        print("TRAINING COMPLETE S&P 500 PREDICTION PIPELINE")
        print("="*80)
        
        # Prepare data
        df_clean = self.prepare_features()
        
        # Split data
        train_data, val_data, test_data = self.split_data_time_series(df_clean)
        
        # Calculate class weights from the training data to handle imbalance
        print("\n" + "="*60)
        print("CLASS WEIGHTS FOR IMBALANCE HANDLING")
        print("="*60)
        neg, pos = np.bincount(train_data[self.target])
        if pos > 0 and neg > 0:
            self.scale_pos_weight = neg / pos
            total = neg + pos
            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)
            self.class_weights = {0: weight_for_0, 1: weight_for_1}
            print(f"Positive samples (Up): {pos}, Negative samples (Down): {neg}")
            print(f"XGBoost scale_pos_weight: {self.scale_pos_weight:.2f}")
            print(f"LSTM class_weights: {{0: {self.class_weights[0]:.2f}, 1: {self.class_weights[1]:.2f}}}")
        else:
            print("Cannot calculate class weights: one class is missing in the training data.")
            self.scale_pos_weight = 1
            self.class_weights = None
        
        # Train models
        self.train_lstm_model(train_data, val_data, test_data)
        self.train_xgboost_model(train_data, val_data, test_data)
        self.train_linear_model(train_data, val_data, test_data)
        self.train_meta_model(train_data, val_data, test_data)
        
        # Evaluate models
        results = self.evaluate_models()
        
        # Plot results
        self.plot_predictions()
        
        # Save models
        self.save_models()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print("Models are ready for S&P 500 price prediction.")
        print("Use the saved models to make predictions on new data.")
        
        return results

def main():
    """Main function to run the complete training pipeline"""
    # Initialize predictor
    predictor = SP500Predictor('sp500_features_for_prediction.csv')
    
    # Train all models
    results = predictor.train_all_models()
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()
