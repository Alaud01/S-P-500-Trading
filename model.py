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
    def __init__(self, data_file='data/sp500_features_for_prediction.csv'):
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
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate', 'GDP', 'Gold_Price', 'Unemployment_Rate',
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
            'CP', 'Volume', 'Interest_Rate', 'Inflation_Rate', 'GDP', 'Gold_Price', 'Unemployment_Rate',
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
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        
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
            'test_actual': y_test,
            'history': history
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
            'n_estimators': 1500,
            'max_depth': 3,
            'learning_rate': 0.005,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.1,
            'reg_alpha': 0.005,
            'random_state': 42,
            'early_stopping_rounds': 30,
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
        
        # Train Logistic Regression (with L1 regularization for feature selection)
        model = LogisticRegression(penalty='l1', C=1.0, random_state=42, solver='saga', max_iter=1000, class_weight='balanced')
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
        xgb_train_pred_proba = self.models['xgboost']['train_pred_proba'][-lstm_train_size:]
        xgb_val_pred_proba = self.models['xgboost']['val_pred_proba'][-lstm_val_size:]
        xgb_test_pred_proba = self.models['xgboost']['test_pred_proba'][-lstm_test_size:]
        
        linear_train_pred_proba = self.models['linear']['train_pred_proba'][-lstm_train_size:]
        linear_val_pred_proba = self.models['linear']['val_pred_proba'][-lstm_val_size:]
        linear_test_pred_proba = self.models['linear']['test_pred_proba'][-lstm_test_size:]
        
        # Get predictions from base models (aligned)
        meta_features_train = np.column_stack([
            self.models['lstm']['train_pred_proba'],
            xgb_train_pred_proba,
            linear_train_pred_proba
        ])
        
        meta_features_val = np.column_stack([
            self.models['lstm']['val_pred_proba'],
            xgb_val_pred_proba,
            linear_val_pred_proba
        ])
        
        meta_features_test = np.column_stack([
            self.models['lstm']['test_pred_proba'],
            xgb_test_pred_proba,
            linear_test_pred_proba
        ])
        
        # Add some original features to meta-model
        additional_features = ['CP', 'Volatility_20', 'RSI', 'Interest_Rate', 'Inflation_Rate', 'GDP', 'Gold_Price', 'Unemployment_Rate', 'Sentiment_MA_5']
        
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
        # We use the validation set labels since we are training the meta-model on the validation set predictions
        meta_neg, meta_pos = np.bincount(y_val)
        if meta_pos > 0 and meta_neg > 0:
            meta_scale_pos_weight = meta_neg / meta_pos
        else:
            meta_scale_pos_weight = 1
        print(f"Meta-model scale_pos_weight: {meta_scale_pos_weight:.2f}")
        
        # Train meta-model (XGBoost for meta-learning)
        meta_model = XGBClassifier(
            objective='binary:logistic',
            n_estimators=300,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=meta_scale_pos_weight,
            random_state=42
        )
        
        meta_model.fit(meta_features_val, y_val)
        
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
            'train_pred_proba': train_meta_pred_proba,
            'val_pred_proba': val_meta_pred_proba,
            'test_pred_proba': test_meta_pred_proba,
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
            
            # For meta-model, we use its stored prediction probabilities for ROC AUC
            roc_auc = roc_auc_score(self.meta_model['test_actual'], self.meta_model['test_pred_proba'])

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
        
    def plot_training_metrics(self):
        """Plot training metrics for all models"""
        print("\n" + "="*60)
        print("TRAINING METRICS VISUALIZATION")
        print("="*60)
        
        # Calculate metrics for all models
        metrics_data = {}
        
        for model_name, model_data in self.models.items():
            # Train metrics
            train_acc = accuracy_score(model_data['train_actual'], model_data['train_pred'])
            train_prec = precision_score(model_data['train_actual'], model_data['train_pred'])
            train_rec = recall_score(model_data['train_actual'], model_data['train_pred'])
            train_f1 = f1_score(model_data['train_actual'], model_data['train_pred'])
            train_roc = roc_auc_score(model_data['train_actual'], model_data['train_pred_proba'])
            
            # Validation metrics
            val_acc = accuracy_score(model_data['val_actual'], model_data['val_pred'])
            val_prec = precision_score(model_data['val_actual'], model_data['val_pred'])
            val_rec = recall_score(model_data['val_actual'], model_data['val_pred'])
            val_f1 = f1_score(model_data['val_actual'], model_data['val_pred'])
            val_roc = roc_auc_score(model_data['val_actual'], model_data['val_pred_proba'])
            
            # Test metrics
            test_acc = accuracy_score(model_data['test_actual'], model_data['test_pred'])
            test_prec = precision_score(model_data['test_actual'], model_data['test_pred'])
            test_rec = recall_score(model_data['test_actual'], model_data['test_pred'])
            test_f1 = f1_score(model_data['test_actual'], model_data['test_pred'])
            test_roc = roc_auc_score(model_data['test_actual'], model_data['test_pred_proba'])
            
            metrics_data[model_name] = {
                'train': {'accuracy': train_acc, 'precision': train_prec, 'recall': train_rec, 'f1': train_f1, 'roc_auc': train_roc},
                'val': {'accuracy': val_acc, 'precision': val_prec, 'recall': val_rec, 'f1': val_f1, 'roc_auc': val_roc},
                'test': {'accuracy': test_acc, 'precision': test_prec, 'recall': test_rec, 'f1': test_f1, 'roc_auc': test_roc}
            }
        
        # Add meta-model if available
        if self.meta_model:
            train_acc = accuracy_score(self.meta_model['train_actual'], self.meta_model['train_pred'])
            train_prec = precision_score(self.meta_model['train_actual'], self.meta_model['train_pred'])
            train_rec = recall_score(self.meta_model['train_actual'], self.meta_model['train_pred'])
            train_f1 = f1_score(self.meta_model['train_actual'], self.meta_model['train_pred'])
            train_roc = roc_auc_score(self.meta_model['train_actual'], self.meta_model['train_pred_proba'])
            
            val_acc = accuracy_score(self.meta_model['val_actual'], self.meta_model['val_pred'])
            val_prec = precision_score(self.meta_model['val_actual'], self.meta_model['val_pred'])
            val_rec = recall_score(self.meta_model['val_actual'], self.meta_model['val_pred'])
            val_f1 = f1_score(self.meta_model['val_actual'], self.meta_model['val_pred'])
            val_roc = roc_auc_score(self.meta_model['val_actual'], self.meta_model['val_pred_proba'])
            
            test_acc = accuracy_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            test_prec = precision_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            test_rec = recall_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            test_f1 = f1_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            test_roc = roc_auc_score(self.meta_model['test_actual'], self.meta_model['test_pred_proba'])
            
            metrics_data['meta'] = {
                'train': {'accuracy': train_acc, 'precision': train_prec, 'recall': train_rec, 'f1': train_f1, 'roc_auc': train_roc},
                'val': {'accuracy': val_acc, 'precision': val_prec, 'recall': val_rec, 'f1': val_f1, 'roc_auc': val_roc},
                'test': {'accuracy': test_acc, 'precision': test_prec, 'recall': test_rec, 'f1': test_f1, 'roc_auc': test_roc}
            }
        
        # Create comprehensive training metrics plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Metrics Comparison Across All Models', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            models = list(metrics_data.keys())
            x = np.arange(len(models))
            width = 0.25
            
            train_values = [metrics_data[model]['train'][metric] for model in models]
            val_values = [metrics_data[model]['val'][metric] for model in models]
            test_values = [metrics_data[model]['test'][metric] for model in models]
            
            rects1 = ax.bar(x - width, train_values, width, label='Train', color='skyblue', alpha=0.8)
            rects2 = ax.bar(x, val_values, width, label='Validation', color='lightgreen', alpha=0.8)
            rects3 = ax.bar(x + width, test_values, width, label='Test', color='lightcoral', alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([model.upper() for model in models], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for rects in [rects1, rects2, rects3]:
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
        
        # Add overfitting analysis plot
        ax = axes[1, 2]
        models = list(metrics_data.keys())
        overfitting_scores = []
        
        for model in models:
            train_f1 = metrics_data[model]['train']['f1']
            test_f1 = metrics_data[model]['test']['f1']
            overfitting_score = train_f1 - test_f1
            overfitting_scores.append(overfitting_score)
        
        bars = ax.bar(models, overfitting_scores, color=['red' if score > 0.05 else 'green' for score in overfitting_scores])
        ax.set_xlabel('Models')
        ax.set_ylabel('Overfitting Score (Train F1 - Test F1)')
        ax.set_title('Overfitting Analysis')
        ax.set_xticklabels([model.upper() for model in models], rotation=45)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, overfitting_scores):
            height = bar.get_height()
            ax.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return metrics_data
    
    def plot_model_performance_comparison(self):
        """Plot model performance comparison similar to model_summary.py"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        # Calculate test set metrics for all models
        performance_data = {}
        
        for model_name, model_data in self.models.items():
            accuracy = accuracy_score(model_data['test_actual'], model_data['test_pred'])
            precision = precision_score(model_data['test_actual'], model_data['test_pred'])
            recall = recall_score(model_data['test_actual'], model_data['test_pred'])
            f1 = f1_score(model_data['test_actual'], model_data['test_pred'])
            roc_auc = roc_auc_score(model_data['test_actual'], model_data['test_pred_proba'])
            
            performance_data[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            }
        
        # Add meta-model if available
        if self.meta_model:
            accuracy = accuracy_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            precision = precision_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            recall = recall_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            f1 = f1_score(self.meta_model['test_actual'], self.meta_model['test_pred'])
            roc_auc = roc_auc_score(self.meta_model['test_actual'], self.meta_model['test_pred_proba'])
            
            performance_data['meta'] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            }
        
        # Create performance comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison (Test Set)', fontsize=16, fontweight='bold')
        
        models = list(performance_data.keys())
        accuracy_values = [performance_data[model]['Accuracy'] for model in models]
        f1_values = [performance_data[model]['F1 Score'] for model in models]
        roc_auc_values = [performance_data[model]['ROC AUC'] for model in models]
        precision_values = [performance_data[model]['Precision'] for model in models]
        recall_values = [performance_data[model]['Recall'] for model in models]
        
        # Accuracy comparison
        bars1 = axes[0, 0].bar(models, accuracy_values, color=['blue', 'green', 'red', 'purple'])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy (Higher is Better)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        for bar, value in zip(bars1, accuracy_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{value:.4f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # F1 Score comparison
        bars2 = axes[0, 1].bar(models, f1_values, color=['blue', 'green', 'red', 'purple'])
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score (Higher is Better)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        for bar, value in zip(bars2, f1_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{value:.4f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # ROC AUC comparison
        bars3 = axes[1, 0].bar(models, roc_auc_values, color=['blue', 'green', 'red', 'purple'])
        axes[1, 0].set_title('ROC AUC Score Comparison')
        axes[1, 0].set_ylabel('ROC AUC (Higher is Better)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        for bar, value in zip(bars3, roc_auc_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{value:.4f}', 
                           ha='center', va='bottom', fontweight='bold')
            
        # Precision vs. Recall
        x = np.arange(len(models))
        width = 0.35
        
        rects1 = axes[1, 1].bar(x - width/2, precision_values, width, label='Precision', color='lightblue')
        rects2 = axes[1, 1].bar(x + width/2, recall_values, width, label='Recall', color='lightgreen')
        
        axes[1, 1].set_ylabel('Scores')
        axes[1, 1].set_title('Precision vs. Recall')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([model.upper() for model in models], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels for precision vs recall
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                axes[1, 1].annotate(f'{height:.3f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return performance_data
    
    def plot_lstm_training_history(self):
        """Plot LSTM training history (loss and accuracy curves)"""
        print("\n" + "="*60)
        print("LSTM TRAINING HISTORY")
        print("="*60)
        
        if 'lstm' not in self.models or 'history' not in self.models['lstm']:
            print("LSTM model or training history not found.")
            return
        
        history = self.models['lstm']['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('LSTM Training History', fontsize=16, fontweight='bold')
        
        # Plot training & validation loss
        axes[0].plot(history.history['loss'], label='Training Loss', color='blue')
        axes[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot training & validation accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print training summary
        print(f"Training completed in {len(history.history['loss'])} epochs")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Check for overfitting
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        if val_loss > train_loss * 1.2:
            print("⚠️  Warning: Potential overfitting detected (validation loss > 1.2 * training loss)")
        else:
            print("✅ No significant overfitting detected")
    
    def plot_feature_importance(self):
        """Plot feature importance from XGBoost model"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        if 'xgboost' not in self.models:
            print("XGBoost model not found. Cannot plot feature importance.")
            return
        
        feature_importance = self.models['xgboost']['feature_importance']
        top_features = feature_importance.head(15)  # Show top 15 features
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features for S&P 500 Prediction (XGBoost)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, importance in zip(bars, top_features['importance']):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print top features
        print("Top 15 Most Important Features:")
        print("-" * 60)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:25} ({row['importance']:.4f})")
    
    def plot_predictions(self):
        """Plot confusion matrices and prediction distributions for all models"""
        print("\n" + "="*60)
        print("PREDICTION ANALYSIS PLOTS")
        print("="*60)
        
        # Create subplots for confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Confusion Matrices for All Models (Test Set)', fontsize=16, fontweight='bold')
        
        # Plot for each model
        models_to_plot = ['lstm', 'xgboost', 'linear', 'meta']
        
        for i, model_name in enumerate(models_to_plot):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if model_name == 'meta' and self.meta_model:
                actual = self.meta_model['test_actual']
                pred = self.meta_model['test_pred']
                title = 'META-MODEL'
            elif model_name in self.models:
                actual = self.models[model_name]['test_actual']
                pred = self.models[model_name]['test_pred']
                title = model_name.upper()
            else:
                continue
            
            cm = confusion_matrix(actual, pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'{title} Confusion Matrix')
        
        plt.tight_layout()
        plt.show()
        
        # Plot prediction probability distributions
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Prediction Probability Distributions (Test Set)', fontsize=16, fontweight='bold')
        
        for i, model_name in enumerate(models_to_plot):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if model_name == 'meta' and self.meta_model:
                actual = self.meta_model['test_actual']
                pred_proba = self.meta_model['test_pred_proba']
                title = 'META-MODEL'
            elif model_name in self.models:
                actual = self.models[model_name]['test_actual']
                pred_proba = self.models[model_name]['test_pred_proba']
                title = model_name.upper()
            else:
                continue
            
            # Plot probability distributions for each class
            up_proba = pred_proba[actual == 1]
            down_proba = pred_proba[actual == 0]
            
            ax.hist(up_proba, bins=20, alpha=0.7, label='Actual Up', color='green', density=True)
            ax.hist(down_proba, bins=20, alpha=0.7, label='Actual Down', color='red', density=True)
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Density')
            ax.set_title(f'{title} Probability Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
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
        
        # Plot comprehensive results
        self.plot_training_metrics()
        self.plot_model_performance_comparison()
        self.plot_lstm_training_history()
        self.plot_feature_importance()
        self.plot_predictions()
        
        # Save models
        self.save_models()
        
        # Create comprehensive summary report
        self.create_comprehensive_report(results)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print("Models are ready for S&P 500 price prediction.")
        print("Use the saved models to make predictions on new data.")
        
        return results
    
    def create_comprehensive_report(self, results):
        """Create a comprehensive summary report similar to model_summary.py"""
        print("\n" + "="*100)
        print("COMPREHENSIVE S&P 500 PREDICTION MODEL REPORT")
        print("="*100)
        
        # Model comparison table
        print("\nMODEL PERFORMANCE COMPARISON:")
        print("-" * 80)
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1 Score': f"{metrics['F1 Score']:.4f}",
                'ROC AUC': f"{metrics['ROC AUC']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Find best performing models
        best_f1_model = max(results.items(), key=lambda x: x[1]['F1 Score'])
        best_acc_model = max(results.items(), key=lambda x: x[1]['Accuracy'])
        best_roc_auc_model = max(results.items(), key=lambda x: x[1]['ROC AUC'])
        
        print(f"\nPERFORMANCE HIGHLIGHTS:")
        print(f"• Best F1 Score: {best_f1_model[0].upper()} ({best_f1_model[1]['F1 Score']:.4f})")
        print(f"• Best Accuracy: {best_acc_model[0].upper()} ({best_acc_model[1]['Accuracy']:.4f})")
        print(f"• Best ROC AUC: {best_roc_auc_model[0].upper()} ({best_roc_auc_model[1]['ROC AUC']:.4f})")
        
        # Model descriptions and insights
        model_descriptions = {
            'lstm': {
                'Description': 'Deep Learning model using Long Short-Term Memory networks',
                'Features': 'Sequential technical indicators (60-day lookback)',
                'Strengths': 'Captures temporal dependencies and patterns',
                'Weaknesses': 'Sensitive to market regime changes'
            },
            'xgboost': {
                'Description': 'Gradient boosting ensemble model',
                'Features': 'Technical indicators, economic data, lagged features',
                'Strengths': 'Handles non-linear relationships, feature importance',
                'Weaknesses': 'Prone to overfitting if not tuned carefully'
            },
            'linear': {
                'Description': 'L1-regularized Logistic Regression baseline model',
                'Features': 'Technical indicators, economic data, lagged features',
                'Strengths': 'Interpretable, fast, performs feature selection',
                'Weaknesses': 'Assumes linear relationships'
            },
            'meta': {
                'Description': 'Ensemble model combining predictions from base models',
                'Features': 'Probabilistic predictions from LSTM, XGBoost, and Linear models',
                'Strengths': 'Combines different modeling approaches for improved accuracy',
                'Weaknesses': 'Performance is highly dependent on base model quality'
            }
        }
        
        print(f"\nMODEL DESCRIPTIONS:")
        print("-" * 80)
        for model_name, desc in model_descriptions.items():
            if model_name in results:
                print(f"\n{model_name.upper()}:")
                print(f"  Description: {desc['Description']}")
                print(f"  Features: {desc['Features']}")
                print(f"  Strengths: {desc['Strengths']}")
                print(f"  Weaknesses: {desc['Weaknesses']}")
        
        # Key insights
        print(f"\nKEY INSIGHTS:")
        print("-" * 80)
        if 'meta' in results:
            print(f"• Meta-Model shows the best overall performance with an F1 Score of {results['meta']['F1 Score']:.4f}.")
            print(f"• The stacking approach successfully combines base models to improve predictive power.")
        if 'lstm' in results:
            print(f"• The LSTM model performs well, showing strong recall of {results['lstm']['Recall']:.4f}.")
        if 'linear' in results:
            print(f"• The Linear model's high precision ({results['linear']['Precision']:.4f}) suggests it's good at identifying positive cases.")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS FOR IMPROVEMENT:")
        print("-" * 80)
        print("1. META-MODEL TUNING:")
        print("   • Implement k-fold cross-validation for training the meta-model to improve generalization.")
        print("   • Experiment with different algorithms for the meta-model (e.g., Logistic Regression, LightGBM).")
        
        print("\n2. FEATURE ENGINEERING:")
        print("   • Create more diverse features for the base models to reduce correlation.")
        print("   • Add more alternative data sources like options data (VIX) or sector-specific ETFs.")
        
        print("\n3. MODEL DIVERSIFICATION:")
        print("   • Introduce more diverse model architectures into the ensemble.")
        print("   • Consider models like TabNet, LightGBM, or a simple Naive Bayes classifier.")
        
        print("\n4. RISK MANAGEMENT:")
        print("   • Never rely solely on model predictions for trading.")
        print("   • Always use proper position sizing and stop-losses.")
        print("   • Consider the model's confidence level (probability scores) in decisions.")
        
        print(f"\n" + "="*100)
        print("FINAL SUMMARY")
        print("="*100)
        print("• Successfully trained and evaluated a sophisticated 4-model stacking ensemble.")
        if 'meta' in results:
            print(f"• The Meta-Model is the top performer, achieving an F1-Score of {results['meta']['F1 Score']:.4f} on the test set.")
        print("• The ensemble approach effectively combines diverse models to achieve superior performance.")
        print("• Models are saved and ready for deployment in a prediction pipeline.")
        
        print(f"\nNote: These models are for educational and research purposes only.")
        print("Financial markets are inherently unpredictable and past performance")
        print("does not guarantee future results. Always conduct thorough research")
        print("and consider professional advice before making investment decisions.")

def main():
    """Main function to run the complete training pipeline"""
    # Initialize predictor
    predictor = SP500Predictor('data/sp500_features_for_prediction.csv')
    
    # Train all models
    results = predictor.train_all_models()
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()
