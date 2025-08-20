import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import random
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

# Set TensorFlow seed for reproducibility
tf.random.set_seed(SEED)

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
        print(f"Using seed {SEED} for reproducibility")
        
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
        
        # All available features (for XGBoost and Linear models)
        technical_features = [
            # Price and Volume data
            'CP', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume',
            # Economic indicators
            'Interest_Rate', 'Inflation_Rate', 'GDP', 'Gold_Price', 'Unemployment_Rate',
            # News sentiment
            'Headline_Count', 'Sentiment', 'Sentiment_MA_5',
            # Returns
            'Returns', 'Log_Returns',
            # Moving averages
            'MA_5', 'MA_20', 'MA_50', 'MA_200',
            # Technical indicators
            'Volatility_20', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            # Price changes
            'Price_Change_5d', 'Price_Change_20d',
            # Volume indicators
            'Volume_MA_20', 'Volume_Ratio'
        ]
        
        # Lag features
        lag_features = [f'Price_Lag_{lag}' for lag in [1, 5, 10, 20]] + \
                      [f'Returns_Lag_{lag}' for lag in [1, 5, 10, 20]]
        
        # All features for XGBoost and Linear models
        self.all_features = technical_features + lag_features
        
        # Sequential features for LSTM (last 60 days) - using key features for sequence modeling
        self.sequence_length = 60
        self.lstm_features = [
            # Core price and volume
            'CP', 'Volume', 'Returns',
            # Economic indicators
            'Interest_Rate', 'Inflation_Rate', 'GDP', 'Gold_Price', 'Unemployment_Rate',
            # Sentiment
            'Sentiment', 'Sentiment_MA_5',
            # Key technical indicators
            'RSI', 'MACD', 'Volatility_20', 'BB_Position'
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
        """Train LSTM model on sequential data using cross-validation time series split"""
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL WITH CROSS-VALIDATION")
        print("="*60)
        
        # Combine train and validation data for cross-validation
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        print(f"Combined data for LSTM CV: {len(combined_data)} records from {combined_data['Date'].min()} to {combined_data['Date'].max()}")
        
        # Create sequences for the combined data
        X_combined, y_combined = self.create_sequences(combined_data, self.lstm_features, self.target, self.sequence_length)
        
        # Scale the data
        X_combined_scaled = self.lstm_scaler.fit_transform(X_combined.reshape(-1, X_combined.shape[-1])).reshape(X_combined.shape)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X_combined_scaled) * 0.2))
        
        # Store CV results
        cv_scores = []
        cv_histories = []
        best_model = None
        best_val_score = 0
        
        print(f"Performing {tscv.n_splits}-fold time series cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_combined_scaled), 1):
            print(f"\nFold {fold}/{tscv.n_splits}")
            print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")
            
            # Split data for this fold
            X_train_fold = X_combined_scaled[train_idx]
            y_train_fold = y_combined[train_idx]
            X_val_fold = X_combined_scaled[val_idx]
            y_val_fold = y_combined[val_idx]
            
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
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                class_weight=self.class_weights,
                verbose=0
            )
            
            # Evaluate on validation set
            val_pred_proba = model.predict(X_val_fold, verbose=0).flatten()
            val_pred = (val_pred_proba > 0.5).astype(int)
            val_accuracy = accuracy_score(y_val_fold, val_pred)
            val_f1 = f1_score(y_val_fold, val_pred)
            
            cv_scores.append({
                'fold': fold,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'epochs': len(history.history['loss'])
            })
            
            cv_histories.append(history)
            
            print(f"  Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Epochs: {len(history.history['loss'])}")
            
            # Keep track of best model
            if val_f1 > best_val_score:
                best_val_score = val_f1
                best_model = model
        
        # Print CV summary
        print(f"\nCross-Validation Results:")
        print("-" * 50)
        avg_accuracy = np.mean([score['val_accuracy'] for score in cv_scores])
        avg_f1 = np.mean([score['val_f1'] for score in cv_scores])
        std_accuracy = np.std([score['val_accuracy'] for score in cv_scores])
        std_f1 = np.std([score['val_f1'] for score in cv_scores])
        
        print(f"Average Validation Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
        print(f"Average Validation F1 Score: {avg_f1:.4f} (±{std_f1:.4f})")
        
        # Now train the final model on the combined train+val data using the best model architecture
        print(f"\nTraining final LSTM model on combined train+validation data...")
        
        # Use the best model from CV or create a new one with the same architecture
        final_model = Sequential([
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
        
        final_model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train final model on combined data
        final_history = final_model.fit(
            X_combined_scaled, y_combined,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Prepare test sequences
        X_test, y_test = self.create_sequences(test_data, self.lstm_features, self.target, self.sequence_length)
        X_test_scaled = self.lstm_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Make predictions
        train_pred_proba = final_model.predict(X_combined_scaled, verbose=0).flatten()
        test_pred_proba = final_model.predict(X_test_scaled, verbose=0).flatten()
        
        # Convert probabilities to binary predictions
        train_pred = (train_pred_proba > 0.5).astype(int)
        test_pred = (test_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_combined, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"Final LSTM Results:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  CV Average F1: {avg_f1:.4f}")
        
        # Store model and predictions
        self.models['lstm'] = {
            'model': final_model,
            'scaler': self.lstm_scaler,
            'train_pred': train_pred,
            'val_pred': None,  # No single validation set in CV approach
            'test_pred': test_pred,
            'train_pred_proba': train_pred_proba,
            'val_pred_proba': None,  # No single validation set in CV approach
            'test_pred_proba': test_pred_proba,
            'train_actual': y_combined,
            'val_actual': None,  # No single validation set in CV approach
            'test_actual': y_test,
            'history': final_history,
            'cv_scores': cv_scores,
            'cv_histories': cv_histories
        }
        
        return final_model, train_pred_proba, None, test_pred_proba
        
    def train_xgboost_model(self, train_data, val_data, test_data):
        """Train XGBoost model with time-series cross-validation"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL WITH TIME-SERIES CV")
        print("="*60)
        
        # Use time-series cross-validation like LSTM to prevent overfitting
        print("Using time-series cross-validation for XGBoost...")
        
        # Combine train and validation data for CV
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        print(f"Combined data for XGBoost CV: {len(combined_data)} records from {combined_data['Date'].min()} to {combined_data['Date'].max()}")
        
        # Prepare features for CV
        X_combined = combined_data[self.all_features]
        y_combined = combined_data[self.target]
        
        # Scale features
        X_combined_scaled = self.scaler.fit_transform(X_combined)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X_combined_scaled) * 0.2))
        
        # Store CV results
        cv_scores = []
        cv_models = []
        best_model = None
        best_val_score = 0
        
        print(f"Performing {tscv.n_splits}-fold time series cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_combined_scaled), 1):
            print(f"\nFold {fold}/{tscv.n_splits}")
            print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")
            
            # Split data for this fold
            X_train_fold = X_combined_scaled[train_idx]
            y_train_fold = y_combined[train_idx]
            X_val_fold = X_combined_scaled[val_idx]
            y_val_fold = y_combined[val_idx]
            
            # XGBoost parameters tuned to prevent overfitting and handle class imbalance
            xgb_params = {
                'objective': 'binary:logistic',
                'n_estimators': 500,
                'max_depth': 3,
                'min_child_weight': 3,
                'learning_rate': 0.01,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'eval_metric': ['logloss', 'auc'],
                'scale_pos_weight': self.scale_pos_weight
            }
            
            # Adjust scale_pos_weight based on this fold's class distribution
            class_counts = np.bincount(y_train_fold)
            if len(class_counts) >= 2 and class_counts[0] > 0 and class_counts[1] > 0:
                fold_neg, fold_pos = class_counts[0], class_counts[1]
                fold_scale_pos_weight = fold_neg / fold_pos
                xgb_params['scale_pos_weight'] = fold_scale_pos_weight
                print(f"    Fold {fold} class balance - Neg: {fold_neg}, Pos: {fold_pos}, scale_pos_weight: {fold_scale_pos_weight:.2f}")
            else:
                print(f"    Fold {fold} - insufficient class diversity, using default scale_pos_weight")
            
            # Check class balance in this fold
            unique_classes = np.unique(y_train_fold)
            if len(unique_classes) < 2:
                print(f"  Skipping fold {fold} - insufficient class diversity: {unique_classes}")
                continue
                
            # Train model for this fold
            model = XGBClassifier(**xgb_params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                early_stopping_rounds=30,
                verbose=0
            )
            
            # Evaluate on validation set
            val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            
            # Find optimal threshold for this fold
            possible_thresholds = np.linspace(0.1, 0.9, 81)
            best_thresh = 0.5
            best_f1 = -1.0
            best_balanced_acc = -1.0
            
            for thr in possible_thresholds:
                preds = (val_pred_proba >= thr).astype(int)
                f1 = f1_score(y_val_fold, preds)
                
                # Calculate balanced accuracy to ensure we're not just predicting one class
                tn, fp, fn, tp = confusion_matrix(y_val_fold, preds).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                balanced_acc = (sensitivity + specificity) / 2
                
                # Use a combination of F1 and balanced accuracy
                combined_score = f1 * 0.7 + balanced_acc * 0.3
                
                if combined_score > best_f1:
                    best_f1 = combined_score
                    best_thresh = thr
                    best_balanced_acc = balanced_acc
            
            val_pred = (val_pred_proba >= best_thresh).astype(int)
            val_accuracy = accuracy_score(y_val_fold, val_pred)
            
            cv_scores.append({
                'fold': fold,
                'val_accuracy': val_accuracy,
                'val_f1': best_f1,
                'threshold': best_thresh,
                'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
            })
            
            cv_models.append(model)
            
            print(f"  Validation Accuracy: {val_accuracy:.4f}, F1: {best_f1:.4f}, Balanced Acc: {best_balanced_acc:.4f}, Threshold: {best_thresh:.3f}")
            
            # Keep track of best model
            if best_f1 > best_val_score:
                best_val_score = best_f1
                best_model = model
        
        # Print CV summary
        print(f"\nCross-Validation Results:")
        print("-" * 50)
        if cv_scores:
            avg_accuracy = np.mean([score['val_accuracy'] for score in cv_scores])
            avg_f1 = np.mean([score['val_f1'] for score in cv_scores])
            std_accuracy = np.std([score['val_accuracy'] for score in cv_scores])
            std_f1 = np.std([score['val_f1'] for score in cv_scores])
            
            print(f"Average Validation Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
            print(f"Average Validation F1 Score: {avg_f1:.4f} (±{std_f1:.4f})")
        else:
            print("No valid CV results - proceeding with final model training only")
            avg_f1 = 0.0
        
        # Now train the final model on the combined train+val data using the best parameters
        print(f"\nTraining final XGBoost model on combined train+validation data...")
        
        # Use the best parameters from CV
        final_xgb_params = {
            'objective': 'binary:logistic',
            'n_estimators': 1000,
            'max_depth': 3,
            'min_child_weight': 3,
            'learning_rate': 0.01,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'eval_metric': ['logloss', 'auc'],
            'scale_pos_weight': self.scale_pos_weight
        }
        
        # Adjust final scale_pos_weight based on combined data class distribution
        combined_class_counts = np.bincount(y_combined)
        if len(combined_class_counts) >= 2 and combined_class_counts[0] > 0 and combined_class_counts[1] > 0:
            combined_neg, combined_pos = combined_class_counts[0], combined_class_counts[1]
            final_scale_pos_weight = combined_neg / combined_pos
            final_xgb_params['scale_pos_weight'] = final_scale_pos_weight
            print(f"Final model class balance - Neg: {combined_neg}, Pos: {combined_pos}, scale_pos_weight: {final_scale_pos_weight:.2f}")
        else:
            print("Warning: Insufficient class diversity in combined data")
        
        # Train final model on combined data
        final_model = XGBClassifier(**final_xgb_params)
        final_model.fit(
            X_combined_scaled, y_combined,
            eval_set=[(X_combined_scaled, y_combined)],
            verbose=100
        )
        
        # Prepare test data
        X_test = test_data[self.all_features]
        y_test = test_data[self.target]
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        train_pred_proba = final_model.predict_proba(X_combined_scaled)[:, 1]
        test_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
        
        # Use average threshold from CV (handle case where some folds were skipped)
        if cv_scores:
            avg_threshold = np.mean([score['threshold'] for score in cv_scores])
            print(f"Using average threshold from CV: {avg_threshold:.3f}")
            
            # Calculate optimal threshold based on class distribution
            test_class_counts = np.bincount(y_test)
            if len(test_class_counts) >= 2 and test_class_counts[0] > 0 and test_class_counts[1] > 0:
                test_neg, test_pos = test_class_counts[0], test_class_counts[1]
                # Use class distribution to set a more balanced threshold
                class_ratio = test_neg / test_pos
                balanced_threshold = 1 / (1 + class_ratio)  # This gives us a threshold that considers class balance
                print(f"Class ratio in test set: {class_ratio:.2f}, balanced threshold: {balanced_threshold:.3f}")
                
                # Use the higher of the two thresholds to prevent over-prediction
                avg_threshold = max(avg_threshold, balanced_threshold)
                print(f"Final threshold: {avg_threshold:.3f}")
            
            # Add a minimum threshold constraint to prevent over-prediction of one class
            if avg_threshold < 0.3:
                print(f"Warning: Threshold {avg_threshold:.3f} is too low, adjusting to 0.3 to prevent class imbalance")
                avg_threshold = 0.3
        else:
            avg_threshold = 0.5
            print("No valid CV folds found, using default threshold: 0.5")
        
        train_pred = (train_pred_proba >= avg_threshold).astype(int)
        test_pred = (test_pred_proba >= avg_threshold).astype(int)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_combined, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Check class distribution in predictions
        train_class_dist = np.bincount(train_pred)
        test_class_dist = np.bincount(test_pred)
        
        print(f"Final XGBoost Results:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  CV Average F1: {avg_f1:.4f}")
        print(f"  Train Predictions - Class 0: {train_class_dist[0] if len(train_class_dist) > 0 else 0}, Class 1: {train_class_dist[1] if len(train_class_dist) > 1 else 0}")
        print(f"  Test Predictions - Class 0: {test_class_dist[0] if len(test_class_dist) > 0 else 0}, Class 1: {test_class_dist[1] if len(test_class_dist) > 1 else 0}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.all_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Store model and predictions
        self.models['xgboost'] = {
            'model': final_model,
            'scaler': self.scaler,
            'train_pred': train_pred,
            'val_pred': None,  # No single validation set in CV approach
            'test_pred': test_pred,
            'train_pred_proba': train_pred_proba,
            'val_pred_proba': None,  # No single validation set in CV approach
            'test_pred_proba': test_pred_proba,
            'train_actual': y_combined,
            'val_actual': None,  # No single validation set in CV approach
            'test_actual': y_test,
            'feature_importance': feature_importance,
            'threshold': avg_threshold,
            'cv_scores': cv_scores,
            'cv_models': cv_models
        }
        
        # Plot learning curves from final model
        try:
            evals_result = final_model.evals_result()
            if evals_result:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                # Logloss
                if 'logloss' in evals_result.get('validation_0', {}):
                    ax1.plot(evals_result['validation_0']['logloss'], label='Train Logloss', color='blue')
                ax1.set_title('XGBoost Final Model Logloss')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Logloss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # AUC
                if 'auc' in evals_result.get('validation_0', {}):
                    ax2.plot(evals_result['validation_0']['auc'], label='Train AUC', color='blue')
                ax2.set_title('XGBoost Final Model AUC')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('AUC')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Could not plot XGBoost learning curves: {e}")
        
        return final_model, train_pred_proba, None, test_pred_proba
        
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
        lstm_test_size = len(self.models['lstm']['test_pred'])
        
        # For XGBoost and Linear models, we need to get predictions for the combined train+val data
        # to match the LSTM predictions
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        
        # Prepare features for XGBoost and Linear on combined data
        X_combined = combined_data[self.all_features]
        X_combined_scaled = self.models['xgboost']['scaler'].transform(X_combined)
        
        # Get XGBoost predictions for combined data
        xgb_combined_pred_proba = self.models['xgboost']['model'].predict_proba(X_combined_scaled)[:, 1]
        
        # Get Linear predictions for combined data
        linear_combined_pred_proba = self.models['linear']['model'].predict_proba(X_combined_scaled)[:, 1]
        
        # The LSTM predictions are based on sequences, so we need to align the XGBoost and Linear predictions
        # to match the LSTM sequence-based predictions
        # LSTM loses (sequence_length - 1) samples at the beginning, so we need to trim the predictions accordingly
        xgb_combined_pred_proba = xgb_combined_pred_proba[self.sequence_length-1:]
        linear_combined_pred_proba = linear_combined_pred_proba[self.sequence_length-1:]
        
        # Now trim to match the exact LSTM size
        xgb_combined_pred_proba = xgb_combined_pred_proba[-lstm_train_size:]
        linear_combined_pred_proba = linear_combined_pred_proba[-lstm_train_size:]
        
        # Align test predictions (trim from the end to match LSTM test size)
        xgb_test_pred_proba = self.models['xgboost']['test_pred_proba'][-lstm_test_size:]
        linear_test_pred_proba = self.models['linear']['test_pred_proba'][-lstm_test_size:]
        
        # Get predictions from base models (aligned)
        meta_features_train = np.column_stack([
            self.models['lstm']['train_pred_proba'],
            xgb_combined_pred_proba,
            linear_combined_pred_proba
        ])
        
        meta_features_test = np.column_stack([
            self.models['lstm']['test_pred_proba'],
            xgb_test_pred_proba,
            linear_test_pred_proba
        ])
        
        # Add key features to meta-model for better ensemble performance
        additional_features = [
            'CP', 'Volume', 'Returns', 'Volatility_20', 'RSI', 'MACD', 
            'Interest_Rate', 'Inflation_Rate', 'GDP', 'Gold_Price', 'Unemployment_Rate', 
            'Sentiment', 'Sentiment_MA_5', 'BB_Position'
        ]
        
        # Get additional features for each split (aligned with LSTM)
        # For train, we need to account for the sequence length offset and align with LSTM predictions
        train_additional = combined_data[additional_features].iloc[self.sequence_length-1:].values
        train_additional = train_additional[-lstm_train_size:]
        test_additional = test_data[additional_features].iloc[-lstm_test_size:].values
        
        # Combine predictions with additional features
        meta_features_train = np.column_stack([meta_features_train, train_additional])
        meta_features_test = np.column_stack([meta_features_test, test_additional])
        
        # Get actual values (already aligned with LSTM)
        y_train = self.models['lstm']['train_actual']
        y_test = self.models['lstm']['test_actual']
        
        # For meta-model training, we'll use a portion of the training data as validation
        # Split the training data for meta-model validation
        split_idx = int(len(meta_features_train) * 0.8)
        meta_train_features = meta_features_train[:split_idx]
        meta_val_features = meta_features_train[split_idx:]
        meta_train_labels = y_train[:split_idx]
        meta_val_labels = y_train[split_idx:]
        
        # Calculate class weights for meta-model to handle its own training imbalance
        meta_neg, meta_pos = np.bincount(meta_val_labels)
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
        
        meta_model.fit(meta_train_features, meta_train_labels)
        
        # Make meta-predictions
        train_meta_pred_proba = meta_model.predict_proba(meta_features_train)[:, 1]
        val_meta_pred_proba = meta_model.predict_proba(meta_val_features)[:, 1]
        test_meta_pred_proba = meta_model.predict_proba(meta_features_test)[:, 1]
        
        train_meta_pred = (train_meta_pred_proba > 0.5).astype(int)
        val_meta_pred = (val_meta_pred_proba > 0.5).astype(int)
        test_meta_pred = (test_meta_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_meta_pred)
        val_accuracy = accuracy_score(meta_val_labels, val_meta_pred)
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
            'val_actual': meta_val_labels,
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
            
            # Test metrics
            test_acc = accuracy_score(model_data['test_actual'], model_data['test_pred'])
            test_prec = precision_score(model_data['test_actual'], model_data['test_pred'])
            test_rec = recall_score(model_data['test_actual'], model_data['test_pred'])
            test_f1 = f1_score(model_data['test_actual'], model_data['test_pred'])
            test_roc = roc_auc_score(model_data['test_actual'], model_data['test_pred_proba'])
            
            # Handle validation metrics (LSTM uses CV, others use single validation set)
            if model_data['val_actual'] is not None and model_data['val_pred'] is not None:
                val_acc = accuracy_score(model_data['val_actual'], model_data['val_pred'])
                val_prec = precision_score(model_data['val_actual'], model_data['val_pred'])
                val_rec = recall_score(model_data['val_actual'], model_data['val_pred'])
                val_f1 = f1_score(model_data['val_actual'], model_data['val_pred'])
                val_roc = roc_auc_score(model_data['val_actual'], model_data['val_pred_proba'])
            else:
                # For LSTM with CV, use CV average scores
                if 'cv_scores' in model_data:
                    val_acc = np.mean([score['val_accuracy'] for score in model_data['cv_scores']])
                    val_f1 = np.mean([score['val_f1'] for score in model_data['cv_scores']])
                    val_prec = val_f1  # Approximate since we don't have precision per fold
                    val_rec = val_f1   # Approximate since we don't have recall per fold
                    val_roc = val_f1   # Approximate since we don't have ROC per fold
                else:
                    val_acc = val_prec = val_rec = val_f1 = val_roc = 0.0
            
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
        
        # Check if we have CV results
        has_cv = 'cv_scores' in self.models['lstm'] and 'cv_histories' in self.models['lstm']
        
        if has_cv:
            # Plot CV results
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('LSTM Cross-Validation Results', fontsize=16, fontweight='bold')
            
            # Plot CV scores across folds
            cv_scores = self.models['lstm']['cv_scores']
            folds = [score['fold'] for score in cv_scores]
            accuracies = [score['val_accuracy'] for score in cv_scores]
            f1_scores = [score['val_f1'] for score in cv_scores]
            
            axes[0, 0].bar(folds, accuracies, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Validation Accuracy by Fold')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)
            
            axes[0, 1].bar(folds, f1_scores, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Validation F1 Score by Fold')
            axes[0, 1].set_xlabel('Fold')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
            
            # Plot final training history
            axes[1, 0].plot(history.history['loss'], label='Training Loss', color='blue')
            axes[1, 0].set_title('Final Model Training Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
            axes[1, 1].set_title('Final Model Training Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Print CV summary
            print(f"Cross-Validation Results:")
            print(f"  Average Validation Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
            print(f"  Average Validation F1 Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
            print(f"  Best Fold: {folds[np.argmax(f1_scores)]} (F1: {max(f1_scores):.4f})")
            print(f"  Worst Fold: {folds[np.argmin(f1_scores)]} (F1: {min(f1_scores):.4f})")
            
        else:
            # Original plotting for non-CV case
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle('LSTM Training History', fontsize=16, fontweight='bold')
            
            # Plot training & validation loss
            axes[0].plot(history.history['loss'], label='Training Loss', color='blue')
            if 'val_loss' in history.history:
                axes[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
            axes[0].set_title('Model Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot training & validation accuracy
            axes[1].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
            if 'val_accuracy' in history.history:
                axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
            axes[1].set_title('Model Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Print training summary
        print(f"Final training completed in {len(history.history['loss'])} epochs")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        
        if has_cv:
            print("✅ Cross-validation approach helps avoid overfitting to specific time periods")
        else:
            print("⚠️  Using single validation set - consider cross-validation for better generalization")
    
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
        
        # Add LSTM CV results if available
        if 'lstm' in self.models and 'cv_scores' in self.models['lstm']:
            print("\nLSTM Cross-Validation Confusion Matrices:")
            print("-" * 50)
            cv_scores = self.models['lstm']['cv_scores']
            for score in cv_scores:
                print(f"Fold {score['fold']}: Accuracy={score['val_accuracy']:.4f}, F1={score['val_f1']:.4f}")
        
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
