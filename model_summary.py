import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class SP500ModelSummary:
    def __init__(self):
        """Initialize the model summary with the latest classification results"""
        self.results = {
            'LSTM': {
                'Accuracy': 0.5712, 'Precision': 0.5703, 'Recall': 0.8328, 'F1 Score': 0.6770, 'ROC AUC': 0.6785,
                'Description': 'Deep Learning model using Long Short-Term Memory networks',
                'Features': 'Sequential technical indicators (60-day lookback)',
                'Strengths': 'Captures temporal dependencies and patterns',
                'Weaknesses': 'Sensitive to market regime changes'
            },
            'XGBoost': {
                'Accuracy': 0.5188, 'Precision': 0.6791, 'Recall': 0.3175, 'F1 Score': 0.4327, 'ROC AUC': 0.5915,
                'Description': 'Gradient boosting ensemble model',
                'Features': 'Technical indicators, economic data, lagged features',
                'Strengths': 'Handles non-linear relationships, feature importance',
                'Weaknesses': 'Prone to overfitting if not tuned carefully'
            },
            'Linear': {
                'Accuracy': 0.4668, 'Precision': 0.7067, 'Recall': 0.1325, 'F1 Score': 0.2232, 'ROC AUC': 0.6633,
                'Description': 'L1-regularized Logistic Regression baseline model',
                'Features': 'Technical indicators, economic data, lagged features',
                'Strengths': 'Interpretable, fast, performs feature selection',
                'Weaknesses': 'Assumes linear relationships'
            },
            'Meta-Model': {
                'Accuracy': 0.6123, 'Precision': 0.5945, 'Recall': 0.8856, 'F1 Score': 0.7114, 'ROC AUC': 0.6208,
                'Description': 'Ensemble model combining predictions from base models',
                'Features': 'Probabilistic predictions from LSTM, XGBoost, and Linear models',
                'Strengths': 'Combines different modeling approaches for improved accuracy',
                'Weaknesses': 'Performance is highly dependent on base model quality'
            }
        }
        
    def display_model_comparison(self):
        """Display comprehensive model comparison for classification"""
        print("="*100)
        print("S&P 500 PREDICTION MODEL COMPARISON (CLASSIFICATION)")
        print("="*100)
        
        # Create comparison table
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1 Score': f"{metrics['F1 Score']:.4f}",
                'ROC AUC': f"{metrics['ROC AUC']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        print(f"\n" + "="*100)
        print("MODEL PERFORMANCE ANALYSIS")
        print("="*100)
        
        # Find best performing model based on F1 Score
        best_f1_model = max(self.results.items(), key=lambda x: x[1]['F1 Score'])
        best_acc_model = max(self.results.items(), key=lambda x: x[1]['Accuracy'])
        best_roc_auc_model = max(self.results.items(), key=lambda x: x[1]['ROC AUC'])
        
        print(f"Best F1 Score: {best_f1_model[0]} ({best_f1_model[1]['F1 Score']:.4f})")
        print(f"Best Accuracy: {best_acc_model[0]} ({best_acc_model[1]['Accuracy']:.4f})")
        print(f"Best ROC AUC: {best_roc_auc_model[0]} ({best_roc_auc_model[1]['ROC AUC']:.4f})")
        
        print(f"\nKey Insights:")
        print(f"• Meta-Model shows the best overall performance with an F1 Score of {self.results['Meta-Model']['F1 Score']:.4f}.")
        print(f"• The stacking approach successfully combines base models to improve predictive power.")
        print(f"• The LSTM model also performs well, showing strong recall.")
        print(f"• The Linear model's high precision suggests it's good at identifying positive cases, but its low recall means it misses many.")
        
    def plot_model_performance(self):
        """Plot model performance comparison for classification"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data for plotting
        models = list(self.results.keys())
        accuracy_values = [self.results[model]['Accuracy'] for model in models]
        f1_values = [self.results[model]['F1 Score'] for model in models]
        roc_auc_values = [self.results[model]['ROC AUC'] for model in models]
        
        # Accuracy comparison
        bars1 = axes[0, 0].bar(models, accuracy_values, color=['blue', 'green', 'red', 'purple'])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy (Higher is Better)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars1, accuracy_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{value:.4f}', ha='center', va='bottom')
        
        # F1 Score comparison
        bars2 = axes[0, 1].bar(models, f1_values, color=['blue', 'green', 'red', 'purple'])
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score (Higher is Better)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars2, f1_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{value:.4f}', ha='center', va='bottom')
        
        # ROC AUC comparison
        bars3 = axes[1, 0].bar(models, roc_auc_values, color=['blue', 'green', 'red', 'purple'])
        axes[1, 0].set_title('ROC AUC Score Comparison')
        axes[1, 0].set_ylabel('ROC AUC (Higher is Better)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars3, roc_auc_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{value:.4f}', ha='center', va='bottom')
            
        # Precision vs. Recall
        precision_values = [self.results[model]['Precision'] for model in models]
        recall_values = [self.results[model]['Recall'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        rects1 = axes[1, 1].bar(x - width/2, precision_values, width, label='Precision')
        rects2 = axes[1, 1].bar(x + width/2, recall_values, width, label='Recall')
        
        axes[1, 1].set_ylabel('Scores')
        axes[1, 1].set_title('Precision vs. Recall')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
    def analyze_feature_importance(self):
        """Analyze feature importance from XGBoost model"""
        print(f"\n" + "="*100)
        print("FEATURE IMPORTANCE ANALYSIS (XGBoost Model)")
        print("="*100)
        
        # Top features from the latest training run
        top_features = [
            ('Interest_Rate', 0.0831, 'Prevailing interest rate'),
            ('Inflation_Rate', 0.0790, 'Trailing 12-month inflation rate'),
            ('Sentiment_MA_5', 0.0680, '5-day moving average of news sentiment'),
            ('MACD_Signal', 0.0642, 'MACD signal line'),
            ('Price_Lag_5', 0.0635, 'Price 5 days ago'),
            ('Volatility_20', 0.0618, '20-day volatility'),
            ('MA_5', 0.0618, '5-day moving average'),
            ('MA_20', 0.0589, '20-day moving average'),
            ('MACD', 0.0582, 'Moving Average Convergence Divergence'),
            ('Price_Lag_10', 0.0565, 'Price 10 days ago')
        ]
        
        print("Top 10 Most Important Features:")
        print("-" * 80)
        for i, (feature, importance, description) in enumerate(top_features, 1):
            print(f"{i:2d}. {feature:20} ({importance:.4f}) - {description}")
            
        # Plot feature importance
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(features, importances, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features for S&P 500 Prediction')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nKey Insights:")
        print(f"• Economic indicators (Interest_Rate, Inflation_Rate) are the most dominant features.")
        print(f"• News sentiment (Sentiment_MA_5) plays a crucial role in prediction.")
        print(f"• Both short-term price history (lags) and technical indicators (MACD, MAs) are highly predictive.")
        
    def provide_recommendations(self):
        """Provide recommendations for improving the models"""
        print(f"\n" + "="*100)
        print("RECOMMENDATIONS FOR IMPROVEMENT")
        print("="*100)
        
        print("1. META-MODEL TUNING:")
        print("   • The Meta-Model performs well but could be overfit to the validation set.")
        print("   • Implement k-fold cross-validation for training the meta-model to improve generalization.")
        print("   • Experiment with different algorithms for the meta-model (e.g., Logistic Regression, LightGBM).")
        
        print("\n2. FEATURE ENGINEERING:")
        print("   • Create more diverse features for the base models to reduce correlation.")
        print("   • For example, create a long-term trend model and a short-term volatility model.")
        print("   • Add more alternative data sources like options data (VIX) or sector-specific ETFs.")
        
        print("\n3. MODEL DIVERSIFICATION:")
        print("   • Introduce more diverse model architectures into the ensemble.")
        print("   • Consider models like TabNet, LightGBM, or a simple Naive Bayes classifier.")
        print("   • The goal is to have uncorrelated errors among the base models.")
        
        print("\n4. RISK MANAGEMENT:")
        print("   • Never rely solely on model predictions for trading.")
        print("   • Always use proper position sizing and stop-losses.")
        print("   • Consider the model's confidence level (probability scores) in decisions.")
        
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("="*100)
        print("COMPREHENSIVE S&P 500 PREDICTION MODEL REPORT")
        print("="*100)
        
        # Model comparison
        self.display_model_comparison()
        
        # Performance plots
        self.plot_model_performance()
        
        # Feature importance
        self.analyze_feature_importance()
        
        # Recommendations
        self.provide_recommendations()
        
        print(f"\n" + "="*100)
        print("FINAL SUMMARY")
        print("="*100)
        print("• Successfully trained and evaluated a sophisticated 4-model stacking ensemble.")
        print(f"• The Meta-Model is the top performer, achieving an F1-Score of {self.results['Meta-Model']['F1 Score']:.4f} on the test set.")
        print("• Corrected implementation issues related to data leakage and feature representation.")
        print("• The ensemble approach effectively combines diverse models to achieve superior performance.")
        print("• Models are saved and ready for deployment in a prediction pipeline.")
        
        print(f"\nNote: These models are for educational and research purposes only.")
        print("Financial markets are inherently unpredictable and past performance")
        print("does not guarantee future results. Always conduct thorough research")
        print("and consider professional advice before making investment decisions.")

def main():
    """Main function to run the comprehensive summary"""
    summary = SP500ModelSummary()
    summary.create_summary_report()

if __name__ == "__main__":
    main() 