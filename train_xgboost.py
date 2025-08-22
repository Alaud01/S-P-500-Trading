import os
import json
import math
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

try:
	import xgboost as xgb
	XGBClassifier = xgb.XGBClassifier
except Exception as e:
	raise RuntimeError("xgboost is required. Please install dependencies via requirements.txt") from e


class _IdentityScaler:
	def fit(self, X):
		return self
	def transform(self, X):
		return X


@dataclass
class XGBParams:
	# Core params
	n_estimators: int = 800
	max_depth: int = 6
	learning_rate: float = 0.03
	subsample: float = 0.8
	colsample_bytree: float = 0.8
	min_child_weight: float = 1.0
	reg_alpha: float = 0.0
	reg_lambda: float = 1.0
	gamma: float = 0.0

	# Training params
	n_jobs: int = -1
	tree_method: str = "hist"
	device: Optional[str] = None  # "cuda" | "cpu"
	verbosity: int = 0
	random_state: int = 42

	# Time-series config
	seq_len: int = 1  # not sequence model; keep 1
	cv_splits: int = 5
	val_window: int = 250
	gap: int = 5
	min_train_window: int = 750
	use_scaler: bool = True

	# Early stopping
	early_stopping_rounds: int = 100
	test_size_holdout: float = 0.15


def set_seed(seed: int = 42) -> None:
	np.random.seed(seed)


def load_dataset(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if 'Date' in df.columns:
		df['Date'] = pd.to_datetime(df['Date'])
		df = df.sort_values('Date').reset_index(drop=True)
	return df.fillna(method='ffill').fillna(method='bfill')


def select_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
	feature_df = df.select_dtypes(include=[np.number]).copy()
	if 'Target' not in feature_df.columns:
		raise ValueError("Target column 'Target' not found in dataset. Run built_dataset.py first.")
	y = feature_df['Target'].astype(int).values
	X = feature_df.drop(columns=['Target'])
	return X, y, list(X.columns)


def time_series_cv_indices(n_rows: int,
						n_splits: int,
						val_window: int,
						gap: int,
						min_train_window: int) -> List[Tuple[np.ndarray, np.ndarray]]:
	splits: List[Tuple[np.ndarray, np.ndarray]] = []
	start_val = max(min_train_window, 0)
	while start_val + gap + val_window <= n_rows and len(splits) < n_splits:
		train_end = start_val  # exclusive
		val_start = start_val + gap
		val_end = val_start + val_window
		train_idx = np.arange(0, train_end)
		val_idx = np.arange(val_start, val_end)
		if len(train_idx) == 0 or len(val_idx) == 0:
			break
		splits.append((train_idx, val_idx))
		start_val = val_end
	if len(splits) == 0:
		raise ValueError("Unable to create any time-series CV splits. Reduce val_window or gap.")
	return splits


def compute_scale_pos_weight(labels: np.ndarray) -> float:
	pos = float((labels == 1).sum())
	neg = float((labels == 0).sum())
	if pos == 0:
		return 1.0
	return max(neg / max(pos, 1.0), 1.0)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
	try:
		auc = roc_auc_score(y_true, y_prob)
	except Exception:
		auc = float('nan')
	y_pred = (y_prob >= threshold).astype(int)
	acc = accuracy_score(y_true, y_pred)
	bal_acc = balanced_accuracy_score(y_true, y_pred)
	precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
	return {
		'auc': float(auc) if not (isinstance(auc, float) and math.isnan(auc)) else 0.0,
		'accuracy': float(acc),
		'balanced_accuracy': float(bal_acc),
		'precision': float(precision),
		'recall': float(recall),
		'f1': float(f1),
	}


def find_best_threshold(y_prob: np.ndarray, y_true: np.ndarray, metric: str = 'f1') -> Tuple[float, float]:
	best_thr = 0.5
	best_score = -1.0
	for thr in np.linspace(0.05, 0.95, 19):
		preds = (y_prob >= thr).astype(int)
		if metric == 'youden':
			# Youden's J = TPR - FPR
			tp = np.logical_and(preds == 1, y_true == 1).sum()
			tn = np.logical_and(preds == 0, y_true == 0).sum()
			fp = np.logical_and(preds == 1, y_true == 0).sum()
			fn = np.logical_and(preds == 0, y_true == 1).sum()
			tpr = tp / max(tp + fn, 1)
			fpr = fp / max(fp + tn, 1)
			score = tpr - fpr
		else:
			_, _, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
			score = float(f1)
		if score > best_score:
			best_score = score
			best_thr = float(thr)
	return best_thr, best_score


def train_xgb_cv(X: np.ndarray,
				y: np.ndarray,
				feature_names: List[str],
				params: XGBParams,
				verbose: bool = True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any], float, Any, Optional[StandardScaler]]:
	# Compute class weight using first training fold to avoid future leakage
	splits = time_series_cv_indices(n_rows=X.shape[0],
							n_splits=params.cv_splits,
							val_window=params.val_window,
							gap=params.gap,
							min_train_window=params.min_train_window)

	best_val_auc = -1.0
	best_model = None
	best_scaler: Optional[StandardScaler] = None
	best_threshold = 0.5
	best_hist: Dict[str, List[float]] = {'val_auc': [], 'val_f1': []}

	for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
		if verbose:
			print(f"Fold {fold_idx}/{len(splits)} | train: [0, {train_idx[-1]}], val: [{val_idx[0]}, {val_idx[-1]}]")

		# Fit scaler only on training window
		scaler = StandardScaler() if params.use_scaler else _IdentityScaler()
		scaler.fit(X[train_idx])
		X_train = scaler.transform(X[train_idx])
		X_val = scaler.transform(X[val_idx])
		y_train, y_val = y[train_idx], y[val_idx]

		scale_pos_weight = compute_scale_pos_weight(y_train)

		model = XGBClassifier(
			n_estimators=params.n_estimators,
			max_depth=params.max_depth,
			learning_rate=params.learning_rate,
			subsample=params.subsample,
			colsample_bytree=params.colsample_bytree,
			min_child_weight=params.min_child_weight,
			reg_alpha=params.reg_alpha,
			reg_lambda=params.reg_lambda,
			gamma=params.gamma,
			objective='binary:logistic',
			eval_metric='auc',
			n_jobs=params.n_jobs,
			tree_method=params.tree_method,
			verbosity=params.verbosity,
			random_state=params.random_state,
			scale_pos_weight=scale_pos_weight,
			device=params.device if params.device else None,
		)

		model.fit(
			X_train,
			y_train,
			eval_set=[(X_val, y_val)],
			verbose=False
		)

		val_prob = model.predict_proba(X_val)[:, 1]
		thr, _ = find_best_threshold(val_prob, y_val, metric='f1')
		metrics = evaluate(y_val, val_prob, threshold=thr)
		best_hist['val_auc'].append(metrics['auc'])
		best_hist['val_f1'].append(metrics['f1'])

		if verbose:
			print(f"  Val AUC: {metrics['auc']:.4f} | F1@thr {thr:.2f}: {metrics['f1']:.4f}")

		if metrics['auc'] > best_val_auc:
			best_val_auc = metrics['auc']
			best_model = model
			best_scaler = scaler
			best_threshold = thr

	# Final train/val metrics correspond to best fold's val performance
	train_metrics = {}
	val_metrics = {
		'auc': float(np.mean(best_hist['val_auc'])),
		'f1': float(np.mean(best_hist['val_f1']))
	}

	artifacts = {
		'feature_names': feature_names,
		'scaler_mean': best_scaler.mean_.tolist() if best_scaler is not None else None,
		'scaler_scale': best_scaler.scale_.tolist() if best_scaler is not None else None,
		'cv_val_auc_mean': float(np.mean(best_hist['val_auc'])),
		'cv_val_f1_mean': float(np.mean(best_hist['val_f1'])),
	}

	return train_metrics, val_metrics, artifacts, best_threshold, best_model, best_scaler


def feature_importance_plot(model: XGBClassifier, feature_names: List[str], out_path: str) -> None:
	booster = model.get_booster()
	score_map = booster.get_score(importance_type='gain')
	# Map 'f{i}' to feature name
	name_map = {}
	for key, val in score_map.items():
		if key.startswith('f') and key[1:].isdigit():
			idx = int(key[1:])
			if 0 <= idx < len(feature_names):
				name_map[feature_names[idx]] = val
			else:
				name_map[key] = val
		else:
			name_map[key] = val

	if len(name_map) == 0:
		return

	items = sorted(name_map.items(), key=lambda kv: kv[1], reverse=True)[:30]
	labels, gains = zip(*items)
	plt.figure(figsize=(10, max(4, len(labels) * 0.3)))
	plt.barh(labels[::-1], gains[::-1])
	plt.xlabel('Gain')
	plt.title('Top Feature Importances (gain)')
	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	plt.close()


def plot_test_predictions_vs_price(model: XGBClassifier,
								scaler: StandardScaler,
								data_csv: str,
								plots_dir: str,
								test_start_date: str,
								verbose: bool = True) -> None:
	"""
	Create a plot showing XGBoost model predictions vs S&P 500 price for the test set (2022 onwards).
	"""
	if not verbose:
		return
		
	print("  📊 Generating test set predictions vs price visualization...")
	
	# Load and prepare data
	df = load_dataset(data_csv)
	X_df, y, feature_names = select_features_and_target(df)
	X = X_df.values.astype(np.float32)
	
	# Split data into train and test based on date
	test_start = pd.to_datetime(test_start_date)
	if 'Date' in df.columns:
		test_mask = df['Date'] >= test_start
		train_mask = ~test_mask
		
		X_train = X[train_mask]
		X_test = X[test_mask]
		y_train = y[train_mask]
		y_test = y[test_mask]
		test_dates = df['Date'][test_mask].values
		test_prices = df['Close'][test_mask].values if 'Close' in df.columns else None
	else:
		# Fallback if no date column
		split_idx = int(len(X) * 0.8)
		X_train = X[:split_idx]
		X_test = X[split_idx:]
		y_train = y[:split_idx]
		y_test = y[split_idx:]
		test_dates = np.arange(len(X_test))
		test_prices = None
	
	# Scale test data using the fitted scaler
	X_test_scaled = scaler.transform(X_test)
	
	# Get predictions for test set
	predictions = model.predict_proba(X_test_scaled)[:, 1]
	
	# Create the visualization
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
	fig.suptitle(f'XGBoost Model Predictions vs S&P 500 Price (Test Set: {test_start_date} onwards)', 
				 fontsize=16, fontweight='bold')
	
	# Plot 1: S&P 500 Price
	if test_prices is not None:
		ax1.plot(test_dates, test_prices, color='blue', linewidth=1.5, alpha=0.8)
		ax1.set_title('S&P 500 Closing Price (Test Set)', fontweight='bold')
		ax1.set_ylabel('Price ($)', fontweight='bold')
		ax1.grid(True, alpha=0.3)
		
		# Add price trend
		if len(test_prices) > 30:
			ma_30 = np.convolve(test_prices, np.ones(30)/30, mode='valid')
			ma_dates = test_dates[29:]
			ax1.plot(ma_dates, ma_30, color='red', linewidth=2, alpha=0.7, label='30-day MA')
			ax1.legend()
	else:
		ax1.text(0.5, 0.5, 'Price data not available', ha='center', va='center', transform=ax1.transAxes)
		ax1.set_title('S&P 500 Closing Price (Test Set)', fontweight='bold')
	
	# Plot 2: Model Predictions (Probability)
	ax2.plot(test_dates, predictions, color='green', linewidth=1.5, alpha=0.8)
	ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold (0.5)')
	ax2.set_title('XGBoost Prediction Probability (14-day forward)', fontweight='bold')
	ax2.set_ylabel('Probability', fontweight='bold')
	ax2.set_ylim(0, 1)
	ax2.grid(True, alpha=0.3)
	ax2.legend()
	
	# Plot 3: Predictions vs Actual (Binary)
	# Convert predictions to binary using 0.5 threshold
	pred_binary = (predictions >= 0.5).astype(int)
	actual_binary = y_test
	
	# Create scatter plot
	correct_up = (pred_binary == 1) & (actual_binary == 1)
	correct_down = (pred_binary == 0) & (actual_binary == 0)
	wrong_up = (pred_binary == 1) & (actual_binary == 0)
	wrong_down = (pred_binary == 0) & (actual_binary == 1)
	
	if test_prices is not None:
		ax3.scatter(test_dates[correct_up], test_prices[correct_up], color='green', alpha=0.6, s=20, label='Correct Up Prediction')
		ax3.scatter(test_dates[correct_down], test_prices[correct_down], color='blue', alpha=0.6, s=20, label='Correct Down Prediction')
		ax3.scatter(test_dates[wrong_up], test_prices[wrong_up], color='red', alpha=0.6, s=20, label='Wrong Up Prediction')
		ax3.scatter(test_dates[wrong_down], test_prices[wrong_down], color='orange', alpha=0.6, s=20, label='Wrong Down Prediction')
	else:
		ax3.scatter(test_dates[correct_up], np.zeros_like(test_dates[correct_up]), color='green', alpha=0.6, s=20, label='Correct Up Prediction')
		ax3.scatter(test_dates[correct_down], np.zeros_like(test_dates[correct_down]), color='blue', alpha=0.6, s=20, label='Correct Down Prediction')
		ax3.scatter(test_dates[wrong_up], np.zeros_like(test_dates[wrong_up]), color='red', alpha=0.6, s=20, label='Wrong Up Prediction')
		ax3.scatter(test_dates[wrong_down], np.zeros_like(test_dates[wrong_down]), color='orange', alpha=0.6, s=20, label='Wrong Down Prediction')
	
	ax3.set_title('Prediction Accuracy vs Price (Test Set)', fontweight='bold')
	ax3.set_ylabel('Price ($)' if test_prices is not None else 'Index', fontweight='bold')
	ax3.set_xlabel('Date', fontweight='bold')
	ax3.grid(True, alpha=0.3)
	ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	
	# Calculate and display accuracy metrics
	accuracy = np.mean(pred_binary == actual_binary)
	precision = np.sum((pred_binary == 1) & (actual_binary == 1)) / max(np.sum(pred_binary == 1), 1)
	recall = np.sum((pred_binary == 1) & (actual_binary == 1)) / max(np.sum(actual_binary == 1), 1)
	f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
	
	# Add text box with metrics
	metrics_text = f'Test Set Metrics:\nAccuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
	ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, fontsize=10,
			 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
	
	# Format x-axis dates
	if len(test_dates) > 0 and isinstance(test_dates[0], pd.Timestamp):
		ax3.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
		ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
		plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
	
	plt.tight_layout()
	
	# Save plot
	os.makedirs(plots_dir, exist_ok=True)
	plot_path = os.path.join(plots_dir, 'xgb_test_predictions_vs_price.png')
	plt.savefig(plot_path, dpi=300, bbox_inches='tight')
	print(f"  📈 Test set predictions vs price plot saved to: {plot_path}")
	plt.show()
	
	# Create additional detailed analysis plot for test set
	fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
	fig2.suptitle(f'XGBoost Model Test Set Analysis ({test_start_date} onwards)', fontsize=16, fontweight='bold')
	
	# Plot 1: Prediction distribution
	ax1.hist(predictions, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
	ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
	ax1.set_title('Distribution of Prediction Probabilities (Test Set)')
	ax1.set_xlabel('Prediction Probability')
	ax1.set_ylabel('Frequency')
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	
	# Plot 2: Prediction confidence over time
	confidence = np.abs(predictions - 0.5) * 2  # Convert to 0-1 confidence scale
	ax2.plot(test_dates, confidence, color='purple', alpha=0.7)
	ax2.set_title('Model Confidence Over Time (Test Set)')
	ax2.set_xlabel('Date')
	ax2.set_ylabel('Confidence (0-1)')
	ax2.grid(True, alpha=0.3)
	
	# Plot 3: Rolling accuracy
	window_size = min(50, len(pred_binary) // 4)  # Adaptive window size
	if window_size > 10:
		rolling_accuracy = []
		rolling_dates = []
		for i in range(window_size, len(pred_binary)):
			window_acc = np.mean(pred_binary[i-window_size:i] == actual_binary[i-window_size:i])
			rolling_accuracy.append(window_acc)
			rolling_dates.append(test_dates[i])
		
		ax3.plot(rolling_dates, rolling_accuracy, color='green', alpha=0.7)
		ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')
		ax3.set_title(f'Rolling Accuracy ({window_size}-day window)')
		ax3.set_xlabel('Date')
		ax3.set_ylabel('Accuracy')
		ax3.legend()
		ax3.grid(True, alpha=0.3)
	else:
		ax3.text(0.5, 0.5, 'Insufficient data for rolling accuracy', ha='center', va='center', transform=ax3.transAxes)
		ax3.set_title('Rolling Accuracy')
	
	# Plot 4: Prediction vs actual correlation
	ax4.scatter(actual_binary, predictions, alpha=0.5, color='blue')
	ax4.set_title('Predictions vs Actual Values (Test Set)')
	ax4.set_xlabel('Actual (0=Down, 1=Up)')
	ax4.set_ylabel('Predicted Probability')
	ax4.grid(True, alpha=0.3)
	
	plt.tight_layout()
	
	# Save detailed analysis plot
	detailed_plot_path = os.path.join(plots_dir, 'xgb_test_detailed_analysis.png')
	plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
	print(f"  📊 Test set detailed analysis plot saved to: {detailed_plot_path}")
	plt.show()


def save_json(obj: Dict[str, Any], path: str) -> None:
	with open(path, 'w') as f:
		json.dump(obj, f, indent=2)


def ensure_dirs(*paths: str) -> None:
	for p in paths:
		os.makedirs(p, exist_ok=True)


def main() -> None:
	parser = argparse.ArgumentParser(description='Train XGBoost to predict Target')
	parser.add_argument('--data', type=str, default='data/final_dataset_for_modeling.csv', help='Path to dataset CSV')
	parser.add_argument('--models_dir', type=str, default='models/xgboost', help='Directory to save models and reports')
	parser.add_argument('--plots_dir', type=str, default='plots/xgboost', help='Directory to save plots')
	parser.add_argument('--device', type=str, default=None, help='xgboost device: cuda or cpu (requires xgboost>=2.0)')
	parser.add_argument('--test_start_date', type=str, default='2022-01-01', help='Start date for test set evaluation')
	parser.add_argument('--use_scaler', action='store_true', help='Apply StandardScaler to features (default off)')
	parser.add_argument('--search_trials', type=int, default=0, help='Randomized hyperparameter search trials (0=disabled)')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--verbose', action='store_true')
	args = parser.parse_args()

	set_seed(args.seed)
	ensure_dirs(args.models_dir, args.plots_dir)

	df = load_dataset(args.data)
	X_df, y, feature_names = select_features_and_target(df)
	X = X_df.values.astype(np.float32)

	params = XGBParams()
	params.device = args.device
	params.use_scaler = bool(args.use_scaler)

	if args.verbose:
		print(f"Rows: {X.shape[0]}, Features: {X.shape[1]}")

	train_metrics, val_metrics, artifacts, best_thr, best_model, best_scaler = train_xgb_cv(
		X=X,
		y=y,
		feature_names=feature_names,
		params=params,
		verbose=args.verbose
	)

	# Holdout backtest on last window after final CV split
	splits = time_series_cv_indices(
		n_rows=X.shape[0],
		n_splits=params.cv_splits,
		val_window=params.val_window,
		gap=params.gap,
		min_train_window=params.min_train_window
	)
	last_train_end = splits[-1][0][-1] + 1
	last_val_end = splits[-1][1][-1] + 1
	holdout_start = last_val_end + params.gap
	holdout_idx = np.arange(holdout_start, X.shape[0]) if holdout_start < X.shape[0] else np.array([], dtype=int)

	holdout_metrics: Dict[str, float] = {}
	preds_df: Optional[pd.DataFrame] = None
	if holdout_idx.size > 0 and best_model is not None:
		# Fit scaler on all data up to holdout start
		scaler = StandardScaler()
		scaler.fit(X[:holdout_start])
		X_holdout = scaler.transform(X[holdout_idx])
		y_holdout = y[holdout_idx]
		y_prob_holdout = best_model.predict_proba(X_holdout)[:, 1]
		thr_holdout, _ = find_best_threshold(y_prob_holdout, y_holdout, metric='f1')
		holdout_metrics = evaluate(y_holdout, y_prob_holdout, threshold=thr_holdout)

		preds_df = pd.DataFrame({
			'Date': df['Date'].iloc[holdout_idx].values if 'Date' in df.columns else holdout_idx,
			'prob': y_prob_holdout,
			'pred': (y_prob_holdout >= thr_holdout).astype(int),
			'target': y_holdout,
		})

	# Save artifacts
	model_tag = f"xgb_best"
	report = {
		'cv': val_metrics,
		'best_threshold_cv': best_thr,
		'holdout': holdout_metrics,
		'params': asdict(params),
		'feature_names': feature_names,
	}
	save_json(report, os.path.join(args.models_dir, f"{model_tag}_report.json"))

	if best_model is not None:
		# Save model via JSON dump (portable); user can also pickle if desired
		best_model.save_model(os.path.join(args.models_dir, f"{model_tag}.json"))
		feature_importance_plot(best_model, feature_names, os.path.join(args.plots_dir, 'xgb_feature_importance.png'))
		
		# Generate test set predictions vs price visualization
		plot_test_predictions_vs_price(
			model=best_model,
			scaler=best_scaler,
			data_csv=args.data,
			plots_dir=args.plots_dir,
			test_start_date=args.test_start_date,
			verbose=args.verbose
		)

	if preds_df is not None:
		preds_df.to_csv(os.path.join(args.models_dir, f"{model_tag}_holdout_predictions.csv"), index=False)

	if args.verbose:
		print("Done. Artifacts saved.")


if __name__ == '__main__':
	main()


