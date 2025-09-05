# run command: python train_xgboost.py --data data/final_dataset_for_modeling.csv --use_scaler --target_cleanup --verbose --threshold_metric accuracy

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
		self.mean_ = np.zeros(X.shape[1])
		self.scale_ = np.ones(X.shape[1])
		return self
	def transform(self, X):
		return X


@dataclass
class XGBParams:
	# Core params
	n_estimators: int = 1500
	max_depth: int = 3
	learning_rate: float = 0.005
	subsample: float = 0.8
	colsample_bytree: float = 0.6
	min_child_weight: float = 5.0
	reg_alpha: float = 2.0
	reg_lambda: float = 8.0
	gamma: float = 2.0

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

	# Time-decay weighting (to reduce overfitting to old regimes)
	time_decay_half_life: Optional[int] = None  # in days/rows; None disables
	time_decay_min_weight: float = 0.05  # floor weight for the oldest samples

	# Early stopping
	early_stopping_rounds: int = 100
	test_size_holdout: float = 0.15
	calibrate: bool = False


def set_seed(seed: int = 42) -> None:
	np.random.seed(seed)


def load_dataset(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if 'Date' in df.columns:
		df['Date'] = pd.to_datetime(df['Date'])
		df = df.sort_values('Date').reset_index(drop=True)
	return df.fillna(method='ffill').fillna(method='bfill')


def apply_target_cleanup(df: pd.DataFrame,
						horizon: int = 14,
						band: float = 0.005,
						drop_ambiguous: bool = True) -> pd.DataFrame:
	"""
	Create a cleaner binary target for a forward horizon using a neutral band.

	- Computes forward cumulative return over `horizon` days: (Close[t+h]/Close[t]) - 1
	- Sets `Target` = 1 if return > band, 0 if return < -band
	- If `drop_ambiguous` is True, drops rows with |return| <= band
	- Adds a helper column `FwdRet_{horizon}` for analysis (removed from features later)
	"""
	if 'Close' not in df.columns:
		raise ValueError("Column 'Close' is required in the dataset to compute forward returns.")

	working = df.copy()
	ret_col = f'FwdRet_{int(horizon)}'
	future_close = working['Close'].shift(-int(horizon))
	working[ret_col] = (future_close / working['Close']) - 1.0

	# Build target by sign with a neutral band
	pos_mask = working[ret_col] > float(band)
	neg_mask = working[ret_col] < -float(band)
	keep_mask = pos_mask | neg_mask

	if drop_ambiguous:
		working = working.loc[keep_mask].copy()
		working['Target'] = (working[ret_col] > 0).astype(int)
	else:
		# Keep ambiguous rows; label by sign only (can add a third class later if needed)
		working['Target'] = (working[ret_col] > 0).astype(int)

	# Drop tail rows where future price is NaN due to shifting
	working = working.dropna(subset=[ret_col]).reset_index(drop=True)
	return working


def select_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
	feature_df = df.select_dtypes(include=[np.number]).copy()
	if 'Target' not in feature_df.columns:
		raise ValueError("Target column 'Target' not found in dataset. Run built_dataset.py first.")
	# Ensure no leakage features from target construction are included
	leak_cols = [c for c in feature_df.columns if c == 'Future_Close' or c.startswith('FwdRet_')]
	cols_to_drop = ['Target'] + leak_cols
	y = feature_df['Target'].astype(int).values
	X = feature_df.drop(columns=cols_to_drop, errors='ignore')
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


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = 'f1', 
                      threshold_range: Tuple[float, float] = (0.1, 0.9), 
                      n_thresholds: int = 50) -> Tuple[float, Dict[str, float]]:
	"""
	Optimize prediction threshold based on specified metric using training data.
	
	Args:
		y_true: True binary labels
		y_prob: Predicted probabilities
		metric: Metric to optimize ('f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall')
		threshold_range: Range of thresholds to test (min, max)
		n_thresholds: Number of threshold values to test
	
	Returns:
		Tuple of (optimal_threshold, metrics_at_optimal_threshold)
	"""
	thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
	best_score = -1.0
	best_threshold = 0.5
	best_metrics = {}
	
	# Calculate metrics for each threshold
	for threshold in thresholds:
		metrics = evaluate(y_true, y_prob, threshold)
		score = metrics.get(metric, 0.0)
		
		if score > best_score:
			best_score = score
			best_threshold = threshold
			best_metrics = metrics
	
	return best_threshold, best_metrics


def random_search_xgb(X: np.ndarray,
					y: np.ndarray,
					feature_names: List[str],
					base_params: XGBParams,
					trials: int,
					verbose: bool = True) -> Tuple[XGBParams, Dict[str, float]]:
	"""Randomized hyperparameter search using time-series CV AUC as objective."""
	param_space = {
		'n_estimators': [400, 600, 800, 1000, 1200],
		'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05],
		'max_depth': [3, 4, 5, 6, 7, 8, 9],
		'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
		'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
		'min_child_weight': [1, 2, 3, 5, 7],
		'reg_alpha': [0.0, 0.001, 0.01, 0.1],
		'reg_lambda': [0.5, 1.0, 2.0, 5.0],
		'gamma': [0.0, 0.05, 0.1, 0.2],
	}

	best_score = -1.0
	best_cfg: Optional[XGBParams] = None
	best_metrics: Dict[str, float] = {}

	for t in range(trials):
		trial = XGBParams(**asdict(base_params))
		for k, values in param_space.items():
			setattr(trial, k, np.random.choice(values))
		if verbose:
			print(f"\n[Search] Trial {t+1}/{trials}: depth={trial.max_depth}, lr={trial.learning_rate}, n_estimators={trial.n_estimators}, subsample={trial.subsample}, colsample={trial.colsample_bytree}, min_child_weight={trial.min_child_weight}, reg_alpha={trial.reg_alpha}, reg_lambda={trial.reg_lambda}, gamma={trial.gamma}")
		_, val_metrics, _, _, _, _ = train_xgb_cv(X, y, feature_names, trial, verbose=False)
		score = val_metrics['auc']
		if verbose:
			print(f"  -> CV AUC: {score:.4f}, F1: {val_metrics['f1']:.4f}")
		if score > best_score:
			best_score = score
			best_cfg = trial
			best_metrics = val_metrics

	if best_cfg is None:
		return base_params, {'auc': 0.0, 'f1': 0.0}
	return best_cfg, best_metrics


def train_xgb_cv(X: np.ndarray,
				y: np.ndarray,
				feature_names: List[str],
				params: XGBParams,
				verbose: bool = True,
				threshold_metric: str = 'f1',
				threshold_range: Tuple[float, float] = (0.1, 0.9),
				threshold_n_tests: int = 50) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any], float, Any, Optional[StandardScaler]]:
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
	fold_histories = []
	# Out-of-fold predictions for calibration/thresholding
	oof_prob = np.full(y.shape[0], np.nan, dtype=np.float32)
	oof_true = np.full(y.shape[0], np.nan, dtype=np.float32)

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

		model, _, history = fit_xgb_with_es_compat(X_train, y_train, X_val, y_val, params, train_row_indices=train_idx)
		fold_histories.append(history)

		val_prob = model.predict_proba(X_val)[:, 1]
		metrics = evaluate(y_val, val_prob, threshold=0.5)
		best_hist['val_auc'].append(metrics['auc'])
		best_hist['val_f1'].append(metrics['f1'])
		# store OOF
		oof_prob[val_idx] = val_prob.astype(np.float32)
		oof_true[val_idx] = y_val.astype(np.float32)

		if verbose:
			print(f"  Val AUC: {metrics['auc']:.4f} | F1@thr {0.5:.2f}: {metrics['f1']:.4f}")

		if metrics['auc'] > best_val_auc:
			best_val_auc = metrics['auc']
			best_model = model
			best_scaler = scaler
			best_threshold = 0.5

	# Optimize threshold using out-of-fold predictions
	oof_mask = ~np.isnan(oof_prob) & ~np.isnan(oof_true)
	if np.sum(oof_mask) > 0:
		oof_prob_clean = oof_prob[oof_mask]
		oof_true_clean = oof_true[oof_mask]
		
		# Optimize threshold based on specified metric
		optimal_threshold, optimal_metrics = optimize_threshold(
			oof_true_clean, oof_prob_clean, metric=threshold_metric, 
			threshold_range=threshold_range, 
			n_thresholds=threshold_n_tests
		)
		
		if verbose:
			metric_value = optimal_metrics.get(threshold_metric, 0.0)
			print(f"  Threshold optimization: optimal={optimal_threshold:.3f}, {threshold_metric.upper()}={metric_value:.4f}")
	else:
		optimal_threshold = 0.5
		optimal_metrics = {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}

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
		'cv_val_auc_per_fold': [float(x) for x in best_hist['val_auc']],
		'cv_val_f1_per_fold': [float(x) for x in best_hist['val_f1']],
		'oof_prob': [None if np.isnan(v) else float(v) for v in oof_prob.tolist()],
		'oof_true': [None if np.isnan(v) else int(v) for v in oof_true.tolist()],
		'fold_histories': fold_histories,
		'optimal_threshold': float(optimal_threshold),
		'optimal_threshold_metrics': optimal_metrics,
	}

	return train_metrics, val_metrics, artifacts, optimal_threshold, best_model, best_scaler


def _clone_xgb_with_params(src: XGBClassifier, override: Dict[str, Any] = None) -> XGBClassifier:
	params = src.get_params()
	params.update(override or {})
	return XGBClassifier(**params)


class BoosterModelAdapter:
	def __init__(self, booster: Any, best_ntree_limit: Optional[int] = None):
		self._booster = booster
		self.best_ntree_limit = best_ntree_limit
	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		dm = xgb.DMatrix(X)
		limit = self.best_ntree_limit if self.best_ntree_limit is not None else 0
		if limit <= 0:
			prob_pos = self._booster.predict(dm)
		else:
			try:
				# For modern xgboost versions (>=1.4.0)
				prob_pos = self._booster.predict(dm, iteration_range=(0, limit))
			except TypeError:
				# For older xgboost versions (<1.4.0)
				prob_pos = self._booster.predict(dm, ntree_limit=limit)

		prob_pos = np.asarray(prob_pos).reshape(-1)
		prob_neg = 1.0 - prob_pos
		return np.column_stack([prob_neg, prob_pos])
	def save_model(self, path: str) -> None:
		self._booster.save_model(path)
	def get_booster(self) -> Any:
		return self._booster


def _build_sklearn_xgb(params: XGBParams, scale_pos_weight: float) -> XGBClassifier:
	return XGBClassifier(
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


def fit_xgb_with_es_compat(X_train: np.ndarray,
					y_train: np.ndarray,
					X_val: np.ndarray,
					y_val: np.ndarray,
					params: XGBParams,
					train_row_indices: Optional[np.ndarray] = None) -> Tuple[Any, Optional[int], Dict]:
	spw = compute_scale_pos_weight(y_train)
	evals_result: Dict = {}

	# Optional time-decay sample weights for training set
	train_sample_weight = None
	if params.time_decay_half_life is not None and train_row_indices is not None and len(train_row_indices) == len(y_train):
		# Newer rows get weight closer to 1, older rows approach time_decay_min_weight
		# Weight formula: w(i) = max(min_weight, 0.5 ** ((age_in_rows) / half_life))
		# where age_in_rows = (max_index - row_index)
		max_idx = int(train_row_indices.max())
		age = (max_idx - train_row_indices).astype(float)
		decay = np.power(0.5, age / float(max(1, params.time_decay_half_life)))
		train_sample_weight = np.clip(decay, float(params.time_decay_min_weight), 1.0)
	# Try modern callbacks API
	try:
		model = _build_sklearn_xgb(params, spw)
		model.fit(
			X_train,
			y_train,
			eval_set=[(X_val, y_val)],
			verbose=False,
			sample_weight=train_sample_weight,
			callbacks=[xgb.callback.EarlyStopping(rounds=params.early_stopping_rounds, metric_name='auc', data_name='validation_0', save_best=True)]
		)
		best_it = getattr(model, 'best_iteration', None)
		if best_it is None:
			best_it = getattr(model, 'best_ntree_limit', None)
		return model, best_it, getattr(model, 'evals_result_', {})
	except TypeError:
		pass
	# Try older sklearn API with early_stopping_rounds
	try:
		model = _build_sklearn_xgb(params, spw)
		model.fit(
			X_train,
			y_train,
			eval_set=[(X_val, y_val)],
			verbose=False,
			sample_weight=train_sample_weight,
			early_stopping_rounds=params.early_stopping_rounds
		)
		best_it = getattr(model, 'best_iteration', None)
		if best_it is None:
			best_it = getattr(model, 'best_ntree_limit', None)
		return model, best_it, getattr(model, 'evals_result_', {})
	except TypeError:
		pass
	# Fallback to xgb.train
	dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_sample_weight)
	dval = xgb.DMatrix(X_val, label=y_val)
	xgb_params = {
		'max_depth': params.max_depth,
		'eta': params.learning_rate,
		'subsample': params.subsample,
		'colsample_bytree': params.colsample_bytree,
		'min_child_weight': params.min_child_weight,
		'alpha': params.reg_alpha,
		'lambda': params.reg_lambda,
		'gamma': params.gamma,
		'objective': 'binary:logistic',
		'eval_metric': 'auc',
		'tree_method': params.tree_method,
		'nthread': params.n_jobs if params.n_jobs is not None else -1,
		'verbosity': params.verbosity,
		'scale_pos_weight': spw,
	}
	booster = xgb.train(
		params=xgb_params,
		dtrain=dtrain,
		num_boost_round=params.n_estimators,
		evals=[(dval, 'validation_0')],
		early_stopping_rounds=params.early_stopping_rounds,
		verbose_eval=False,
		evals_result=evals_result,
	)
	best_ntree = getattr(booster, 'best_ntree_limit', None)
	return BoosterModelAdapter(booster, best_ntree), best_ntree, evals_result


def fit_xgb_no_es(X: np.ndarray, y: np.ndarray, params: XGBParams, n_estimators: int, train_row_indices: Optional[np.ndarray] = None) -> Any:
	spw = compute_scale_pos_weight(y)
	# Try sklearn wrapper
	try:
		model = _build_sklearn_xgb(params, spw)
		model.set_params(n_estimators=int(n_estimators))
		# Optional time-decay weights
		sample_weight = None
		if params.time_decay_half_life is not None and train_row_indices is not None and len(train_row_indices) == len(y):
			max_idx = int(train_row_indices.max())
			age = (max_idx - train_row_indices).astype(float)
			decay = np.power(0.5, age / float(max(1, params.time_decay_half_life)))
			sample_weight = np.clip(decay, float(params.time_decay_min_weight), 1.0)
		model.fit(X, y, verbose=False, sample_weight=sample_weight)
		return model
	except TypeError:
		pass
	# Fallback to xgb.train
	# Compute weights if requested
	weights = None
	if params.time_decay_half_life is not None and train_row_indices is not None and len(train_row_indices) == len(y):
		max_idx = int(train_row_indices.max())
		age = (max_idx - train_row_indices).astype(float)
		decay = np.power(0.5, age / float(max(1, params.time_decay_half_life)))
		weights = np.clip(decay, float(params.time_decay_min_weight), 1.0)
	dtrain = xgb.DMatrix(X, label=y, weight=weights)
	xgb_params = {
		'max_depth': params.max_depth,
		'eta': params.learning_rate,
		'subsample': params.subsample,
		'colsample_bytree': params.colsample_bytree,
		'min_child_weight': params.min_child_weight,
		'alpha': params.reg_alpha,
		'lambda': params.reg_lambda,
		'gamma': params.gamma,
		'objective': 'binary:logistic',
		'eval_metric': 'auc',
		'tree_method': params.tree_method,
		'nthread': params.n_jobs if params.n_jobs is not None else -1,
		'verbosity': params.verbosity,
		'scale_pos_weight': spw,
	}
	booster = xgb.train(
		params=xgb_params,
		dtrain=dtrain,
		num_boost_round=int(n_estimators),
		verbose_eval=False
	)
	return BoosterModelAdapter(booster, int(n_estimators))


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


def plot_training_vs_test_accuracy(model: XGBClassifier,
							scaler: StandardScaler,
							data_csv: str,
							plots_dir: str,
							test_start_date: str,
							target_cleanup_enabled: bool = False,
							target_horizon: int = 14,
							target_band: float = 0.005,
							keep_ambiguous: bool = False,
							threshold: float = 0.5,
							window_size: int = 50,
							verbose: bool = True) -> None:
	"""
	Plot rolling training vs test accuracy to visualize potential overfitting.
	"""
	if not verbose:
		return

	print("  📊 Generating training vs test accuracy comparison...")

	# Load and prepare data
	df = load_dataset(data_csv)
	if target_cleanup_enabled:
		df = apply_target_cleanup(df, horizon=int(target_horizon), band=float(target_band), drop_ambiguous=not bool(keep_ambiguous))
	X_df, y, feature_names = select_features_and_target(df)
	X = X_df.values.astype(np.float32)

	# Split by date
	test_start = pd.to_datetime(test_start_date)
	if 'Date' in df.columns:
		test_mask = df['Date'] >= test_start
		train_mask = ~test_mask
		train_dates = df['Date'][train_mask].values
		test_dates = df['Date'][test_mask].values
	else:
		split_idx = int(len(X) * 0.8)
		train_mask = np.arange(len(X)) < split_idx
		test_mask = ~train_mask
		train_dates = np.arange(train_mask.sum())
		test_dates = np.arange(test_mask.sum())

	X_train = X[train_mask]
	X_test = X[test_mask]
	y_train = y[train_mask]
	y_test = y[test_mask]

	# Scale using provided scaler (fit upstream on pre-holdout)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# Predictions
	train_prob = model.predict_proba(X_train_scaled)[:, 1]
	test_prob = model.predict_proba(X_test_scaled)[:, 1]
	train_bin = (train_prob >= threshold).astype(int)
	test_bin = (test_prob >= threshold).astype(int)

	# Rolling accuracy
	train_roll, train_roll_dates = [], []
	test_roll, test_roll_dates = [], []
	if len(train_bin) >= window_size:
		for i in range(window_size, len(train_bin)):
			train_roll.append(np.mean(train_bin[i-window_size:i] == y_train[i-window_size:i]))
			train_roll_dates.append(train_dates[i])
	if len(test_bin) >= window_size:
		for i in range(window_size, len(test_bin)):
			test_roll.append(np.mean(test_bin[i-window_size:i] == y_test[i-window_size:i]))
			test_roll_dates.append(test_dates[i])

	# Plot
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
	fig.suptitle(f'Training vs Test Accuracy ({window_size}-day rolling)', fontsize=16, fontweight='bold')

	if train_roll:
		ax1.plot(train_roll_dates, train_roll, color='tab:blue', label='Train', linewidth=2)
	if test_roll:
		ax1.plot(test_roll_dates, test_roll, color='tab:red', label='Test', linewidth=2)
	ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
	ax1.set_ylabel('Accuracy')
	ax1.grid(True, alpha=0.3)
	ax1.legend()

	# Add vertical line to show train/test split
	if 'Date' in df.columns:
		ax1.axvline(x=test_start, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Train/Test Split')
		ax1.legend()

	# Histogram comparison
	if train_roll:
		ax2.hist(train_roll, bins=20, alpha=0.6, density=True, label='Train', color='tab:blue')
	if test_roll:
		ax2.hist(test_roll, bins=20, alpha=0.6, density=True, label='Test', color='tab:red')
	ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
	ax2.set_xlabel('Accuracy')
	ax2.set_ylabel('Density')
	ax2.grid(True, alpha=0.3)
	ax2.legend()

	# Format x-axis dates if we have date data
	if 'Date' in df.columns and (train_roll_dates or test_roll_dates):
		# Get all dates to set proper range
		all_dates = []
		if train_roll_dates:
			all_dates.extend(train_roll_dates)
		if test_roll_dates:
			all_dates.extend(test_roll_dates)
		
		if all_dates:
			# Set x-axis formatting
			ax1.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))  # Every 2 years
			ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
			ax1.xaxis.set_minor_locator(plt.matplotlib.dates.YearLocator(1))  # Every year
			plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
			
			# Set x-axis limits to span full date range
			min_date = min(all_dates)
			max_date = max(all_dates)
			ax1.set_xlim(min_date, max_date)

	plt.tight_layout()
	os.makedirs(plots_dir, exist_ok=True)
	plot_path = os.path.join(plots_dir, 'xgb_training_vs_test_accuracy.png')
	plt.savefig(plot_path, dpi=300, bbox_inches='tight')
	print(f"  📈 Training vs test accuracy plot saved to: {plot_path}")
	plt.show()

def plot_test_predictions_vs_price(model: XGBClassifier,
						scaler: StandardScaler,
						data_csv: str,
						plots_dir: str,
						test_start_date: str,
						target_cleanup_enabled: bool = False,
						target_horizon: int = 14,
						target_band: float = 0.005,
						keep_ambiguous: bool = False,
						threshold: float = 0.5,
						verbose: bool = True) -> None:
	"""
	Create a plot showing XGBoost model predictions vs S&P 500 price for the test set (2022 onwards).
	"""
	if not verbose:
		return
		
	print("  📊 Generating test set predictions vs price visualization...")
	
	# Load and prepare data
	df = load_dataset(data_csv)
	if target_cleanup_enabled:
		df = apply_target_cleanup(df, horizon=int(target_horizon), band=float(target_band), drop_ambiguous=not bool(keep_ambiguous))
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
	ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Decision Threshold ({threshold:.2f})')
	ax2.set_title('XGBoost Prediction Probability (14-day forward)', fontweight='bold')
	ax2.set_ylabel('Probability', fontweight='bold')
	ax2.set_ylim(0, 1)
	ax2.grid(True, alpha=0.3)
	ax2.legend()
	
	# Plot 3: Predictions vs Actual (Binary)
	# Convert predictions to binary using selected threshold
	pred_binary = (predictions >= threshold).astype(int)
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
	ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
	ax1.set_title('Distribution of Prediction Probabilities (Test Set)')
	ax1.set_xlabel('Prediction Probability')
	ax1.set_ylabel('Frequency')
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	
	# Plot 2: Prediction confidence over time
	confidence = np.abs(predictions - threshold) * 2  # Convert to 0-1 confidence scale
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


def plot_training_history_xgb(fold_histories: List[Dict],
							val_metrics: List[Dict[str, float]],
							plots_dir: str,
							verbose: bool = True):
	if not verbose or not fold_histories:
		return
	print("  📊 Generating training history plots...")
	
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	fig.suptitle('XGBoost Training History - Cross-Validation', fontsize=16, fontweight='bold')
	
	# Plot 1: AUC over boosting rounds
	for i, history in enumerate(fold_histories):
		if 'validation_0' in history and 'auc' in history['validation_0']:
			rounds = len(history['validation_0']['auc'])
			axes[0].plot(range(rounds), history['validation_0']['auc'], label=f'Fold {i+1} Val AUC', alpha=0.8)
	axes[0].set_title('Validation AUC vs Boosting Rounds')
	axes[0].set_xlabel('Boosting Round')
	axes[0].set_ylabel('AUC')
	axes[0].legend()
	axes[0].grid(True, alpha=0.3)

	# Plot 2: Final CV metrics
	metrics_names = list(val_metrics[0].keys())
	avg_metrics = [np.mean([m[k] for m in val_metrics]) for k in metrics_names]
	bars = axes[1].bar(metrics_names, avg_metrics, alpha=0.7, color='skyblue', edgecolor='navy')
	axes[1].set_title('Average CV Metrics')
	axes[1].set_ylabel('Score')
	axes[1].tick_params(axis='x', rotation=45)
	axes[1].grid(True, alpha=0.3, axis='y')
	for bar, value in zip(bars, avg_metrics):
		axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
					f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

	# Plot 3: Individual fold performance
	fold_aucs = [m['auc'] for m in val_metrics]
	fold_f1s = [m['f1'] for m in val_metrics]
	fold_numbers = list(range(1, len(val_metrics) + 1))
	x = np.arange(len(fold_numbers))
	width = 0.35
	bars1 = axes[2].bar(x - width/2, fold_aucs, width, label='AUC', alpha=0.7, color='lightcoral')
	bars2 = axes[2].bar(x + width/2, fold_f1s, width, label='F1', alpha=0.7, color='lightgreen')
	axes[2].set_title('Individual Fold Performance')
	axes[2].set_xlabel('Fold')
	axes[2].set_ylabel('Score')
	axes[2].set_xticks(x)
	axes[2].set_xticklabels(fold_numbers)
	axes[2].legend()
	axes[2].grid(True, alpha=0.3, axis='y')

	plt.tight_layout()
	plot_path = os.path.join(plots_dir, 'xgb_training_history.png')
	plt.savefig(plot_path, dpi=300, bbox_inches='tight')
	print(f"  📈 Training history plot saved to: {plot_path}")
	plt.show()


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(obj: Dict[str, Any], path: str) -> None:
	with open(path, 'w') as f:
		json.dump(obj, f, indent=2, cls=NumpyEncoder)


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
	parser.add_argument('--calibrate', action='store_true', help='Enable Platt scaling using OOF predictions')
	parser.add_argument('--smoke', action='store_true', help='Run a quick, lightweight training (reduced folds/trees)')
	# Regularization and weighting controls
	parser.add_argument('--time_decay_half_life', type=int, default=0, help='Half-life (in rows) for time-decay sample weighting (0=disabled)')
	parser.add_argument('--time_decay_min_weight', type=float, default=0.05, help='Minimum floor weight for oldest samples [0,1]')
	parser.add_argument('--final_train_trailing_window', type=int, default=0, help='If >0, refit final model on only last N pre-holdout rows')
	parser.add_argument('--auto_time_decay_years', type=int, default=3, help='If >0 and half-life not set, auto-apply time decay with half-life ~= years*252 rows')
	parser.add_argument('--final_train_trailing_years', type=int, default=0, help='If >0 and trailing_window not set, refit final model on last years*252 rows')
	# Target-side cleanup options
	parser.add_argument('--target_cleanup', action='store_true', help='Recompute target via forward return with neutral band and drop ambiguous rows')
	parser.add_argument('--target_horizon', type=int, default=14, help='Forward horizon in days for target construction')
	parser.add_argument('--target_band', type=float, default=0.005, help='Neutral band for forward return; |ret| <= band is ambiguous')
	parser.add_argument('--keep_ambiguous', action='store_true', help='Keep rows within the neutral band instead of dropping')
	# Threshold optimization controls
	parser.add_argument('--threshold_metric', type=str, default='f1', choices=['f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall'], help='Metric to optimize threshold on')
	parser.add_argument('--threshold_range_min', type=float, default=0.1, help='Minimum threshold value for optimization')
	parser.add_argument('--threshold_range_max', type=float, default=0.9, help='Maximum threshold value for optimization')
	parser.add_argument('--threshold_n_tests', type=int, default=50, help='Number of threshold values to test during optimization')
	args = parser.parse_args()

	set_seed(args.seed)
	ensure_dirs(args.models_dir, args.plots_dir)

	df = load_dataset(args.data)
	if args.target_cleanup:
		df = apply_target_cleanup(df, horizon=int(args.target_horizon), band=float(args.target_band), drop_ambiguous=not bool(args.keep_ambiguous))
	X_df, y, feature_names = select_features_and_target(df)
	X = X_df.values.astype(np.float32)

	# Check for presence of new engineered features and log
	required_features = [
		'Sentiment_Score', 'Headline_Count',
		'GDP_QoQ_Pct_Change', 'Unemployment_MoM_Change', 'Interest_MoM_Change', 'Inflation_MoM_Change',
		'Price_vs_MA30', 'RSI_Overbought', 'RSI_Oversold', 'MACD_Crossover'
	]
	present = [f for f in required_features if f in feature_names]
	missing = [f for f in required_features if f not in feature_names]
	if args.verbose:
		print(f"Detected {len(feature_names)} features. Using all numeric features by default.")
		if present:
			print(f"New engineered features present: {present}")
		if missing:
			print(f"Warning: The following expected features are missing: {missing}")
	
	params = XGBParams()
	params.device = args.device
	params.use_scaler = bool(args.use_scaler)
	params.calibrate = bool(args.calibrate)
	# Time-decay controls
	params.time_decay_half_life = int(args.time_decay_half_life) if int(args.time_decay_half_life) > 0 else None
	params.time_decay_min_weight = float(args.time_decay_min_weight)

	# Auto-apply time decay if not explicitly provided
	if params.time_decay_half_life is None and int(args.auto_time_decay_years) > 0:
		rows_per_year = 252  # approximate trading days per year
		params.time_decay_half_life = int(rows_per_year * int(args.auto_time_decay_years))
		if args.verbose:
			print(f"Auto time-decay enabled: half-life={params.time_decay_half_life} rows (~{int(args.auto_time_decay_years)} years), min_weight={params.time_decay_min_weight}")

	# Smoke mode adjustments for faster validation
	if args.smoke:
		params.n_estimators = 150
		params.cv_splits = 3
		params.val_window = 120
		params.min_train_window = 240
		params.early_stopping_rounds = 30
		params.learning_rate = max(params.learning_rate, 0.03)
		if args.verbose:
			print("Smoke mode enabled: reduced trees/folds and smaller windows for quick validation.")

	# Enforce a purged gap at least equal to the target horizon to avoid label leakage
	# across train/validation boundaries in time-series CV (Purged Walk-Forward CV).
	# This is critical when targets depend on future windows of length `target_horizon`.
	params.gap = int(max(params.gap, int(args.target_horizon)))
	if args.verbose:
		print(f"Using purged gap={params.gap} (>= target_horizon={int(args.target_horizon)})")

	if args.verbose:
		print(f"Rows: {X.shape[0]}, Features: {X.shape[1]}")
	
	# Optional randomized search to improve base hyperparameters (time-series aware)
	if args.search_trials and args.search_trials > 0:
		best_cfg, best_metrics = random_search_xgb(X, y, feature_names, params, trials=args.search_trials, verbose=args.verbose)
		if args.verbose:
			print(f"Best search CV AUC={best_metrics.get('auc', float('nan')):.4f}, F1={best_metrics.get('f1', float('nan')):.4f}")
		params = best_cfg

	train_metrics, val_metrics, artifacts, best_thr_cv, best_model_cv, best_scaler_cv = train_xgb_cv(
		X=X,
		y=y,
		feature_names=feature_names,
		params=params,
		verbose=args.verbose,
		threshold_metric=args.threshold_metric,
		threshold_range=(args.threshold_range_min, args.threshold_range_max),
		threshold_n_tests=args.threshold_n_tests
	)

	# Derive OOF predictions for robust threshold selection and optional calibration
	oof_prob_list = artifacts.get('oof_prob', [])
	oof_true_list = artifacts.get('oof_true', [])
	oof_mask = [p is not None and t is not None for p, t in zip(oof_prob_list, oof_true_list)]

	# Use the optimized threshold from CV process
	chosen_thr = artifacts.get('optimal_threshold', 0.5)
	optimal_threshold_metrics = artifacts.get('optimal_threshold_metrics', {})
	
	if args.verbose:
		print(f"Using optimized threshold: {chosen_thr:.3f}")
		print(f"Optimized threshold metrics: F1={optimal_threshold_metrics.get('f1', 0.0):.4f}, "
			  f"Accuracy={optimal_threshold_metrics.get('accuracy', 0.0):.4f}, "
			  f"Precision={optimal_threshold_metrics.get('precision', 0.0):.4f}, "
			  f"Recall={optimal_threshold_metrics.get('recall', 0.0):.4f}")

	calibrator = None
	if params.calibrate and any(oof_mask):
		# Platt scaling: logistic regression on OOF probabilities
		oof_prob = np.array([oof_prob_list[i] for i in range(len(oof_prob_list)) if oof_mask[i]], dtype=np.float32)
		oof_true = np.array([oof_true_list[i] for i in range(len(oof_true_list)) if oof_mask[i]], dtype=int)
		calibrator = LogisticRegression(max_iter=1000)
		calibrator.fit(oof_prob.reshape(-1, 1), oof_true)
		if args.verbose:
			print("Calibration enabled: Platt scaling fitted on OOF predictions.")

	# Build splits once for pre-holdout training and holdout selection
	splits = time_series_cv_indices(
		n_rows=X.shape[0],
		n_splits=params.cv_splits,
		val_window=params.val_window,
		gap=params.gap,
		min_train_window=params.min_train_window
	)
	last_train_idx, last_val_idx = splits[-1]
	last_train_end = last_train_idx[-1] + 1
	last_val_end = last_val_idx[-1] + 1
	holdout_start = last_val_end + params.gap
	holdout_idx = np.arange(holdout_start, X.shape[0]) if holdout_start < X.shape[0] else np.array([], dtype=int)

	# Train final model on all pre-holdout data using early stopping window from the last CV fold
	final_model = None
	final_scaler = None
	best_n_estimators = None
	if last_val_idx.size > 0:
		# Scale using only pre-holdout data up to the start of the ES validation set
		scaler = StandardScaler() if params.use_scaler else _IdentityScaler()
		scaler.fit(X[last_train_idx])
		X_train_es = scaler.transform(X[last_train_idx])
		X_val_es = scaler.transform(X[last_val_idx])
		y_train_es = y[last_train_idx]
		y_val_es = y[last_val_idx]

		_, best_n, _ = fit_xgb_with_es_compat(X_train_es, y_train_es, X_val_es, y_val_es, params, train_row_indices=last_train_idx)
		if best_n is None:
			best_n = params.n_estimators
		# Refit on full pre-holdout data using the best number of trees
		full_idx = np.concatenate([last_train_idx, last_val_idx])
		full_idx = np.unique(full_idx)
		# Optionally restrict to a trailing window to reduce reliance on very old regimes
		# If years-based arg is provided and window not set, convert to rows (~252 trading days per year)
		if int(args.final_train_trailing_window) <= 0 and int(args.final_train_trailing_years) > 0:
			args.final_train_trailing_window = int(252 * int(args.final_train_trailing_years))
		if int(args.final_train_trailing_window) > 0 and len(full_idx) > int(args.final_train_trailing_window):
			full_idx = full_idx[-int(args.final_train_trailing_window):]
		# The scaler was already fit on the training portion, so we just transform the selected pre-holdout data
		X_full = scaler.transform(X[full_idx])
		y_full = y[full_idx]
		model_final = fit_xgb_no_es(X_full, y_full, params, n_estimators=int(best_n), train_row_indices=full_idx)
		if args.verbose:
			print(f"Final model trained with n_estimators={int(best_n)} on pre-holdout data size={len(full_idx)}")
		final_model = model_final
		final_scaler = scaler
		best_n_estimators = int(best_n)

	# Holdout backtest using the final pre-holdout model
	holdout_metrics: Dict[str, float] = {}
	preds_df: Optional[pd.DataFrame] = None
	if holdout_idx.size > 0 and final_model is not None and final_scaler is not None:
		X_holdout = final_scaler.transform(X[holdout_idx])
		y_holdout = y[holdout_idx]
		y_prob_holdout = final_model.predict_proba(X_holdout)[:, 1]
		if calibrator is not None:
			y_prob_holdout = calibrator.predict_proba(y_prob_holdout.reshape(-1, 1))[:, 1]
		holdout_metrics = evaluate(y_holdout, y_prob_holdout, threshold=chosen_thr)

		preds_df = pd.DataFrame({
			'Date': df['Date'].iloc[holdout_idx].values if 'Date' in df.columns else holdout_idx,
			'prob': y_prob_holdout,
			'pred': (y_prob_holdout >= chosen_thr).astype(int),
			'target': y_holdout,
		})

	# Save artifacts
	model_tag = f"xgb_best"
	report = {
		'cv': val_metrics,
		'best_threshold_cv': best_thr_cv,
		'oof_threshold': chosen_thr,
		'threshold_optimization': {
			'optimal_threshold': float(chosen_thr),
			'optimization_metric': args.threshold_metric,
			'optimized_metrics': optimal_threshold_metrics,
		},
		'holdout': holdout_metrics,
		'params': asdict(params),
		'feature_names': feature_names,
		'best_n_estimators_pre_holdout': int(best_n_estimators) if best_n_estimators is not None else None,
		'calibration': {
			'enabled': bool(calibrator is not None),
			'coef': calibrator.coef_.ravel().tolist() if calibrator is not None else None,
			'intercept': calibrator.intercept_.ravel().tolist() if calibrator is not None else None,
		},
	}
	save_json(report, os.path.join(args.models_dir, f"{model_tag}_report.json"))

	# Generate training history plots
	fold_histories = artifacts.get('fold_histories', [])
	cv_auc_per_fold = artifacts.get('cv_val_auc_per_fold', [])
	cv_f1_per_fold = artifacts.get('cv_val_f1_per_fold', [])
	if cv_auc_per_fold and cv_f1_per_fold:
		fold_metrics_for_plot = [{'auc': a, 'f1': f} for a, f in zip(cv_auc_per_fold, cv_f1_per_fold)]
		plot_training_history_xgb(fold_histories, fold_metrics_for_plot, args.plots_dir, args.verbose)

	# Save model and plots using the final model (not a CV fold model)
	if final_model is not None:
		final_model.save_model(os.path.join(args.models_dir, f"{model_tag}.json"))
		feature_importance_plot(final_model, feature_names, os.path.join(args.plots_dir, 'xgb_feature_importance.png'))
		plot_test_predictions_vs_price(
			model=final_model,
			scaler=final_scaler,
			data_csv=args.data,
			plots_dir=args.plots_dir,
			test_start_date=args.test_start_date,
			target_cleanup_enabled=bool(args.target_cleanup),
			target_horizon=int(args.target_horizon),
			target_band=float(args.target_band),
			keep_ambiguous=bool(args.keep_ambiguous),
			threshold=chosen_thr,
			verbose=args.verbose
		)
		plot_training_vs_test_accuracy(
			model=final_model,
			scaler=final_scaler,
			data_csv=args.data,
			plots_dir=args.plots_dir,
			test_start_date=args.test_start_date,
			target_cleanup_enabled=bool(args.target_cleanup),
			target_horizon=int(args.target_horizon),
			target_band=float(args.target_band),
			keep_ambiguous=bool(args.keep_ambiguous),
			threshold=chosen_thr,
			verbose=args.verbose
		)

	if preds_df is not None:
		preds_df.to_csv(os.path.join(args.models_dir, f"{model_tag}_holdout_predictions.csv"), index=False)

	if args.verbose:
		print("Done. Artifacts saved.")


if __name__ == '__main__':
	main()


