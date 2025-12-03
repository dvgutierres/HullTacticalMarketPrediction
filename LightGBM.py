"""
LightGBM.py

Minimal LightGBM MVP for Hull Tactical (predict forward_returns).
This version calls the official competition metric.score(...) when metric.py is present.
Usage:
    python LightGBM.py --data-path ./Data/train.csv --output-dir ./Output
"""

import os
import argparse
import joblib
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr

import lightgbm as lgb

def time_train_valid_split(X: pd.DataFrame, y: pd.Series, valid_frac: float = 0.2):
    n = len(X)
    split = int(np.floor(n * (1.0 - valid_frac)))
    X_train = X.iloc[:split]
    X_valid = X.iloc[split:]
    y_train = y.iloc[:split]
    y_valid = y.iloc[split:]
    print(f"[INFO] Train rows: {len(X_train)}; Valid rows: {len(X_valid)}; split_index={split}")
    return X_train, X_valid, y_train, y_valid

def prepare_features(df: pd.DataFrame, target_col: str = "forward_returns") -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Returns X, y, index. Drops target_col and other derived targets from X.
    """
    df = df.sort_index()
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")
    y = df[target_col].copy()
    drop_cols = [target_col, "risk_free_rate", "market_forward_excess_returns"]
    features = [c for c in df.columns if c not in drop_cols]
    X = df[features].copy()
    return X, y, X.index

def simple_impute(X: pd.DataFrame, add_missing_indicators: bool = True) -> pd.DataFrame:
    """
    Median imputation + missing indicators (Option D).
    Returns an imputed DataFrame. Indicators are named <col>_missing (0/1).
    """
    X = X.copy()
    if add_missing_indicators:
        for col in X.columns:
            X[col + "_missing"] = X[col].isna().astype(int)

    # median impute numeric columns
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    # final fallback
    X = X.fillna(0.0)
    return X

def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
):
    if params is None:
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.02,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 50,
            "verbose": -1,
            "seed": 42,
        }

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    print("[INFO] Starting LightGBM training...")

    # callbacks for compatibility
    cbs = []
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        cbs.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
    cbs.append(lgb.log_evaluation(period=100))

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=cbs,
    )

    best_it = getattr(model, "best_iteration", None) or getattr(model, "best_iteration_", None)
    if best_it is not None:
        print(f"[INFO] Best iteration: {best_it}")
    else:
        print("[INFO] Best iteration: (not available from this LightGBM build)")

    return model

def evaluate(y_true: pd.Series, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    try:
        corr, _ = pearsonr(y_true, y_pred)
    except Exception:
        corr = np.nan
    return {"mae": mae, "rmse": rmse, "pearson": corr}

def compute_competition_score_from_solution(solution_df: pd.DataFrame, preds: np.ndarray):
    """
    Call the competition metric.score(solution_df, submission_df, None).
    solution_df must contain columns: forward_returns and risk_free_rate.
    submission_df must contain column 'prediction' and align with solution_df row-wise (index).
    """
    # build submission dataframe (same index)
    submission_df = pd.DataFrame({"prediction": preds}, index=solution_df.index)

    # try to import metric.py
    try:
        # direct import (if metric.py is in the same folder or installed)
        from metric import score as metric_score_fn
    except Exception:
        # attempt dynamic import from common locations
        import importlib.util, sys
        candidate_paths = [
            Path("./metric.py"),
            Path("./kaggle_competition_files/metric.py"),
            Path("/kaggle/input/hull-tactical-market-prediction/metric.py"),
            Path("/kaggle/input/metric.py"),
        ]
        metric_score_fn = None
        for p in candidate_paths:
            try:
                if p.exists():
                    spec = importlib.util.spec_from_file_location("metric_local", str(p))
                    metric_local = importlib.util.module_from_spec(spec)
                    sys.modules["metric_local"] = metric_local
                    spec.loader.exec_module(metric_local)  # type: ignore
                    if hasattr(metric_local, "score"):
                        metric_score_fn = metric_local.score
                        print(f"[INFO] Using metric.score from {p}")
                        break
            except Exception as e:
                print(f"[WARN] Could not import metric from {p}: {e}")
        if metric_score_fn is None:
            print("[WARN] metric.score not found in candidate paths.")
            metric_score_fn = None

    if metric_score_fn is None:
        # fallback: return dict with Pearson (for diagnostics)
        try:
            corr, _ = pearsonr(solution_df["forward_returns"].values, preds)
        except Exception:
            corr = float("nan")
        return {"pearson_fallback": float(corr), "note": "official metric.score not available"}
    else:
        # The metric expects a DataFrame solution that includes forward_returns and risk_free_rate
        # Ensure those columns exist:
        expected_cols = {"forward_returns", "risk_free_rate"}
        if not expected_cols.issubset(solution_df.columns):
            raise KeyError(f"solution_df must contain columns: {expected_cols}. Got: {solution_df.columns.tolist()}")

        # Call metric.score and return the numeric result
        try:
            # Some implementations expect plain DataFrames (not necessarily indices), but the metric you pasted uses columns directly.
            result = metric_score_fn(solution_df.reset_index(drop=True), submission_df.reset_index(drop=True), None)
            return float(result)
        except Exception as e:
            print(f"[ERROR] metric.score call failed: {e}")
            # fallback
            try:
                corr, _ = pearsonr(solution_df["forward_returns"].values, preds)
            except Exception:
                corr = float("nan")
            return {"pearson_fallback": float(corr), "note": f"metric.score failed: {e}"}

def save_outputs(output_dir: str, model, X_valid: pd.DataFrame, y_valid: pd.Series, preds: np.ndarray, metrics: dict, metric_score=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save model with joblib (if available)
    try:
        model_path = output_dir / "lgbm_mvp.joblib"
        joblib.dump(model, model_path)
        print(f"[INFO] Saved model to {model_path}")
    except Exception as e:
        # fallback to LightGBM native save
        fallback = output_dir / "lgbm_mvp.txt"
        try:
            model.save_model(str(fallback))
            print(f"[WARN] joblib save failed ({e}). Saved LightGBM model to {fallback}")
        except Exception as e2:
            print(f"[ERROR] Failed to save model with both joblib and save_model: {e2}")

    df_out = pd.DataFrame({"y_true": y_valid.values, "y_pred": preds}, index=y_valid.index)
    df_out.to_csv(output_dir / "valid_predictions.csv", index_label="date_id")
    print(f"[INFO] Saved validation predictions to {output_dir/'valid_predictions.csv'}")

    with open(output_dir / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        if metric_score is not None:
            f.write(f"metric_score: {metric_score}\n")
    print(f"[INFO] Saved metrics to {output_dir/'metrics.txt'}")

def main(args):
    #Define filepath
    filepath = os.path.join("Data", "train.csv")

    #Read data
    df = pd.read_csv(filepath, index_col="date_id")

    #Print path and shape
    print(f"[INFO] Loaded data from {filepath} with shape {df.shape}")

    #Build df without unwanted columns
    clean_df = df.drop(columns=["forward_returns", "risk_free_rate", "market_forward_excess_returns"])

    #Show new df
    print(f"[INFO] Data shape after dropping unwanted columns: {clean_df.shape}")

    X, y, idx = prepare_features(df, target_col=args.target)

    # Impute + missing indicators
    X = simple_impute(X, add_missing_indicators=True)

    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"[INFO] Dropping {len(const_cols)} constant columns.")
        X = X.drop(columns=const_cols)

    X_train, X_valid, y_train, y_valid = time_train_valid_split(X, y, valid_frac=args.valid_frac)

    model = train_lgbm(
        X_train,
        y_train,
        X_valid,
        y_valid,
        params=None,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
    )

    preds_valid = model.predict(X_valid, num_iteration=getattr(model, "best_iteration", None) or None)

    if args.min_pred is not None or args.max_pred is not None:
        lo = -np.inf if args.min_pred is None else args.min_pred
        hi = np.inf if args.max_pred is None else args.max_pred
        preds_valid = np.clip(preds_valid, lo, hi)

    # Evaluate standard metrics
    metrics = evaluate(y_valid, preds_valid)

    # Build the exact solution slice (must include forward_returns and risk_free_rate) for metric.score
    solution_df = df.loc[y_valid.index, ["forward_returns", "risk_free_rate"]].copy()

    metric_score = compute_competition_score_from_solution(solution_df, preds_valid)

    print("[RESULTS] Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Print metric_score: could be a float (official score) or a dict fallback
    if isinstance(metric_score, dict):
        # fallback info
        for kk, vv in metric_score.items():
            print(f"  {kk}: {vv}")
    else:
        print(f"  metric_score (competition): {metric_score:.6f}")

    save_outputs(args.output_dir, model, X_valid, y_valid, preds_valid, metrics, metric_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM MVP for Hull Tactical - LightGBM.py")
    parser.add_argument("--data-path", type=str, default="./Data/train.csv", help="Path to train.csv")
    parser.add_argument("--output-dir", type=str, default="./Output", help="Directory to store outputs")
    parser.add_argument("--target", type=str, default="forward_returns", help="Target column")
    parser.add_argument("--valid-frac", type=float, default=0.2, help="Fraction of data used for validation (time-based)")
    parser.add_argument("--num-boost-round", type=int, default=2000, help="LightGBM num_boost_round")
    parser.add_argument("--early-stopping", type=int, default=100, help="Early stopping rounds")
    parser.add_argument("--min-pred", type=float, default=None, help="Optional lower bound to clip predictions")
    parser.add_argument("--max-pred", type=float, default=None, help="Optional upper bound to clip predictions")
    args = parser.parse_args()

    main(args)