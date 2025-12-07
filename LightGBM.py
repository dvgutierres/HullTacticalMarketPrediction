import os
import argparse
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

import lightgbm as lgb

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
    # Define filepath
    filepath = os.path.join("Data", "train.csv")

    # Read data
    df = pd.read_csv(filepath, index_col="date_id")

    # Define target column
    target_column = "forward_returns"

    # Print path and shape
    print(f"[INFO] Loaded data from {filepath} with shape {df.shape}")

    # Build df without unwanted columns (features only)
    clean_df = df.drop(columns=["forward_returns", "risk_free_rate", "market_forward_excess_returns"])
    print(f"[INFO] Data shape after dropping unwanted columns: {clean_df.shape}")

    # Time-ordered index
    clean_df = clean_df.sort_index()
    y = df[target_column].sort_index()

    n = len(clean_df)
    print(f"[INFO] Total rows: {n}")

    # -----------------------------
    # Walk-forward fold definition
    # -----------------------------
    # Example: 4 folds with 60/70/80/90% train cut points and 10% validation windows.
    cut_1 = int(n * 0.6)
    cut_2 = int(n * 0.7)
    cut_3 = int(n * 0.8)
    cut_4 = int(n * 0.9)

    folds = [
        ("fold1", 0,      cut_1, cut_1, cut_2),  # train [0:cut_1), valid [cut_1:cut_2)
        ("fold2", 0,      cut_2, cut_2, cut_3),  # train [0:cut_2), valid [cut_2:cut_3)
        ("fold3", 0,      cut_3, cut_3, cut_4),  # train [0:cut_3), valid [cut_3:cut_4)
        ("fold4", 0,      cut_4, cut_4, n),      # train [0:cut_4), valid [cut_4:n)
    ]

    # Hyperparameters
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
    num_boost_round = 200

    all_metrics = []
    all_metric_scores = []

    last_model = None
    last_X_valid = None
    last_y_valid = None
    last_preds_valid = None

    for fold_name, tr_start, tr_end, val_start, val_end in folds:
        print(f"\n[INFO] ===== {fold_name} =====")
        print(f"[INFO] Train idx: [{tr_start}:{tr_end})  Valid idx: [{val_start}:{val_end})")

        X_train = clean_df.iloc[tr_start:tr_end]
        y_train = y.iloc[tr_start:tr_end]
        X_valid = clean_df.iloc[val_start:val_end]
        y_valid = y.iloc[val_start:val_end]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        print("[INFO] Starting model training...")
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=100)]
        )
        print("[INFO] Model training completed.")
        print(f"[INFO] Best iteration ({fold_name}): {model.best_iteration}")

        importance = model.feature_importance(importance_type="gain")
        features = X_train.columns

        fi = pd.DataFrame({
            "feature": features,
            "importance": importance
        }).sort_values("importance", ascending=False)

        print(fi.head(20))
        
        # Predict on validation set (use best_iteration if available)
        preds_valid = model.predict(X_valid, num_iteration=model.best_iteration or None)
        preds_valid = np.clip(preds_valid, 0.0, 2.0)
        print("[INFO] Completed predictions on validation set.")

        # Evaluate standard metrics
        metrics = evaluate(y_valid, preds_valid)

        # Build solution slice for metric.score
        solution_df = df.sort_index().iloc[val_start:val_end][["forward_returns", "risk_free_rate"]].copy()
        metric_score = compute_competition_score_from_solution(solution_df, preds_valid)

        print(f"[RESULTS] {fold_name} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        if isinstance(metric_score, dict):
            for kk, vv in metric_score.items():
                print(f"  {kk}: {vv}")
        else:
            print(f"  metric_score (competition): {metric_score:.6f}")

        all_metrics.append(metrics)
        all_metric_scores.append(metric_score if isinstance(metric_score, float) else np.nan)

        # Keep last fold artifacts to optionally save
        last_model = model
        last_X_valid = X_valid
        last_y_valid = y_valid
        last_preds_valid = preds_valid

    # -----------------------------
    # Aggregate results across folds
    # -----------------------------
    print("\n[INFO] ===== Aggregate walk-forward results =====")
    # Aggregate MAE / RMSE / Pearson
    mae_list = [m["mae"] for m in all_metrics]
    rmse_list = [m["rmse"] for m in all_metrics]
    pearson_list = [m["pearson"] for m in all_metrics]

    print(f"  MAE mean:     {np.mean(mae_list):.6f}  (per-fold: {[f'{x:.6f}' for x in mae_list]})")
    print(f"  RMSE mean:    {np.mean(rmse_list):.6f}  (per-fold: {[f'{x:.6f}' for x in rmse_list]})")
    print(f"  Pearson mean: {np.mean(pearson_list):.6f}  (per-fold: {[f'{x:.66f}' for x in pearson_list]})")

    # Aggregate competition metric (ignore NaNs from fallback)
    metric_array = np.array(all_metric_scores, dtype=float)
    metric_mean = np.nanmean(metric_array)
    print(f"  metric_score mean (competition): {metric_mean:.6f}  (per-fold: {[f'{x:.6f}' for x in metric_array]})")

    # Optionally save last fold model & preds
    if last_model is not None and last_X_valid is not None:
        save_outputs(args.output_dir, last_model, last_X_valid, last_y_valid, last_preds_valid,
                     {"mae": np.mean(mae_list), "rmse": np.mean(rmse_list), "pearson": np.mean(pearson_list)},
                     metric_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM MVP for Hull Tactical - LightGBM.py")
    parser.add_argument("--output-dir", type=str, default="./Output", help="Directory to store outputs")
    args = parser.parse_args()
    main(args)