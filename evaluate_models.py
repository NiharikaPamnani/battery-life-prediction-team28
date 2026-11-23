"""Compare all project regressors on the shared, deterministic validation split."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from app.model import FEATURES, BatteryNet
from train import clean_data, normalisation, split_frame


DATA_PATH = Path("data/team_28_train.csv")
MODEL_DIR = Path("model_comparison")


def score(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
    return {
        "validation_mae": float(mean_absolute_error(y_true, y_predicted)),
        "validation_rmse": float(np.sqrt(mean_squared_error(y_true, y_predicted))),
        "validation_r2": float(r2_score(y_true, y_predicted)),
    }


def inference_ms_per_row(predict, example: np.ndarray, repeats: int = 500) -> float:
    sample = example.reshape(1, -1)
    for _ in range(10):
        predict(sample)
    started = time.perf_counter()
    for _ in range(repeats):
        predict(sample)
    return (time.perf_counter() - started) * 1000 / repeats


def evaluate_tree_candidates(model_name, candidates, x_train, y_train, x_val, y_val):
    results = []
    for parameters in candidates:
        model = model_name(**parameters)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - started
        predictions = model.predict(x_val)
        results.append({
            "model": model,
            "parameters": parameters,
            "training_seconds": training_seconds,
            "predictions": predictions,
            "mae": mean_absolute_error(y_val, predictions),
        })
    return min(results, key=lambda result: result["mae"]), sum(
        result["training_seconds"] for result in results
    )


def save_comparison_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    chart_specs = (
        ("validation_mae", "Validation MAE (days)", "mae.png"),
        ("validation_rmse", "Validation RMSE (days)", "rmse.png"),
        ("validation_r2", "Validation R²", "r2.png"),
    )
    for metric, label, filename in chart_specs:
        ordered = metrics.sort_values(metric, ascending=(metric != "validation_r2"))
        figure, axis = plt.subplots(figsize=(8, 4.5))
        bars = axis.bar(ordered["model"], ordered[metric], color=plt.cm.viridis(np.linspace(0.18, 0.86, len(ordered))))
        axis.set_ylabel(label)
        axis.set_title(f"Model comparison: {label}")
        axis.tick_params(axis="x", rotation=18)
        axis.bar_label(bars, fmt="%.2f", padding=3)
        figure.tight_layout()
        figure.savefig(plots_dir / filename, dpi=160)
        plt.close(figure)

    model_names = [name for name in predictions if name not in ("row_index", "actual_days_left")]
    actual = predictions["actual_days_left"].to_numpy()
    lower = min(actual.min(), *(predictions[name].min() for name in model_names))
    upper = max(actual.max(), *(predictions[name].max() for name in model_names))
    figure, axis = plt.subplots(figsize=(8, 6))
    for color, name in zip(plt.cm.tab10.colors, model_names):
        axis.scatter(actual, predictions[name], label=name, s=12, alpha=0.35, color=color)
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", label="perfect prediction")
    axis.set(xlabel="Actual days left", ylabel="Predicted days left", title="Validation: actual vs predicted")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for color, name in zip(plt.cm.tab10.colors, model_names):
        residual = predictions[name].to_numpy() - actual
        axis.hist(residual, bins=35, alpha=0.38, label=name, color=color)
    axis.axvline(0, color="black", linestyle="--")
    axis.set(xlabel="Prediction error (days)", ylabel="Validation rows", title="Validation residual distributions")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots_dir / "residuals.png", dpi=160)
    plt.close(figure)


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    cleaned, glitch_counts = clean_data(raw)
    training, validation = split_frame(cleaned)
    x_train = training[FEATURES].to_numpy(dtype=np.float32)
    y_train = training["days_left"].to_numpy(dtype=np.float32)
    x_val = validation[FEATURES].to_numpy(dtype=np.float32)
    y_val = validation["days_left"].to_numpy(dtype=np.float32)

    metrics_rows = []
    predictions = {"row_index": validation.index.to_numpy(), "actual_days_left": y_val}

    def add_result(name, hyperparameters, train_seconds, predict, y_hat):
        metrics_rows.append({
            "model": name,
            "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
            "training_time_seconds": float(train_seconds),
            "inference_ms_per_sample": inference_ms_per_row(predict, x_val[0]),
            **score(y_val, y_hat),
        })
        predictions[name] = y_hat

    mean_started = time.perf_counter()
    mean_value = float(y_train.mean())
    mean_training_seconds = time.perf_counter() - mean_started
    mean_predict = lambda rows: np.full(len(rows), mean_value, dtype=np.float32)
    add_result(
        "Mean baseline",
        {"prediction": "training-target mean"},
        mean_training_seconds,
        mean_predict,
        mean_predict(x_val),
    )

    linear_started = time.perf_counter()
    design_train = np.column_stack((x_train, np.ones(len(x_train), dtype=np.float32)))
    design_val = np.column_stack((x_val, np.ones(len(x_val), dtype=np.float32)))
    linear_coefficients, _, _, _ = np.linalg.lstsq(design_train, y_train, rcond=None)
    linear_training_seconds = time.perf_counter() - linear_started
    linear_predict = lambda rows: np.column_stack(
        (rows, np.ones(len(rows), dtype=np.float32))
    ) @ linear_coefficients
    add_result(
        "Linear regression",
        {"solver": "numpy.linalg.lstsq", "intercept": True},
        linear_training_seconds,
        linear_predict,
        linear_predict(x_val),
    )

    with open("model/config.json", encoding="utf-8") as file:
        configuration = json.load(file)
    network = BatteryNet(tuple(configuration["hidden"]))
    network.load_state_dict(torch.load("model/model.pt", map_location="cpu", weights_only=True))
    network.eval()
    mean, scale = np.asarray(configuration["mu"], np.float32), np.asarray(configuration["sd"], np.float32)

    def mlp_predict(rows):
        with torch.no_grad():
            return network(torch.from_numpy((rows - mean) / scale)).numpy()

    mlp_train_prediction = mlp_predict(x_train)
    mlp_val_prediction = mlp_predict(x_val)
    add_result(
        "Neural network",
        {"hidden": configuration["hidden"], "normalisation": "training mean/std"},
        configuration["training_time_seconds"],
        mlp_predict,
        mlp_val_prediction,
    )
    metrics_rows[-1]["training_mae"] = float(mean_absolute_error(y_train, mlp_train_prediction))

    random_forest_candidates = [
        {
            "n_estimators": trees,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "random_state": 0,
            "n_jobs": 1,
        }
        for trees in (100, 300)
        for depth in (None, 12)
        for leaf in (1, 2)
    ]
    best_forest, forest_search_seconds = evaluate_tree_candidates(
        RandomForestRegressor, random_forest_candidates, x_train, y_train, x_val, y_val
    )
    forest_predict = best_forest["model"].predict
    add_result(
        "Random Forest",
        best_forest["parameters"],
        best_forest["training_seconds"],
        forest_predict,
        best_forest["predictions"],
    )
    metrics_rows[-1]["hyperparameter_search_seconds"] = forest_search_seconds

    xgboost_candidates = [
        {
            "n_estimators": trees,
            "max_depth": depth,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 0,
            "n_jobs": 1,
            "verbosity": 0,
        }
        for trees in (200, 500)
        for depth in (3, 5)
    ]
    best_boosted, xgboost_search_seconds = evaluate_tree_candidates(
        XGBRegressor, xgboost_candidates, x_train, y_train, x_val, y_val
    )
    boosted_predict = best_boosted["model"].predict
    add_result(
        "XGBoost",
        best_boosted["parameters"],
        best_boosted["training_seconds"],
        boosted_predict,
        best_boosted["predictions"],
    )
    metrics_rows[-1]["hyperparameter_search_seconds"] = xgboost_search_seconds

    MODEL_DIR.mkdir(exist_ok=True)
    model_dir = MODEL_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(mean_value, model_dir / "mean_baseline.joblib")
    np.savez(model_dir / "linear_regression.npz", coefficients=linear_coefficients)
    joblib.dump(best_forest["model"], model_dir / "random_forest.joblib")
    best_boosted["model"].save_model(model_dir / "xgboost.json")
    artifact_paths = {
        "Mean baseline": model_dir / "mean_baseline.joblib",
        "Linear regression": model_dir / "linear_regression.npz",
        "Neural network": Path("model/model.pt"),
        "Random Forest": model_dir / "random_forest.joblib",
        "XGBoost": model_dir / "xgboost.json",
    }
    for row in metrics_rows:
        artifact = artifact_paths[row["model"]]
        row["model_artifact"] = artifact.as_posix()
        row["artifact_file_bytes"] = artifact.stat().st_size
    with (MODEL_DIR / "baselines.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"mean_prediction": mean_value, "linear_coefficients": [float(value) for value in linear_coefficients]},
            file,
            indent=2,
        )
        file.write("\n")

    metrics_frame = pd.DataFrame(metrics_rows).sort_values("validation_mae")
    metrics_frame.to_csv(MODEL_DIR / "metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(predictions).to_csv(MODEL_DIR / "validation_predictions.csv", index=False, float_format="%.6f")
    summary = {
        "team": 28,
        "rows_raw": len(raw),
        "rows_cleaned": len(cleaned),
        "validation_rows": len(validation),
        "glitch_counts": glitch_counts,
        "selection_metric": "validation_mae",
        "deployed_model": "Neural network; retained for small edge footprint",
        "best_validation_model": metrics_frame.iloc[0]["model"],
    }
    with (MODEL_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    save_comparison_plots(metrics_frame, pd.DataFrame(predictions), MODEL_DIR)
    print(metrics_frame.to_string(index=False))
    print(f"\nBest validation MAE: {summary['best_validation_model']}")
    print(f"Cleaned {len(raw) - len(cleaned)} of {len(raw)} rows: {glitch_counts}")


if __name__ == "__main__":
    main()
