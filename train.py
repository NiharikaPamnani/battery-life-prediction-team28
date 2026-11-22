"""Clean team data, compare baselines, train a network, and save it for the API."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from app.model import FEATURES, FEATURE_RANGES, BatteryNet, model_metrics


def parse_hidden(value: str) -> tuple[int, ...]:
    try:
        hidden = tuple(int(width) for width in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden widths must be comma-separated integers") from exc
    if not hidden or any(width <= 0 for width in hidden):
        raise argparse.ArgumentTypeError("provide one or more positive hidden widths")
    return hidden


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = FEATURES + ["days_left"]
    values = frame[columns].to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(values).all(axis=1)
    reasons = {"non_finite": int(invalid.sum())}

    for feature in FEATURES:
        low, high = FEATURE_RANGES[feature]
        bad = (frame[feature] < low) | (frame[feature] > high)
        reasons[feature] = int(bad.sum())
        invalid |= bad.to_numpy()

    bad_target = ~np.isfinite(frame["days_left"].to_numpy(dtype=np.float64)) | (frame["days_left"] < 0)
    reasons["days_left"] = int(bad_target.sum())
    invalid |= bad_target
    reasons["total_rows_removed"] = int(invalid.sum())
    return frame.loc[~invalid].copy(), reasons


def split_data(frame: pd.DataFrame):
    rng = np.random.default_rng(0)
    indices = rng.permutation(len(frame))
    validation_count = len(frame) // 5
    validation = frame.iloc[indices[:validation_count]]
    training = frame.iloc[indices[validation_count:]]
    x_train = training[FEATURES].to_numpy(dtype=np.float32)
    y_train = training["days_left"].to_numpy(dtype=np.float32)
    x_val = validation[FEATURES].to_numpy(dtype=np.float32)
    y_val = validation["days_left"].to_numpy(dtype=np.float32)
    return x_train, y_train, x_val, y_val


def normalisation(x_train: np.ndarray, enabled: bool):
    if not enabled:
        return np.zeros(x_train.shape[1], dtype=np.float32), np.ones(x_train.shape[1], dtype=np.float32)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def mean_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - targets)))


def fit_network(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    hidden: tuple[int, ...],
    epochs: int,
    log_progress: bool = False,
):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    network = BatteryNet(hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)
    train_x = torch.from_numpy((x_train - mean) / scale)
    train_y = torch.from_numpy(y_train)
    val_x = torch.from_numpy((x_val - mean) / scale)

    for epoch in range(epochs):
        permutation = torch.randperm(len(train_x))
        for start in range(0, len(train_x), 128):
            batch = permutation[start:start + 128]
            loss = nn.functional.mse_loss(network(train_x[batch]), train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if log_progress and (epoch % 300 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                val_prediction = network(val_x).numpy()
            print(f"  epoch {epoch:4d}  val MAE {mean_absolute_error(val_prediction, y_val):.2f} days")

    with torch.no_grad():
        train_prediction = network(train_x).numpy()
        val_prediction = network(val_x).numpy()
    return network, mean_absolute_error(train_prediction, y_train), mean_absolute_error(val_prediction, y_val)


def plot_data(frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = FEATURES + ["days_left"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, column in zip(axes.flat, columns):
        axis.hist(frame[column], bins=35, color="#4472c4", edgecolor="white")
        axis.set(title=column, xlabel=column, ylabel="rows")
    figure.tight_layout()
    figure.savefig(output_dir / "column_histograms.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, feature in zip(axes.flat, FEATURES):
        axis.scatter(frame[feature], frame["days_left"], s=8, alpha=0.35)
        axis.set(title=f"days_left vs {feature}", xlabel=feature, ylabel="days_left")
    axes.flat[-1].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "target_vs_inputs.png", dpi=150)
    plt.close(figure)


def plot_predictions(actual: np.ndarray, predicted: np.ndarray, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(actual, predicted, s=12, alpha=0.5)
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axis.set(xlabel="true days_left", ylabel="predicted days_left", title="Linear baseline")
    figure.tight_layout()
    figure.savefig(output_dir / "linear_baseline.png", dpi=150)
    plt.close(figure)


def run_sweep(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    epochs: int,
    output_dir: Path,
) -> None:
    architectures = [(width,) for width in (4, 8, 16, 32, 64)]
    architectures += [(width, width) for width in (4, 8, 16, 32, 64)]
    rows = []
    for hidden in architectures:
        network, train_mae, val_mae = fit_network(
            x_train, y_train, x_val, y_val, mean, scale, hidden, epochs
        )
        metrics = model_metrics(network)
        rows.append({
            "hidden": ",".join(map(str, hidden)),
            "total_params": metrics["total_params"],
            "train_mae": train_mae,
            "val_mae": val_mae,
        })
        print(f"hidden={hidden!s:12s} params={metrics['total_params']:5d} "
              f"train MAE={train_mae:6.2f}  val MAE={val_mae:6.2f} days")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model_sweep.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for depth in (1, 2):
        selected = [row for row in rows if len(row["hidden"].split(",")) == depth]
        axis.plot(
            [row["total_params"] for row in selected],
            [row["val_mae"] for row in selected],
            marker="o",
            label=f"{depth} hidden layer{'s' if depth > 1 else ''}",
        )
    axis.set(xlabel="parameters", ylabel="validation MAE (days)", title="Network size sweep")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "model_sweep.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", type=int, required=True)
    parser.add_argument("--hidden", type=parse_hidden, default=(32, 32))
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--plots-dir", type=Path)
    parser.add_argument("--sweep", action="store_true", help="train the required width/depth comparison")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    data_path = Path("data") / f"team_{args.team:02d}_train.csv"
    if not data_path.is_file():
        parser.error(f"team data file not found: {data_path}")
    raw = pd.read_csv(data_path)
    expected = set(FEATURES + ["days_left"])
    missing = expected.difference(raw.columns)
    if missing:
        parser.error(f"training data is missing columns: {', '.join(sorted(missing))}")

    cleaned, glitch_counts = clean_data(raw)
    if len(cleaned) < 5:
        parser.error("not enough valid rows remain after removing sensor glitches")
    print(f"{len(raw)} rows; removed {glitch_counts['total_rows_removed']} glitch rows")
    for name, count in glitch_counts.items():
        if name != "total_rows_removed" and count:
            print(f"  {name}: {count}")

    if args.plots_dir:
        plot_data(raw, args.plots_dir)

    x_train, y_train, x_val, y_val = split_data(cleaned)
    mean, scale = normalisation(x_train, enabled=not args.no_normalise)
    baseline_mean = mean_absolute_error(np.full_like(y_val, y_train.mean()), y_val)
    design_train = np.column_stack((x_train, np.ones(len(x_train), dtype=np.float32)))
    design_val = np.column_stack((x_val, np.ones(len(x_val), dtype=np.float32)))
    coefficients, _, _, _ = np.linalg.lstsq(design_train, y_train, rcond=None)
    linear_prediction = design_val @ coefficients
    baseline_linear = mean_absolute_error(linear_prediction, y_val)
    print(f"baseline (predict the training mean): {baseline_mean:.2f} days")
    print(f"baseline (linear least squares):      {baseline_linear:.2f} days")

    if args.plots_dir:
        plot_predictions(y_val, linear_prediction, args.plots_dir)

    if args.sweep:
        if args.no_normalise:
            parser.error("--sweep requires normalisation; omit --no-normalise")
        run_sweep(x_train, y_train, x_val, y_val, mean, scale, args.epochs, args.plots_dir or Path("plots"))
        return

    network, train_mae, val_mae = fit_network(
        x_train, y_train, x_val, y_val, mean, scale, args.hidden, args.epochs, log_progress=True
    )
    print(f"network {args.hidden}: train MAE {train_mae:.2f} days; validation MAE {val_mae:.2f} days")

    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    torch.save(network.state_dict(), model_dir / "model.pt")
    configuration = {
        "team": args.team,
        "hidden": list(args.hidden),
        "mu": mean.tolist(),
        "sd": scale.tolist(),
        "normalised": not args.no_normalise,
        "features": FEATURES,
        "train_mae": round(train_mae, 2),
        "val_mae": round(val_mae, 2),
        "glitch_rows_removed": glitch_counts["total_rows_removed"],
    }
    with (model_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)
        file.write("\n")
    print(f"saved {model_dir / 'model.pt'} and {model_dir / 'config.json'}")


if __name__ == "__main__":
    main()
