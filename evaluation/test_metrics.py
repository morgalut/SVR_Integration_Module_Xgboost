import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

log = logging.getLogger("ctr_eval")

def evaluate_and_plot(y_true, y_pred, name: str, save_dir: Path):
    """
    Compute evaluation metrics and save a prediction vs actual plot.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        name (str): Name of the model being evaluated (used in logs and plot filename).
        save_dir (Path): Directory to save the plot.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # ✅ FIXED
    r2 = r2_score(y_true, y_pred)

    # Log metrics
    log.info(f"{name:<9} | MAE = {mae:.4f} | RMSE = {rmse:.4f} | R² = {r2:.4f}")

    # Plot: prediction vs actual
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors='k', linewidth=0.3)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
            "--", color="black", label="Perfect Prediction")
    ax.set_title(f"{name} Prediction vs Actual\nR²={r2:.3f} | MAE={mae:.4f} | RMSE={rmse:.4f}")
    ax.set_xlabel("Actual CTR")
    ax.set_ylabel("Predicted CTR")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # Save plot
    plot_path = save_dir / f"{name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.close()

    return mae, rmse, r2
