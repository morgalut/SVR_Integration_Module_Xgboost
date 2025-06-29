import joblib
import numpy as np
import scipy.sparse as sp
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

def load_test_data(processed_dir: Path):
    """Load preprocessed features and labels."""
    X_test = sp.load_npz(processed_dir / "test_X.npz").astype(np.float32)
    y_test = np.load(processed_dir / "test_y.npy").astype(np.float32)
    return X_test, y_test

def batch_predict(model, X, batch_size=1000, desc="Predicting"):
    preds = []
    n_batches = int(np.ceil(X.shape[0] / batch_size))
    is_native_booster = isinstance(model, xgb.Booster)

    for i in tqdm(range(0, X.shape[0], batch_size), total=n_batches, desc=desc):
        batch = X[i:i + batch_size]
        if is_native_booster:
            batch = xgb.DMatrix(batch)
        preds.append(model.predict(batch))
    return np.concatenate(preds)

def evaluate(y_true, y_pred, label="Model"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label:>10} | R²={r2:.4f} | MAE={mae:.4f} | RMSE={np.sqrt(mse):.4f}")
    return {"R2": r2, "MAE": mae, "RMSE": np.sqrt(mse)}

def main():
    processed_dir = Path("processed_out")      # change if needed
    model_dir = Path("results_out")            # trained model dir
    output_csv = model_dir / "test_predictions.csv"

    print("🔁 Loading test data...")
    X_test, y_test = load_test_data(processed_dir)

    print("📦 Loading models...")
    xgb_model = joblib.load(model_dir / "xgboost_model.joblib")
    svr_model = joblib.load(model_dir / "svr_model.joblib")
    meta_model = joblib.load(model_dir / "meta_model.joblib")

    print("📊 Running predictions...")
    xgb_preds = batch_predict(xgb_model, X_test, desc="XGBoost")
    svr_preds = batch_predict(svr_model, X_test, desc="SVR")

    blend_input = np.column_stack([xgb_preds, svr_preds])
    blended_preds = meta_model.predict(blend_input)

    print("\n📈 Evaluation:")
    evaluate(y_test, xgb_preds, "XGBoost")
    evaluate(y_test, svr_preds, "SVR")
    evaluate(y_test, blended_preds, "Ensemble")

    # Optionally save
    try:
        import pandas as pd
        df = pd.DataFrame({
            "y_true": y_test,
            "xgboost_pred": xgb_preds,
            "svr_pred": svr_preds,
            "ensemble_pred": blended_preds
        })
        df.to_csv(output_csv, index=False)
        print(f"✅ Predictions saved to: {output_csv}")
    except ImportError:
        print("⚠️ pandas not installed, skipping CSV export")

if __name__ == "__main__":
    main()
