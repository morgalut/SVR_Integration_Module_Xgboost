from __future__ import annotations
import argparse, logging, joblib, json, time, numpy as np, scipy.sparse as sp
import gc, subprocess, sys, os
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from xgboost_settings import train_xgboost_robust
from svr_settings import train_svr_incremental
from evaluation.test_metrics import evaluate_and_plot
import xgboost as xgb

try:
    import torch
    print("PyTorch CUDA Available:", torch.cuda.is_available())
    print("PyTorch GPU Count:", torch.cuda.device_count())
except ImportError:
    print("PyTorch not available (not required)")
except Exception as e:
    print(f"PyTorch CUDA check failed: {e}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("ctr_train")

_ps = None
try:
    import psutil
    _ps = psutil.Process()
except ImportError:
    psutil = None

def _mem():
    if psutil and _ps is not None:
        mb = _ps.memory_info().rss / 1024**2
        return f"{mb:,.0f} MB"
    return "n/a"

def _load(path: Path, pfx: str):
    return (sp.load_npz(path / f"{pfx}_X.npz").astype(np.float32),
            np.load(path / f"{pfx}_y.npy").astype(np.float32))

def batch_predict(model, X, batch_size=1000, desc="Predicting"):
    preds = []
    n_batches = int(np.ceil(X.shape[0] / batch_size))
    is_native_booster = isinstance(model, xgb.Booster)
    for i in tqdm(range(0, X.shape[0], batch_size), total=n_batches, desc=desc):
        batch = X[i:i+batch_size]
        if is_native_booster:
            batch = xgb.DMatrix(batch)
        preds.append(model.predict(batch))
    return np.concatenate(preds)

def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description="CTR Ensemble Training (Robust GPU/CPU)")
    ap.add_argument("--proc_dir", default="processed_out", help="Directory containing processed data")
    ap.add_argument("--device", default="cuda:0", help='Device for XGBoost')
    ap.add_argument("--svr_subsample", type=float, default=0.5, help="Fraction of training data for SVR")
    ap.add_argument("--force_cpu", action="store_true", help="Force CPU training for XGBoost")
    ap.add_argument("--output_dir", default="trained_models_out", help="Output directory for model artifacts")
    args = ap.parse_args()

    P = Path(args.proc_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagram_dir = out_dir / "diagram"
    diagram_dir.mkdir(parents=True, exist_ok=True)

    log.info("🚀 Starting CTR Ensemble Training (Robust)")
    log.info("Arguments: %s", vars(args))
    log.info("Environment:")
    log.info("  CUDA_VISIBLE_DEVICES: %s", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))

    if args.force_cpu:
        args.device = "cpu"
        log.info("🔧 Forced CPU mode enabled")

    log.info("📂 Loading training data...")
    Xtr, ytr = _load(P, "train")
    Xva, yva = _load(P, "eval")
    log.info("✅ Data loaded - Train: %s rows, Eval: %s rows, Features: %s",
             Xtr.shape[0], Xva.shape[0], Xtr.shape[1])
    log.info("Memory after data loading: %s", _mem())

    if args.svr_subsample < 1.0:
        n_svr_samples = int(Xtr.shape[0] * args.svr_subsample)
        keep_indices = np.random.default_rng(42).choice(
            Xtr.shape[0], size=n_svr_samples, replace=False
        )
        Xtr_svr, ytr_svr = Xtr[keep_indices], ytr[keep_indices]
        log.info("📊 SVR subsample: %s/%s rows (%.1f%%)",
                 n_svr_samples, Xtr.shape[0], args.svr_subsample * 100)
    else:
        Xtr_svr, ytr_svr = Xtr, ytr
        log.info("📊 SVR using full training set: %s rows", Xtr.shape[0])

    xgb, xgb_train_time, device_used = train_xgboost_robust(Xtr, ytr, Xva, yva, args.device)
    log.info("Memory after XGBoost training: %s", _mem())

    svr, svr_train_time = train_svr_incremental(Xtr_svr, ytr_svr)
    log.info("Memory after SVR training: %s", _mem())

    log.info("🔮 Generating meta-features...")
    xgb_train_preds = batch_predict(xgb, Xtr, desc="XGBoost train preds")
    svr_train_preds = batch_predict(svr, Xtr if args.svr_subsample < 1.0 else Xtr_svr, desc="SVR train preds")
    meta_train = np.column_stack([xgb_train_preds, svr_train_preds])

    xgb_val_preds = batch_predict(xgb, Xva, desc="XGBoost val preds")
    svr_val_preds = batch_predict(svr, Xva, desc="SVR val preds")
    meta_val = np.column_stack([xgb_val_preds, svr_val_preds])

    log.info("✅ Meta-features generated (mem: %s)", _mem())

    log.info("🧠 [Meta-Model] Training RidgeCV...")
    t_meta = time.time()
    meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5, scoring='neg_mean_squared_error')
    meta_model.fit(meta_train, ytr if args.svr_subsample < 1.0 else ytr_svr)
    meta_train_time = time.time() - t_meta
    log.info("✅ [Meta-Model] Training completed in %.2fs", meta_train_time)
    log.info("   📊 Best alpha: %.3f", meta_model.alpha_)

    blend_preds = meta_model.predict(meta_val)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    models_metrics = {}
    for name, preds in [("Ensemble", blend_preds), ("XGBoost", xgb_val_preds), ("SVR", svr_val_preds)]:
        mse = mean_squared_error(yva, preds)
        mae = mean_absolute_error(yva, preds)
        r2 = r2_score(yva, preds)
        models_metrics[name] = {"MSE": mse, "MAE": mae, "R2": r2}
        log.info(f"   {name:>10}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")

    evaluate_and_plot(yva, blend_preds, "Ensemble_Blend", diagram_dir)
    evaluate_and_plot(yva, xgb_val_preds, "XGBoost_Only", diagram_dir)
    evaluate_and_plot(yva, svr_val_preds, "SVR_Only", diagram_dir)
    log.info("✅ Evaluation plots saved to: %s", diagram_dir)

    log.info("💾 Saving trained models...")
    joblib.dump(xgb, out_dir / "xgboost_model.joblib", compress=3)
    joblib.dump(svr, out_dir / "svr_model.joblib", compress=3)
    joblib.dump(meta_model, out_dir / "meta_model.joblib", compress=3)
    joblib.dump({
        'training_info': {
            'xgb_train_time': xgb_train_time,
            'svr_train_time': svr_train_time,
            'meta_train_time': meta_train_time,
            'device_requested': args.device,
            'device_used': device_used,
            'svr_subsample_ratio': args.svr_subsample,
            'training_samples': Xtr.shape[0],
            'validation_samples': Xva.shape[0],
            'features': Xtr.shape[1],
            'models_metrics': models_metrics,
            'meta_alpha': float(meta_model.alpha_)
        }
    }, out_dir / "training_info.joblib", compress=3)

    total_runtime = time.time() - t0
    log.info("🎉 Training completed successfully!")
    log.info("⏱️  Training times:")
    log.info("   - XGBoost (%s): %.1fs", device_used, xgb_train_time)
    log.info("   - SVR (CPU): %.1fs", svr_train_time) 
    log.info("   - Meta-model: %.2fs", meta_train_time)
    log.info("🏁 Total runtime: %.1fs", total_runtime)
    log.info("💾 Final memory usage: %s", _mem())
    log.info("🎯 Best model: %s (R² = %.4f)", 
             max(models_metrics.items(), key=lambda x: x[1]['R2'])[0],
             max(models_metrics.values(), key=lambda x: x['R2'])['R2'])

if __name__ == "__main__":
    main()
