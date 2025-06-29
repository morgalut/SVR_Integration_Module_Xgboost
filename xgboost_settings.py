import logging
import subprocess
import time
import json
import numpy as np
import sys
from typing import Tuple, Dict, Optional

import torch
import xgboost as xgb
from xgboost.callback import EarlyStopping

# === Log setup ===
log = logging.getLogger("ctr_train")

# === Show GPU info ===
try:
    torch_available = torch.cuda.is_available()
    log.info(f"✅ PyTorch CUDA Available: {torch_available}")
    if torch_available:
        log.info(f"   PyTorch GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    log.warning(f"⚠️ PyTorch CUDA check failed: {e}")

try:
    log.info(f"✅ XGBoost Version: {xgb.__version__}")
    log.info(f"   Built with CUDA: {getattr(xgb, 'BUILT_WITH_CUDA', 'Unknown')}")
except Exception as e:
    log.warning(f"⚠️ XGBoost info failed: {e}")

# === CUDA Check ===
def check_cuda_environment() -> bool:
    log.info("🔍 Checking CUDA environment...")
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        if result.returncode == 0:
            log.info("✅ NVIDIA drivers detected")
            return True
        else:
            log.warning(f"❌ nvidia-smi failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        log.warning(f"❌ CUDA environment error: {e}")
        return False

# === Validate XGBoost has GPU support ===
def validate_gpu_setup(device: str) -> Dict:
    log.info("🔍 Validating XGBoost GPU availability...")
    if not getattr(xgb, "BUILT_WITH_CUDA", False):
        raise RuntimeError("❌ XGBoost is not compiled with CUDA support.")
    return {"device": device}

# === XGBoost Params ===
def get_xgboost_params(device: str, working_gpu_config: Optional[Dict] = None, use_gpu: bool = True) -> Dict:
    """Return XGBoost config based on GPU or CPU"""
    base_params = {
        'n_estimators': 800,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'max_bin': 256,
        'reg_lambda': 1.0,
        'objective': "reg:squarederror",
        'eval_metric': ["rmse", "mae"],
        'verbosity': 1,
        'random_state': 42,
        'max_leaves': 0,
        'grow_policy': 'depthwise',
        'tree_method': 'hist',
        'device': 'cuda' if use_gpu else 'cpu',
        'n_jobs': 1 if use_gpu else -1
    }
    log.info(f"🔧 Using {'GPU' if use_gpu else 'CPU'} device")
    return base_params

# === Main Training Function ===
def train_xgboost_robust(Xtr, ytr, Xva, yva, device: str) -> Tuple[xgb.Booster, float, str]:
    """Train XGBoost with GPU if available, fallback to CPU if not"""

    use_gpu = False
    working_gpu_config = None

    if device.startswith("cuda") and check_cuda_environment():
        try:
            working_gpu_config = validate_gpu_setup(device)
            use_gpu = True
            log.info("🎯 Using GPU for training.")
        except Exception as gpu_error:
            log.warning(f"⚠️ GPU validation failed: {gpu_error}")
            log.info("🔄 Falling back to CPU.")
    else:
        log.info("🔄 Using CPU for training.")

    xgb_params = get_xgboost_params(device, working_gpu_config, use_gpu)
    device_type = "GPU" if use_gpu else "CPU"

    log.info(f"🚀 Starting XGBoost training on {device_type}")
    log.info("   Parameters:\n%s", json.dumps(xgb_params, indent=2))

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xva, label=yva)
    evals = [(dtrain, "train"), (dvalid, "eval")]

    num_boost_round = xgb_params.pop("n_estimators", 800)
    t_xgb = time.time()

    try:
        booster = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        train_time = time.time() - t_xgb
        log.info(f"✅ {device_type} training completed in {train_time:.1f}s")
        return booster, train_time, device_type

    except Exception as e:
        log.error(f"❌ {device_type} training failed: {e}")

        if use_gpu:
            log.info("🔄 Retrying training on CPU...")
            cpu_params = get_xgboost_params("cpu", None, use_gpu=False)
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dvalid = xgb.DMatrix(Xva, label=yva)
            evals = [(dtrain, "train"), (dvalid, "eval")]
            num_boost_round = cpu_params.pop("n_estimators", 800)

            booster = xgb.train(
                params=cpu_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=50
            )
            train_time = time.time() - t_xgb
            log.info("✅ CPU fallback training completed in %.1fs", train_time)
            return booster, train_time, "CPU (fallback)"
        else:
            raise RuntimeError(f"XGBoost training failed: {e}") from e
