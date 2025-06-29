import logging
import time
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVR
import gc

# Set up logging
log = logging.getLogger("ctr_train")

def train_svr_incremental(X, y, batch_size=2000, max_epochs=5):
    """Train SVR with incremental learning for large datasets"""
    
    log.info("🚀 [SVR] Starting incremental training...")
    log.info("   Training samples: %s, Batch size: %s", X.shape[0], batch_size)
    
    svr = LinearSVR(
        epsilon=0.02,
        C=1.0,
        loss="squared_epsilon_insensitive",
        max_iter=2000,
        random_state=42,
        dual=False,
        verbose=1
    )
    
    t_svr = time.time()
    n_batches = int(np.ceil(X.shape[0] / batch_size))
    
    prev_loss = float('inf')
    convergence_checks = 0
    tol = 1e-4
    
    for epoch in range(1, max_epochs + 1):
        log.info(f"   📊 Epoch {epoch}/{max_epochs}")
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        for i in tqdm(range(0, X.shape[0], batch_size), 
                     total=n_batches, 
                     desc=f"SVR Epoch {epoch}"):
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            if epoch == 1 and i == 0:
                svr.fit(X_batch, y_batch)
            else:
                svr.fit(X_batch, y_batch)
            
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        if X.shape[0] > 10000:
            sample_size = 5000
            sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            sample_preds = svr.predict(X[sample_indices])
            current_loss = np.mean((sample_preds - y[sample_indices])**2)
        else:
            sample_preds = svr.predict(X)
            current_loss = np.mean((sample_preds - y)**2)
            
        log.info(f"   📉 Epoch {epoch} Sample MSE: {current_loss:.6f}")
        
        if abs(prev_loss - current_loss) < tol:
            convergence_checks += 1
            if convergence_checks >= 2:
                log.info(f"✅ SVR converged at epoch {epoch}")
                break
        else:
            convergence_checks = 0
            
        prev_loss = current_loss
    
    svr_train_time = time.time() - t_svr
    log.info("✅ [SVR] Training completed in %.1fs", svr_train_time)
    
    return svr, svr_train_time