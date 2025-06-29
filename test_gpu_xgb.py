from xgboost import XGBRegressor
import numpy as np

X = np.random.rand(100, 10).astype(np.float32)
y = np.random.rand(100).astype(np.float32)

model = XGBRegressor(tree_method="hist", device="cuda")
model.fit(X, y)
print("✅ XGBoost GPU training successful!")
