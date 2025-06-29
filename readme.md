# ✅ ** How to Install Requirements**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```


### ✅ ** Preprocessing Step**

First preprocess the raw CSV files into `processed_

```bash
python preprocess.py --data_folder data \
                     --ctr_file hebrew_word_ctr_effects.csv \
                     --save_dir processed_out

```

```
processed_ X_train.npz
├── X_test.npz
├── X_eval.npz
├── y_train.npy
├── y_test.npy
├── y_eval.npy
└── preprocessor.joblib
```

---



#### 🚀 **XGBoost + SVR**

```bash
python main.py --proc_dir processed_out --device cuda:0 --svr_subsample 0.5 --output_dir results_out
```
# 🧩 **Folder Structure After Training**

After running, `dagromote/` will contain:

```
dagromote/
├── xgboost_model.joblib
├── xgboost_train.png
├── xgboost_eval.png
├── xgboost_comparison.png
├── catboost_model.joblib
├── catboost_train.png
├── catboost_eval.png
├── catboost_comparison.png
├── svr_model.joblib
├── svr_train.png
├── svr_eval.png
├── svr_comparison.png
```

---

✅ **All commands now include:**

* `--data_dir processed_data`
* `--output_dir dagromote`
* Optional `--log_transform` for log target

