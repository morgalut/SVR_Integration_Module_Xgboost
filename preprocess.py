#!/usr/bin/env python
# preprocess.py – Prepare CTR dataset for SVR/XGBoost

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from packaging import version
from sklearn import __version__ as skl_version
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ───────────────────────────  Logging  ────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ctr_prepare")

# ────────────────────── CTR Weight Utilities ──────────────────────
# ── replace the current load_ctr_weights() ──
def load_ctr_weights(csv_path: Path,
                     base: float = 1.15,
                     max_boost: float = 3.0) -> Dict[str, float]:
    """
    Map each word → repeat-factor used by CTRTfidfVectorizer.

    * Positive diff  ⇒ weight > 1
    * Negative diff  ⇒ weight < 1   (rarely helpful, so we gentle-penalize)
    """
    df = pd.read_csv(csv_path)
    if not {"word", "diff"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain 'word' and 'diff' columns")

    def _w(d):
        signed_log = np.sign(d) * np.log1p(abs(d))
        return float(np.clip(1 + base * signed_log, 0.5, max_boost))

    return {w: _w(d) for w, d in df[["word", "diff"]].itertuples(index=False)}


class CTRWeightedTokenizer:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def __call__(self, doc: str) -> List[str]:
        weighted_tokens = []
        for tok in doc.split():
            weight = self.weights.get(tok, 1.0)
            repeat_count = int(np.ceil(weight))
            weighted_tokens.extend([tok] * repeat_count)
        return weighted_tokens


class CTRTfidfVectorizer(TfidfVectorizer):
    def __init__(self, word_weights: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.word_weights = word_weights

    def build_analyzer(self):
        base = super().build_analyzer()

        def analyzer(doc):
            weighted_tokens = []
            for token in base(doc):
                weight = self.word_weights.get(token, 1.0)
                repeat_count = int(np.ceil(weight))
                weighted_tokens.extend([token] * repeat_count)
            return weighted_tokens

        return analyzer


# ─────────────────────── Preprocessor Setup ───────────────────────
def build_preprocessor(weights: Dict[str, float]) -> ColumnTransformer:
    text_pipe = Pipeline([
        ("tfidf", CTRTfidfVectorizer(weights, max_features=10000, ngram_range=(1, 2)))
    ])

    ohe_args = dict(handle_unknown="ignore")
    if version.parse(skl_version) < version.parse("1.4"):
        ohe_args["sparse"] = True
    else:
        ohe_args["sparse_output"] = True

    cat_pipe = Pipeline([("onehot", OneHotEncoder(**ohe_args))])

    return ColumnTransformer([
        ("text", text_pipe, "text"),
        ("cat", cat_pipe, ["category"]),
    ])

# ─────────────────────── Data Split Helper ────────────────────────
def split_three(
    X: sp.csr_matrix, y: np.ndarray, test_size: float, eval_size: float, seed: int
) -> Tuple[Tuple[sp.csr_matrix, np.ndarray], ...]:
    X_temp, X_eval, y_temp, y_eval = train_test_split(X, y, test_size=eval_size, random_state=seed)
    rel_test = test_size / (1 - eval_size)
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=rel_test, random_state=seed)
    return (X_train, y_train), (X_test, y_test), (X_eval, y_eval)

def save_split(split, prefix: str, out_dir: Path):
    X, y = split
    sp.save_npz(out_dir / f"{prefix}_X.npz", X)
    np.save(out_dir / f"{prefix}_y.npy", y)

# ────────────────────────── Main CLI ──────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Prepare CTR data for SVR/XGBoost")
    p.add_argument("--data_folder", default="data")
    p.add_argument("--ctr_file", default="hebrew_word_ctr_effects.csv")
    p.add_argument("--save_dir", default="processed_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--eval_size", type=float, default=0.20)
    args = p.parse_args()

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Load CTR word weights
    weights = load_ctr_weights(Path(args.ctr_file))
    log.info("Loaded CTR weights: %d words", len(weights))

    # 2️⃣ Load CSV files and normalize schema
    csvs = sorted(Path(args.data_folder).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {args.data_folder}")

    frames = []
    for f in csvs:
        df = pd.read_csv(f)

        # 🔁 Rename columns to match required schema
        df = df.rename(columns={
            "title": "text",
            "URL CTR": "ctr"
        })

        # Optional: combine title + subtitle
        if "subtitle" in df.columns:
            df["text"] = df["text"].fillna("") + " " + df["subtitle"].fillna("")

        missing = {"text", "ctr"} - set(df.columns)
        if missing:
            raise ValueError(f"File {f.name} is missing required columns: {missing}")

        df["category"] = f.stem
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d rows from %d CSV files", len(data), len(csvs))

    # 3️⃣ Fit preprocessor
    preproc = build_preprocessor(weights)
    X_full = preproc.fit_transform(data)
    y_full = data["ctr"].astype(np.float32).values

    # 4️⃣ Train/Test/Eval split
    train, test, eval_ = split_three(X_full, y_full, args.test_size, args.eval_size, args.seed)

    # 5️⃣ Save
    save_split(train, "train", save_dir)
    save_split(test,  "test",  save_dir)
    save_split(eval_, "eval",  save_dir)
    joblib.dump(preproc, save_dir / "preprocessor.joblib")

    log.info("✅ Saved dataset splits and preprocessor to %s", save_dir)
    log.info("   - train_X.npz / train_y.npy")
    log.info("   - test_X.npz  / test_y.npy")
    log.info("   - eval_X.npz  / eval_y.npy")
    log.info("   - preprocessor.joblib")

if __name__ == "__main__":
    main()
