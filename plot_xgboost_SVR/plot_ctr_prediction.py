import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import ttest_ind, f_oneway
import numpy as np

# === Utility: ensure output folder ===
def ensure_output_folder(path: str):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        print(f"📁 Creating missing folder: {abs_path}")
        os.makedirs(abs_path)
    else:
        print(f"📂 Output folder exists: {abs_path}")

    try:
        with open(os.path.join(abs_path, "write_test.txt"), "w") as f:
            f.write("✅ Folder is writable.")
    except Exception as e:
        raise RuntimeError(f"❌ Cannot write to {abs_path}: {e}")

    return abs_path

# === Setup paths ===
print("📁 Setting up paths...")
data_dir = "data"
diagram_root_dir = ensure_output_folder("dagromote")

print(f"📂 Data directory: {data_dir}")
print(f"📂 Diagram root directory: {diagram_root_dir}")

# === Load CSVs ===
print("📄 Loading CSV files...")
all_data = []
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        path = os.path.join(data_dir, filename)
        print(f"🔍 Reading {path}")
        try:
            df = pd.read_csv(path)
            if 'category' not in df.columns:
                df['category'] = os.path.splitext(filename)[0]
            print(f"✅ Loaded {filename} with {df.shape[0]} rows")
            all_data.append(df)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

if not all_data:
    raise ValueError("❌ No CSV files were loaded from the data directory.")

print("🧩 Concatenating CSV data...")
data = pd.concat(all_data, ignore_index=True)
data.columns = [col.lower() for col in data.columns]

# === Ensure essential columns ===
print("🔍 Verifying CTR column...")
if 'ctr' not in data.columns:
    if 'url clicks' in data.columns and 'impressions' in data.columns:
        print("✅ Generating CTR from 'url clicks' / 'impressions'")
        data['ctr'] = data['url clicks'] / data['impressions']
    else:
        ctr_col = next((col for col in data.columns if 'click' in col), None)
        if ctr_col:
            print(f"✅ Found CTR fallback column: {ctr_col}")
            data['ctr'] = data[ctr_col]
        else:
            raise ValueError("❌ 'ctr' column not found and no fallback with 'click' in name.")

data['ctr'] = pd.to_numeric(data['ctr'], errors='coerce')

print("🧹 Cleaning data...")
print(f"🔎 Non-null CTR entries: {data['ctr'].notnull().sum()}")
data = data.dropna(subset=['title', 'ctr'])

print(f"📊 Min CTR: {data['ctr'].min()}, Max CTR: {data['ctr'].max()}")
print("📉 Filtering CTR between 0.01 and 0.2...")
data = data[(data['ctr'] >= 0.01) & (data['ctr'] <= 0.2)]

if 'subtitle' not in data.columns:
    print("ℹ️ Adding missing 'subtitle' column")
    data['subtitle'] = ""
if 'category' not in data.columns:
    print("ℹ️ Adding missing 'category' column")
    data['category'] = "Other"

print(f"📊 Total records after filtering: {len(data)}")
print(data['category'].value_counts())

# === Feature Engineering ===
print("🔧 Feature engineering...")
hebrew_punctuation = r'[!?,.؟؛״׳:]'
def count_uppercase_words(text):
    words = str(text).split()
    return sum(1 for w in words if w.isupper() and re.search("[א-ת]", w))
def is_hebrew(text):
    return bool(re.search(r'[\u0590-\u05FF]', str(text)))
def avg_word_length(text):
    words = str(text).split()
    return sum(len(w) for w in words) / (len(words) + 1e-5)

for field in ['title', 'subtitle']:
    if field in data.columns:
        print(f"⚙️ Processing features for: {field}")
        data[f'{field}_len'] = data[field].str.len()
        data[f'{field}_word_count'] = data[field].apply(lambda x: len(str(x).split()))
        data[f'{field}_has_exclamation'] = data[field].str.contains("!", regex=False)
        data[f'{field}_has_question'] = data[field].str.contains(r"\\?", regex=True)
        data[f'{field}_has_number'] = data[field].apply(lambda x: bool(re.search(r'\\d', str(x))))
        data[f'{field}_uppercase_word_count'] = data[field].apply(count_uppercase_words)
        data[f'{field}_avg_word_len'] = data[field].apply(avg_word_length)
        data[f'{field}_starts_with_number'] = data[field].apply(lambda x: str(x).strip()[0].isdigit() if str(x).strip() else False)
        data[f'{field}_ends_with_punctuation'] = data[field].apply(lambda x: bool(re.search(f"{hebrew_punctuation}$", str(x).strip())))
        data[f'{field}_hebrew_detected'] = data[field].apply(is_hebrew)

# === Hypothesis Features ===
print("🧠 Extracting hypothesis features...")
hypothesis_features = [col for col in data.columns if any(
    keyword in col for keyword in ['_len', '_count', '_has_', '_avg_', '_starts_', '_ends_', '_hebrew_detected'])]
print(f"📌 Total hypothesis features: {len(hypothesis_features)}")

# === Diagram Generation ===
print("📈 Generating diagrams...")
for feature in hypothesis_features:
    feature_dir = ensure_output_folder(os.path.join(diagram_root_dir, feature))
    for cat in sorted(data['category'].unique()):
        subset = data[data['category'] == cat]
        if subset.shape[0] == 0 or feature not in subset.columns:
            print(f"   ⚠️ Skipped {cat}: rows=0, has_col={feature in subset.columns}")
            continue

        plt.figure(figsize=(10, 4))
        sns.histplot(subset[feature], kde=True, bins=30)
        plt.title(f"{feature} Distribution - {cat}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        filename = f"{cat}.png".replace(" ", "_")
        filepath = os.path.join(feature_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"   📊 Category: {cat} | Rows: {len(subset)}")
        print(f"   ✅ Saved: {filepath}")

print("✅ All done.")