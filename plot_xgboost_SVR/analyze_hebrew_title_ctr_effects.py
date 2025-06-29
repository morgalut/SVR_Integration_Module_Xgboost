import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import matplotlib as mpl
import matplotlib.font_manager as fm


# === Configuration ===
DATA_DIR = "data"
OUTPUT_CSV = "hebrew_word_ctr_effects.csv"
OUTPUT_PLOT = "hebrew_ctr_word_impact.png"
TOP_N_WORDS = 1000
MIN_OCCURRENCE = 10
PLOT_TOP_N = 300

def load_all_csvs(data_dir):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(path)
                if 'category' not in df.columns:
                    df['category'] = os.path.splitext(filename)[0]
                all_data.append(df)
                print(f"✅ Loaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
    if not all_data:
        raise FileNotFoundError(f"No valid CSVs found in {data_dir}")
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(df):
    df.columns = [col.lower() for col in df.columns]
    if 'ctr' not in df.columns:
        if 'url clicks' in df.columns and 'impressions' in df.columns:
            df['ctr'] = df['url clicks'] / df['impressions']
        else:
            raise ValueError("Missing 'ctr' or click/impression columns")
    df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce')
    df = df.dropna(subset=['title', 'ctr'])
    df = df[(df['ctr'] >= 0.01) & (df['ctr'] <= 0.2)]
    return df

def extract_hebrew_words(df):
    hebrew_regex = re.compile(r'[\u0590-\u05FF]+')
    df['hebrew_words'] = df['title'].apply(lambda x: hebrew_regex.findall(str(x)))
    return df

def analyze_word_ctr_effects(df, top_n=TOP_N_WORDS, min_occurrence=MIN_OCCURRENCE):
    all_words = [word for words in df['hebrew_words'] for word in words]
    word_freq = pd.Series(all_words).value_counts().head(top_n)

    result_rows = []
    for word in word_freq.index:
        has_word = df['title'].apply(lambda x: word in str(x))
        if has_word.sum() > min_occurrence and (~has_word).sum() > min_occurrence:
            ctr_with = df[has_word]['ctr']
            ctr_without = df[~has_word]['ctr']
            stat, pval = ttest_ind(ctr_with, ctr_without, equal_var=False)
            result_rows.append({
                "word": word,
                "avg_ctr_with": ctr_with.mean(),
                "avg_ctr_without": ctr_without.mean(),
                "diff": ctr_with.mean() - ctr_without.mean(),
                "pval": pval,
                "count_with": len(ctr_with),
                "count_without": len(ctr_without)
            })
    return pd.DataFrame(result_rows).sort_values(by="diff", ascending=False)

def plot_top_words(df, top_n, output_path):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import os

    top_words = df.head(top_n).sort_values(by="diff", ascending=True)

    # Reverse Hebrew words for RTL rendering
    top_words["word"] = top_words["word"].apply(lambda w: w[::-1])

    # Load fonts
    hebrew_font = fm.FontProperties(family="Noto Sans Hebrew")
    latin_font = fm.FontProperties(family="DejaVu Sans")

    # Plot
    plt.figure(figsize=(10, max(8, top_n * 0.4)))
    plt.barh(top_words["word"], top_words["diff"], color="skyblue")

    plt.xlabel("CTR Impact", fontsize=12, fontproperties=latin_font)
    plt.title("30 המילים שמגבירות הקלקות בכותרות בעברית", fontsize=14,
              fontproperties=hebrew_font, loc='right')
    plt.xticks(fontproperties=latin_font)
    plt.yticks(fontproperties=hebrew_font)

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"📊 Hebrew plot saved to: {output_path}")



def main():
    print(f"📂 Loading data from: {DATA_DIR}")
    df = load_all_csvs(DATA_DIR)
    df = preprocess_data(df)
    df = extract_hebrew_words(df)

    print("📊 Analyzing word impact on CTR...")
    result_df = analyze_word_ctr_effects(df)

    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Results saved to: {OUTPUT_CSV}")
    print(result_df.head(10))

    print(f"📈 Creating plot of top {PLOT_TOP_N} CTR-driving words...")
    plot_top_words(result_df, top_n=PLOT_TOP_N, output_path=OUTPUT_PLOT)

if __name__ == "__main__":
    main()
