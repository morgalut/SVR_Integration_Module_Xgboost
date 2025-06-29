import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# === Setup ===
data_dir = "data"
output_dir = "dagromote"
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
dfs = []
for fname in os.listdir(data_dir):
    if fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_dir, fname))
        if 'category' not in df.columns:
            df['category'] = os.path.splitext(fname)[0]
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data.columns = [c.lower() for c in data.columns]

# === CTR Column Handling ===
if 'ctr' not in data.columns:
    if 'url clicks' in data.columns and 'impressions' in data.columns:
        data['ctr'] = data['url clicks'] / data['impressions']
    else:
        raise ValueError("CTR column missing and cannot be derived.")

data = data[(data['ctr'] >= 0.01) & (data['ctr'] <= 0.2)]
data['subtitle'] = data.get('subtitle', "")
data['title_word_count'] = data['title'].apply(lambda x: len(str(x).split()))
data['subtitle_word_count'] = data['subtitle'].apply(lambda x: len(str(x).split()))

# === Binning ===
bin_edges = [0, 3, 6, 9, 12, 15, 20]
bin_labels = ["1–3", "4–6", "7–9", "10–12", "13–15", "16–20"]
data['title_wc_bin'] = pd.cut(data['title_word_count'], bins=bin_edges, labels=bin_labels)
data['subtitle_wc_bin'] = pd.cut(data['subtitle_word_count'], bins=bin_edges, labels=bin_labels)

# === Aggregate CTR ===
title_grouped = data.groupby(['category', 'title_wc_bin'], observed=False)['ctr'].mean().reset_index().dropna()
subtitle_grouped = data.groupby(['category', 'subtitle_wc_bin'], observed=False)['ctr'].mean().reset_index().dropna()
title_grouped = title_grouped.rename(columns={"title_wc_bin": "bin"})
subtitle_grouped = subtitle_grouped.rename(columns={"subtitle_wc_bin": "bin"})

# Ensure bin is string for compatibility
title_grouped['bin'] = title_grouped['bin'].astype(str)
subtitle_grouped['bin'] = subtitle_grouped['bin'].astype(str)

# === Extrema Function ===
def prepare_extrema(df):
    min_df = df.groupby("category")["ctr"].min().reset_index().rename(columns={"ctr": "ctr_min"})
    max_df = df.groupby("category")["ctr"].max().reset_index().rename(columns={"ctr": "ctr_max"})
    return pd.merge(min_df, max_df, on="category")

# === Plotting Function ===
def plot_ctr_vs_wordcount(grouped_df, extrema_df, category_colors, ylabel, title, output_path):
    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.25)
    ax_plot = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis("off")

    sns.lineplot(
        data=grouped_df,
        x="bin",
        y="ctr",
        hue="category",
        marker="o",
        ax=ax_plot,
        palette=category_colors,
        legend=False
    )

    ax_plot.set_title(title)
    ax_plot.set_xlabel("Word Count Range")
    ax_plot.set_ylabel(ylabel)
    ax_plot.tick_params(axis='x', rotation=45)
    ax_plot.grid(True)

    # === Legend Panel ===
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    y_start = 1.0
    line_height = 1.0 / (len(extrema_df) + 2)

    ax_legend.text(0.05, y_start, "📁 Category CTR Summary", fontsize=12, fontweight='bold', va="top")

    for i, (_, row) in enumerate(extrema_df.sort_values(by="category").iterrows()):
        color = category_colors[row["category"]]
        y = y_start - (i + 1) * line_height

        ax_legend.add_patch(Rectangle((0.02, y - 0.015), 0.02, 0.03, facecolor=color, edgecolor='black', linewidth=0.5))
        ax_legend.text(0.07, y, f"{row['category']}", fontsize=9, va='center')
        ax_legend.text(0.47, y, f"Min: {row['ctr_min']:.3f}", fontsize=9, va='center')
        ax_legend.text(0.72, y, f"Max: {row['ctr_max']:.3f}", fontsize=9, va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")

# === Color Mapping ===
palette = sns.color_palette("tab20", n_colors=title_grouped["category"].nunique())
category_colors = dict(zip(sorted(title_grouped["category"].unique()), palette))

# === Generate Plots ===
title_extrema = prepare_extrema(title_grouped)
subtitle_extrema = prepare_extrema(subtitle_grouped)

plot_ctr_vs_wordcount(
    grouped_df=title_grouped,
    extrema_df=title_extrema,
    category_colors=category_colors,
    ylabel="Average CTR",
    title="📘 CTR vs. Title Word Count Range by Category",
    output_path=os.path.join(output_dir, "ctr_vs_title_wordcount.png")
)

plot_ctr_vs_wordcount(
    grouped_df=subtitle_grouped,
    extrema_df=subtitle_extrema,
    category_colors=category_colors,
    ylabel="CTR",
    title="📙 CTR vs. Subtitle Word Count Range by Category",
    output_path=os.path.join(output_dir, "ctr_vs_subtitle_wordcount.png")
)
