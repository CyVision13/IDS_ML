import matplotlib.pyplot as plt  # ✅ FIXED
import pandas as pd
from pathlib import Path


def plot_feature_rankings(ranking_files, n_top=10):
    fig, axes = plt.subplots(1, len(ranking_files), figsize=(6 * len(ranking_files), 6))

    for ax, (title, filepath) in zip(axes, ranking_files.items()):
        df = pd.read_csv(filepath)

        # Auto-detect ranking column (skip 'Feature', 'P_value' etc.)
        ranking_cols = [col for col in df.columns if col.lower() not in ['feature', 'p_value']]
        if len(ranking_cols) != 1:
            raise ValueError(f"Expected exactly one ranking column in {filepath}. Found: {ranking_cols}")
        metric = ranking_cols[0]

        top_features = df.head(n_top)
        ax.barh(top_features['Feature'], top_features[metric], color='skyblue')
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.invert_yaxis()

    plt.tight_layout()
    return fig
