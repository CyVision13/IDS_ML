import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_rankings(paths, score_columns):
    """Load and align feature rankings from different criteria"""
    dfs = []

    for path, score_col in zip(paths, score_columns):
        df = pd.read_csv(path)
        if "Feature" not in df.columns:
            raise ValueError(f"'Feature' column not found in {path}")
        if score_col not in df.columns:
            raise ValueError(f"'{score_col}' column not found in {path}")
        df = df[["Feature", score_col]].rename(columns={score_col: "Score"})
        dfs.append(df)

    # Inner join all DataFrames on 'Feature'
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Feature", suffixes=('', '_dup'))

    feature_names = merged["Feature"].values
    score_matrix = merged.drop(columns=["Feature"]).values

    return score_matrix, feature_names


def normalize_matrix(matrix):
    """Normalize decision matrix using vector normalization (Eq. 13)"""
    norm = np.linalg.norm(matrix, axis=0)
    return matrix / norm


def compute_ideal_solutions(norm_matrix):
    """Determine positive (A+) and negative (A−) ideal solutions"""
    ideal_positive = np.max(norm_matrix, axis=0)
    ideal_negative = np.min(norm_matrix, axis=0)
    return ideal_positive, ideal_negative


def calculate_separation_measures(norm_matrix, ideal_pos, ideal_neg):
    """Calculate separation from ideal solutions (Eq. 14 and 15)"""
    dist_pos = np.linalg.norm(norm_matrix - ideal_pos, axis=1)
    dist_neg = np.linalg.norm(norm_matrix - ideal_neg, axis=1)
    return dist_pos, dist_neg


def compute_closeness_coefficients(d_pos, d_neg):
    """Compute closeness coefficient (Eq. 16)"""
    return d_neg / (d_pos + d_neg)


def fuzzy_topsis_main(paths, score_columns, top_k=20, weights=None):
    # Load rankings
    matrix, features = load_rankings(paths, score_columns)

    # Use equal weights if none provided
    if weights is None:
        weights = np.ones(matrix.shape[1]) / matrix.shape[1]

    # Step 1: Create weighted matrix
    weighted_matrix = matrix * weights

    # Step 2: Normalize matrix
    norm_matrix = normalize_matrix(weighted_matrix)

    # Step 3: Ideal solutions
    ideal_pos, ideal_neg = compute_ideal_solutions(norm_matrix)

    # Step 4: Separation measures
    d_pos, d_neg = calculate_separation_measures(norm_matrix, ideal_pos, ideal_neg)

    # Step 5: Closeness coefficient
    closeness = compute_closeness_coefficients(d_pos, d_neg)

    # Step 6: Rank and select top-K features
    result_df = pd.DataFrame({
        'Feature': features,
        'Closeness': closeness
    }).sort_values(by='Closeness', ascending=False).reset_index(drop=True)

    return result_df.head(top_k)
