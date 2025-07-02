import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys

# Add project root to python path (more reliable method)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import using full package path
try:
    from code.utils.plotting_functions import plot_feature_rankings
except ImportError as e:
    print(f"Import error: {e}")
    print("Current Python path:")
    for path in sys.path:
        print(f" - {path}")
    raise

# Set paths
PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "tables"


def load_data():
    """Load processed training data"""
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_processed.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_processed.csv")
    return train_df, test_df


def chi_square_ranking(X, y):
    """Calculate Chi-square statistics for feature ranking"""
    chi2_stats = []
    p_values = []

    for feature in X.columns:
        contingency_table = pd.crosstab(X[feature], y)
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi2_stats.append(chi2)
        p_values.append(p)

    ranking = pd.DataFrame({
        'Feature': X.columns,
        'Chi2_statistic': chi2_stats,
        'P_value': p_values
    }).sort_values('Chi2_statistic', ascending=False)

    return ranking


def mad_ranking(X, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['id', 'attack_cat']  # add any other columns to exclude

    print("Input columns:", X.columns)
    print("Excluding columns:", exclude_cols)

    # Drop columns that should not be ranked
    X_filtered = X.drop(columns=[col for col in exclude_cols if col in X.columns])

    # Select numeric columns after exclusion
    X_numeric = X_filtered.select_dtypes(include=np.number)
    print("Numeric columns considered:", X_numeric.columns)

    means = X_numeric.mean(axis=0)
    print("Means per feature:\n", means)

    abs_dev = X_numeric.sub(means, axis=1).abs()
    mad_values = abs_dev.mean(axis=0)

    ranking = pd.DataFrame({
        'Feature': mad_values.index,
        'MAD': mad_values.values
    }).sort_values('MAD', ascending=False).reset_index(drop=True)

    print("Ranking:\n", ranking)
    return ranking


def pcc_ranking(X, y):
    """Calculate Pearson Correlation Coefficient for feature ranking"""
    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y

    pcc_values = []
    p_values = []

    for feature in X.columns:
        corr, p = pearsonr(X[feature], y_encoded)
        pcc_values.append(abs(corr))  # Using absolute value for ranking
        p_values.append(p)

    ranking = pd.DataFrame({
        'Feature': X.columns,
        'PCC': pcc_values,
        'P_value': p_values
    }).sort_values('PCC', ascending=False)

    return ranking


def save_rankings(chi2_rank, mad_rank, pcc_rank):
    """Save ranking results to files"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    chi2_path = RESULTS_DIR / "chi2_ranking.csv"
    mad_path = RESULTS_DIR / "mad_ranking.csv"
    pcc_path = RESULTS_DIR / "pcc_ranking.csv"

    chi2_rank.to_csv(chi2_path, index=False)
    mad_rank.to_csv(mad_path, index=False)
    pcc_rank.to_csv(pcc_path, index=False)

    # Plot rankings
    ranking_files = {
        "Chi-square Ranking": chi2_path,
        "MAD Ranking": mad_path,
        "PCC Ranking": pcc_path
    }

    fig = plot_feature_rankings(ranking_files)
    fig.savefig(RESULTS_DIR.parent / "figures" / "feature_rankings.png")
    plt.close(fig)


def feature_selection():
    # Load data
    train_df, _ = load_data()

    # Separate features and target (assuming last column is target)
    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]

    # Calculate rankings
    print("Calculating Chi-square rankings...")
    chi2_rank = chi_square_ranking(X, y)

    print("Calculating MAD rankings...")
    mad_rank = mad_ranking(X)

    print("Calculating PCC rankings...")
    pcc_rank = pcc_ranking(X, y)

    # Save results
    save_rankings(chi2_rank, mad_rank, pcc_rank)
    print("Feature rankings saved to results/tables/")


if __name__ == "__main__":
    feature_selection()
