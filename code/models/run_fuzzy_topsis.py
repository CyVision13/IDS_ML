import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.utils.fuzzy_topsis import fuzzy_topsis_main


import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.utils.fuzzy_topsis import fuzzy_topsis_main


def run_fuzzy_topsis_feature_selection():
    # Define ranking files and score columns using Path (relative to project root)
    paths = [
        PROJECT_ROOT / "results" / "tables" / "chi2_ranking.csv",
        PROJECT_ROOT / "results" / "tables" / "mad_ranking.csv",
        PROJECT_ROOT / "results" / "tables" / "pcc_ranking.csv",
    ]

    score_columns = ["Chi2_statistic", "MAD", "PCC"]  # <-- Required

    # Convert Path objects to strings
    paths = [str(p) for p in paths]

    # Run Fuzzy TOPSIS
    top_features_df = fuzzy_topsis_main(paths, score_columns=score_columns, top_k=20)

    print("Top 20 features selected by Fuzzy TOPSIS:")
    print(top_features_df)

    # Save output
    output_path = PROJECT_ROOT / "results" / "tables" / "fuzzy_topsis_top20.csv"
    top_features_df.to_csv(output_path, index=False)
    # Filter train/test sets to top 20 features
    train_path = PROJECT_ROOT / "data" / "processed" / "train_processed.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_processed.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    top_features = top_features_df["Feature"].tolist()

    # Keep target column if exists
    if 'label' in train_df.columns:
        top_features.append('label')
    elif 'Label' in train_df.columns:
        top_features.append('Label')

    # Filter and save
    train_df_filtered = train_df[[col for col in top_features if col in train_df.columns]]
    test_df_filtered = test_df[[col for col in top_features if col in test_df.columns]]

    train_df_filtered.to_csv(PROJECT_ROOT / "data" / "processed" / "train_top20.csv", index=False)
    test_df_filtered.to_csv(PROJECT_ROOT / "data" / "processed" / "test_top20.csv", index=False)

    print("\nFiltered train and test sets saved with top 20 features.")
