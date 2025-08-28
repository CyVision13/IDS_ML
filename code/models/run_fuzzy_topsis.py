import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Assuming this file contains the correct TOPSIS mathematical implementation
from code.utils.fuzzy_topsis import fuzzy_topsis_main


def run_fuzzy_topsis_feature_selection():
    """
    Runs the Fuzzy TOPSIS method to select top-K features from the ranker outputs.
    This corresponds to the second part of Phase 2 in the article.
    """
    print("=" * 60)
    print("🎯 PHASE 2b: FUZZY TOPSIS CONSENSUS RANKING")
    print("=" * 60)

    # Define paths to the ranking files
    ranking_files = [
        str(PROJECT_ROOT / "results" / "tables" / "chi2_ranking.csv"),
        str(PROJECT_ROOT / "results" / "tables" / "mad_ranking.csv"),
        str(PROJECT_ROOT / "results" / "tables" / "pcc_ranking.csv"),
    ]
    score_columns = ["Chi2_statistic", "MAD", "PCC"]

    # Run Fuzzy TOPSIS to get the top 20 features
    top_features_df = fuzzy_topsis_main(ranking_files, score_columns=score_columns, top_k=20)

    print("\n🏆 Top 20 features selected by Fuzzy TOPSIS:")
    print(top_features_df)

    # Save the list of selected features
    output_path = PROJECT_ROOT / "results" / "tables" / "fuzzy_topsis_top20.csv"
    top_features_df.to_csv(output_path, index=False)
    print(f"\n✅ Top 20 feature list saved to: {output_path}")

    # --- Prepare datasets for the next phase (DGWA) ---
    print("\n🔧 Filtering datasets to keep only the top 20 features...")

    # Load the ONE-HOT encoded files
    train_path = PROJECT_ROOT / "data" / "processed" / "train_processed_5class_onehot.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_processed_5class_onehot.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    top_feature_names = top_features_df["Feature"].tolist()

    # *** START OF CORRECTION ***
    # Map original feature names to one-hot encoded column names
    columns_to_keep = []
    original_categorical_features = ['protocol_type', 'service', 'flag']

    for feature in top_feature_names:
        if feature in original_categorical_features:
            # If it's a categorical feature, find all its one-hot columns
            one_hot_cols = [col for col in train_df.columns if col.startswith(feature + '_')]
            columns_to_keep.extend(one_hot_cols)
        else:
            # If it's a numerical/binary feature, its name is the same
            columns_to_keep.append(feature)

    # Add the label column
    columns_to_keep.append('label')
    # *** END OF CORRECTION ***

    # Filter the DataFrames using the new, complete list of columns
    train_df_filtered = train_df[columns_to_keep]
    test_df_filtered = test_df[columns_to_keep]

    # Define paths for the new filtered datasets
    train_filtered_path = PROJECT_ROOT / "data" / "processed" / "train_top20_filtered.csv"
    test_filtered_path = PROJECT_ROOT / "data" / "processed" / "test_top20_filtered.csv"

    # Save the filtered datasets
    train_df_filtered.to_csv(train_filtered_path, index=False)
    test_df_filtered.to_csv(test_filtered_path, index=False)

    print(f"✅ Filtered training set ({train_df_filtered.shape}) saved to: {train_filtered_path}")
    print(f"✅ Filtered test set ({test_df_filtered.shape}) saved to: {test_filtered_path}")
    print("\nReady for Phase 3: DGWA Optimization")


if __name__ == "__main__":
    run_fuzzy_topsis_feature_selection()