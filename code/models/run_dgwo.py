import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.utils.dgwo import DGWA


def run_dgwa_feature_selection():
    # Load filtered dataset (after Fuzzy TOPSIS)
    train_path = PROJECT_ROOT / "data" / "processed" / "train_top20.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_top20.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Prepare inputs
    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values

    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values

    # Classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42)

    # Run DGWA
    dgwa = DGWA(
        classifier=clf,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        population_size=20,
        max_iter=30,
        feature_count=X_train.shape[1],
        verbose=True
    )

    best_features_mask, best_score = dgwa.optimize()

    selected_features = train_df.drop(columns=['label']).columns[best_features_mask == 1]
    print("\nSelected features by DGWA:")
    print(list(selected_features))
    print(f"Best classification accuracy (DGWA): {best_score:.4f}")

    # Save selected features
    output_path = PROJECT_ROOT / "results" / "tables" / "dgwa_selected_features.csv"
    pd.DataFrame({
        "Selected_Features": list(selected_features)
    }).to_csv(output_path, index=False)

    print(f"\nSaved selected features to: {output_path}")
