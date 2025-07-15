import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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

    # ✅ NEW: Evaluate using final selected features
    clf.fit(X_train[:, best_features_mask == 1], y_train)
    y_pred = clf.predict(X_test[:, best_features_mask == 1])
    evaluate_model(y_test, y_pred)

    # Save selected features
    output_path = PROJECT_ROOT / "results" / "tables" / "dgwa_selected_features.csv"
    pd.DataFrame({
        "Selected_Features": list(selected_features)
    }).to_csv(output_path, index=False)

    print(f"\nSaved selected features to: {output_path}")


def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("Confusion Matrix:")
    print(cm)