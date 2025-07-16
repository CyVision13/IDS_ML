from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from code.utils.dgwo import DGWA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_mgwa_feature_selection():
    print("\n🔹 Running Modified GWA (MGWA)")

    # Load data
    train_path = PROJECT_ROOT / "data" / "processed" / "train_top20.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_top20.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values

    # Classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42)

    # Run MGWA (exploration disabled)
    mgwa = DGWA(
        classifier=clf,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        population_size=20,
        max_iter=30,
        feature_count=X_train.shape[1],
        verbose=True,
        use_exploration_ops=False
    )

    best_mask, best_score = mgwa.optimize()
    selected_features = train_df.drop(columns=['label']).columns[best_mask == 1]

    print(f"\nSelected features by MGWA:")
    print(list(selected_features))
    print(f"Best classification accuracy (MGWA): {best_score:.4f}")

    clf.fit(X_train[:, best_mask == 1], y_train)
    y_pred = clf.predict(X_test[:, best_mask == 1])
    metrics = evaluate_model(y_test, y_pred)

    # Save results
    pd.DataFrame({"Selected_Features": list(selected_features)}).to_csv(
        PROJECT_ROOT / "results" / "tables" / "mgwa_selected_features.csv", index=False
    )


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
