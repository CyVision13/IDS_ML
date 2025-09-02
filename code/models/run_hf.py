import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_hf_model(y_true, y_pred):
    # --- CORRECTION: Use multi-class metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    specificity = np.mean(tn / (tn + fp)) if np.all((tn + fp) > 0) else 0

    print(f"\n[HF] Accuracy: {acc:.4f}")
    print(f"[HF] Precision: {prec:.4f}")
    print(f"[HF] Recall: {rec:.4f}")
    print(f"[HF] F1-score: {f1:.4f}")
    print(f"[HF] Specificity: {specificity:.4f}")
    print("[HF] Confusion Matrix:")
    print(cm)

    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-score": f1, "Specificity": specificity
    }


def run_hf_feature_selection(k=10):
    print(f"\n🔹 Running Heuristic Filter (Chi2) Comparison with top-{k} features")

    # Load the data pre-filtered by TOPSIS to ensure a fair comparison
    train_path = PROJECT_ROOT / "data" / "processed" / "train_top20_filtered.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_top20_filtered.csv"
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "hf_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    # Apply chi2 ranking on the input features
    selector = SelectKBest(score_func=chi2, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_columns = X_train.columns[selector.get_support()]
    print(f"\nSelected features by HF (Chi2):")
    print(list(selected_columns))

    # Train & evaluate classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)

    metrics = evaluate_hf_model(y_test, y_pred)

    # Save results
    (PROJECT_ROOT / "results" / "tables").mkdir(exist_ok=True)
    pd.DataFrame({"Selected_Features": list(selected_columns)}).to_csv(
        PROJECT_ROOT / "results" / "tables" / "hf_selected_features.csv", index=False
    )
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"\nSaved results and metrics for HF.")