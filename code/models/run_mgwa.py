from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.utils.dgwo import DGWA  # Assuming DGWA is in this location


def evaluate_mgwa_model(y_true, y_pred):
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

    print(f"\n[MGWA] Accuracy: {acc:.4f}")
    print(f"[MGWA] Precision: {prec:.4f}")
    print(f"[MGWA] Recall: {rec:.4f}")
    print(f"[MGWA] F1-score: {f1:.4f}")
    print(f"[MGWA] Specificity: {specificity:.4f}")
    print("[MGWA] Confusion Matrix:")
    print(cm)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1, "Specificity": specificity}


def run_mgwa_feature_selection():
    print("\n🔹 Running Modified GWA (MGWA) Comparison")

    train_path = PROJECT_ROOT / "data" / "processed" / "train_top20_filtered.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_top20_filtered.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_full = train_df.drop(columns=['label'])
    y_train_full = train_df['label']
    X_test_final = test_df.drop(columns=['label'])
    y_test_final = test_df['label']

    # --- CORRECTION: Create a validation set from the training data to avoid leakage ---
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    clf_for_mgwa = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

    # Run MGWA (simulated by disabling the novel exploration operators)
    mgwa = DGWA(
        classifier=clf_for_mgwa,
        X_train=X_train_opt.values,
        y_train=y_train_opt.values,
        X_val=X_val_opt.values,  # Use the validation set
        y_val=y_val_opt.values,  # Use the validation set
        population_size=20,
        max_iter=30,
        feature_count=X_train_full.shape[1],
        verbose=True,
        use_exploration_ops=False  # Key difference for MGWA
    )

    best_mask, best_score = mgwa.optimize()
    best_mask = np.array(best_mask, dtype=bool)
    selected_features = X_train_full.columns[best_mask]

    print(f"\nSelected features by MGWA: {len(selected_features)}")
    print(f"Best validation accuracy (MGWA): {best_score:.4f}")

    # Final evaluation on the held-out test set
    final_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_clf.fit(X_train_full[selected_features], y_train_full)
    y_pred = final_clf.predict(X_test_final[selected_features])

    evaluate_mgwa_model(y_test_final, y_pred)

    pd.DataFrame({"Selected_Features": list(selected_features)}).to_csv(
        PROJECT_ROOT / "results" / "tables" / "mgwa_selected_features.csv", index=False
    )