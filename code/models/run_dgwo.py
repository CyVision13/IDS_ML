import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.utils.dgwo import SimplifiedDGWA  # Changed from DGWA to SimplifiedDGWA

def run_dgwa_feature_selection():
    """
    Runs SimplifiedDGWA optimization (Phase 3) and final evaluation (Phase 4).
    """
    print("=" * 60)
    print("🎯 PHASE 3 & 4: SIMPLIFIED DGWA OPTIMIZATION & FINAL EVALUATION")
    print("=" * 60)

    # Load the filtered dataset created by the Fuzzy TOPSIS script
    train_path = PROJECT_ROOT / "data" / "processed" / "train_top20_filtered.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_top20_filtered.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Prepare training and final test data
    X_train_full = train_df.drop(columns=['label'])
    y_train_full = train_df['label']
    X_test_final = test_df.drop(columns=['label'])
    y_test_final = test_df['label']

    # Create a validation set from the training data
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f"Data split for SimplifiedDGWA: Training Opt={X_train_opt.shape}, Validation Opt={X_val_opt.shape}")

    # The classifier that the SimplifiedDGWA will use for fitness evaluation
    clf_for_dgwa = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

    # Run SimplifiedDGWA using the new training/validation split
    simplified_dgwa = SimplifiedDGWA(
        classifier=clf_for_dgwa,
        X_train=X_train_opt.values,
        y_train=y_train_opt.values,
        X_val=X_val_opt.values,
        y_val=y_val_opt.values,
        population_size=20,
        max_iter=30,
        verbose=True
    )

    best_features_mask, best_score = simplified_dgwa.optimize()
    best_features_mask = np.array(best_features_mask, dtype=bool)

    selected_features = X_train_full.columns[best_features_mask]
    print("\n🏆 Optimal features selected by SimplifiedDGWA:")
    print(list(selected_features))
    print(f"Best validation accuracy during SimplifiedDGWA: {best_score:.4f}")

    # --- PHASE 4: FINAL EVALUATION ---
    print("\n🔬 Evaluating final model on the held-out test set...")
    final_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Select the best features from the full training and final test sets
    X_train_final_selected = X_train_full[selected_features]
    X_test_final_selected = X_test_final[selected_features]

    # Train on the FULL training data and predict on the FINAL test data
    final_clf.fit(X_train_final_selected, y_train_full)
    y_pred_final = final_clf.predict(X_test_final_selected)

    evaluate_model(y_test_final, y_pred_final)

    # Save the list of selected features
    output_path = PROJECT_ROOT / "results" / "tables" / "simplified_dgwa_selected_features.csv"
    pd.DataFrame({"Selected_Features": list(selected_features)}).to_csv(output_path, index=False)
    print(f"\n✅ Saved selected features to: {output_path}")


def evaluate_model(y_true, y_pred):
    """Calculates and prints multi-class evaluation metrics."""
    # --- CORRECTION 2: Use 'weighted' average for multi-class metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Robust multi-class specificity calculation
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    specificity = np.mean(tn / (tn + fp))

    print("\n--- Final Evaluation Metrics ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    run_dgwa_feature_selection()