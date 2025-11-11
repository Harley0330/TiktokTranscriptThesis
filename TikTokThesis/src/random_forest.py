"""
Random Forest with RandomizedSearchCV
Baseline: TF-IDF features from prepare_data()
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from src.train import prepare_data, get_folds
from src.utils import RAW_DIR, RESULTS_DIR, MODELS_DIR, LOG_DIR, SEED, set_seed, save_model
from src.preprocessing import preprocess_dataset
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import numpy as np

def run_rf(csv_path, random_state=42):
    """
    Train Random Forest with hyperparameter tuning using RandomizedSearchCV
    """

    set_seed(random_state)
    # Load features and labels
    dataset_path = RAW_DIR
    df = preprocess_dataset(dataset_path)
    X, y, vectorizer = prepare_data(df,csv_path)

    # Train/test split (keep 20% for final eval)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Parameter distributions for RandomizedSearchCV
    # param_dist = {
    #     "n_estimators": [100, 200, 400, 600],
    #     "max_depth": [None, 10, 20, 30, 50],
    #     "max_features": ["sqrt", "log2", None],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4],
    #     "bootstrap": [True, False],
    # }

    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        bootstrap=False
    )

    """
    USED TO CHECK FOR BEST PARAMETERS
    """
    # rf_random = RandomizedSearchCV(
    #     estimator=rf,
    #     param_distributions=param_dist,
    #     n_iter=30,                 # number of random combos
    #     cv=cv,
    #     scoring="f1",              # optimize F1 (can change to "accuracy")
    #     verbose=2,
    #     random_state=random_state,
    #     n_jobs=-1
    # )

    # # Fit RandomizedSearchCV
    # rf_random.fit(X_train, y_train)

    # # Best parameters & CV score
    # print("\nBest Params:", rf_random.best_params_)
    # print("Best CV F1 Score:", rf_random.best_score_)

    # Evaluate best model on test set
    # best_model = rf_random.best_estimator_

    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\nTest Set Performance:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:\n", cm)

    model_path = MODELS_DIR / "rf_final.pkl"
    save_model(rf, model_path)

    # Save results
    results = pd.DataFrame([{
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }])
    results_csv = RESULTS_DIR / "rf_final_results.csv"
    results.to_csv(results_csv, index=False)
    print(f"\nSaved test results to {results_csv}")

    return rf


def run_rf_baseline_cv(df, random_state=42, n_splits=15, max_features=5000):
    """
    Runs a Random Forest baseline using TF-IDF features only,
    with 15-fold Stratified CV (to match the hybrid model setup).
    Saves per-fold metrics and full predictions for statistical testing.
    """

    set_seed(random_state)
    X, y, vectorizer = prepare_data(df, RAW_DIR / "data_cleaned_formatted.csv", max_features=max_features)
    print(f"Loaded {len(y)} samples | TF-IDF shape: {X.shape}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics, y_pred_all, y_proba_all = [], np.zeros_like(y), np.zeros_like(y, dtype=float)

    print(f"\n=== Step 2: Training Random Forest ({n_splits}-fold CV) ===")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Model definition ---
        rf = RandomForestClassifier(
            n_estimators=700,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="log2",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
        )

        rf.fit(X_train, y_train)

        # --- Predictions ---
        y_train_pred = rf.predict(X_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]

        y_pred_all[test_idx], y_proba_all[test_idx] = y_pred, y_proba

        # --- Train/Test Metrics ---
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)

        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            y_train, y_train_pred, average="binary", pos_label=1, zero_division=0
        )
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0
        )

        fold_metrics.append({
            "fold": fold,
            "train_acc": train_acc,
            "train_prec": train_prec,
            "train_rec": train_rec,
            "train_f1": train_f1,
            "test_acc": test_acc,
            "test_prec": test_prec,
            "test_rec": test_rec,
            "test_f1": test_f1,
            "acc_gap": abs(train_acc - test_acc),
            "f1_gap": abs(train_f1 - test_f1),
            "oob": rf.oob_score_,
        })

        print(f"\nFold {fold:02d}")
        print(f"  🔹 Train — Acc:{train_acc:.4f} Prec:{train_prec:.4f} Rec:{train_rec:.4f} F1:{train_f1:.4f}")
        print(f"  🔹 Test  — Acc:{test_acc:.4f} Prec:{test_prec:.4f} Rec:{test_rec:.4f} F1:{test_f1:.4f}")
        print(f"  🔹 Gaps  — Acc Gap:{abs(train_acc - test_acc):.4f} | F1 Gap:{abs(train_f1 - test_f1):.4f}")
        print(f"  🔹 OOB Score:{rf.oob_score_:.4f}")

    # === Aggregate Mean Metrics ===
    metrics_df = pd.DataFrame(fold_metrics)
    mean_train_acc, mean_test_acc = metrics_df["train_acc"].mean(), metrics_df["test_acc"].mean()
    mean_train_prec, mean_test_prec = metrics_df["train_prec"].mean(), metrics_df["test_prec"].mean()
    mean_train_rec, mean_test_rec = metrics_df["train_rec"].mean(), metrics_df["test_rec"].mean()
    mean_train_f1, mean_test_f1 = metrics_df["train_f1"].mean(), metrics_df["test_f1"].mean()
    mean_oob = metrics_df["oob"].mean()
    mean_acc_gap = abs(mean_train_acc - mean_test_acc)
    mean_f1_gap = abs(mean_train_f1 - mean_test_f1)

    print(f"\n📊 Mean Train Accuracy: {mean_train_acc:.4f} | Test Accuracy: {mean_test_acc:.4f}")
    print(f"📊 Mean Train F1: {mean_train_f1:.4f} | Test F1: {mean_test_f1:.4f}")
    print(f"📉 Accuracy Gap: {mean_acc_gap*100:.2f}% | F1 Gap: {mean_f1_gap*100:.2f}%")
    print(f"📊 Mean OOB Score: {mean_oob:.4f}")

    # === Append Summary Row ===
    summary_row = pd.DataFrame([{
        "fold": "mean",
        "train_acc": mean_train_acc,
        "train_prec": mean_train_prec,
        "train_rec": mean_train_rec,
        "train_f1": mean_train_f1,
        "test_acc": mean_test_acc,
        "test_prec": mean_test_prec,
        "test_rec": mean_test_rec,
        "test_f1": mean_test_f1,
        "acc_gap": mean_acc_gap,
        "f1_gap": mean_f1_gap,
        "oob": mean_oob,
    }])
    metrics_df = pd.concat([metrics_df, summary_row], ignore_index=True)

    # === Save Metrics and Predictions ===
    metrics_path = RESULTS_DIR / "baseline_fold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n💾 Saved per-fold + mean metrics to {metrics_path}")

    df["predicted_label_baseline"] = y_pred_all
    df["predicted_label_baseline"] = df["predicted_label_baseline"].map({1: "fake", 0: "real"})
    df["predicted_proba_baseline"] = y_proba_all
    df.to_csv(RESULTS_DIR / "baseline_predictions_full.csv", index=False)
    print(f"💾 Saved full predictions to baseline_predictions_full.csv")

    return metrics_df





def run_rf_with_features(X_train, X_test, y_train, y_test, *, random_state=42):
    """
    Train and evaluate a Random Forest on given features, returning both train and test metrics.
    Method used for training the hybrid model
    """
    rf = RandomForestClassifier(
        n_estimators=1500,
        max_depth=30,
        min_samples_split=20,
        min_samples_leaf=3,  # or 2
        max_samples=0.85,
        max_features="log2",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )

    # Train the model
    rf.fit(X_train, y_train)

    # --- Train predictions ---
    y_train_pred = rf.predict(X_train)
    y_train_proba = rf.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average="binary", pos_label=1, zero_division=0
    )

    # --- Test predictions ---
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )

    return rf, y_pred, y_proba, acc, prec, rec, f1, train_acc, train_prec, train_rec, train_f1

if __name__ == "__main__":
    dataset_path =RAW_DIR 
    #run_rf_baseline_cv(dataset_path)
    run_rf(dataset_path)