"""
Random Forest with RandomizedSearchCV
Baseline: TF-IDF features from prepare_data()
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from src.train import prepare_data  
from src.utils import RAW_DIR, RESULTS_DIR, MODELS_DIR, LOG_DIR, SEED, set_seed, save_model
from src.preprocessing import preprocess_dataset

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
    Saves per-fold metrics for statistical testing.
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

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features="log2",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
        )

        rf.fit(X_train, y_train)

        # Parameter distributions for RandomizedSearchCV
        # param_dist = {
        #     "n_estimators": [100, 200, 400, 600],
        #     "max_depth": [None, 10, 20, 30, 50],
        #     "max_features": ["sqrt", "log2", None],
        #     "min_samples_split": [2, 5, 10],
        #     "min_samples_leaf": [1, 2, 4],
        #     "bootstrap": [True, False],
        # }

        # Predictions
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        y_pred_all[test_idx], y_proba_all[test_idx] = y_pred, y_proba

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0
        )
        fold_metrics.append({
            "fold": fold,
            "test_acc": acc,
            "test_prec": prec,
            "test_rec": rec,
            "test_f1": f1,
        })

        print(f"Fold {fold:02d} — Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f}")

    # Aggregate mean metrics
    metrics_df = pd.DataFrame(fold_metrics)
    mean_acc, mean_f1 = metrics_df["test_acc"].mean(), metrics_df["test_f1"].mean()
    print(f"\n📊 Mean CV Accuracy: {mean_acc:.4f} | F1: {mean_f1:.4f}")

    # Save results
    df["predicted_label_baseline"] = y_pred_all
    df["predicted_label_baseline"] = df["predicted_label_baseline"].map({1: "fake", 0: "real"})
    df["predicted_proba_baseline"] = y_proba_all
    df.to_csv(RESULTS_DIR / "baseline_predictions_full.csv", index=False)
    metrics_df.to_csv(RESULTS_DIR / "baseline_fold_metrics.csv", index=False)

    print(f"\n💾 Saved per-fold metrics to baseline_fold_metrics.csv")
    print(f"💾 Saved full predictions to baseline_predictions_full.csv")

    return metrics_df

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


def run_rf_with_features(X_train, X_test, y_train, y_test, *, random_state=42):
    """
    Train and evaluate a Random Forest on given features, returning both train and test metrics.
    Method used for training the hybrid model
    """
    rf = RandomForestClassifier(
        n_estimators=2000,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=1,  # or 2
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