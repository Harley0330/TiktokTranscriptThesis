"""
Handles Random Forest training and Evaluation with Stratified K-Fold CV
Baseline model: TF-IDF features only.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from train import prepare_data, get_folds

def run_rf_baseline_cv(csv_path, n_splits=5, random_state=42):
    """
    Train and evals Random Forest using Stratified K-Fold CV
    """
    X,y, vectorizer = prepare_data(csv_path)
    accs, precs, recs, f1s = [], [], [], []
    fold = 1

    for train_idx, test_idx, in get_folds(X,y,n_splits=n_splits, random_state=random_state):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #Initialize Random Forest

        rf = RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state
        )
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)

        #Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0
        )

        print(f"Fold {fold}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision :{prec:.4f} Recall {rec:.4f} F1: {f1:.4f}")

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        fold += 1
    
    print("Mean Performance over Folds")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"F1-score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__ == "__main__":
    dataset_path = "../data/data_cleaned.csv"   # adjust if needed
    run_rf_baseline_cv(dataset_path, n_splits=5)