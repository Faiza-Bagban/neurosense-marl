"""
eval/baselines.py
Baseline models: single-agent, no-fusion comparison.
Uses standard sklearn classifiers (Logistic Regression, Random Forest)
directly on the same features the RL agents use — no RL, no LLM fusion.
Physio uses LOSO (matches physio agent protocol); Behavior uses a
stratified train/test split (matches behavior agent protocol).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from agents.physio_agent import load_physio_data
from agents.behavior_agent import load_behavior_data

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}


def run_physio_baselines():
    X, y, subjects = load_physio_data()
    unique_subjects = sorted(set(subjects))
    results = []

    for model_name, model_fn in MODELS.items():
        accs, f1s, aucs = [], [], []
        for test_subj in unique_subjects:
            train_mask = subjects != test_subj
            test_mask = subjects == test_subj
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            if len(set(y_test)) < 2:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = model_fn()
            clf.fit(X_train_s, y_train)
            preds = clf.predict(X_test_s)
            probs = clf.predict_proba(X_test_s)[:, 1]

            accs.append(accuracy_score(y_test, preds))
            f1s.append(f1_score(y_test, preds, zero_division=0))
            try:
                aucs.append(roc_auc_score(y_test, probs))
            except ValueError:
                pass

        results.append({
            "modality": "physio", "model": model_name,
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
            "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
        })
        print(f"[Physio/{model_name}] acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
              f"f1={np.mean(f1s):.3f}±{np.std(f1s):.3f} auc={np.mean(aucs):.3f}±{np.std(aucs):.3f}")

    return results


def run_behavior_baselines():
    X, y = load_behavior_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    results = []

    for model_name, model_fn in MODELS.items():
        clf = model_fn()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)

        results.append({
            "modality": "behavior", "model": model_name,
            "acc_mean": acc, "acc_std": 0.0,
            "f1_mean": f1, "f1_std": 0.0,
            "auc_mean": auc, "auc_std": 0.0,
        })
        print(f"[Behavior/{model_name}] acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")

    return results


if __name__ == "__main__":
    all_results = []
    print("=== Physio baselines (LOSO) ===")
    all_results += run_physio_baselines()
    print("\n=== Behavior baselines (stratified split) ===")
    all_results += run_behavior_baselines()

    df = pd.DataFrame(all_results)
    df.to_csv("results/baseline_results.csv", index=False)
    print("\nSaved to results/baseline_results.csv")