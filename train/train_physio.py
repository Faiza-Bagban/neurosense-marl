"""
train/train_physio.py
Train the Physio Agent (PPO) on WESAD windowed features using
Leave-One-Subject-Out (LOSO) cross-validation — standard practice
for WESAD, avoids subject leakage and gives a robust performance estimate
across all 15 subjects instead of one fixed small held-out split.
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from agents.physio_agent import PhysioAgent, load_physio_data, FEATURE_COLS

N_EPISODES = 200  # per fold — kept lower than single-split version since LOSO runs 15x
BATCH_SIZE = 64


def make_episode_batch(X, y, batch_size):
    idx = np.random.choice(len(X), size=min(batch_size, len(X)), replace=False)
    return X[idx], y[idx]


def train_one_fold(X_train, y_train, X_test, y_test):
    agent = PhysioAgent(input_dim=len(FEATURE_COLS))
    agent.fit_normalizer(X_train)
    X_train_norm = (X_train - agent.feature_mean) / agent.feature_std

    for episode in range(N_EPISODES):
        batch_X, batch_y = make_episode_batch(X_train_norm, y_train, BATCH_SIZE)
        actions, log_probs, rewards = [], [], []
        for x, true_label in zip(batch_X, batch_y):
            action, log_prob, _ = agent.agent.select_action(x)
            reward = 1.0 if action == true_label else -1.0
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
        agent.agent.update(batch_X, actions, log_probs, rewards)

    preds, confidences = [], []
    for x in X_test:
        action, _ = agent.predict(x)
        x_norm = (x - agent.feature_mean) / agent.feature_std
        conf = agent.agent.predict_proba_positive(x_norm)
        preds.append(action)
        confidences.append(conf)
    preds = np.array(preds)

    acc = np.mean(preds == y_test)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, confidences)
    except ValueError:
        auc = float("nan")  # happens if test fold has only one class

    return agent, acc, f1, auc


def loso_train():
    X, y, subjects = load_physio_data()
    unique_subjects = sorted(set(subjects))
    print(f"Running LOSO across {len(unique_subjects)} subjects: {unique_subjects}")

    results = []
    best_agent, best_acc = None, -1

    for test_subj in unique_subjects:
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(set(y_test)) < 2:
            print(f"Fold {test_subj}: skipped (only one class in test fold, {len(X_test)} windows)")
            continue

        agent, acc, f1, auc = train_one_fold(X_train, y_train, X_test, y_test)
        print(f"Fold {test_subj}: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f} (n_test={len(X_test)})")
        results.append({"subject": test_subj, "acc": acc, "f1": f1, "auc": auc, "n_test": len(X_test)})

        if acc > best_acc:
            best_acc = acc
            best_agent = agent

    accs = [r["acc"] for r in results]
    f1s = [r["f1"] for r in results]
    aucs = [r["auc"] for r in results if not np.isnan(r["auc"])]

    print("\n=== LOSO Summary (mean ± std across folds) ===")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"F1:       {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"AUC:      {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (n_folds={len(aucs)})")

    import pandas as pd
    pd.DataFrame(results).to_csv("results/physio_loso_results.csv", index=False)
    print("Saved per-fold results to results/physio_loso_results.csv")

    best_agent.save("results/physio_agent.pt")
    print("Saved best-fold model to results/physio_agent.pt")


if __name__ == "__main__":
    loso_train()