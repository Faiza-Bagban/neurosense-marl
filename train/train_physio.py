"""
train/train_physio.py
Train the Physio Agent (PPO) on WESAD windowed features.
Subject-wise train/test split to avoid leakage (same subject's windows
never split across train and test).
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from agents.physio_agent import PhysioAgent, load_physio_data, FEATURE_COLS

N_EPISODES = 300
BATCH_SIZE = 64
TEST_SUBJECTS = ["S16", "S17"]  # held out for eval


def make_episode_batch(X, y, batch_size):
    idx = np.random.choice(len(X), size=min(batch_size, len(X)), replace=False)
    return X[idx], y[idx]


def train():
    X, y, subjects = load_physio_data()

    train_mask = ~np.isin(subjects, TEST_SUBJECTS)
    test_mask = np.isin(subjects, TEST_SUBJECTS)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train windows: {len(X_train)}, Test windows: {len(X_test)}")

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

        p_loss, v_loss = agent.agent.update(batch_X, actions, log_probs, rewards)

        if (episode + 1) % 50 == 0:
            train_acc = np.mean(np.array(actions) == batch_y)
            print(f"Episode {episode+1}: policy_loss={p_loss:.4f}, value_loss={v_loss:.4f}, batch_acc={train_acc:.3f}")

    # ---- eval on held-out subjects ----
    preds, confidences = [], []
    for x in X_test:
        action, conf = agent.predict(x)
        preds.append(action)
        x_norm = (x - agent.feature_mean) / agent.feature_std
        confidences.append(agent.agent.predict_proba_positive(x_norm))
    preds = np.array(preds)

    acc = np.mean(preds == y_test)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, confidences)
    except ValueError:
        auc = float("nan")

    print(f"\nTest results (held-out subjects {TEST_SUBJECTS}):")
    print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    agent.save("results/physio_agent.pt")
    print("Saved model to results/physio_agent.pt")


if __name__ == "__main__":
    train()