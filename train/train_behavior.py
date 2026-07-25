"""
train/train_behavior.py
Train the Behavior Agent (PPO) on Reddit TF-IDF text features.
Random train/test split (no subject/user grouping available in this dataset).
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from agents.behavior_agent import BehaviorAgent, load_behavior_data

N_EPISODES = 400
BATCH_SIZE = 128
TEST_SIZE = 0.15
RANDOM_SEED = 42


def make_episode_batch(X, y, batch_size):
    idx = np.random.choice(len(X), size=min(batch_size, len(X)), replace=False)
    return X[idx], y[idx]


def train():
    X, y = load_behavior_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    agent = BehaviorAgent(input_dim=X.shape[1])

    for episode in range(N_EPISODES):
        batch_X, batch_y = make_episode_batch(X_train, y_train, BATCH_SIZE)

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

    # ---- eval ----
    preds, confidences = [], []
    for x in X_test:
        action, conf = agent.predict(x)
        preds.append(action)
        confidences.append(agent.agent.predict_proba_positive(x))
    preds = np.array(preds)

    acc = np.mean(preds == y_test)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, confidences)

    # print(f"\nTest results:")
    # print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    # agent.save("results/behavior_agent.pt")
    # print("Saved model to results/behavior_agent.pt")

    print(f"\nTest results:")
    print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    import pandas as pd
    pd.DataFrame([{"acc": acc, "f1": f1, "auc": auc, "n_test": len(X_test)}]).to_csv(
        "results/behavior_results.csv", index=False
    )
    print("Saved results to results/behavior_results.csv")

    agent.save("results/behavior_agent.pt")
    print("Saved model to results/behavior_agent.pt")


if __name__ == "__main__":
    train()