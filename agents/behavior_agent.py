"""
agents/behavior_agent.py
Behavior Agent — wraps PPOAgent for Reddit TF-IDF text features
to predict depression risk + confidence.
"""
import numpy as np
from scipy import sparse
import pandas as pd
from agents.base_agent import PPOAgent


def load_behavior_data(
    tfidf_path: str = "data/processed/reddit_tfidf.npz",
    labels_path: str = "data/processed/reddit_labels.csv",
):
    X = sparse.load_npz(tfidf_path).toarray().astype(np.float32)
    y = pd.read_csv(labels_path)["label"].values.astype(np.int64)
    return X, y


class BehaviorAgent:
    def __init__(self, input_dim: int):
        self.agent = PPOAgent(input_dim=input_dim, n_actions=2)

    def predict(self, x_raw: np.ndarray):
        return self.agent.predict(x_raw)  # (action, confidence)

    def select_action(self, x_raw: np.ndarray):
        return self.agent.select_action(x_raw)

    def save(self, path: str = "results/behavior_agent.pt"):
        self.agent.save(path)

    def load(self, path: str = "results/behavior_agent.pt"):
        self.agent.load(path)