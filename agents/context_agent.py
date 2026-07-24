"""
agents/context_agent.py
Context Agent — lightweight policy on temporal/session-derived features.
WESAD has no real wall-clock timestamps, so context features are derived
from window position within each subject's recording session:
  - window_index: position of this window in the subject's sequence
  - session_progress: window_index / total_windows_for_subject (0-1)
  - elapsed_sec: window_index * STEP_SEC (proxy for "time into session")
These act as a stand-in for time-of-day / session-gap context signals.
"""
import numpy as np
import pandas as pd
from agents.base_agent import PPOAgent

STEP_SEC = 30  # must match preprocess_wesad.py STEP_SEC
CONTEXT_FEATURE_COLS = ["window_index", "session_progress", "elapsed_sec"]


def load_context_data(path: str = "data/processed/wesad_features.csv"):
    df = pd.read_csv(path)
    df["window_index"] = df.groupby("subject").cumcount()
    max_idx = df.groupby("subject")["window_index"].transform("max")
    df["session_progress"] = df["window_index"] / max_idx.replace(0, 1)
    df["elapsed_sec"] = df["window_index"] * STEP_SEC

    X = df[CONTEXT_FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    subjects = df["subject"].values
    return X, y, subjects


class ContextAgent:
    def __init__(self, input_dim: int = len(CONTEXT_FEATURE_COLS)):
        self.agent = PPOAgent(input_dim=input_dim, n_actions=2)
        self.feature_mean = None
        self.feature_std = None

    def fit_normalizer(self, X: np.ndarray):
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8

    def predict(self, x_raw: np.ndarray):
        x_norm = (x_raw - self.feature_mean) / self.feature_std
        return self.agent.predict(x_norm)

    def select_action(self, x_raw: np.ndarray):
        x_norm = (x_raw - self.feature_mean) / self.feature_std
        return self.agent.select_action(x_norm)

    def save(self, path: str = "results/context_agent.pt"):
        self.agent.save(path)

    def load(self, path: str = "results/context_agent.pt"):
        self.agent.load(path)