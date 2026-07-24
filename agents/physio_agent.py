"""
agents/physio_agent.py
Physio Agent — wraps PPOAgent for WESAD physio features
(EDA, EMG, Temp, Resp, HRV) to predict stress risk + confidence.
"""
import numpy as np
import pandas as pd
from agents.base_agent import PPOAgent

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_slope",
    "emg_mean_abs", "emg_std",
    "temp_mean", "temp_std",
    "resp_mean", "resp_std",
    "hrv_mean_rr", "hrv_std_rr", "hrv_rmssd",
]


def load_physio_data(path: str = "data/processed/wesad_features.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURE_COLS)  # drop windows with failed HRV extraction
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    subjects = df["subject"].values
    return X, y, subjects


def normalize_features(X: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


class PhysioAgent:
    def __init__(self, input_dim: int = len(FEATURE_COLS)):
        self.agent = PPOAgent(input_dim=input_dim, n_actions=2)
        self.feature_mean = None
        self.feature_std = None

    def fit_normalizer(self, X: np.ndarray):
        _, self.feature_mean, self.feature_std = normalize_features(X)

    def predict(self, x_raw: np.ndarray):
        x_norm = (x_raw - self.feature_mean) / self.feature_std
        return self.agent.predict(x_norm)  # (action, confidence)

    def select_action(self, x_raw: np.ndarray):
        x_norm = (x_raw - self.feature_mean) / self.feature_std
        return self.agent.select_action(x_norm)

    def save(self, path: str = "results/physio_agent.pt"):
        self.agent.save(path)

    def load(self, path: str = "results/physio_agent.pt"):
        self.agent.load(path)