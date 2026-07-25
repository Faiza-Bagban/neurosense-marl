"""
explainability/shap_utils.py
SHAP-based explainability for the physio and context agents' policy
networks. Uses shap.DeepExplainer directly on the PyTorch policy net.
"""
import numpy as np
import torch
import shap


def compute_shap_values(policy_net: torch.nn.Module, X_background: np.ndarray,
                         X_explain: np.ndarray, device: str = None):
    """
    policy_net: the PPOAgent's .policy network (outputs logits for 2 classes)
    X_background: normalized feature array used as SHAP background distribution
                   (sample ~50-100 rows, full set is slow)
    X_explain: normalized feature array of samples to explain
    Returns: shap_values for class 1 (risk/stress class), shape (n_explain, n_features)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = policy_net.to(device)
    policy_net.eval()

    bg_t = torch.tensor(X_background, dtype=torch.float32, device=device)
    explain_t = torch.tensor(X_explain, dtype=torch.float32, device=device)

    explainer = shap.DeepExplainer(policy_net, bg_t)
    shap_values = explainer.shap_values(explain_t)

    # shap_values shape: (n_explain, n_features, n_classes) in recent shap versions
    if isinstance(shap_values, list):
        class1_values = shap_values[1]  # class index 1 = risk/stress
    else:
        class1_values = shap_values[:, :, 1]

    return class1_values


def top_features_for_sample(shap_row: np.ndarray, feature_names: list, top_k: int = 3):
    """Return top-k features by absolute SHAP contribution for one sample."""
    idx_sorted = np.argsort(-np.abs(shap_row))[:top_k]
    return [(feature_names[i], float(shap_row[i])) for i in idx_sorted]


def summarize_feature_importance(shap_values: np.ndarray, feature_names: list):
    """Mean absolute SHAP value per feature across all explained samples."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranking = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])
    return ranking