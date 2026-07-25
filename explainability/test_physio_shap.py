"""
explainability/test_physio_shap.py
Quick test: compute and print SHAP feature importance for the physio agent.
"""
import numpy as np
from agents.physio_agent import PhysioAgent, load_physio_data, FEATURE_COLS
from explainability.shap_utils import compute_shap_values, summarize_feature_importance, top_features_for_sample

agent = PhysioAgent(input_dim=len(FEATURE_COLS))
agent.load("results/physio_agent.pt")

X, y, subjects = load_physio_data()
agent.fit_normalizer(X)
X_norm = (X - agent.feature_mean) / agent.feature_std

np.random.seed(0)
bg_idx = np.random.choice(len(X_norm), size=50, replace=False)
explain_idx = np.random.choice(len(X_norm), size=20, replace=False)

X_background = X_norm[bg_idx]
X_explain = X_norm[explain_idx]

shap_values = compute_shap_values(agent.agent.policy, X_background, X_explain)

print("=== Global feature importance (mean |SHAP|) ===")
ranking = summarize_feature_importance(shap_values, FEATURE_COLS)
for name, val in ranking:
    print(f"  {name}: {val:.4f}")

print("\n=== Example: top-3 features for sample 0 ===")
top3 = top_features_for_sample(shap_values[0], FEATURE_COLS)
for name, val in top3:
    print(f"  {name}: {val:+.4f}")