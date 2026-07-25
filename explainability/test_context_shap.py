"""
explainability/test_context_shap.py
SHAP feature importance for the context agent.
"""
import numpy as np
from agents.context_agent import ContextAgent, load_context_data, CONTEXT_FEATURE_COLS
from explainability.shap_utils import compute_shap_values, summarize_feature_importance, top_features_for_sample

agent = ContextAgent(input_dim=len(CONTEXT_FEATURE_COLS))
agent.load("results/context_agent.pt")

X, y, subjects = load_context_data()
agent.fit_normalizer(X)
X_norm = (X - agent.feature_mean) / agent.feature_std

np.random.seed(0)
bg_idx = np.random.choice(len(X_norm), size=50, replace=False)
explain_idx = np.random.choice(len(X_norm), size=20, replace=False)

X_background = X_norm[bg_idx]
X_explain = X_norm[explain_idx]

shap_values = compute_shap_values(agent.agent.policy, X_background, X_explain)

print("=== Global feature importance (mean |SHAP|) ===")
ranking = summarize_feature_importance(shap_values, CONTEXT_FEATURE_COLS)
for name, val in ranking:
    print(f"  {name}: {val:.4f}")

print("\n=== Example: top features for sample 0 ===")
top3 = top_features_for_sample(shap_values[0], CONTEXT_FEATURE_COLS, top_k=len(CONTEXT_FEATURE_COLS))
for name, val in top3:
    print(f"  {name}: {val:+.4f}")