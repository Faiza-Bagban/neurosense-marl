"""
eval/ablation.py
Ablation study (2-3 configs, per project scope — not exhaustive).
Compares fusion quality on WESAD windows, where physio and context
share genuine ground truth (behavior/Reddit lacks matched ground truth
against WESAD, per the documented cross-dataset limitation — so this
ablation isolates the physio+context pairing where fusion accuracy is
actually verifiable).

Configs:
  1. Physio-only
  2. Context-only
  3. Physio+Context fusion (confidence-weighted average)

Uses the saved best-fold models (from LOSO) — this is a scope-limited
demonstration of fusion benefit, not a fresh LOSO run for each config.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from agents.physio_agent import PhysioAgent, load_physio_data, FEATURE_COLS as PHYSIO_COLS
from agents.context_agent import ContextAgent, load_context_data, CONTEXT_FEATURE_COLS


def fused_prediction(physio_conf: float, context_conf: float, threshold: float = 0.5) -> int:
    """Confidence-weighted average of P(risk=1) from both agents."""
    avg_conf = (physio_conf + context_conf) / 2
    return 1 if avg_conf >= threshold else 0


def run_ablation():
    physio = PhysioAgent(input_dim=len(PHYSIO_COLS))
    physio.load("results/physio_agent.pt")
    X_physio, y_physio, _ = load_physio_data()
    physio.fit_normalizer(X_physio)

    context = ContextAgent(input_dim=len(CONTEXT_FEATURE_COLS))
    context.load("results/context_agent.pt")
    X_context, y_context, _ = load_context_data()
    context.fit_normalizer(X_context)

    # X_physio and X_context share row order/index (same source windows)
    assert len(X_physio) == len(X_context), "row mismatch — physio/context should share window index"
    y_true = y_physio  # same as y_context by construction

    physio_preds, context_preds, fused_preds = [], [], []

    
    for i in range(len(X_physio)):
        p_action, _ = physio.predict(X_physio[i])
        c_action, _ = context.predict(X_context[i])

        p_x_norm = (X_physio[i] - physio.feature_mean) / physio.feature_std
        c_x_norm = (X_context[i] - context.feature_mean) / context.feature_std
        p_conf_pos = physio.agent.predict_proba_positive(p_x_norm)
        c_conf_pos = context.agent.predict_proba_positive(c_x_norm)

        physio_preds.append(p_action)
        context_preds.append(c_action)
        fused_preds.append(fused_prediction(p_conf_pos, c_conf_pos))
        
    configs = {
        "Physio-only": physio_preds,
        "Context-only": context_preds,
        "Physio+Context fusion": fused_preds,
    }

    results = []
    for name, preds in configs.items():
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        print(f"[{name}] acc={acc:.3f} f1={f1:.3f}")
        results.append({"config": name, "acc": acc, "f1": f1})

    df = pd.DataFrame(results)
    df.to_csv("results/ablation_results.csv", index=False)
    print("\nSaved to results/ablation_results.csv")
    print("\nNote: models used here are the best-LOSO-fold checkpoints, evaluated on "
          "the full dataset (not a held-out set) — this ablation demonstrates relative "
          "fusion benefit, not a generalization estimate (see LOSO results for that).")


if __name__ == "__main__":
    run_ablation()