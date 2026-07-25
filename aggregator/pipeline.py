"""
aggregator/pipeline.py
End-to-end integration: loads trained physio, behavior, context agents,
runs one sample through each, and fuses via the LLM aggregator.

LIMITATION (documented for paper): WESAD (physio+context) and Reddit
(behavior) are separate datasets from different individuals — there is
no single dataset with synchronized physio+behavior+context from the
same person. Physio and context predictions here come from the SAME
WESAD window (genuinely paired). Behavior prediction comes from an
independently sampled Reddit example. This demonstrates the fusion
mechanism, not literal same-person real-time monitoring.
"""
import numpy as np
from agents.physio_agent import PhysioAgent, load_physio_data, FEATURE_COLS as PHYSIO_COLS
from agents.behavior_agent import BehaviorAgent, load_behavior_data
from agents.context_agent import ContextAgent, load_context_data, CONTEXT_FEATURE_COLS
from aggregator.llm_aggregator import aggregate
from explainability.shap_utils import compute_shap_values, top_features_for_sample
from explainability.rationale_eval import build_explanation_record, print_explanation_record


def load_trained_agents():
    physio = PhysioAgent(input_dim=len(PHYSIO_COLS))
    physio.load("results/physio_agent.pt")
    X_physio, _, _ = load_physio_data()
    physio.fit_normalizer(X_physio)  # refit normalizer stats (not saved in checkpoint)

    context = ContextAgent(input_dim=len(CONTEXT_FEATURE_COLS))
    context.load("results/context_agent.pt")
    X_context, _, _ = load_context_data()
    context.fit_normalizer(X_context)

    X_behavior, _ = load_behavior_data()
    behavior = BehaviorAgent(input_dim=X_behavior.shape[1])
    behavior.load("results/behavior_agent.pt")

    return physio, behavior, context


def run_pipeline_once(seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    physio, behavior, context = load_trained_agents()

    # sample one paired WESAD window (physio + context, same window index)
    X_physio, y_physio, subjects = load_physio_data()
    idx = np.random.randint(len(X_physio))
    physio_action, physio_conf = physio.predict(X_physio[idx])

    X_context, y_context, _ = load_context_data()
    context_action, context_conf = context.predict(X_context[idx])  # same row = same window

    # sample one independent Reddit example
    X_behavior, y_behavior = load_behavior_data()
    b_idx = np.random.randint(len(X_behavior))
    behavior_action, behavior_conf = behavior.predict(X_behavior[b_idx])

    print("Agent predictions:")
    print(f"  Physio:   risk={physio_action}, conf={physio_conf:.3f} (true label={y_physio[idx]}, subject={subjects[idx]})")
    print(f"  Behavior: risk={behavior_action}, conf={behavior_conf:.3f} (true label={y_behavior[b_idx]})")
    print(f"  Context:  risk={context_action}, conf={context_conf:.3f} (true label={y_context[idx]})")

    result = aggregate(
        physio_risk=physio_action, physio_conf=physio_conf,
        behavior_risk=behavior_action, behavior_conf=behavior_conf,
        context_risk=context_action, context_conf=context_conf,
    )

    print("\nAggregated result:")
    print(f"  Risk level: {result['risk_level']}")
    print(f"  Rationale:  {result['rationale']}")

    # SHAP explanations for physio + context
    bg_idx = np.random.choice(len(X_physio), size=50, replace=False)
    X_physio_norm = (X_physio - physio.feature_mean) / physio.feature_std
    physio_shap = compute_shap_values(physio.agent.policy, X_physio_norm[bg_idx], X_physio_norm[idx:idx+1])
    physio_top = top_features_for_sample(physio_shap[0], PHYSIO_COLS)

    X_context_norm = (X_context - context.feature_mean) / context.feature_std
    context_shap = compute_shap_values(context.agent.policy, X_context_norm[bg_idx], X_context_norm[idx:idx+1])
    context_top = top_features_for_sample(context_shap[0], CONTEXT_FEATURE_COLS)

    record = build_explanation_record(
        physio_action, physio_conf, physio_top,
        behavior_action, behavior_conf,
        context_action, context_conf, context_top,
        result,
    )
    print("\n=== Full Explanation Record ===")
    print_explanation_record(record)

    return record

if __name__ == "__main__":
    run_pipeline_once(seed=42)