"""
explainability/rationale_eval.py
Combines numeric SHAP explanations with LLM natural-language rationale
into one dual-explainability record, and checks basic consistency
between the LLM's stated reasoning and the actual agent confidences.
"""
import json

AGENT_NAMES = {"physio": "Physio Agent", "behavior": "Behavior Agent", "context": "Context Agent"}


def build_explanation_record(physio_risk, physio_conf, physio_shap_top,
                              behavior_risk, behavior_conf,
                              context_risk, context_conf, context_shap_top,
                              llm_result: dict):
    """
    Assembles one combined explanation record:
      - numeric agent outputs
      - SHAP top features (physio, context — behavior uses TF-IDF, less
        interpretable per-token without extra vocab mapping, so SHAP
        applied to physio/context only, as per project scope)
      - LLM risk level + rationale
      - consistency flag: does LLM's most-influential-agent claim match
        the agent with highest confidence numerically
    """
    confidences = {"physio": physio_conf, "behavior": behavior_conf, "context": context_conf}
    most_confident_agent = max(confidences, key=confidences.get)
    most_confident_name = AGENT_NAMES[most_confident_agent]

    rationale_text = llm_result.get("rationale", "")
    mentions_most_confident = most_confident_name.lower() in rationale_text.lower()

    record = {
        "agent_outputs": {
            "physio": {"risk": physio_risk, "confidence": physio_conf, "top_shap_features": physio_shap_top},
            "behavior": {"risk": behavior_risk, "confidence": behavior_conf},
            "context": {"risk": context_risk, "confidence": context_conf, "top_shap_features": context_shap_top},
        },
        "llm_risk_level": llm_result.get("risk_level"),
        "llm_rationale": rationale_text,
        "most_confident_agent": most_confident_name,
        "rationale_mentions_most_confident_agent": mentions_most_confident,
    }
    return record


def print_explanation_record(record: dict):
    print(json.dumps(record, indent=2))
    if not record["rationale_mentions_most_confident_agent"]:
        print(f"\n[NOTE] LLM rationale did not explicitly name the most-confident agent "
              f"({record['most_confident_agent']}) — worth flagging in qualitative review.")