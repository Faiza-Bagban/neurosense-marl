"""
aggregator/llm_aggregator.py
Fuses outputs from physio, behavior, and context agents (risk score +
confidence each) using a local Ollama LLM to produce a final risk level
and natural-language rationale (explainability).
"""
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"

PROMPT_TEMPLATE = """You are a mental health risk aggregation assistant. You are given outputs
from three independent agents monitoring different signals of the same person:

- Physio Agent (physiological signals: HRV, EDA, EMG, temperature, respiration): risk={physio_risk}, confidence={physio_conf:.2f}
- Behavior Agent (text/language patterns): risk={behavior_risk}, confidence={behavior_conf:.2f}
- Context Agent (session/temporal context): risk={context_risk}, confidence={context_conf:.2f}

Each risk value is 0 (low risk / non-stress) or 1 (elevated risk / stress-depression indicator).

Based on these three signals, provide:
1. A final risk level: "low", "moderate", or "high"
2. A short 2-3 sentence rationale explaining how the agents' signals were weighed to reach this conclusion, mentioning which agent(s) most influenced the decision.

Respond ONLY in this exact JSON format, no other text:
{{"risk_level": "...", "rationale": "..."}}
"""


def call_ollama(prompt: str, model: str = MODEL_NAME, timeout: int = 60) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]


def aggregate(physio_risk: int, physio_conf: float,
              behavior_risk: int, behavior_conf: float,
              context_risk: int, context_conf: float) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        physio_risk=physio_risk, physio_conf=physio_conf,
        behavior_risk=behavior_risk, behavior_conf=behavior_conf,
        context_risk=context_risk, context_conf=context_conf,
    )
    raw_output = call_ollama(prompt)

    # LLM sometimes wraps JSON in markdown fences — strip those first
    cleaned = raw_output.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {"risk_level": "unknown", "rationale": f"[parse error] raw output: {raw_output}"}
    return parsed


if __name__ == "__main__":
    # quick sanity test with dummy values
    result = aggregate(
        physio_risk=1, physio_conf=0.91,
        behavior_risk=1, behavior_conf=0.78,
        context_risk=0, context_conf=0.55,
    )
    print(json.dumps(result, indent=2))