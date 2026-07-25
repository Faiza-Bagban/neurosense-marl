"""
eval/metrics.py
Consolidates all results (LOSO physio/context, behavior test split,
baselines, ablation) into one master summary table for the paper.
"""
import pandas as pd


def load_or_none(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[warn] missing: {path}")
        return None


def build_summary():
    rows = []

    physio_loso = load_or_none("results/physio_loso_results.csv")
    if physio_loso is not None:
        rows.append({
            "component": "Physio Agent (PPO, LOSO)",
            "accuracy": f"{physio_loso['acc'].mean():.3f} ± {physio_loso['acc'].std():.3f}",
            "f1": f"{physio_loso['f1'].mean():.3f} ± {physio_loso['f1'].std():.3f}",
            "auc": f"{physio_loso['auc'].mean():.3f} ± {physio_loso['auc'].std():.3f}",
        })

    context_loso = load_or_none("results/context_loso_results.csv")
    if context_loso is not None:
        rows.append({
            "component": "Context Agent (PPO, LOSO)",
            "accuracy": f"{context_loso['acc'].mean():.3f} ± {context_loso['acc'].std():.3f}",
            "f1": f"{context_loso['f1'].mean():.3f} ± {context_loso['f1'].std():.3f}",
            "auc": f"{context_loso['auc'].mean():.3f} ± {context_loso['auc'].std():.3f}",
        })

    behavior = load_or_none("results/behavior_results.csv")
    if behavior is not None:
        rows.append({
            "component": "Behavior Agent (PPO)",
            "accuracy": f"{behavior['acc'].iloc[0]:.3f}",
            "f1": f"{behavior['f1'].iloc[0]:.3f}",
            "auc": f"{behavior['auc'].iloc[0]:.3f}",
        })

    baselines = load_or_none("results/baseline_results.csv")
    if baselines is not None:
        for _, r in baselines.iterrows():
            rows.append({
                "component": f"Baseline: {r['modality']} / {r['model']}",
                "accuracy": f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f}",
                "f1": f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}",
                "auc": f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}",
            })

    ablation = load_or_none("results/ablation_results.csv")
    if ablation is not None:
        for _, r in ablation.iterrows():
            rows.append({
                "component": f"Ablation: {r['config']}",
                "accuracy": f"{r['acc']:.3f}",
                "f1": f"{r['f1']:.3f}",
                "auc": "-",
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("results/final_summary.csv", index=False)
    return summary_df


if __name__ == "__main__":
    df = build_summary()
    print(df.to_string(index=False))
    print("\nSaved to results/final_summary.csv")