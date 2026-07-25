"""
eval/generate_figures.py
Generates paper-ready figures from saved results CSVs.
Saves all as PNG into paper/figures/.
Styling: bold text, visible spines, font sizes tuned per-figure to avoid
legend/label clashes on dense plots.
"""
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "paper/figures"

plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.5,
    "axes.edgecolor": "black",
})


def style_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")


def bold_ticks(ax, size=11):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(size)


def fig_loso_per_subject():
    df = pd.read_csv("results/physio_loso_results.csv")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(df))
    ax.bar(x, df["acc"], label="Accuracy", alpha=0.7, width=0.6)
    ax.bar(x, df["f1"], label="F1", alpha=0.7, width=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(df["subject"], rotation=45, fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Physio Agent — Per-Subject LOSO Performance", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    bold_ticks(ax, size=10)
    leg = ax.legend(fontsize=10, loc="upper center", ncol=2,
                     bbox_to_anchor=(0.5, 1.0), frameon=True)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    style_spines(ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/loso_per_subject.png", dpi=150)
    plt.close()
    print("Saved loso_per_subject.png")


def fig_baseline_comparison():
    baselines = pd.read_csv("results/baseline_results.csv")
    physio_loso = pd.read_csv("results/physio_loso_results.csv")
    behavior = pd.read_csv("results/behavior_results.csv")

    labels, accs, f1s = [], [], []
    for _, r in baselines.iterrows():
        labels.append(f"{r['modality']}\n{r['model']}")
        accs.append(r["acc_mean"])
        f1s.append(r["f1_mean"])

    labels += ["physio\nPPO (ours)", "behavior\nPPO (ours)"]
    accs += [physio_loso["acc"].mean(), behavior["acc"].iloc[0]]
    f1s += [physio_loso["f1"].mean(), behavior["f1"].iloc[0]]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.35
    ax.bar([i - width/2 for i in x], accs, width, label="Accuracy")
    ax.bar([i + width/2 for i in x], f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Baseline Classifiers vs MARL (PPO) Agents", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(10)
    leg = ax.legend(fontsize=10, loc="upper center", ncol=2,
                     bbox_to_anchor=(0.5, 1.0), frameon=True)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    style_spines(ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/baseline_comparison.png", dpi=150)
    plt.close()
    print("Saved baseline_comparison.png")


def fig_ablation():
    df = pd.read_csv("results/ablation_results.csv")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df["acc"], width, label="Accuracy")
    ax.bar([i + width/2 for i in x], df["f1"], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=15, fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Ablation: Physio-only vs Context-only vs Fusion", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(10)
    leg = ax.legend(fontsize=10, loc="upper center", ncol=2,
                     bbox_to_anchor=(0.5, 1.0), frameon=True)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    style_spines(ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ablation.png", dpi=150)
    plt.close()
    print("Saved ablation.png")


def fig_shap_summary():
    import numpy as np
    from agents.physio_agent import PhysioAgent, load_physio_data, FEATURE_COLS
    from explainability.shap_utils import compute_shap_values

    agent = PhysioAgent(input_dim=len(FEATURE_COLS))
    agent.load("results/physio_agent.pt")
    X, y, _ = load_physio_data()
    agent.fit_normalizer(X)
    X_norm = (X - agent.feature_mean) / agent.feature_std

    np.random.seed(0)
    bg_idx = np.random.choice(len(X_norm), size=50, replace=False)
    explain_idx = np.random.choice(len(X_norm), size=100, replace=False)
    shap_values = compute_shap_values(agent.agent.policy, X_norm[bg_idx], X_norm[explain_idx])

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([FEATURE_COLS[i] for i in order], mean_abs[order])
    ax.set_xlabel("Mean |SHAP value|", fontsize=13, fontweight="bold")
    ax.set_title("Physio Agent — SHAP Feature Importance", fontsize=14, fontweight="bold")
    bold_ticks(ax, size=10)
    style_spines(ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_physio_summary.png", dpi=150)
    plt.close()
    print("Saved shap_physio_summary.png")


if __name__ == "__main__":
    fig_loso_per_subject()
    fig_baseline_comparison()
    fig_ablation()
    fig_shap_summary()
    print("\nAll figures saved to paper/figures/")