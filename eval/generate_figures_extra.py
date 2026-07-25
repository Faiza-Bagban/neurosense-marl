"""
eval/generate_figures_extra.py
Additional figure variety: confusion matrices, ROC curves, feature
correlation heatmap.
Styling: bold text throughout, font size 14-16, visible spines.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

OUT_DIR = "paper/figures"

plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.5,
    "axes.edgecolor": "black",
})


def style_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")


def fig_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-risk", "Risk"], fontweight="bold")
    ax.set_yticklabels(["Non-risk", "Risk"], fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")
    cbar = plt.colorbar(im)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    style_spines(ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"Saved {filename}")


def fig_roc_curve(y_true, y_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance", linewidth=2)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    leg = ax.legend()
    for text in leg.get_texts():
        text.set_fontweight("bold")
    style_spines(ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"Saved {filename}")


def fig_feature_correlation_heatmap():
    from agents.physio_agent import load_physio_data, FEATURE_COLS
    X, y, _ = load_physio_data()
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=90, fontsize=12, fontweight="bold")
    ax.set_yticklabels(FEATURE_COLS, fontsize=12, fontweight="bold")
    ax.set_title("Physio Feature Correlation Heatmap", fontweight="bold")
    cbar = plt.colorbar(im)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    style_spines(ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/physio_feature_correlation.png", dpi=150)
    plt.close()
    print("Saved physio_feature_correlation.png")


def fig_physio_confusion_and_roc():
    from agents.physio_agent import PhysioAgent, load_physio_data, FEATURE_COLS
    X, y, subjects = load_physio_data()

    test_mask = np.isin(subjects, ["S16", "S17"])
    train_mask = ~test_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    agent = PhysioAgent(input_dim=len(FEATURE_COLS))
    agent.load("results/physio_agent.pt")
    agent.fit_normalizer(X_train)

    preds, scores = [], []
    for x in X_test:
        action, _ = agent.predict(x)
        x_norm = (x - agent.feature_mean) / agent.feature_std
        conf_pos = agent.agent.predict_proba_positive(x_norm)
        preds.append(action)
        scores.append(conf_pos)

    fig_confusion_matrix(y_test, preds, "Physio Agent — Confusion Matrix", "physio_confusion_matrix.png")
    fig_roc_curve(y_test, scores, "Physio Agent — ROC Curve", "physio_roc_curve.png")


def fig_behavior_confusion_and_roc():
    from sklearn.model_selection import train_test_split
    from agents.behavior_agent import BehaviorAgent, load_behavior_data

    X, y = load_behavior_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    agent = BehaviorAgent(input_dim=X.shape[1])
    agent.load("results/behavior_agent.pt")

    preds, scores = [], []
    for x in X_test:
        action, _ = agent.predict(x)
        conf_pos = agent.agent.predict_proba_positive(x)
        preds.append(action)
        scores.append(conf_pos)

    fig_confusion_matrix(y_test, preds, "Behavior Agent — Confusion Matrix", "behavior_confusion_matrix.png")
    fig_roc_curve(y_test, scores, "Behavior Agent — ROC Curve", "behavior_roc_curve.png")


if __name__ == "__main__":
    fig_physio_confusion_and_roc()
    fig_behavior_confusion_and_roc()
    fig_feature_correlation_heatmap()
    print("\nAll extra figures saved to paper/figures/")