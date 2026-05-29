"""
visualization.py - Generate EDA plots for the Streamlit app and output/figures/.

All functions return matplotlib Figure objects so Streamlit can display them
with st.pyplot(fig) without writing to disk.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
from utils import CATEGORIES, PATHS

PALETTE = {
    "intern":           "#6EC6F5",
    "junior":           "#4A90D9",
    "senior":           "#2C5F8A",
    "lead":             "#1A2F4A",
    "template_or_clone":"#E07B4A",
    "low_value":        "#C0392B",
}

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── 1. Label distribution ─────────────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df["label"].value_counts().reindex(CATEGORIES, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[l] for l in counts.index], edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_title("Repository Category Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


# ── 2. Stars by category ──────────────────────────────────────────────────────

def plot_stars_by_category(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = [c for c in CATEGORIES if c in df["label"].unique()]
    sns.boxplot(
        data=df, x="label", y="stargazers_count", order=order,
        palette=PALETTE, ax=ax, showfliers=False,
    )
    ax.set_yscale("symlog")
    ax.set_title("Stars Distribution by Category (log scale)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Stars (log)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


# ── 3. Maturity score heatmap ─────────────────────────────────────────────────

def plot_signal_heatmap(df: pd.DataFrame) -> plt.Figure:
    signal_cols = [
        "stargazers_count", "contributors_count", "commit_count_last_year",
        "release_count", "has_ci_workflow", "readme_size_bytes",
        "maturity_score", "doc_score", "community_score",
    ]
    available = [c for c in signal_cols if c in df.columns]
    order = [c for c in CATEGORIES if c in df["label"].unique()]

    # Mean per category, then normalize per column
    grouped = df.groupby("label")[available].mean().reindex(order)
    normalized = (grouped - grouped.min()) / (grouped.max() - grouped.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        normalized.T, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=order, yticklabels=available, ax=ax, linewidths=0.5
    )
    ax.set_title("Normalized Signal Means per Category", fontsize=14, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


# ── 4. Contributors vs Stars scatter ─────────────────────────────────────────

def plot_contributors_vs_stars(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    for label in CATEGORIES:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        ax.scatter(
            np.log1p(sub["contributors_count"]),
            np.log1p(sub["stargazers_count"]),
            label=label, alpha=0.55, s=30,
            color=PALETTE[label],
        )
    ax.set_xlabel("log(1 + Contributors)")
    ax.set_ylabel("log(1 + Stars)")
    ax.set_title("Contributors vs Stars by Category", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, markerscale=1.5)
    plt.tight_layout()
    return fig


# ── 5. CI/CD and README presence ─────────────────────────────────────────────

def plot_quality_signals(df: pd.DataFrame) -> plt.Figure:
    order = [c for c in CATEGORIES if c in df["label"].unique()]
    ci_pct  = df.groupby("label")["has_ci_workflow"].mean().reindex(order) * 100
    readme_pct = df.groupby("label")["has_readme"].mean().reindex(order) * 100

    x = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, ci_pct,     width, label="Has CI/CD (%)",  color="#4A90D9", alpha=0.85)
    ax.bar(x + width/2, readme_pct, width, label="Has README (%)", color="#E07B4A", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("% of Repositories")
    ax.set_title("CI/CD and README Presence by Category", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    return fig


# ── 6. Commit activity distribution ──────────────────────────────────────────

def plot_commit_activity(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = [c for c in CATEGORIES if c in df["label"].unique()]
    sns.violinplot(
        data=df[df["commit_count_last_year"] < df["commit_count_last_year"].quantile(0.95)],
        x="label", y="commit_count_last_year", order=order,
        palette=PALETTE, ax=ax, inner="box", cut=0,
    )
    ax.set_title("Annual Commit Activity by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Commits (last year)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


# ── Save all to disk (called from CLI) ───────────────────────────────────────

def save_all(df: pd.DataFrame):
    plots = {
        "label_distribution.png":      plot_label_distribution,
        "stars_by_category.png":       plot_stars_by_category,
        "signal_heatmap.png":          plot_signal_heatmap,
        "contributors_vs_stars.png":   plot_contributors_vs_stars,
        "quality_signals.png":         plot_quality_signals,
        "commit_activity.png":         plot_commit_activity,
    }
    for fname, fn in plots.items():
        fig = fn(df)
        path = PATHS["figures"] / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    labeled_path = PATHS["labeled"] / "repositories_labeled.csv"
    df = pd.read_csv(labeled_path)
    save_all(df)
