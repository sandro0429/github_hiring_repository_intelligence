"""
evaluation.py - Stage 6: Comprehensive evaluation and error analysis.

Produces:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix (saved as figure)
  - Error analysis: most common misclassifications
  - Baseline comparison: TF-IDF + LogisticRegression vs DistilBERT
  - Saves all metrics to output/metrics/
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import sys
sys.path.append(str(Path(__file__).parent))
from utils import get_logger, PATHS, CATEGORIES, save_json

logger = get_logger("evaluation")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_predictions() -> pd.DataFrame:
    path = PATHS["metrics"] / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test predictions not found at {path}. Run train.py first.")
    return pd.read_csv(path)


def load_splits():
    train = pd.read_csv(PATHS["splits"] / "train.csv")
    val   = pd.read_csv(PATHS["splits"] / "val.csv")
    test  = pd.read_csv(PATHS["splits"] / "test.csv")
    return train, val, test


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close(fig)
    return cm


# ── Full evaluation ───────────────────────────────────────────────────────────

def evaluate_bert_predictions(df: pd.DataFrame) -> dict:
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()

    report_dict = classification_report(
        y_true, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, labels=CATEGORIES, zero_division=0
    )
    logger.info(f"\nClassification Report (DistilBERT):\n{report_str}")

    # Save report
    save_json(report_dict, PATHS["metrics"] / "classification_report_bert.json")
    with open(PATHS["metrics"] / "classification_report_bert.txt", "w") as f:
        f.write(report_str)

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, CATEGORIES,
        title="DistilBERT Confusion Matrix",
        save_path=PATHS["figures"] / "confusion_matrix_bert.png",
    )

    return report_dict


# ── Baseline: TF-IDF + Logistic Regression ───────────────────────────────────

def run_baseline(train_df, test_df) -> dict:
    logger.info("Running baseline: TF-IDF + Logistic Regression …")

    X_train = train_df["summary_full"].fillna("").tolist()
    y_train = train_df["label"].tolist()
    X_test  = test_df["summary_full"].fillna("").tolist()
    y_test  = test_df["label"].tolist()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report_dict = classification_report(
        y_test, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_test, y_pred, labels=CATEGORIES, zero_division=0
    )
    logger.info(f"\nClassification Report (Baseline TF-IDF+LR):\n{report_str}")

    save_json(report_dict, PATHS["metrics"] / "classification_report_baseline.json")
    with open(PATHS["metrics"] / "classification_report_baseline.txt", "w") as f:
        f.write(report_str)

    plot_confusion_matrix(
        y_test, y_pred, CATEGORIES,
        title="Baseline (TF-IDF + LR) Confusion Matrix",
        save_path=PATHS["figures"] / "confusion_matrix_baseline.png",
    )

    # Save predictions for comparison
    test_df = test_df.copy()
    test_df["baseline_pred"] = y_pred
    test_df.to_csv(PATHS["metrics"] / "baseline_predictions.csv", index=False)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "report":   report_dict,
    }


# ── Error analysis ────────────────────────────────────────────────────────────

def error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    errors = df[df["true_label"] != df["predicted_label"]].copy()
    logger.info(f"\nError analysis: {len(errors)} misclassified out of {len(df)} test samples")

    # Most common confusion pairs
    pairs = errors.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")
    pairs = pairs.sort_values("count", ascending=False)
    logger.info(f"\nTop misclassification pairs:\n{pairs.head(10).to_string(index=False)}")

    pairs.to_csv(PATHS["metrics"] / "error_pairs.csv", index=False)

    # Sample errors
    sample = errors.sample(min(10, len(errors)), random_state=42)
    cols = ["full_name", "true_label", "predicted_label", "summary_short"]
    available = [c for c in cols if c in sample.columns]
    sample[available].to_csv(PATHS["metrics"] / "error_samples.csv", index=False)

    return errors


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_model_comparison(bert_report: dict, baseline_report: dict):
    labels = CATEGORIES
    bert_f1     = [bert_report.get(l, {}).get("f1-score", 0) for l in labels]
    baseline_f1 = [baseline_report.get(l, {}).get("f1-score", 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, baseline_f1, width, label="Baseline (TF-IDF+LR)", color="#4A90D9", alpha=0.85)
    ax.bar(x + width/2, bert_f1,     width, label="DistilBERT",           color="#E07B4A", alpha=0.85)

    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("F1 Score by Category: Baseline vs DistilBERT", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = PATHS["figures"] / "model_comparison_f1.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Comparison plot saved to {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    test_preds = load_test_predictions()
    train_df, val_df, test_df = load_splits()

    logger.info("=== DistilBERT evaluation ===")
    bert_report = evaluate_bert_predictions(test_preds)

    logger.info("=== Baseline evaluation ===")
    baseline_results = run_baseline(train_df, test_df)

    logger.info("=== Error analysis ===")
    error_analysis(test_preds)

    logger.info("=== Model comparison ===")
    plot_model_comparison(bert_report, baseline_results["report"])

    # Summary comparison
    summary = {
        "distilbert": {
            "accuracy": bert_report["accuracy"],
            "f1_macro": bert_report["macro avg"]["f1-score"],
        },
        "baseline_tfidf_lr": {
            "accuracy": baseline_results["accuracy"],
            "f1_macro": baseline_results["f1_macro"],
        },
    }
    save_json(summary, PATHS["metrics"] / "model_comparison_summary.json")
    logger.info(f"\nModel comparison:\n{json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    run()
