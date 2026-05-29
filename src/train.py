"""
train.py - Stages 4 & 5: Split data and fine-tune DistilBERT classifier.

Model choice: distilbert-base-uncased
  - 40% smaller and 60% faster than BERT-base
  - Retains 97% of BERT's performance on classification tasks
  - Ideal for academic projects: fits in free Colab GPU (T4)

Input:  summary_full  (richer text than summary_short — better for BERT)
Output: one of 6 category labels

Training details:
  - 70/15/15 split (stratified by label)
  - 3 epochs (enough to converge on this dataset size)
  - AdamW optimizer, linear scheduler with warmup
  - Batch size 16 (safe for T4 16GB / local MPS)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent))
from utils import get_logger, PATHS, CATEGORIES, save_json

logger = get_logger("train")

MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 512
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
WARMUP_FRAC = 0.1
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

class RepoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Splits ────────────────────────────────────────────────────────────────────

def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """70/15/15 stratified split. Returns train, val, test DataFrames + encoder."""
    le = LabelEncoder()
    le.classes_ = np.array(CATEGORIES)   # fixed order
    df = df.copy()
    df["label_id"] = le.transform(df["label"])

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label_id"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=SEED
    )

    # Save splits
    train_df.to_csv(PATHS["splits"] / "train.csv", index=False)
    val_df.to_csv(PATHS["splits"]   / "val.csv",   index=False)
    test_df.to_csv(PATHS["splits"]  / "test.csv",  index=False)

    logger.info(f"Splits — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Save encoder classes
    save_json(le.classes_.tolist(), PATHS["models"] / "label_classes.json")
    return train_df, val_df, test_df, le


# ── Training ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            loss    = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score
    return {
        "loss":     total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "preds":    all_preds,
        "labels":   all_labels,
    }


def train(df: pd.DataFrame = None):
    """Full training pipeline."""

    # Load data
    if df is None:
        labeled_path = PATHS["labeled"] / "repositories_labeled.csv"
        if not labeled_path.exists():
            raise FileNotFoundError(f"Labeled data not found at {labeled_path}. Run llm_labeling.py first.")
        df = pd.read_csv(labeled_path)

    # Ensure all labels are valid
    df = df[df["label"].isin(CATEGORIES)].reset_index(drop=True)
    logger.info(f"Training on {len(df)} labeled repos")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # Create splits
    train_df, val_df, test_df, le = create_splits(df)

    # Tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def make_loader(split_df, shuffle=False):
        texts  = split_df["summary_full"].fillna("").tolist()
        labels = split_df["label_id"].tolist()
        dataset = RepoDataset(texts, labels, tokenizer)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df)
    test_loader  = make_loader(test_df)

    # Model
    device = get_device()
    logger.info(f"Device: {device}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CATEGORIES)
    )
    model.to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    history = []
    best_val_f1 = 0.0
    model_save_path = PATHS["models"] / "distilbert_repo_classifier"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if k not in ("preds", "labels")},
        })

        # Save best model
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"  -> Best model saved (val_f1={best_val_f1:.4f})")

    # Final evaluation on test set
    logger.info("Evaluating best model on test set …")
    model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device)

    logger.info(
        f"TEST | acc={test_metrics['accuracy']:.4f} | f1={test_metrics['f1_macro']:.4f}"
    )

    # Save results
    results = {
        "training_history": history,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["f1_macro"],
            "loss":     test_metrics["loss"],
        },
        "label_classes": le.classes_.tolist(),
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
    }
    save_json(results, PATHS["metrics"] / "training_results.json")

    # Save test predictions for error analysis
    test_df = test_df.copy()
    test_df["predicted_label_id"] = test_metrics["preds"]
    test_df["predicted_label"]    = le.inverse_transform(test_metrics["preds"])
    test_df["true_label"]         = le.inverse_transform(test_metrics["labels"])
    test_df.to_csv(PATHS["metrics"] / "test_predictions.csv", index=False)

    logger.info("Training complete.")
    return results


if __name__ == "__main__":
    train()
