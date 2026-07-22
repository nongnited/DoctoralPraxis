"""
RAGShield — Fine-tune SecureBERT 2.0 for Injection Detection
Security-domain pre-trained transformer (cisco-ai/SecureBERT2.0-base).
Same hyperparameters as BERT Base for controlled comparison.

Output: models/securebert/ (HuggingFace format)

Usage:
    python 3_training/train_securebert.py
"""

import os
import time
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import pandas as pd
from sklearn.metrics import (
    f1_score, matthews_corrcoef, precision_score,
    recall_score, accuracy_score
)
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR    = Path(os.getenv("DATA_DIR",   "./data"))
MODELS_DIR  = Path(os.getenv("MODELS_DIR", "./models"))
PROC_DIR    = DATA_DIR / "processed"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

MODEL_ID    = "cisco-ai/SecureBERT2.0-base"
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5


def detect_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"  Using CUDA: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("  Using Apple MPS")
    else:
        device = torch.device("cpu")
        log.info("  Using CPU")
    return device


class InjectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def main():
    torch.manual_seed(RANDOM_SEED)
    device = detect_device()

    log.info(f"Loading tokenizer: {MODEL_ID}")
    # use_fast=False for compatibility with SecureBERT custom vocab
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    log.info("Loading data...")
    train_df = pd.read_parquet(PROC_DIR / "train.parquet")
    val_df   = pd.read_parquet(PROC_DIR / "val.parquet")
    log.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    log.info("Tokenizing datasets...")
    train_dataset = InjectionDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LEN
    )
    val_dataset = InjectionDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )

    log.info(f"Loading model: {MODEL_ID}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    history = {"train_loss": [], "train_acc": [], "val_f1": [], "val_mcc": []}
    best_mcc = -1.0
    best_epoch = 0
    t_total = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        log.info(f"\n── Epoch {epoch}/{EPOCHS} ──")
        model.train()
        total_loss = 0
        correct = 0
        n = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            n += len(labels)

            if step % 500 == 0:
                log.info(f"  Step {step}/{len(train_loader)} | "
                         f"loss={total_loss/step:.4f} | acc={correct/n:.4f}")

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / n
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        y_true, y_pred = evaluate(model, val_loader, device)
        val_f1  = f1_score(y_true, y_pred, zero_division=0)
        val_mcc = matthews_corrcoef(y_true, y_pred)
        history["val_f1"].append(val_f1)
        history["val_mcc"].append(val_mcc)
        log.info(f"  Epoch {epoch} val F1={val_f1:.4f} MCC={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc   = val_mcc
            best_epoch = epoch

    elapsed = time.perf_counter() - t_total
    log.info(f"\nTraining complete: {elapsed:.1f}s | Best MCC={best_mcc:.4f} at epoch {best_epoch}")

    out_dir = MODELS_DIR / "securebert"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    y_true, y_pred = evaluate(model, val_loader, device)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    metrics = {
        "model": MODEL_ID, "model_label": "SecureBERT 2.0",
        "best_epoch": best_epoch, "epochs_run": EPOCHS,
        "total_train_time": round(elapsed, 1),
        "f1":               float(f1_score(y_true, y_pred)),
        "mcc":              float(matthews_corrcoef(y_true, y_pred)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "benign_recall":    float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "learning_rate": LR,
        "device": str(device), "history": history,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Model saved to {out_dir}")
    log.info(f"MCC={metrics['mcc']:.4f} F1={metrics['f1']:.4f} "
             f"FAR={metrics['false_alarm_rate']:.4f}")


if __name__ == "__main__":
    main()
