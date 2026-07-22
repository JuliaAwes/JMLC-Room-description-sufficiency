#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_distilmbert.py

Дообучение multilingual DistilBERT для бинарной классификации
названий номеров отелей (supplier_room_name -> target).

Логика полностью соответствует исходному train_bert.py / bert-final.ipynb,
пути и гиперпараметры вынесены в аргументы командной строки, чтобы скрипт
можно было запускать не только на Kaggle с зашитыми путями, а где угодно.

Запуск с путями по умолчанию (как было в ноутбуке, для Kaggle):
    python scripts/train_distilmbert.py

Запуск со своими путями (например, локально):
    python scripts/train_distilmbert.py \
        --train-path data/raw/public_dataset.csv \
        --test-path data/raw/submission_sample.csv \
        --output-dir models/distilmbert \
        --epochs 9
"""

import os
import json
import random
import argparse
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm


# =========================================================
# CLI / CONFIG
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train DistilBERT room-name classifier")

    # --- пути к данным и артефактам ---
    p.add_argument(
        "--train-path", type=str,
        default="/kaggle/input/datasets/liaawies/public-dataset4/public_dataset.csv",
        help="CSV с колонками supplier_room_name и target",
    )
    p.add_argument(
        "--test-path", type=str,
        default="/kaggle/input/datasets/liaawies/submission-sample/new_submission_sample (3) (1).csv",
        help="CSV для сабмита (без target)",
    )
    p.add_argument(
        "--output-dir", type=str, default="room_text_distilmbert_full",
        help="Куда сохранять чекпоинты, токенизатор, submission и .npy с вероятностями",
    )
    p.add_argument(
        "--resume-checkpoint", type=str, default=None,
        help="Путь к last_checkpoint.pt для продолжения обучения (по умолчанию — с нуля)",
    )

    # --- модель / гиперпараметры (значения = как в исходном ноутбуке) ---
    p.add_argument("--model-name", type=str, default="distilbert-base-multilingual-cased")
    p.add_argument("--text-col", type=str, default="supplier_room_name")
    p.add_argument("--target-col", type=str, default="target")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--valid-batch-size", type=int, default=32)
    p.add_argument("--pred-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=9)
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--min-precision", type=float, default=0.90)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # --- быстрый прогон для отладки на маленькой доле данных ---
    p.add_argument("--data-fraction", type=float, default=1.0,
                    help="Доля train/valid для отладки, 1.0 = весь датасет")
    p.add_argument("--skip-submission", action="store_true",
                    help="Не строить submission/test_bert_probs.npy после обучения")

    return p.parse_args()


ARGS = parse_args()

SEED = ARGS.seed
MODEL_NAME = ARGS.model_name
TEXT_COL = ARGS.text_col
TARGET_COL = ARGS.target_col

TRAIN_PATH = ARGS.train_path
TEST_PATH = ARGS.test_path
OUTPUT_DIR = ARGS.output_dir

MAX_LEN = ARGS.max_len
TRAIN_BATCH_SIZE = ARGS.train_batch_size
VALID_BATCH_SIZE = ARGS.valid_batch_size
PRED_BATCH_SIZE = ARGS.pred_batch_size
LR = ARGS.lr
WEIGHT_DECAY = ARGS.weight_decay
TOTAL_EPOCHS = ARGS.epochs
VALID_SIZE = ARGS.valid_size
MIN_PRECISION = ARGS.min_precision
GRAD_CLIP = ARGS.grad_clip
NUM_WORKERS = os.cpu_count()

USE_DATA_FRACTION = ARGS.data_fraction < 1.0
DATA_FRACTION = ARGS.data_fraction
SKIP_SUBMISSION = ARGS.skip_submission
RESUME_CHECKPOINT = ARGS.resume_checkpoint

# =========================================================
# PATHS
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
LAST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "last_checkpoint.pt")
SPLIT_PATH = os.path.join(OUTPUT_DIR, "split_indices.npz")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "train_config.json")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
# ВАЖНО: это имя используется дальше в src/ensemble для сборки ансамбля —
# не переименовывать без синхронизации с predict_distilmbert.py и ensemble-скриптом.
TEST_PROBS_PATH = os.path.join(OUTPUT_DIR, "test_bert_probs.npy")

# =========================================================
# DEVICE / AMP
# =========================================================
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

USE_BF16 = USE_CUDA and torch.cuda.is_bf16_supported()
USE_FP16 = USE_CUDA and not USE_BF16
USE_AMP = USE_CUDA
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
USE_SCALER = USE_FP16


def amp_autocast():
    if not USE_AMP:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=AMP_DTYPE)


# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# =========================================================
# TEXT NORMALIZATION
# =========================================================
def normalize_text(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\xa0", " ")
    x = x.replace("&amp;", "&")
    x = x.replace("\u201c", '"').replace("\u201d", '"')  # “ ”
    x = x.replace("\u2019", "'")  # ’
    x = " ".join(x.split())
    return x.strip()


def sample_fraction_df(df: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    if frac >= 1.0:
        return df.reset_index(drop=True)
    if frac <= 0:
        raise ValueError("frac must be > 0")
    n = max(1, int(len(df) * frac))
    return df.sample(n=n, random_state=seed, replace=False).reset_index(drop=True)


# =========================================================
# DATASET
# =========================================================
class RoomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        labels = None
        if "labels" in batch[0]:
            labels = torch.stack([x["labels"] for x in batch])

        features = [{k: v for k, v in x.items() if k != "labels"} for x in batch]
        padded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        if labels is not None:
            padded["labels"] = labels
        return padded


# =========================================================
# METRICS
# =========================================================
def find_threshold_for_precision(y_true, y_prob, min_precision=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]

    mask = precision >= min_precision
    if not mask.any():
        return 1.0, 0.0, 0.0

    valid_thresholds = thresholds[mask]
    valid_precision = precision[mask]
    valid_recall = recall[mask]
    best_idx = np.argmax(valid_recall)

    return (
        float(valid_thresholds[best_idx]),
        float(valid_precision[best_idx]),
        float(valid_recall[best_idx]),
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses = []
    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="valid", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with amp_autocast():
            outputs = model(**batch)

        loss = outputs.loss
        probs = F.softmax(outputs.logits.float(), dim=1)[:, 1]

        losses.append(float(loss.item()))
        all_probs.append(probs.detach().float().cpu().numpy())
        all_labels.append(batch["labels"].detach().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    pr_auc = average_precision_score(all_labels, all_probs)
    thr, prec, rec = find_threshold_for_precision(
        all_labels, all_probs, min_precision=MIN_PRECISION
    )

    return {
        "loss": float(np.mean(losses)),
        "pr_auc": float(pr_auc),
        "threshold_p90": thr,
        "precision_at_thr": prec,
        "recall_at_thr": rec,
        "probs": all_probs,
        "labels": all_labels,
    }


# =========================================================
# SAVE / LOAD
# =========================================================
def save_json_config():
    config = vars(ARGS)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def save_split_indices(train_idx: np.ndarray, valid_idx: np.ndarray, path: str):
    np.savez(path, train_idx=train_idx, valid_idx=valid_idx)


def load_split_indices(path: str):
    data = np.load(path)
    return data["train_idx"], data["valid_idx"]


def save_best_model(model, tokenizer, best_pr_auc: float):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "max_len": MAX_LEN,
            "best_pr_auc": best_pr_auc,
        },
        BEST_MODEL_PATH,
    )
    tokenizer.save_pretrained(OUTPUT_DIR)


def save_last_checkpoint(model, optimizer, scaler, epoch: int, best_pr_auc: float):
    checkpoint = {
        "epoch": epoch,
        "best_pr_auc": best_pr_auc,
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "seed": SEED,
    }
    torch.save(checkpoint, LAST_CHECKPOINT_PATH)


def load_full_checkpoint(checkpoint_path: str, model, optimizer, scaler):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.float()

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    best_pr_auc = ckpt["best_pr_auc"]
    return start_epoch, best_pr_auc


def load_best_model_for_inference():
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt["model_name"],
        num_labels=2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.float()
    model.eval()
    return tokenizer, model


# =========================================================
# DATALOADERS
# =========================================================
def build_dataloaders(tokenizer, train_part: pd.DataFrame, valid_part: pd.DataFrame):
    train_ds = RoomTextDataset(
        texts=train_part[TEXT_COL].tolist(),
        labels=train_part[TARGET_COL].tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )
    valid_ds = RoomTextDataset(
        texts=valid_part[TEXT_COL].tolist(),
        labels=valid_part[TARGET_COL].tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )

    collator = Collator(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA,
        collate_fn=collator,
    )
    return train_loader, valid_loader


# =========================================================
# PREDICT
# =========================================================
@torch.no_grad()
def predict_proba(texts, tokenizer, model):
    ds = RoomTextDataset(
        texts=[normalize_text(x) for x in texts],
        labels=None,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )
    collator = Collator(tokenizer)
    loader = DataLoader(
        ds,
        batch_size=PRED_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA,
        collate_fn=collator,
    )

    probs = []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with amp_autocast():
            logits = model(**batch).logits
        batch_probs = F.softmax(logits.float(), dim=1)[:, 1]
        probs.append(batch_probs.detach().float().cpu().numpy())
    return np.concatenate(probs)


# =========================================================
# TRAIN
# =========================================================
def train_model():
    save_json_config()

    df = pd.read_csv(TRAIN_PATH)
    df[TEXT_COL] = df[TEXT_COL].map(normalize_text)
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    if os.path.exists(SPLIT_PATH):
        train_idx, valid_idx = load_split_indices(SPLIT_PATH)
        print(f"Loaded existing split: {SPLIT_PATH}")
    else:
        all_idx = np.arange(len(df))
        train_idx, valid_idx = train_test_split(
            all_idx,
            test_size=VALID_SIZE,
            random_state=SEED,
            stratify=df[TARGET_COL].values,
        )
        save_split_indices(train_idx, valid_idx, SPLIT_PATH)
        print(f"Saved split: {SPLIT_PATH}")

    train_part = df.iloc[train_idx].copy().reset_index(drop=True)
    valid_part = df.iloc[valid_idx].copy().reset_index(drop=True)

    if USE_DATA_FRACTION:
        train_part = sample_fraction_df(train_part, DATA_FRACTION, SEED)
        valid_part = sample_fraction_df(valid_part, DATA_FRACTION, SEED)
        print("FAST CHECK MODE")
        print(f"DATA_FRACTION = {DATA_FRACTION:.2%}")

    print("train:", train_part.shape, "valid:", valid_part.shape)
    print("target rate train:", train_part[TARGET_COL].mean())
    print("target rate valid:", valid_part[TARGET_COL].mean())

    if os.path.exists(os.path.join(OUTPUT_DIR, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model.to(DEVICE)
    model.float()

    print("first param dtype:", next(model.parameters()).dtype)

    train_loader, valid_loader = build_dataloaders(tokenizer, train_part, valid_part)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER) if USE_CUDA else None

    start_epoch = 1
    best_pr_auc = -1.0

    if RESUME_CHECKPOINT is not None:
        if not os.path.exists(RESUME_CHECKPOINT):
            raise FileNotFoundError(f"Checkpoint not found: {RESUME_CHECKPOINT}")
        print(f"Resuming from: {RESUME_CHECKPOINT}")
        start_epoch, best_pr_auc = load_full_checkpoint(
            checkpoint_path=RESUME_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
        )
        print(f"Resume start_epoch={start_epoch}, best_pr_auc={best_pr_auc:.6f}")
        print("first param dtype after resume:", next(model.parameters()).dtype)

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train()
        train_losses = []

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{TOTAL_EPOCHS}")
        for batch in progress:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with amp_autocast():
                outputs = model(**batch)
                loss = outputs.loss

            if USE_SCALER:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            train_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        valid_metrics = evaluate(model, valid_loader)

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_PR_AUC={valid_metrics['pr_auc']:.6f} | "
            f"thr@P>={MIN_PRECISION:.2f}={valid_metrics['threshold_p90']:.4f} | "
            f"precision={valid_metrics['precision_at_thr']:.4f} | "
            f"recall={valid_metrics['recall_at_thr']:.4f}"
        )

        save_last_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_pr_auc=max(best_pr_auc, valid_metrics["pr_auc"]),
        )
        print(f"Saved last checkpoint: {LAST_CHECKPOINT_PATH}")

        if valid_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = valid_metrics["pr_auc"]
            save_best_model(model, tokenizer, best_pr_auc)
            print(f"Saved best model: {BEST_MODEL_PATH}")


# =========================================================
# SUBMISSION
# =========================================================
def build_submission():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}")

    tokenizer, model = load_best_model_for_inference()

    test_df = pd.read_csv(TEST_PATH)
    test_df[TEXT_COL] = test_df[TEXT_COL].map(normalize_text)
    test_df[TEXT_COL] = test_df[TEXT_COL].fillna("")

    probs = predict_proba(test_df[TEXT_COL].tolist(), tokenizer, model)

    if "Unnamed: 0" in test_df.columns:
        row_id = test_df["Unnamed: 0"]
    elif "row_id" in test_df.columns:
        row_id = test_df["row_id"]
    else:
        row_id = np.arange(len(test_df))

    submission = pd.DataFrame({
        "row_id": row_id,
        "target": probs,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    np.save(TEST_PROBS_PATH, probs)

    print("\nSubmission preview:")
    print(submission.head())
    print(f"Saved submission: {SUBMISSION_PATH}")
    print(f"Saved probs: {TEST_PROBS_PATH}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    print("MODEL_NAME:", MODEL_NAME)
    print("TRAIN_PATH:", TRAIN_PATH)
    print("TEST_PATH:", TEST_PATH)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("USE_AMP:", USE_AMP)
    print("RESUME_CHECKPOINT:", RESUME_CHECKPOINT)
    print("DATA_FRACTION:", DATA_FRACTION)

    train_model()

    if not SKIP_SUBMISSION:
        build_submission()
