#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_distilmbert.py

Отдельный инференс-скрипт: загружает уже обученную модель
(best_model.pt + токенизатор из --model-dir, сохранённые
train_distilmbert.py) и считает вероятности для нового CSV.

Не требует sklearn/train_test_split — используется только для предсказаний,
поэтому его можно гонять отдельно от обучения (например, при пересчёте
test_bert_probs.npy для ансамбля).

Запуск с путями по умолчанию (как в исходном ноутбуке, для Kaggle):
    python scripts/predict_distilmbert.py

Запуск со своими путями:
    python scripts/predict_distilmbert.py \
        --model-dir models/distilmbert \
        --test-path data/raw/submission_sample.csv \
        --output-csv outputs/submission_predict.csv
"""

import os
import argparse
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm


# =========================================================
# CLI / CONFIG
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Predict with a trained DistilBERT room-name classifier")

    p.add_argument(
        "--model-dir", type=str, default="room_text_distilmbert_full",
        help="Папка с best_model.pt и токенизатором (OUTPUT_DIR из train_distilmbert.py)",
    )
    p.add_argument(
        "--test-path", type=str,
        default="/kaggle/input/datasets/liaawies/submission-sample/new_submission_sample (3) (1).csv",
        help="CSV, для которого нужно посчитать вероятности",
    )
    p.add_argument(
        "--output-csv", type=str, default=None,
        help="Куда сохранить submission (по умолчанию: <model-dir>/submission_predict.csv)",
    )
    p.add_argument(
        "--output-probs", type=str, default=None,
        # ВАЖНО: имя должно совпадать с тем, что использует src/ensemble
        # (по умолчанию — test_bert_probs.npy, как и в train_distilmbert.py)
        help="Куда сохранить .npy с вероятностями (по умолчанию: <model-dir>/test_bert_probs.npy)",
    )
    p.add_argument("--text-col", type=str, default="supplier_room_name")
    p.add_argument("--pred-batch-size", type=int, default=64)

    return p.parse_args()


ARGS = parse_args()

MODEL_DIR = ARGS.model_dir
TEST_PATH = ARGS.test_path
TEXT_COL = ARGS.text_col
PRED_BATCH_SIZE = ARGS.pred_batch_size

OUTPUT_SUBMISSION = ARGS.output_csv or os.path.join(MODEL_DIR, "submission_predict.csv")
# Совпадает по умолчанию с train_distilmbert.py -> ансамблю не важно, каким
# скриптом файл был получен, лишь бы имя было одно и то же.
OUTPUT_PROBS = ARGS.output_probs or os.path.join(MODEL_DIR, "test_bert_probs.npy")

NUM_WORKERS = os.cpu_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()
USE_BF16 = USE_CUDA and torch.cuda.is_bf16_supported()
USE_FP16 = USE_CUDA and not USE_BF16
USE_AMP = USE_CUDA
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def amp_autocast():
    if not USE_AMP:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=AMP_DTYPE)


# =========================================================
# TEXT NORMALIZATION (должна совпадать с train_distilmbert.py)
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


# =========================================================
# DATASET / COLLATOR
# =========================================================
class RoomTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len: int):
        self.texts = list(texts)
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
        return item


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        features = [{k: v for k, v in x.items()} for x in batch]
        padded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        return padded


# =========================================================
# LOAD MODEL
# =========================================================
def load_model(model_dir):
    """
    Загружает сохранённую модель и токенизатор из model_dir.
    Ожидается, что там лежат:
      - best_model.pt (словарь с состоянием модели, сохранён train_distilmbert.py)
      - tokenizer_config.json, vocab.txt и др. (tokenizer.save_pretrained)
    """
    best_model_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Файл best_model.pt не найден в {model_dir}")

    ckpt = torch.load(best_model_path, map_location=DEVICE)
    model_name = ckpt.get("model_name", "distilbert-base-multilingual-cased")
    max_len = ckpt.get("max_len", 64)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.float()
    model.eval()

    print(f"Модель загружена из {model_dir}, device={DEVICE}")
    return tokenizer, model, max_len


# =========================================================
# PREDICT
# =========================================================
@torch.no_grad()
def predict_proba(texts, tokenizer, model, max_len=64, batch_size=64):
    """Список текстов -> массив вероятностей класса 1."""
    texts = [normalize_text(x) for x in texts]

    ds = RoomTextDataset(texts, tokenizer, max_len)
    collator = Collator(tokenizer)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
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
# MAIN
# =========================================================
def main():
    tokenizer, model, max_len = load_model(MODEL_DIR)

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Тестовый файл не найден: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)

    if TEXT_COL not in test_df.columns:
        raise ValueError(f"В тестовом файле нет колонки '{TEXT_COL}'")

    test_df[TEXT_COL] = test_df[TEXT_COL].fillna("")

    probs = predict_proba(test_df[TEXT_COL].tolist(), tokenizer, model, max_len, PRED_BATCH_SIZE)

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

    os.makedirs(os.path.dirname(OUTPUT_SUBMISSION) or ".", exist_ok=True)
    submission.to_csv(OUTPUT_SUBMISSION, index=False)

    os.makedirs(os.path.dirname(OUTPUT_PROBS) or ".", exist_ok=True)
    np.save(OUTPUT_PROBS, probs)

    print(f"\n✅ Предсказания сохранены в {OUTPUT_SUBMISSION}")
    print(f"✅ Вероятности сохранены в {OUTPUT_PROBS}")
    print("Первые 5 строк:")
    print(submission.head())


if __name__ == "__main__":
    main()
