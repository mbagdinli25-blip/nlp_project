"""BERTurk multi-output regression fine-tuning.

Fine-tunes `dbmdz/bert-base-turkish-cased` to predict 4 frame scores
(technical, political, development, sustainability) jointly.
Labels are normalized 0–3 → 0–1 for training stability.
Stratified train/test split by `province`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    BATCH_SIZE,
    EPOCHS,
    FRAMES,
    HF_MODEL,
    LABEL_MAX,
    LR,
    MAX_LEN,
    MODEL_DIR,
    OUTPUT_ROOT,
    RANDOM_STATE,
    STRATIFY_COL,
    TEST_SIZE,
    TEXT_COL,
    TRAIN_CSV,
    get_device,
)


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=[TEXT_COL] + FRAMES + [STRATIFY_COL]).reset_index(drop=True)
    for f in FRAMES:
        df[f] = df[f].astype(int)

    # Collapse rare provinces so stratified split is possible.
    counts = df[STRATIFY_COL].value_counts()
    rare = counts[counts < 2].index
    strat = df[STRATIFY_COL].where(~df[STRATIFY_COL].isin(rare), other="__rare__")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat,
    )
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def to_hf_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Tokenize with head truncation, attach normalized 4-d label vector."""
    labels = (df[FRAMES].values.astype(np.float32)) / LABEL_MAX

    def tokenize(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,  # dynamic padding via Trainer's default collator
        )

    ds = Dataset.from_dict({TEXT_COL: df[TEXT_COL].tolist()})
    ds = ds.map(tokenize, batched=True, remove_columns=[TEXT_COL])
    ds = ds.add_column("labels", labels.tolist())
    return ds


def compute_metrics(eval_pred):
    """Per-frame MAE on 0–1 scale + QWK after rounding pred*3 to 0..3."""
    preds, labels = eval_pred
    preds = np.asarray(preds, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    metrics = {}
    mae_all = []
    qwk_all = []
    for i, frame in enumerate(FRAMES):
        mae = mean_absolute_error(labels[:, i], preds[:, i])
        true_int = np.clip(np.rint(labels[:, i] * LABEL_MAX), 0, 3).astype(int)
        pred_int = np.clip(np.rint(preds[:, i] * LABEL_MAX), 0, 3).astype(int)
        qwk = cohen_kappa_score(true_int, pred_int, weights="quadratic", labels=[0, 1, 2, 3])
        metrics[f"mae_{frame}"] = float(mae)
        metrics[f"qwk_{frame}"] = float(qwk)
        mae_all.append(mae)
        qwk_all.append(qwk)
    metrics["mae_mean"] = float(np.mean(mae_all))
    metrics["qwk_mean"] = float(np.mean(qwk_all))
    return metrics


def main():
    device = get_device()
    print(f"Device: {device}")

    train_df, test_df = load_and_split()

    print(f"Loading tokenizer + model: {HF_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL,
        num_labels=len(FRAMES),
        problem_type="regression",
    )

    train_ds = to_hf_dataset(train_df, tokenizer)
    test_ds = to_hf_dataset(test_df, tokenizer)

    args = TrainingArguments(
        output_dir=str(OUTPUT_ROOT / "berturk_run"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae_mean",
        greater_is_better=False,
        logging_steps=20,
        save_total_limit=2,
        seed=RANDOM_STATE,
        report_to="none",
        fp16=(device.type == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n=== Training ===")
    trainer.train()

    print("\n=== Final eval on held-out test set ===")
    final = trainer.evaluate()
    for k, v in final.items():
        if k.startswith("eval_") and (k.endswith(("_technical", "_political",
                                                   "_development", "_sustainability",
                                                   "mae_mean", "qwk_mean"))):
            print(f"  {k}: {v:.4f}")

    print(f"\nSaving model + tokenizer to: {MODEL_DIR}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    with open(MODEL_DIR / "eval_metrics.json", "w") as fh:
        json.dump(final, fh, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
