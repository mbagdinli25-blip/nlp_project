"""BERTurk 5-fold CV with configurable hyperparameters.

Trains BERTurk K times on K-1 folds, evaluates on the held-out fold,
aggregates MAE + QWK per frame across folds. Logs to experiments.csv.

Run examples (edit the EXPERIMENT_NAME / hyperparameters at the bottom):
    !python pipeline_berturk_cv.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    BATCH_SIZE,
    FRAMES,
    HF_MODEL,
    LABEL_MAX,
    MAX_LEN,
    OUTPUT_ROOT,
    RANDOM_STATE,
    TEXT_COL,
    TRAIN_CSV,
    get_device,
)
from experiments_logger import log_results


def load_training() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=[TEXT_COL] + FRAMES).reset_index(drop=True)
    for f in FRAMES:
        df[f] = df[f].astype(int)
    return df


def to_hf_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    labels = (df[FRAMES].values.astype(np.float32)) / LABEL_MAX

    def tokenize(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )

    ds = Dataset.from_dict({TEXT_COL: df[TEXT_COL].tolist()})
    ds = ds.map(tokenize, batched=True, remove_columns=[TEXT_COL])
    ds = ds.add_column("labels", labels.tolist())
    return ds


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    metrics = {}
    for i, frame in enumerate(FRAMES):
        mae = mean_absolute_error(labels[:, i], preds[:, i])
        true_int = np.clip(np.rint(labels[:, i] * LABEL_MAX), 0, 3).astype(int)
        pred_int = np.clip(np.rint(preds[:, i] * LABEL_MAX), 0, 3).astype(int)
        qwk = cohen_kappa_score(true_int, pred_int, weights="quadratic", labels=[0, 1, 2, 3])
        metrics[f"mae_{frame}"] = float(mae)
        metrics[f"qwk_{frame}"] = float(qwk)
    return metrics


def run_one_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    fold_dir: Path,
    epochs: int,
    lr: float,
    hf_model: str,
    device,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model,
        num_labels=len(FRAMES),
        problem_type="regression",
    )

    train_ds = to_hf_dataset(train_df, tokenizer)
    test_ds = to_hf_dataset(test_df, tokenizer)

    args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",          # no checkpoints to save Drive space during CV
        logging_steps=50,
        seed=RANDOM_STATE,
        report_to="none",
        fp16=(device.type == "cuda"),
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    # cleanup transient checkpoint dir
    try:
        shutil.rmtree(fold_dir, ignore_errors=True)
    except Exception:
        pass
    # also free GPU memory between folds
    del trainer, model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return metrics


def main(
    experiment_name: str,
    epochs: int,
    lr: float,
    n_folds: int = 5,
    hf_model: str | None = None,
    notes: str = "",
):
    device = get_device()
    hf_model = hf_model or HF_MODEL
    print(f"Device: {device}")
    print(f"Experiment: {experiment_name}")
    print(f"Model: {hf_model} | epochs={epochs} | lr={lr} | folds={n_folds}")

    df = load_training()
    print(f"Training data: {len(df)} docs")

    print(f"Loading tokenizer: {hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    # collect per-fold metrics: per_frame_mae[frame] = [fold1, fold2, ...]
    per_frame_mae = {f: [] for f in FRAMES}
    per_frame_qwk = {f: [] for f in FRAMES}

    for fold, (tr_idx, te_idx) in enumerate(kf.split(df), start=1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)
        print(f"  train={len(train_df)} test={len(test_df)}")

        fold_dir = OUTPUT_ROOT / f"_cv_tmp_{experiment_name}_fold{fold}"
        metrics = run_one_fold(
            train_df, test_df, tokenizer, fold_dir,
            epochs=epochs, lr=lr, hf_model=hf_model, device=device,
        )

        for f in FRAMES:
            per_frame_mae[f].append(metrics[f"eval_mae_{f}"])
            per_frame_qwk[f].append(metrics[f"eval_qwk_{f}"])
            print(f"  {f}: MAE={metrics[f'eval_mae_{f}']:.4f}  QWK={metrics[f'eval_qwk_{f}']:.4f}")

    # aggregate
    print(f"\n=== {experiment_name} | {n_folds}-fold CV summary ===")
    per_frame_summary = {}
    for f in FRAMES:
        mae_mean = float(np.mean(per_frame_mae[f]))
        mae_std = float(np.std(per_frame_mae[f]))
        qwk_mean = float(np.mean(per_frame_qwk[f]))
        qwk_std = float(np.std(per_frame_qwk[f]))
        per_frame_summary[f] = {
            "mae_mean": mae_mean, "mae_std": mae_std,
            "qwk_mean": qwk_mean, "qwk_std": qwk_std,
        }
        print(f"  [{f}] MAE={mae_mean:.3f} ± {mae_std:.3f} | QWK={qwk_mean:.3f} ± {qwk_std:.3f}")

    macro_mae = np.mean([per_frame_summary[f]["mae_mean"] for f in FRAMES])
    macro_qwk = np.mean([per_frame_summary[f]["qwk_mean"] for f in FRAMES])
    print(f"  [MEAN] MAE={macro_mae:.3f} | QWK={macro_qwk:.3f}")

    variant = f"model={hf_model.split('/')[-1]}, epochs={epochs}, lr={lr}"
    log_results(
        experiment=experiment_name,
        model_family="berturk",
        variant=variant,
        per_frame=per_frame_summary,
        n_folds=n_folds,
        notes=notes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="berturk_cv_ep10_lr1e-5", help="experiment name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", default=None, help="override HF_MODEL (e.g. dbmdz/electra-small-turkish-mc4-cased-discriminator)")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    main(
        experiment_name=args.name,
        epochs=args.epochs,
        lr=args.lr,
        n_folds=args.folds,
        hf_model=args.model,
        notes=args.notes,
    )
