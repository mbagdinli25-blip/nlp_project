
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


# ---- Per-sample, per-frame inverse-frequency weights -----------------------
def compute_frame_weights(df: pd.DataFrame, imbalance_threshold: float = 5.0) -> np.ndarray:
    
    n = len(df)
    weights = np.ones((n, len(FRAMES)), dtype=np.float32)
    for j, frame in enumerate(FRAMES):
        y = df[frame].values.astype(int)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) <= 1:
            continue
        if counts.max() / max(counts.min(), 1) < imbalance_threshold:
            continue  # frame is balanced enough
        k = len(classes)
        cls_to_count = dict(zip(classes, counts))
        for i, yi in enumerate(y):
            weights[i, j] = n / (k * cls_to_count[yi])
        # report
        print(f"  [{frame}] reweighting: counts={dict(zip(classes.tolist(), counts.tolist()))} "
              f"-> weight range [{weights[:, j].min():.2f}, {weights[:, j].max():.2f}]")
    return weights


def load_training() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=[TEXT_COL] + FRAMES).reset_index(drop=True)
    for f in FRAMES:
        df[f] = df[f].astype(int)
    return df


def to_hf_dataset(df: pd.DataFrame, tokenizer, sample_weights: np.ndarray) -> Dataset:
    """Tokenize, attach normalized labels and per-sample-per-frame weights."""
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
    ds = ds.add_column("sample_weight", sample_weights.tolist())
    return ds


# ---- Custom Trainer with per-sample weighted MSE ---------------------------
class WeightedRegressionTrainer(Trainer):
    """Override compute_loss to apply per-sample, per-frame weights to MSE.

    Standard HF regression head returns logits with shape (B, num_labels).
    We compute (logits - labels)^2 * sample_weight and average.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight")  # shape: (B, num_labels)
        outputs = model(**inputs)
        logits = outputs.logits  # (B, num_labels)

        # per-element squared error
        sq_err = (logits.float() - labels.float()) ** 2  # (B, num_labels)
        weighted = sq_err * sample_weight.float()
        loss = weighted.mean()

        return (loss, outputs) if return_outputs else loss


def data_collator_with_weights(features):
    """Custom collator that pads input_ids/attention_mask and stacks weights/labels."""
    from transformers.data.data_collator import default_data_collator
    # pop float fields, let default collator handle the tokenized fields
    weights = [f.pop("sample_weight") for f in features]
    labels = [f.pop("labels") for f in features]
    batch = default_data_collator(features)
    batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
    batch["labels"] = torch.tensor(labels, dtype=torch.float32)
    return batch


# ---- Metrics (same as pipeline_berturk_cv.py) ------------------------------
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


# ---- Per-fold training -----------------------------------------------------
def run_one_fold(train_df, test_df, tokenizer, fold_dir, epochs, lr, hf_model, device):
    print("  computing frame weights on train fold...")
    train_weights = compute_frame_weights(train_df)
    # eval set: weights of 1 (we don't reweight evaluation loss)
    eval_weights = np.ones((len(test_df), len(FRAMES)), dtype=np.float32)

    train_ds = to_hf_dataset(train_df, tokenizer, train_weights)
    test_ds = to_hf_dataset(test_df, tokenizer, eval_weights)

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model,
        num_labels=len(FRAMES),
        problem_type="regression",
    )

    args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=RANDOM_STATE,
        report_to="none",
        fp16=(device.type == "cuda"),
        disable_tqdm=True,
        remove_unused_columns=False,  # keep sample_weight column
    )

    trainer = WeightedRegressionTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator_with_weights,
    )
    trainer.train()
    metrics = trainer.evaluate()

    shutil.rmtree(fold_dir, ignore_errors=True)
    del trainer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


# ---- Main ------------------------------------------------------------------
def main(experiment_name, epochs, lr, n_folds=5, hf_model=None, notes=""):
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

    variant = f"weighted-MSE, model={hf_model.split('/')[-1]}, epochs={epochs}, lr={lr}"
    log_results(
        experiment=experiment_name,
        model_family="berturk",
        variant=variant,
        per_frame=per_frame_summary,
        n_folds=n_folds,
        notes=notes or "per-sample inverse-frequency weighted MSE for imbalanced frames",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="berturk_cv_weighted_ep10_lr1e5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", default=None)
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
