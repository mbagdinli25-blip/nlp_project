
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import (
    BATCH_SIZE,
    FRAMES,
    LABEL_MAX,
    MAX_LEN,
    OUTPUT_ROOT,
    RANDOM_STATE,
    TEXT_COL,
    TRAIN_CSV,
    get_device,
)
from pipeline_tfidf_v2 import TURKISH_STOPWORDS, make_vectorizers, sample_weights
from pipeline_berturk_cv_weighted import (
    WeightedRegressionTrainer,
    compute_frame_weights,
    make_data_collator,
    to_hf_dataset,
)


FINAL_TFIDF_DIR = OUTPUT_ROOT / "final_tfidf"
FINAL_DISTILBERT_DIR = OUTPUT_ROOT / "final_distilbert"
DISTILBERT_MODEL = "dbmdz/distilbert-base-turkish-cased"
DISTILBERT_EPOCHS = 10
DISTILBERT_LR = 3e-5


def load_training() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=[TEXT_COL] + FRAMES).reset_index(drop=True)
    for f in FRAMES:
        df[f] = df[f].astype(int)
    return df


def train_tfidf_final(df: pd.DataFrame):
    print("\n=== Training final TF-IDF v2 (LinearSVR + class weights) on full 300 docs ===")
    word_vec, char_vec = make_vectorizers()
    Xw = word_vec.fit_transform(df[TEXT_COL])
    Xc = char_vec.fit_transform(df[TEXT_COL])
    X = hstack([Xw, Xc]).tocsr()
    print(f"  feature matrix: {X.shape}")

    models = {}
    for frame in FRAMES:
        y = df[frame].values.astype(float)
        w = sample_weights(y, frame)
        m = LinearSVR(random_state=RANDOM_STATE, max_iter=10000)
        m.fit(X, y, sample_weight=w)
        models[frame] = m
        wmin, wmax = float(w.min()), float(w.max())
        print(f"  [{frame}] trained (weight range [{wmin:.2f}, {wmax:.2f}])")

    FINAL_TFIDF_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "word_vec": word_vec,
        "char_vec": char_vec,
        "models": models,
        "frames": FRAMES,
    }
    out_path = FINAL_TFIDF_DIR / "tfidf_bundle.joblib"
    joblib.dump(bundle, out_path)
    print(f"  saved -> {out_path}")



def train_distilbert_final(df: pd.DataFrame):
    print(f"\n=== Training final DistilBERTurk (weighted MSE) on full 300 docs ===")
    device = get_device()
    print(f"  device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    weights = compute_frame_weights(df)
    print(f"  weight matrix shape: {weights.shape}")

    train_ds = to_hf_dataset(df, tokenizer, weights)

    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL,
        num_labels=len(FRAMES),
        problem_type="regression",
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_ROOT / "_final_distilbert_run"),
        num_train_epochs=DISTILBERT_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=DISTILBERT_LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        seed=RANDOM_STATE,
        report_to="none",
        fp16=(device.type == "cuda"),
        disable_tqdm=True,
        remove_unused_columns=False,
    )

    trainer = WeightedRegressionTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=make_data_collator(tokenizer),
    )
    trainer.train()

    FINAL_DISTILBERT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_DISTILBERT_DIR))
    tokenizer.save_pretrained(str(FINAL_DISTILBERT_DIR))
    print(f"  saved -> {FINAL_DISTILBERT_DIR}")

    # cleanup transient checkpoint dir if any
    import shutil
    shutil.rmtree(OUTPUT_ROOT / "_final_distilbert_run", ignore_errors=True)


def main():
    df = load_training()
    print(f"Loaded {len(df)} labeled docs")
    train_tfidf_final(df)
    train_distilbert_final(df)
    print("\nAll final models saved. Now run predict_corpus_both.py to score the corpus.")


if __name__ == "__main__":
    main()
