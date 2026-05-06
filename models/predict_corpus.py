"""Apply the fine-tuned BERTurk model to the full 2139-doc corpus.

Outputs `analytic_with_scores.csv`: original corpus columns plus
`score_technical, score_political, score_development, score_sustainability`
on the 0–1 scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CORPUS_CSV,
    FRAMES,
    MAX_LEN,
    MODEL_DIR,
    SCORED_CORPUS_CSV,
    TEXT_COL,
    get_device,
)

INFER_BATCH_SIZE = 16


@torch.no_grad()
def score_texts(texts: list[str], model, tokenizer, device) -> np.ndarray:
    out = np.zeros((len(texts), len(FRAMES)), dtype=np.float32)
    model.eval()
    for start in range(0, len(texts), INFER_BATCH_SIZE):
        batch = texts[start:start + INFER_BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits.float().cpu().numpy()
        out[start:start + len(batch)] = logits
        if (start // INFER_BATCH_SIZE) % 10 == 0:
            print(f"  scored {min(start + INFER_BATCH_SIZE, len(texts))}/{len(texts)}")
    # Clamp to [0, 1] in case the regression head drifted slightly outside.
    return np.clip(out, 0.0, 1.0)


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)

    print(f"Loading corpus: {CORPUS_CSV}")
    df = pd.read_csv(CORPUS_CSV, encoding="utf-8-sig")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    print(f"  {len(df)} docs")

    print("\n=== Scoring ===")
    scores = score_texts(df[TEXT_COL].tolist(), model, tokenizer, device)
    for i, frame in enumerate(FRAMES):
        df[f"score_{frame}"] = scores[:, i]

    print(f"\nWriting: {SCORED_CORPUS_CSV}")
    df.to_csv(SCORED_CORPUS_CSV, index=False, encoding="utf-8-sig")
    print("Done.")
    print("\nScore summary (0–1 scale):")
    print(df[[f"score_{f}" for f in FRAMES]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
