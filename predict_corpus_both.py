
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CORPUS_CSV,
    FRAMES,
    LABEL_MAX,
    MAX_LEN,
    OUTPUT_ROOT,
    SCORED_CORPUS_CSV,
    TEXT_COL,
    get_device,
)

FINAL_TFIDF_PATH = OUTPUT_ROOT / "final_tfidf" / "tfidf_bundle.joblib"
FINAL_DISTILBERT_DIR = OUTPUT_ROOT / "final_distilbert"

INFER_BATCH = 16


def score_with_tfidf(texts: list[str]) -> dict[str, np.ndarray]:
    print(f"Loading TF-IDF bundle: {FINAL_TFIDF_PATH}")
    bundle = joblib.load(FINAL_TFIDF_PATH)
    word_vec = bundle["word_vec"]
    char_vec = bundle["char_vec"]
    models = bundle["models"]

    Xw = word_vec.transform(texts)
    Xc = char_vec.transform(texts)
    X = hstack([Xw, Xc]).tocsr()

    preds = {}
    for frame in FRAMES:
        y = models[frame].predict(X)
        # clip to plausible range; LinearSVR can overshoot
        preds[frame] = np.clip(y, 0.0, 3.0)
    return preds

@torch.no_grad()
def score_with_distilbert(texts: list[str]) -> np.ndarray:
    """Returns array shape (n, len(FRAMES)) on the 0-1 scale."""
    device = get_device()
    print(f"Loading DistilBERTurk: {FINAL_DISTILBERT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_DISTILBERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_DISTILBERT_DIR)).to(device)
    model.eval()

    out = np.zeros((len(texts), len(FRAMES)), dtype=np.float32)
    for start in range(0, len(texts), INFER_BATCH):
        batch = texts[start:start + INFER_BATCH]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits.float().cpu().numpy()
        out[start:start + len(batch)] = logits
        if (start // INFER_BATCH) % 20 == 0:
            print(f"  distilbert: scored {min(start + INFER_BATCH, len(texts))}/{len(texts)}")
    # the head was trained on 0-1 normalized labels; clip to that range
    return np.clip(out, 0.0, 1.0)


def main():
    print(f"Loading corpus: {CORPUS_CSV}")
    df = pd.read_csv(CORPUS_CSV, encoding="utf-8-sig")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    n = len(df)
    print(f"  {n} docs")

    texts = df[TEXT_COL].tolist()

    # TF-IDF
    print("\n=== Scoring with TF-IDF v2 weighted ===")
    tfidf_preds = score_with_tfidf(texts)
    for frame in FRAMES:
        df[f"tfidf_score_{frame}"] = tfidf_preds[frame]

    # DistilBERT
    print("\n=== Scoring with DistilBERTurk weighted ===")
    distil_preds_01 = score_with_distilbert(texts)
    for i, frame in enumerate(FRAMES):
        df[f"distilbert_score_{frame}"] = distil_preds_01[:, i]              # 0-1 native
        df[f"distilbert_score_{frame}_0to3"] = distil_preds_01[:, i] * LABEL_MAX  # rescaled

    # Save
    print(f"\nWriting: {SCORED_CORPUS_CSV}")
    df.to_csv(SCORED_CORPUS_CSV, index=False, encoding="utf-8-sig")
    print(f"  {n} rows, {len(df.columns)} columns")

    # Summary
    print("\n=== Summary statistics ===")
    print("\nTF-IDF (0-3 scale):")
    print(df[[f"tfidf_score_{f}" for f in FRAMES]].describe().round(3).to_string())
    print("\nDistilBERTurk (0-3 scale, rescaled):")
    print(df[[f"distilbert_score_{f}_0to3" for f in FRAMES]].describe().round(3).to_string())

    # Agreement: Pearson correlation per frame
    print("\nPer-frame correlation between the two models (Pearson r on 0-3 scale):")
    for f in FRAMES:
        r = df[f"tfidf_score_{f}"].corr(df[f"distilbert_score_{f}_0to3"])
        print(f"  {f}: r = {r:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
