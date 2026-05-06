"""TF-IDF + Linear SVR baseline.

Trains one LinearSVR per frame on the 300-doc training set, reports
5-fold CV MAE and quadratic-weighted Cohen's kappa, and prints the
top-20 positive features per frame for codebook validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR

from config import FRAMES, RANDOM_STATE, TEXT_COL, TRAIN_CSV

# Compact Turkish stopword list (HF and NLTK lists agree on this core set).
TURKISH_STOPWORDS = [
    "acaba", "ama", "ancak", "artık", "asla", "aslında", "az", "bana", "bazen",
    "bazı", "belki", "ben", "benden", "beni", "benim", "beri", "beş", "bile",
    "bir", "biraz", "birçok", "biri", "birkaç", "birşey", "biz", "bizden",
    "bizi", "bizim", "böyle", "böylece", "bu", "buna", "bunda", "bundan",
    "bunlar", "bunları", "bunların", "bunu", "bunun", "burada", "çok", "çünkü",
    "da", "daha", "dahi", "de", "defa", "değil", "diğer", "diye", "doksan",
    "dokuz", "dolayı", "dolayısıyla", "dört", "elli", "en", "fakat", "falan",
    "filan", "gene", "gibi", "hala", "hangi", "hatta", "hem", "henüz", "hep",
    "hepsi", "her", "herhangi", "herkes", "hiç", "hiçbir", "için", "iki", "ile",
    "ilgili", "ise", "işte", "itibaren", "itibariyle", "kadar", "karşın",
    "kendi", "kendilerine", "kendini", "kendisi", "kendisine", "kendisini",
    "kez", "ki", "kim", "kimden", "kime", "kimi", "kimse", "madem", "mı", "mi",
    "mu", "mü", "nasıl", "ne", "neden", "nedenle", "nerde", "nerede", "nereye",
    "niçin", "niye", "o", "olan", "olarak", "oldu", "olduğu", "olduğunu",
    "olduklarını", "olmadı", "olmadığı", "olmak", "olması", "olmayan",
    "olmaz", "olsa", "olsun", "olup", "olur", "olursa", "oluyor", "on", "ona",
    "ondan", "onlar", "onlardan", "onları", "onların", "onu", "onun", "otuz",
    "oysa", "öyle", "pek", "rağmen", "sana", "sanki", "sekiz", "seksen", "sen",
    "senden", "seni", "senin", "siz", "sizden", "sizi", "sizin", "sonra",
    "şayet", "şey", "şeyden", "şeyi", "şeyler", "şimdi", "şu", "şuna", "şunda",
    "şundan", "şunları", "şunu", "tüm", "üç", "üzere", "var", "vardı", "ve",
    "veya", "ya", "yani", "yedi", "yerine", "yetmiş", "yine", "yirmi", "yoksa",
    "yüz", "zaten",
]


def load_training() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=[TEXT_COL] + FRAMES).reset_index(drop=True)
    for f in FRAMES:
        df[f] = df[f].astype(int)
    return df


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QWK with rounded preds clipped to the 0..3 ordinal range."""
    y_pred_int = np.clip(np.rint(y_pred), 0, 3).astype(int)
    return cohen_kappa_score(y_true, y_pred_int, weights="quadratic", labels=[0, 1, 2, 3])


def cross_validate_frame(X_text: pd.Series, y: np.ndarray, frame: str):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    maes, qwks = [], []
    for fold, (tr, te) in enumerate(kf.split(X_text), start=1):
        vec = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=3,
            stop_words=TURKISH_STOPWORDS,
            lowercase=True,
        )
        Xtr = vec.fit_transform(X_text.iloc[tr])
        Xte = vec.transform(X_text.iloc[te])
        model = LinearSVR(random_state=RANDOM_STATE, max_iter=10000)
        model.fit(Xtr, y[tr])
        preds = model.predict(Xte)
        maes.append(mean_absolute_error(y[te], preds))
        qwks.append(quadratic_weighted_kappa(y[te], preds))
    print(f"  [{frame}] CV MAE={np.mean(maes):.3f} ± {np.std(maes):.3f} | "
          f"QWK={np.mean(qwks):.3f} ± {np.std(qwks):.3f}")
    return np.mean(maes), np.mean(qwks)


def fit_full_and_top_features(X_text: pd.Series, y: np.ndarray, frame: str, k: int = 20):
    vec = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        stop_words=TURKISH_STOPWORDS,
        lowercase=True,
    )
    X = vec.fit_transform(X_text)
    model = LinearSVR(random_state=RANDOM_STATE, max_iter=10000)
    model.fit(X, y)
    coefs = model.coef_
    vocab = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(coefs)[-k:][::-1]
    print(f"\n  Top-{k} positive features for [{frame}]:")
    for i in top_idx:
        print(f"    {coefs[i]:+.3f}  {vocab[i]}")


def main():
    print(f"Loading training data: {TRAIN_CSV}")
    df = load_training()
    print(f"  {len(df)} docs, frames: {FRAMES}")

    print("\n=== 5-fold CV (TF-IDF + LinearSVR) ===")
    summary = {}
    for frame in FRAMES:
        y = df[frame].values.astype(float)
        mae, qwk = cross_validate_frame(df[TEXT_COL], y, frame)
        summary[frame] = {"mae": mae, "qwk": qwk}

    print("\n=== Summary ===")
    print(pd.DataFrame(summary).T.round(3).to_string())

    print("\n=== Refitting on full training set for feature inspection ===")
    for frame in FRAMES:
        y = df[frame].values.astype(float)
        fit_full_and_top_features(df[TEXT_COL], y, frame)


if __name__ == "__main__":
    main()
