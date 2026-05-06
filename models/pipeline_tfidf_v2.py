"""TF-IDF v2: char+word n-grams, sample weighting, multiple regressors.

Compares LinearSVR / Ridge / GradientBoosting on the same 5-fold CV protocol.
For sustainability (very imbalanced), upweights non-zero examples so the
regressor doesn't collapse to predicting 0 everywhere.

Logs every (model, frame) result to experiments.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR

from config import FRAMES, RANDOM_STATE, TEXT_COL, TRAIN_CSV
from experiments_logger import log_results

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


def make_vectorizers():
    """Word 1-2 grams + char 3-5 grams stacked.

    Char n-grams help with Turkish morphology (suffix-rich language)
    without committing to a lemmatizer.
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=3,
        stop_words=TURKISH_STOPWORDS,
        lowercase=True,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=20000,
        min_df=3,
        lowercase=True,
        sublinear_tf=True,
    )
    return word_vec, char_vec


def fit_transform_combined(word_vec, char_vec, X_train_text, X_test_text):
    Xtr_w = word_vec.fit_transform(X_train_text)
    Xte_w = word_vec.transform(X_test_text)
    Xtr_c = char_vec.fit_transform(X_train_text)
    Xte_c = char_vec.transform(X_test_text)
    return hstack([Xtr_w, Xtr_c]).tocsr(), hstack([Xte_w, Xte_c]).tocsr()


def sample_weights(y: np.ndarray, frame: str) -> np.ndarray:
    """For sparse frames (mostly zeros), upweight non-zero rows.

    Inverse-frequency-style: weight_i = max(1, n_total / (k * n_class_i))
    Only applied when frame has class imbalance > ~5x.
    """
    weights = np.ones_like(y, dtype=np.float32)
    classes, counts = np.unique(y.astype(int), return_counts=True)
    if len(classes) <= 1:
        return weights
    max_c, min_c = counts.max(), counts.min()
    if max_c / max(min_c, 1) < 5:
        return weights  # no need

    n = len(y)
    k = len(classes)
    cls_to_count = dict(zip(classes, counts))
    for i, yi in enumerate(y.astype(int)):
        weights[i] = n / (k * cls_to_count[yi])
    return weights


def quadratic_weighted_kappa(y_true, y_pred):
    y_pred_int = np.clip(np.rint(y_pred), 0, 3).astype(int)
    return cohen_kappa_score(y_true.astype(int), y_pred_int, weights="quadratic", labels=[0, 1, 2, 3])


def make_model(name: str):
    if name == "linearsvr":
        return LinearSVR(random_state=RANDOM_STATE, max_iter=10000)
    if name == "ridge":
        return Ridge(random_state=RANDOM_STATE, alpha=1.0)
    if name == "gbr":
        # GBR doesn't accept sparse matrices well — handled by densifying in fit
        return GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=200, max_depth=3)
    raise ValueError(name)


def cv_one_model(df: pd.DataFrame, model_name: str, use_weights: bool):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    per_frame_mae = {f: [] for f in FRAMES}
    per_frame_qwk = {f: [] for f in FRAMES}

    for fold, (tr, te) in enumerate(kf.split(df), start=1):
        word_vec, char_vec = make_vectorizers()
        Xtr, Xte = fit_transform_combined(
            word_vec, char_vec,
            df[TEXT_COL].iloc[tr], df[TEXT_COL].iloc[te],
        )
        if model_name == "gbr":
            Xtr_use, Xte_use = Xtr.toarray(), Xte.toarray()
        else:
            Xtr_use, Xte_use = Xtr, Xte

        for frame in FRAMES:
            y_tr = df[frame].iloc[tr].values.astype(float)
            y_te = df[frame].iloc[te].values.astype(float)

            model = make_model(model_name)
            if use_weights:
                w = sample_weights(y_tr, frame)
                try:
                    model.fit(Xtr_use, y_tr, sample_weight=w)
                except TypeError:
                    model.fit(Xtr_use, y_tr)
            else:
                model.fit(Xtr_use, y_tr)
            preds = model.predict(Xte_use)

            per_frame_mae[frame].append(mean_absolute_error(y_te, preds))
            per_frame_qwk[frame].append(quadratic_weighted_kappa(y_te, preds))

    summary = {}
    for f in FRAMES:
        summary[f] = {
            "mae_mean": float(np.mean(per_frame_mae[f])),
            "mae_std": float(np.std(per_frame_mae[f])),
            "qwk_mean": float(np.mean(per_frame_qwk[f])),
            "qwk_std": float(np.std(per_frame_qwk[f])),
        }
    return summary


def main():
    df = load_training()
    print(f"Loaded {len(df)} docs")

    configs = [
        ("tfidf_v2_linearsvr",       "linearsvr", False, "word1-2 + char3-5, no weights"),
        ("tfidf_v2_linearsvr_w",     "linearsvr", True,  "word1-2 + char3-5, sample weights"),
        ("tfidf_v2_ridge",           "ridge",     False, "Ridge alpha=1"),
        ("tfidf_v2_ridge_w",         "ridge",     True,  "Ridge with sample weights"),
        ("tfidf_v2_gbr",             "gbr",       False, "GradientBoostingRegressor 200 trees"),
    ]

    for exp, model_name, use_weights, notes in configs:
        print(f"\n=== {exp} ===")
        summary = cv_one_model(df, model_name, use_weights)
        for f in FRAMES:
            s = summary[f]
            print(f"  [{f}] MAE={s['mae_mean']:.3f} ± {s['mae_std']:.3f}  "
                  f"QWK={s['qwk_mean']:.3f} ± {s['qwk_std']:.3f}")
        macro_qwk = np.mean([summary[f]["qwk_mean"] for f in FRAMES])
        macro_mae = np.mean([summary[f]["mae_mean"] for f in FRAMES])
        print(f"  [MEAN] MAE={macro_mae:.3f}  QWK={macro_qwk:.3f}")

        log_results(
            experiment=exp,
            model_family="tfidf",
            variant=f"{model_name}, weights={use_weights}",
            per_frame=summary,
            n_folds=5,
            notes=notes,
        )

    print("\nAll TF-IDF v2 experiments logged to experiments.csv")


if __name__ == "__main__":
    main()
