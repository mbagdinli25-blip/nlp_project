
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from config import OUTPUT_ROOT

EXPERIMENTS_CSV = OUTPUT_ROOT / "experiments.csv"

FIELDS = [
    "timestamp",
    "experiment",   # short name, e.g. "tfidf_baseline", "berturk_cv_ep10_lr1e-5"
    "model_family", # "tfidf" or "berturk"
    "variant",      # human-readable description of what changed
    "frame",        # technical / political / development / sustainability / mean
    "mae_mean",
    "mae_std",
    "qwk_mean",
    "qwk_std",
    "n_folds",
    "notes",
]


def log_results(
    experiment: str,
    model_family: str,
    variant: str,
    per_frame: dict[str, dict[str, float]],
    n_folds: int = 5,
    notes: str = "",
) -> None:
    """Append per-frame results + the macro mean row to experiments.csv.

    per_frame: {"technical": {"mae_mean": .., "mae_std": .., "qwk_mean": .., "qwk_std": ..}, ...}
    """
    new_file = not EXPERIMENTS_CSV.exists()
    ts = datetime.now().isoformat(timespec="seconds")

    rows = []
    mae_means, qwk_means = [], []
    for frame, m in per_frame.items():
        rows.append({
            "timestamp": ts,
            "experiment": experiment,
            "model_family": model_family,
            "variant": variant,
            "frame": frame,
            "mae_mean": round(m["mae_mean"], 4),
            "mae_std": round(m.get("mae_std", 0.0), 4),
            "qwk_mean": round(m["qwk_mean"], 4),
            "qwk_std": round(m.get("qwk_std", 0.0), 4),
            "n_folds": n_folds,
            "notes": notes,
        })
        mae_means.append(m["mae_mean"])
        qwk_means.append(m["qwk_mean"])

    # macro mean row
    if mae_means:
        rows.append({
            "timestamp": ts,
            "experiment": experiment,
            "model_family": model_family,
            "variant": variant,
            "frame": "MEAN",
            "mae_mean": round(sum(mae_means) / len(mae_means), 4),
            "mae_std": 0.0,
            "qwk_mean": round(sum(qwk_means) / len(qwk_means), 4),
            "qwk_std": 0.0,
            "n_folds": n_folds,
            "notes": notes,
        })

    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[experiments_logger] appended {len(rows)} rows -> {EXPERIMENTS_CSV}")
