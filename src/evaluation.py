"""
Domain shift analysis for TCGA pathology foundation models.

Experiment conditions
─────────────────────
┌─────────────────┬──────────────────────────────────┬────────────────────────────────┐
│ Condition       │ Train set                        │ Test set                       │
├─────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ within_valid    │ 80% of external valid split      │ 20% of external valid split    │
│                 │ (same TSS distribution → no shift│                                │
├─────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ train_to_valid  │ external train split             │ external valid split (baseline)│
├─────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ train_to_test   │ external train split             │ external test split (shifted)  │
└─────────────────┴──────────────────────────────────┴────────────────────────────────┘

Classification tasks
─────────────────────
  cancer_type : predict cancer type label   (31 TCGA classes)
  tss         : predict Tissue Source Site  (hospital / sequencing centre)

Classifiers
─────────────
  logistic : L2 logistic regression
  knn      : KNN with cosine distance, k ∈ [50, 100, 200, 500, 1000]
  lda      : Linear Discriminant Analysis (canonical linear classifier)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.data_processing import split_within_valid
from src.feature_engineering import load_cache
from src.models import (
    DEFAULT_KNN_VALUES,
    build_classifier,
    evaluate_classifier,
    run_knn_sweep,
)

log = logging.getLogger(__name__)

Task = Literal["cancer_type", "tss"]
Condition = Literal["within_valid", "train_to_valid", "train_to_test"]

ALL_CONDITIONS: list[Condition] = ["within_valid", "train_to_valid", "train_to_test"]
ALL_TASKS: list[Task] = ["cancer_type", "tss"]
ALL_CLASSIFIERS = ["logistic", "knn", "lda"]


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _get_labels(cache: dict, task: Task) -> np.ndarray:
    if task == "cancer_type":
        return cache["labels"]
    elif task == "tss":
        return cache["tss_codes"]
    else:
        raise ValueError(f"Unknown task: {task!r}")


# ---------------------------------------------------------------------------
# Per-condition data loading
# ---------------------------------------------------------------------------

def load_condition_data(
    model_name: str,
    condition: Condition,
    task: Task,
    within_valid_train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, X_test, y_train, y_test) for a given condition.

    Embeddings are loaded from the .npz cache; call extract_and_cache()
    for each required split before running the analysis.
    """
    if condition == "within_valid":
        valid = load_cache(model_name, "valid")
        X = valid["embeddings"]
        y = _get_labels(valid, task)
        return split_within_valid(X, y, train_frac=within_valid_train_frac, seed=seed)

    elif condition == "train_to_valid":
        train = load_cache(model_name, "train")
        valid = load_cache(model_name, "valid")
        return (
            train["embeddings"],
            valid["embeddings"],
            _get_labels(train, task),
            _get_labels(valid, task),
        )

    elif condition == "train_to_test":
        train = load_cache(model_name, "train")
        test = load_cache(model_name, "test")
        return (
            train["embeddings"],
            test["embeddings"],
            _get_labels(train, task),
            _get_labels(test, task),
        )

    else:
        raise ValueError(f"Unknown condition: {condition!r}")


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_experiment(
    model_name: str,
    condition: Condition,
    task: Task,
    classifiers: list[str] | None = None,
    knn_values: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all classifiers for one (model, condition, task) combination.

    Returns a DataFrame with columns:
      model, condition, task, classifier, k, accuracy, macro_f1, n_classes,
      n_train, n_test
    """
    classifiers = classifiers or ALL_CLASSIFIERS
    knn_values = knn_values or DEFAULT_KNN_VALUES

    log.info("── %s | %s | %s ──", model_name, condition, task)
    X_train, X_test, y_train, y_test = load_condition_data(
        model_name, condition, task, seed=seed
    )

    rows = []
    base = {"model": model_name, "condition": condition, "task": task}

    for clf_name in classifiers:
        if clf_name == "knn":
            # Sweep over k values
            sweep = run_knn_sweep(knn_values, X_train, y_train, X_test, y_test, seed=seed)
            for r in sweep:
                rows.append({**base, "classifier": "knn", **r})
        else:
            clf = build_classifier(clf_name, seed=seed)
            metrics = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
            rows.append({**base, "classifier": clf_name, "k": None, **metrics})
            log.info(
                "  %s  acc=%.4f  f1=%.4f",
                clf_name, metrics["accuracy"], metrics["macro_f1"]
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full domain shift analysis
# ---------------------------------------------------------------------------

def run_domain_shift_analysis(
    model_names: list[str],
    conditions: list[Condition] | None = None,
    tasks: list[Task] | None = None,
    classifiers: list[str] | None = None,
    knn_values: list[int] | None = None,
    seed: int = 42,
    results_dir: Path = Path("results/logs"),
) -> pd.DataFrame:
    """Run the full domain shift analysis across all (model × condition × task) combos.

    Results are also saved as CSV to results_dir/domain_shift_results.csv.
    """
    conditions = conditions or ALL_CONDITIONS
    tasks = tasks or ALL_TASKS

    all_results = []
    for model_name in model_names:
        for condition in conditions:
            for task in tasks:
                try:
                    df = run_experiment(
                        model_name=model_name,
                        condition=condition,
                        task=task,
                        classifiers=classifiers,
                        knn_values=knn_values,
                        seed=seed,
                    )
                    all_results.append(df)
                except FileNotFoundError as e:
                    log.warning("Skipping %s/%s/%s — cache missing: %s", model_name, condition, task, e)

    if not all_results:
        log.error("No results produced — check that embeddings are cached.")
        return pd.DataFrame()

    results = pd.concat(all_results, ignore_index=True)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "domain_shift_results.csv"
    results.to_csv(out_path, index=False)
    log.info("Results saved → %s", out_path)

    return results


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def format_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Pivot results into a readable summary table.

    Rows: (model, condition, task)
    Columns: classifier/k → accuracy
    """
    results = results.copy()
    # Create a readable classifier label (knn-50, knn-100, ..., logistic, lda)
    results["clf_label"] = results.apply(
        lambda r: f"knn-{int(r['k'])}" if r["classifier"] == "knn" and pd.notna(r["k"])
        else r["classifier"],
        axis=1,
    )
    pivot = results.pivot_table(
        index=["model", "condition", "task"],
        columns="clf_label",
        values="accuracy",
        aggfunc="first",
    )
    return pivot.round(4)
