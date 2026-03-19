"""
Classifier definitions for linear probing and KNN evaluation.

Supported classifiers:
  - logistic   : L2-regularised logistic regression (linear probe)
  - knn        : k-Nearest Neighbours (k is a hyperparameter)
  - lda        : Linear Discriminant Analysis ("canonical" linear classifier)
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

ClassifierName = Literal["logistic", "knn", "lda"]

# Default k-values for KNN sweep (following Dako / pathology FM eval convention)
DEFAULT_KNN_VALUES = [50, 100, 200, 500, 1000]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_classifier(
    name: ClassifierName,
    k: int = 20,
    max_iter: int = 1000,
    n_jobs: int = -1,
    seed: int = 42,
) -> Any:
    """Return a scikit-learn classifier ready to fit.

    Args:
        name     : 'logistic' | 'knn' | 'lda'
        k        : neighbour count (KNN only)
        max_iter : max iterations (logistic only)
        n_jobs   : parallelism (-1 = all cores)
        seed     : random state (logistic / lda)
    """
    if name == "logistic":
        return LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            multi_class="multinomial",
            C=1.0,
            n_jobs=n_jobs,
            random_state=seed,
        )
    elif name == "knn":
        return KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",   # embeddings are L2-normalised → cosine = dot product
            n_jobs=n_jobs,
        )
    elif name == "lda":
        # LinearDiscriminantAnalysis is the canonical linear discriminant
        return LinearDiscriminantAnalysis(solver="svd")
    else:
        raise ValueError(f"Unknown classifier: {name!r}. Choose from logistic/knn/lda.")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_classifier(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Fit classifier and return accuracy + macro-F1 on the test set.

    Labels are auto-encoded to integers internally.
    Returns: {'accuracy': float, 'macro_f1': float, 'n_train': int, 'n_test': int}
    """
    from sklearn.metrics import accuracy_score, f1_score

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    clf.fit(X_train, y_train_enc)
    preds = clf.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test_enc, preds)),
        "macro_f1": float(f1_score(y_test_enc, preds, average="macro", zero_division=0)),
        "n_classes": int(len(le.classes_)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def run_knn_sweep(
    k_values: list[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> list[dict]:
    """Run KNN for each k and return list of result dicts (includes k field)."""
    results = []
    for k in k_values:
        clf = build_classifier("knn", k=k)
        metrics = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
        results.append({"k": k, **metrics})
        log.info("KNN k=%d  acc=%.4f  f1=%.4f", k, metrics["accuracy"], metrics["macro_f1"])
    return results
