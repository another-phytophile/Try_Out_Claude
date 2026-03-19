#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.18",
#   "torch>=2.1",
#   "torchvision>=0.16",
#   "transformers>=4.38",
#   "timm>=0.9.16",
#   "open-clip-torch>=2.24",
#   "scikit-learn>=1.4",
#   "numpy>=1.26",
#   "pandas>=2.1",
#   "tqdm>=4.66",
#   "pyyaml>=6.0",
#   "python-dotenv>=1.0",
#   "huggingface-hub>=0.21",
#   "Pillow>=10.0",
# ]
# ///
"""
Domain Shift Analysis — Orchestration Script
============================================
Runs embedding extraction + classifier evaluation for Kaiko, CONCH, and Virchow
on the dakomura/tcga-ut dataset, across three domain shift conditions and two
classification tasks (cancer type + TSS).

Usage
-----
# Using UV inline runner (no venv needed):
    uv run scripts/run_domain_shift.py

# With a custom config:
    uv run scripts/run_domain_shift.py --config configs/domain_shift.yaml

# Skip extraction (use cached embeddings only):
    uv run scripts/run_domain_shift.py --skip-extraction

# Evaluate a single model:
    uv run scripts/run_domain_shift.py --models kaiko

Output
------
  results/logs/domain_shift_results.csv  — full results table
  results/logs/domain_shift_summary.csv  — pivot table (accuracy per classifier)
  results/logs/run.log                   — console log

Pipeline
--------
  1. For each model × split: extract embeddings → cache to data/untracked/embeddings/
  2. For each model × condition × task × classifier: train + evaluate
  3. Print summary table + save CSVs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path (supports both `uv run` and direct call)
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Domain shift analysis for TCGA pathology FMs")
    p.add_argument(
        "--config",
        default="configs/domain_shift.yaml",
        help="Path to YAML config (default: configs/domain_shift.yaml)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override model list, e.g. --models kaiko virchow",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=["within_valid", "train_to_valid", "train_to_test"],
        help="Override domain shift conditions",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=["cancer_type", "tss"],
        help="Override classification tasks",
    )
    p.add_argument(
        "--classifiers",
        nargs="+",
        default=None,
        choices=["logistic", "knn", "lda"],
        help="Override classifiers to run",
    )
    p.add_argument(
        "--knn-values",
        nargs="+",
        type=int,
        default=None,
        help="Override KNN k-values, e.g. --knn-values 50 100 500",
    )
    p.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip embedding extraction (use existing caches only)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Override compute device (cuda/cpu)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -- Config ---------------------------------------------------------------
    from src.utils.config import load_config
    from src.utils.logging import setup_logging

    cfg = load_config(args.config)
    exp = cfg["experiment"]

    setup_logging(log_dir=exp.get("log_dir", "results/logs"))
    import logging
    log = logging.getLogger(__name__)
    log.info("Config: %s", args.config)

    seed = exp.get("seed", 42)
    device = args.device or cfg.get("device", "cuda")
    results_dir = Path(exp.get("results_dir", "results/logs"))

    # -- Resolve model list ---------------------------------------------------
    if args.models:
        model_specs = [{"name": n} for n in args.models]
    else:
        model_specs = cfg.get("models", [{"name": "kaiko"}])

    model_names = [m["name"] for m in model_specs]
    log.info("Models: %s", model_names)

    # -- Resolve experiment parameters ----------------------------------------
    conditions = args.conditions or cfg.get("conditions")
    tasks = args.tasks or cfg.get("tasks")
    classifiers = args.classifiers or cfg.get("classifiers")
    knn_values = args.knn_values or cfg.get("knn_values")

    log.info("Conditions : %s", conditions)
    log.info("Tasks      : %s", tasks)
    log.info("Classifiers: %s", classifiers)
    log.info("KNN values : %s", knn_values)

    # =========================================================================
    # STEP 1 — Embedding extraction
    # =========================================================================
    if not args.skip_extraction:
        from src.feature_engineering import extract_and_cache, load_model

        extr_cfg = cfg.get("extraction", {})
        splits = extr_cfg.get("splits", ["train", "valid", "test"])
        batch_size = extr_cfg.get("batch_size", 64)
        force = extr_cfg.get("force_reextract", False)

        log.info("=" * 60)
        log.info("STEP 1: Embedding Extraction")
        log.info("=" * 60)

        for spec in model_specs:
            model_name = spec["name"]
            model_id = spec.get("model_id")

            log.info("Loading model: %s", model_name)
            model = load_model(model_name, device=device, model_id=model_id)

            for split in splits:
                log.info("  Extracting split=%s ...", split)
                extract_and_cache(model, split, batch_size=batch_size, force=force)

            # Free GPU memory between models
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    else:
        log.info("STEP 1: Skipped (--skip-extraction set).")

    # =========================================================================
    # STEP 2 — Domain shift analysis
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 2: Domain Shift Analysis")
    log.info("=" * 60)

    from src.evaluation import format_summary_table, run_domain_shift_analysis

    results = run_domain_shift_analysis(
        model_names=model_names,
        conditions=conditions,
        tasks=tasks,
        classifiers=classifiers,
        knn_values=knn_values,
        seed=seed,
        results_dir=results_dir,
    )

    if results.empty:
        log.error("No results — check extraction logs above.")
        sys.exit(1)

    # =========================================================================
    # STEP 3 — Print & save summary
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 3: Summary")
    log.info("=" * 60)

    summary = format_summary_table(results)
    summary_path = results_dir / "domain_shift_summary.csv"
    summary.to_csv(summary_path)

    print("\n" + "=" * 80)
    print("DOMAIN SHIFT ANALYSIS — ACCURACY SUMMARY")
    print("=" * 80)
    print(summary.to_string())
    print()

    # Also print a quick comparison: cancer_type vs TSS
    for task in (results["task"].unique() if not results.empty else []):
        sub = results[results["task"] == task]
        best_rows = (
            sub.sort_values("accuracy", ascending=False)
            .groupby(["model", "condition", "classifier"])
            .first()
            .reset_index()
        )
        print(f"\n── Best accuracy per (model, condition, classifier) | task={task} ──")
        print(
            best_rows[["model", "condition", "classifier", "k", "accuracy", "macro_f1"]]
            .to_string(index=False)
        )

    print(f"\nFull results  → {results_dir / 'domain_shift_results.csv'}")
    print(f"Summary table → {summary_path}")


if __name__ == "__main__":
    main()
