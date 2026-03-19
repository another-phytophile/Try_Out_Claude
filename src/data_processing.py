"""
Dako (dakomura/tcga-ut) dataset loading and metadata extraction.

Dataset: WebDataset/TAR format on HuggingFace.
Config 'external' splits: train(192,680), valid(39,670), test(39,360).

Per-sample fields:
  __key__  : str  — e.g. 'TCGA-A1-0001/00001'  (TCGA barcode / patch index)
  json     : dict — {'label': <cancer_type>, ...}
  jpg      : bytes — JPEG image data (224x224 patch)
"""
from __future__ import annotations

import io
from typing import Iterator, Literal

import numpy as np
from datasets import load_dataset
from PIL import Image

Split = Literal["train", "valid", "test"]

# WebDataset may store the image under one of these keys
_IMAGE_KEYS = ("jpg", "jpeg", "png")

HF_REPO = "dakomura/tcga-ut"
HF_CONFIG = "external"


# ---------------------------------------------------------------------------
# Key / metadata helpers
# ---------------------------------------------------------------------------

def get_tss(key: str) -> str:
    """Return Tissue Source Site code from a TCGA barcode key.

    Key format: 'TCGA-{TSS}-{patient_id}/{patch_index}'
    e.g. 'TCGA-A1-AAAA/00001' → 'A1'
    """
    return key.split("/")[0].split("-")[1]


def get_label(sample: dict) -> str:
    """Return cancer-type label from a Dako sample."""
    return sample["json"]["label"]


def get_image(sample: dict) -> Image.Image | None:
    """Decode PIL Image from a Dako sample dict (tries jpg/jpeg/png keys)."""
    for key in _IMAGE_KEYS:
        raw = sample.get(key)
        if raw is not None:
            if isinstance(raw, bytes):
                return Image.open(io.BytesIO(raw)).convert("RGB")
            if isinstance(raw, Image.Image):
                return raw.convert("RGB")
    return None


# ---------------------------------------------------------------------------
# Streaming loader
# ---------------------------------------------------------------------------

def stream_dako(split: Split, config: str = HF_CONFIG) -> Iterator[dict]:
    """Yield raw samples from dakomura/tcga-ut (streaming, no local download)."""
    ds = load_dataset(HF_REPO, name=config, split=split, streaming=True)
    yield from ds


def stream_dako_meta(split: Split, config: str = HF_CONFIG) -> Iterator[dict]:
    """Yield lightweight metadata dicts (no image bytes) for fast scanning.

    Yields: {'key': str, 'label': str, 'tss': str}
    """
    for sample in stream_dako(split, config):
        key = sample["__key__"]
        yield {
            "key": key,
            "label": get_label(sample),
            "tss": get_tss(key),
        }


def stream_dako_images(
    split: Split,
    transform=None,
    config: str = HF_CONFIG,
) -> Iterator[dict]:
    """Yield samples with decoded PIL images (and optional transform applied).

    Yields: {'key', 'label', 'tss', 'image'}
    image is a PIL.Image (or transformed tensor if transform is given).
    """
    for sample in stream_dako(split, config):
        img = get_image(sample)
        if img is None:
            continue
        key = sample["__key__"]
        yield {
            "key": key,
            "label": get_label(sample),
            "tss": get_tss(key),
            "image": transform(img) if transform is not None else img,
        }


# ---------------------------------------------------------------------------
# Metadata summary (streaming scan — slow but no GPU needed)
# ---------------------------------------------------------------------------

def collect_meta_df(splits: list[Split] = ("train", "valid", "test")):
    """Scan all requested splits and return a pandas DataFrame with metadata.

    Columns: split, key, label, tss
    This downloads NO images — only the json / __key__ fields are touched.
    Note: scans ~270k samples; takes several minutes on first call.
    """
    import pandas as pd
    from tqdm import tqdm

    rows = []
    for split in splits:
        for meta in tqdm(stream_dako_meta(split), desc=split):
            rows.append({"split": split, **meta})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Within-valid split (no-domain-shift condition)
# ---------------------------------------------------------------------------

def split_within_valid(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split cached valid embeddings into train/test with fixed random seed.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    idx = rng.permutation(n)
    cut = int(n * train_frac)
    train_idx, test_idx = idx[:cut], idx[cut:]
    return (
        embeddings[train_idx],
        embeddings[test_idx],
        labels[train_idx],
        labels[test_idx],
    )
