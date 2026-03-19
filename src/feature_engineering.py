"""
Foundation-model embedding extraction for Kaiko, CONCH, and Virchow.

Each model is wrapped in a uniform EmbeddingModel interface:
  model.transform   — torchvision transform to apply before forward pass
  model.embed(imgs) — torch.Tensor[B, D] (float32, L2-normalised)

Embeddings are cached in data/untracked/embeddings/<model>/<split>.npz
Each .npz contains:
  embeddings : float32 [N, D]
  labels     : str     [N]   (cancer type)
  tss_codes  : str     [N]   (tissue source site)
  keys       : str     [N]   (__key__ from WebDataset)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data_processing import Split, stream_dako_images

log = logging.getLogger(__name__)

ModelName = Literal["kaiko", "conch", "virchow"]

CACHE_ROOT = Path("data/untracked/embeddings")
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """Base wrapper: .transform + .embed(imgs) → normalised embeddings."""

    name: str
    dim: int

    def transform(self, pil_image):
        raise NotImplementedError

    @torch.no_grad()
    def embed(self, img_batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class KaikoModel(EmbeddingModel):
    """Kaiko pathology ViT loaded via timm.

    Default: kaiko-ai/vit-b-16 (768-dim).
    Override with model_id in config if a different variant is needed.
    """
    name = "kaiko"
    dim = 768

    def __init__(self, model_id: str = "hf-hub:kaiko-ai/vit-b-16", device: str = "cuda"):
        import timm
        from torchvision import transforms

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info("Loading Kaiko: %s on %s", model_id, self.device)
        self._model = timm.create_model(model_id, pretrained=True).eval().to(self.device)

        # Standard imagenet-style normalisation used by most ViT pathology FMs
        self._transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def transform(self, pil_image):
        return self._transform(pil_image)

    @torch.no_grad()
    def embed(self, img_batch: torch.Tensor) -> torch.Tensor:
        img_batch = img_batch.to(self.device)
        feats = self._model.forward_features(img_batch)
        # timm ViT: feats[:, 0] is the CLS token
        cls = feats[:, 0] if feats.ndim == 3 else feats
        return F.normalize(cls.float(), dim=-1).cpu()


class ConchModel(EmbeddingModel):
    """CONCH (MahmoodLab) vision encoder loaded via open_clip.

    CONCH is a CoCa-based model; we use only the image encoder here.
    HF model: MahmoodLab/conch  (may require HF token / gated access).
    """
    name = "conch"
    dim = 512

    def __init__(self, model_id: str = "hf-hub:MahmoodLab/conch", device: str = "cuda"):
        import open_clip
        from torchvision import transforms

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info("Loading CONCH: %s on %s", model_id, self.device)

        # open_clip knows about the conch_ViT-B_16 variant
        model, _, preprocess = open_clip.create_model_and_transforms(
            "conch_ViT-B_16",
            pretrained=model_id,
        )
        self._model = model.eval().to(self.device)
        self._transform = preprocess

    def transform(self, pil_image):
        return self._transform(pil_image)

    @torch.no_grad()
    def embed(self, img_batch: torch.Tensor) -> torch.Tensor:
        img_batch = img_batch.to(self.device)
        feats = self._model.encode_image(img_batch, normalize=True)
        return feats.float().cpu()


class VirchowModel(EmbeddingModel):
    """Virchow (Paige AI) ViT-H/14 loaded via timm.

    Embedding: CLS token concatenated with mean of patch tokens → 2560-dim.
    HF model: paige-ai/Virchow  (may require HF token / gated access).
    """
    name = "virchow"
    dim = 2560  # 1280 (cls) + 1280 (mean patch)

    def __init__(self, model_id: str = "hf-hub:paige-ai/Virchow", device: str = "cuda"):
        import timm
        from torchvision import transforms

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info("Loading Virchow: %s on %s", model_id, self.device)

        self._model = timm.create_model(
            model_id,
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval().to(self.device)

        self._transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def transform(self, pil_image):
        return self._transform(pil_image)

    @torch.no_grad()
    def embed(self, img_batch: torch.Tensor) -> torch.Tensor:
        img_batch = img_batch.to(self.device)
        feats = self._model.forward_features(img_batch)  # [B, 1+num_patches, 1280]
        cls = feats[:, 0]                                 # [B, 1280]
        patches = feats[:, 1:].mean(dim=1)               # [B, 1280]
        combined = torch.cat([cls, patches], dim=-1)      # [B, 2560]
        return F.normalize(combined.float(), dim=-1).cpu()


_REGISTRY: dict[str, type[EmbeddingModel]] = {
    "kaiko": KaikoModel,
    "conch": ConchModel,
    "virchow": VirchowModel,
}


def load_model(
    name: ModelName,
    device: str = "cuda",
    model_id: str | None = None,
) -> EmbeddingModel:
    """Instantiate an embedding model by name."""
    cls = _REGISTRY[name]
    kwargs = {"device": device}
    if model_id:
        kwargs["model_id"] = model_id
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Embedding extraction + caching
# ---------------------------------------------------------------------------

def cache_path(model_name: str, split: str) -> Path:
    return CACHE_ROOT / model_name / f"{split}.npz"


def embeddings_cached(model_name: str, split: str) -> bool:
    return cache_path(model_name, split).exists()


def extract_and_cache(
    model: EmbeddingModel,
    split: Split,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> Path:
    """Extract embeddings for a dataset split and save to .npz cache.

    Skips extraction if cache already exists (set force=True to overwrite).
    Returns path to the .npz file.
    """
    out = cache_path(model.name, split)
    if out.exists() and not force:
        log.info("Cache hit: %s — skipping extraction.", out)
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s embeddings for split=%s ...", model.name, split)

    all_emb, all_labels, all_tss, all_keys = [], [], [], []
    batch_imgs, batch_meta = [], []

    def flush_batch():
        if not batch_imgs:
            return
        tensor = torch.stack(batch_imgs)
        embs = model.embed(tensor)
        all_emb.append(embs.numpy())
        for m in batch_meta:
            all_labels.append(m["label"])
            all_tss.append(m["tss"])
            all_keys.append(m["key"])
        batch_imgs.clear()
        batch_meta.clear()

    for sample in tqdm(
        stream_dako_images(split, transform=model.transform),
        desc=f"{model.name}/{split}",
    ):
        batch_imgs.append(sample["image"])
        batch_meta.append(sample)
        if len(batch_imgs) == batch_size:
            flush_batch()
    flush_batch()

    embeddings = np.concatenate(all_emb, axis=0)
    np.savez_compressed(
        out,
        embeddings=embeddings,
        labels=np.array(all_labels),
        tss_codes=np.array(all_tss),
        keys=np.array(all_keys),
    )
    log.info("Saved %d embeddings → %s", len(embeddings), out)
    return out


def load_cache(model_name: str, split: str) -> dict:
    """Load a cached .npz file. Returns dict with embeddings/labels/tss_codes/keys."""
    p = cache_path(model_name, split)
    if not p.exists():
        raise FileNotFoundError(
            f"No cache at {p}. Run extract_and_cache() first."
        )
    data = np.load(p, allow_pickle=True)
    return {k: data[k] for k in data.files}
