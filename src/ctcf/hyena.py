"""HyenaDNA-tiny embeddings, following the authors' Hyena_embed.ipynb.

Model: LongSafari/hyenadna-tiny-1k-seqlen-d256 (2 Hyena layers, 256 dimensions), built
with HyenaDNA's standalone code (src/ctcf/hyenadna_standalone.py) and loaded the same
way as the authors' HyenaDNAPreTrainedModel.from_pretrained. The embedding of a sequence
is the backbone's output, one 256-number vector per token.

Tokenization matches HyenaDNA's CharacterTokenizer, which tokenizes each base and wraps
the sequence as [CLS] + bases + [SEP], so a 200-nt sequence gives 202 tokens. The
CharacterTokenizer class itself fails to build on transformers 5.x, so the same
vocabulary is applied here directly.
"""

import json
import re

import numpy as np
import torch

from ctcf.data import ROOT
from ctcf.hyenadna_standalone import HyenaDNAModel

MODEL_NAME = "hyenadna-tiny-1k-seqlen-d256"
MODEL_DIR = ROOT / "data" / "models" / MODEL_NAME

# Vocabulary of HyenaDNA's CharacterTokenizer(characters=["A", "C", "G", "T", "N"]).
CLS_ID, SEP_ID = 0, 1
BASE_IDS = np.array([7, 8, 9, 10])  # A, C, G, T in one-hot channel order


def download_weights():
    from huggingface_hub import hf_hub_download
    for name in ("config.json", "weights.ckpt"):
        hf_hub_download(f"LongSafari/{MODEL_NAME}", name, local_dir=MODEL_DIR)


def load_model(device="cpu"):
    """HyenaDNA backbone with pretrained weights, in eval mode."""
    if not (MODEL_DIR / "weights.ckpt").exists():
        download_weights()
    config = json.loads((MODEL_DIR / "config.json").read_text())
    model = HyenaDNAModel(**config, use_head=False)
    # The checkpoint stores PyTorch Lightning training settings as omegaconf objects. Load it
    # in weights-only mode, allowing just those config classes, so no other code can run.
    from collections import defaultdict
    from typing import Any
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    safe = [ListConfig, DictConfig, ContainerMetadata, Metadata, AnyNode, Any, list, dict, int, defaultdict]
    with torch.serialization.safe_globals(safe):
        ckpt = torch.load(MODEL_DIR / "weights.ckpt", map_location=device, weights_only=True)
    pretrained = ckpt["state_dict"]
    checkpointing = bool(config.get("checkpoint_mixer", False))
    state = model.state_dict()
    for key in state:
        if "backbone" not in key:
            continue
        loaded = "model." + key
        if checkpointing:  # same key surgery as the authors' load_weights helper
            loaded = re.sub(r"\.mlp", ".mlp.layer", re.sub(r"\.mixer", ".mixer.layer", loaded))
        state[key] = pretrained[loaded]
    model.load_state_dict(state)
    return model.to(device).eval()


def tokenize(onehot):
    """One-hot (N, 4, L) -> token ids (N, L + 2): [CLS] + bases + [SEP]."""
    onehot = np.asarray(onehot)
    if onehot.ndim != 3 or onehot.shape[1] != 4 or not np.all(onehot.sum(axis=1) == 1):
        raise ValueError("expected a one-hot array of shape (N, 4, L) with exactly one base per position")
    ids = BASE_IDS[onehot.argmax(axis=1)]
    n = len(ids)
    return np.concatenate([np.full((n, 1), CLS_ID), ids, np.full((n, 1), SEP_ID)], axis=1)


@torch.inference_mode()
def embed(model, onehot, batch_size=128, device="cpu"):
    """Full embeddings, shape (N, L + 2, 256), float32."""
    ids = torch.from_numpy(tokenize(onehot))
    out = [model(ids[i:i + batch_size].to(device)).float().cpu().numpy()
           for i in range(0, len(ids), batch_size)]
    return np.concatenate(out)
