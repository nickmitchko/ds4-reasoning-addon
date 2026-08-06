"""Load reasoning-head bundles produced by the trainer.

Reads the v2/v3 self-describing bundle (head + decoder + target_proj +
geometry), from either a torch ``.pt`` pickle or a ``.safetensors`` file --
published model repos ship the latter. Legacy bare-head state_dicts are accepted
for the head alone, but injection REQUIRES a decoder, so those raise a clear
error when used for hosting.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .models import LatentDecoder, ReasoningCompressionHead

logger = logging.getLogger(__name__)


@dataclass
class ReasoningBundle:
    """A loaded reasoning head ready for injection."""

    reasoning_head: ReasoningCompressionHead
    decoder: Optional[LatentDecoder]
    target_proj: Optional[nn.Linear]
    hidden_size: int
    latent_dim: int
    source_layer: Optional[int]
    target_layer: Optional[int]
    compression_factor: Optional[int]

    @property
    def injectable(self) -> bool:
        """True if this bundle can map a latent back into hidden space."""
        return self.decoder is not None


def _load_safetensors_bundle(path: str, map_location="cpu") -> dict:
    """Read a ``.safetensors`` bundle into the same dict shape as a ``.pt`` one.

    Published model repos ship the head as safetensors: one flat tensor dict
    whose submodules are distinguished by key prefix (``reasoning_head.``,
    ``decoder.``, ``target_proj.``), with format_version/config/subdicts carried
    in the file's metadata. Rebuilding the nested dict here means the rest of
    load_bundle is format-agnostic.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - depends on the host env
        raise ImportError(
            f"{path} is a safetensors bundle but the 'safetensors' package is "
            "not installed. pip install safetensors"
        ) from exc

    with safe_open(path, framework="pt", device=map_location) as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    if "config" not in meta:
        raise ValueError(
            f"{path} has no 'config' in its safetensors metadata, so its geometry "
            "is unknown. Re-export the bundle with metadata, or use the .pt file."
        )

    obj: dict = {
        "format_version": int(meta.get("format_version", 3)),
        "config": json.loads(meta["config"]),
    }

    # Prefer the declared submodule list; fall back to whatever prefixes the
    # tensor keys actually carry, so an added submodule still loads.
    subdicts = json.loads(meta["subdicts"]) if "subdicts" in meta else sorted(
        {k.split(".", 1)[0] for k in tensors if "." in k}
    )
    for name in subdicts:
        prefix = f"{name}."
        sub = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        obj[name] = sub or None
    return obj


def load_bundle(path: str, map_location="cpu") -> ReasoningBundle:
    """Load a reasoning bundle from ``path``.

    Accepts a ``.safetensors`` bundle, a v2/v3 ``.pt`` bundle dict, or a legacy
    bare head state_dict. For legacy checkpoints the decoder/target_proj are
    absent (``injectable`` is False).
    """
    if os.fspath(path).endswith(".safetensors"):
        obj = _load_safetensors_bundle(path, map_location=map_location)
    else:
        obj = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(obj, dict) or "format_version" not in obj:
        # Legacy: bare head state_dict keyed like "net.0.weight".
        w_in = obj["net.0.weight"]
        w_out = obj["net.4.weight"]
        hidden_size = w_in.shape[1]
        latent_dim = w_out.shape[0] // 2
        head = ReasoningCompressionHead(hidden_size=hidden_size, latent_dim=latent_dim)
        head.load_state_dict(obj)
        head.eval()
        logger.warning(
            "Loaded LEGACY reasoning head from %s (no decoder). Injection is "
            "unavailable; retrain with train_decoder=True to host.", path,
        )
        return ReasoningBundle(
            reasoning_head=head, decoder=None, target_proj=None,
            hidden_size=hidden_size, latent_dim=latent_dim,
            source_layer=None, target_layer=None, compression_factor=None,
        )

    cfg = obj["config"]
    hidden_size, latent_dim = cfg["hidden_size"], cfg["latent_dim"]
    # Honour the bundle's own mlp_dim. It happens to equal the hidden_size//2
    # default for the shipped heads, so omitting it works by coincidence rather
    # than by design -- and stops working the moment a head is fit at a
    # different width.
    mlp_dim = cfg.get("mlp_dim")

    head = ReasoningCompressionHead(
        hidden_size=hidden_size, latent_dim=latent_dim, mlp_dim=mlp_dim
    )
    # strict=False: v2 bundles have no stop_head; it keeps its fresh init.
    incompatible = head.load_state_dict(obj["reasoning_head"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"{path} carries reasoning_head keys this addon does not know: "
            f"{sorted(incompatible.unexpected_keys)[:6]}. The checkpoint and the "
            "addon disagree on the head architecture."
        )
    # A v3 bundle claims a trained stop head, and serving trusts it to end the
    # latent phase. If those weights did NOT land, the stop head keeps its random
    # init and fires at arbitrary depths -- fail loudly instead.
    stop_missing = [k for k in incompatible.missing_keys if k.startswith("stop_head.")]
    if stop_missing and int(obj.get("format_version", 0)) >= 3:
        raise ValueError(
            f"{path} declares format_version>=3 (trained stop head) but these "
            f"stop_head weights are absent: {sorted(stop_missing)}. Serving with "
            "USE_STOP=1 would run a randomly-initialised stop head."
        )
    head.eval()

    decoder = None
    if obj.get("decoder") is not None:
        decoder = LatentDecoder(
            hidden_size=hidden_size, latent_dim=latent_dim, mlp_dim=mlp_dim
        )
        decoder.load_state_dict(obj["decoder"])
        decoder.eval()

    target_proj = None
    if obj.get("target_proj") is not None:
        target_proj = nn.Linear(hidden_size, latent_dim, bias=False)
        target_proj.load_state_dict(obj["target_proj"])
        target_proj.eval()

    return ReasoningBundle(
        reasoning_head=head, decoder=decoder, target_proj=target_proj,
        hidden_size=hidden_size, latent_dim=latent_dim,
        source_layer=cfg.get("source_layer"),
        target_layer=cfg.get("target_layer"),
        compression_factor=cfg.get("compression_factor"),
    )
