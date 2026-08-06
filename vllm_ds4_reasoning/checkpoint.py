"""Load reasoning-head bundles produced by the trainer.

Reads the v2 self-describing bundle (head + decoder + target_proj + geometry).
Legacy bare-head state_dicts are accepted for the head alone, but injection
REQUIRES a decoder, so those raise a clear error when used for hosting.
"""

import logging
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


def load_bundle(path: str, map_location="cpu") -> ReasoningBundle:
    """Load a reasoning bundle from ``path``.

    Handles both the v2 bundle dict and legacy bare head state_dicts. For legacy
    checkpoints the decoder/target_proj are absent (``injectable`` is False).
    """
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

    head = ReasoningCompressionHead(hidden_size=hidden_size, latent_dim=latent_dim)
    # strict=False: v2 bundles have no stop_head; it keeps its fresh init.
    head.load_state_dict(obj["reasoning_head"], strict=False)
    head.eval()

    decoder = None
    if obj.get("decoder") is not None:
        decoder = LatentDecoder(hidden_size=hidden_size, latent_dim=latent_dim)
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
