"""Standalone reasoning-head modules for the vLLM addon.

These mirror ``modules/reasoning_head.py`` in the training repo but are vendored
here so the addon is independently installable (a hosting box needs only this
package + vLLM, not the training code). Keep the layer geometry in sync with the
trainer -- the checkpoint bundle carries the dims, so a shape mismatch surfaces
loudly at load time.
"""

from typing import Optional

import torch
from torch import nn


class ReasoningCompressionHead(nn.Module):
    """3-layer MLP: hidden_size -> mlp_dim -> mlp_dim -> 2*latent_dim.

    Predicts (mu, log_sigma) of the next compressed reasoning latent from a
    source-layer hidden state. Inference uses ``mu`` (the MAP estimate).
    """

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int = 1024,
        mlp_dim: Optional[int] = None,
    ):
        super().__init__()
        mlp_dim = mlp_dim or hidden_size // 2
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 2 * latent_dim),
        )

        # Stop head (format_version 3): predict the </think> boundary from the
        # source hidden. Independent MLP so v2 bundles load net with strict=False.
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim // 2),
            nn.SiLU(),
            nn.Linear(mlp_dim // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"ReasoningCompressionHead expected hidden_size={self.hidden_size}, "
                f"got input dim={hidden_states.shape[-1]}."
            )
        mu, log_sigma = self.net(hidden_states).chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(min=-10.0, max=2.0)
        return mu, log_sigma

    def stop_logit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Raw stop logit(s) from the source hidden (..., 1). sigmoid > 0.5 at
        inference ends the latent phase (Phase 3)."""
        return self.stop_head(hidden_states)


class LatentDecoder(nn.Module):
    """Map a compressed latent (latent_dim) back to hidden space (hidden_size).

    The piece that makes the head injectable: without it the latent has no path
    back into the backbone. Symmetric to the head's MLP.
    """

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int = 1024,
        mlp_dim: Optional[int] = None,
    ):
        super().__init__()
        mlp_dim = mlp_dim or hidden_size // 2
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"LatentDecoder expected latent_dim={self.latent_dim}, "
                f"got input dim={z.shape[-1]}."
            )
        return self.net(z)
