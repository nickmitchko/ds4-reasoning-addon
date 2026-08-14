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


# ---------------------------------------------------------------------------
# Aux Head v4 ("v2") architecture -- VENDORED VERBATIM from
# modules/reasoning_head_v2.py in the training repo.
#
# Vendored rather than imported because the addon must stand alone: the serve
# process puts only $ROOT/vllm-ds4-reasoning on PYTHONPATH, so ``modules`` is
# NOT importable there (verified: ModuleNotFoundError from a clean cwd). Before
# this, load_bundle always built the v1 class, so every v2 bundle died with a
# size mismatch (net.4 is 2*M*latent wide, and the v2 stop head takes
# hidden+latent+2 inputs) -- i.e. a trained v2 head could not be served at all.
#
# These classes are copied, not re-derived: keep them in sync with the trainer.
# The bundle's config carries arch/bank_m, so a mismatch fails loudly at load.
# ---------------------------------------------------------------------------

class ReasoningCompressionHeadV2(nn.Module):
    """Encoder: source hidden -> (mu, log_sigma) over an M-slot latent bank."""

    def __init__(self, hidden_size: int = 4096, latent_dim: int = 1024,
                 mlp_dim: Optional[int] = None, bank_m: int = 4,
                 use_len_feats: bool = True):
        super().__init__()
        mlp_dim = mlp_dim or hidden_size // 2
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.bank_m = bank_m
        self.use_len_feats = use_len_feats

        # Same trunk as v1; only the output width changes (2 * M * latent_dim).
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 2 * bank_m * latent_dim),
        )

        # Stop head: h_src (+ pooled bank + 2 scalar length/step feats).
        stop_in = hidden_size + latent_dim + (2 if use_len_feats else 0)
        self.stop_head = nn.Sequential(
            nn.Linear(stop_in, mlp_dim // 2),
            nn.SiLU(),
            nn.Linear(mlp_dim // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor):
        """(..., H) -> mu, log_sigma each (..., M, latent_dim)."""
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"ReasoningCompressionHeadV2 expected hidden_size={self.hidden_size}, "
                f"got {hidden_states.shape[-1]}.")
        out = self.net(hidden_states)
        lead = out.shape[:-1]
        out = out.view(*lead, self.bank_m, 2 * self.latent_dim)
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(min=-10.0, max=2.0)  # v1 stability clamp
        return mu, log_sigma

    def stop_logit(self, hidden_states: torch.Tensor,
                   bank: Optional[torch.Tensor] = None,
                   n_prompt: Optional[torch.Tensor] = None,
                   step_frac: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Raw stop logit (..., 1). ``bank`` is the (..., M, latent) latent bank;
        it is mean-pooled. ``n_prompt`` / ``step_frac`` are the length and
        step-position features. Missing features are zero-filled, so this stays
        callable with the v1 signature (h only)."""
        feats = [hidden_states]
        lead = hidden_states.shape[:-1]
        dev, dt = hidden_states.device, hidden_states.dtype

        if bank is not None:
            feats.append(bank.mean(dim=-2))
        else:
            feats.append(torch.zeros(*lead, self.latent_dim, device=dev, dtype=dt))

        if self.use_len_feats:
            def _f(x):
                if x is None:
                    return torch.zeros(*lead, 1, device=dev, dtype=dt)
                t = torch.as_tensor(x, device=dev, dtype=dt)
                return t.expand(*lead).unsqueeze(-1) if t.dim() == 0 else t.reshape(*lead, 1)
            # log1p(n_prompt) scaled to ~O(1); step_frac already in [0, 1].
            feats.append(_f(None if n_prompt is None else torch.log1p(
                torch.as_tensor(n_prompt, device=dev, dtype=dt)) / 12.0))
            feats.append(_f(step_frac))

        return self.stop_head(torch.cat(feats, dim=-1))

    def reparameterize(self, mu, log_sigma):
        return mu + log_sigma.exp() * torch.randn_like(mu)


class LatentDecoderV2(nn.Module):
    """Bank -> one absolute embed-space vector, via self-attention over the bank.

    Emits a vector in layer-0 input-embedding space (NOT a layer-42 hidden and NOT
    a delta) -- see the contract note in the module docstring.
    """

    def __init__(self, hidden_size: int = 4096, latent_dim: int = 1024,
                 mlp_dim: Optional[int] = None, n_blocks: int = 1,
                 n_heads: int = 8, ffn_mult: int = 2):
        super().__init__()
        mlp_dim = mlp_dim or hidden_size // 2
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.n_blocks = n_blocks

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(latent_dim),
                "attn": nn.MultiheadAttention(latent_dim, n_heads, batch_first=True),
                "ln2": nn.LayerNorm(latent_dim),
                "ffn": nn.Sequential(
                    nn.Linear(latent_dim, ffn_mult * latent_dim),
                    nn.SiLU(),
                    nn.Linear(ffn_mult * latent_dim, latent_dim),
                ),
            }) for _ in range(n_blocks)
        ])
        # Pool the bank, then project to hidden space (v1-style output stage).
        self.out = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(..., M, latent) -> (..., hidden). Also accepts (..., latent) for M=1."""
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"LatentDecoderV2 expected latent_dim={self.latent_dim}, "
                             f"got {z.shape[-1]}.")
        squeeze_bank = False
        if z.dim() == 1:  # (latent,) -> (1, 1, latent)
            z = z.view(1, 1, -1)
            squeeze_bank = True
        elif z.dim() == 2:  # (M, latent) -> (1, M, latent)
            z = z.unsqueeze(0)
            squeeze_bank = True
        lead = z.shape[:-2]
        x = z.reshape(-1, z.shape[-2], self.latent_dim)  # (N, M, latent)

        for b in self.blocks:
            h = b["ln1"](x)
            a, _ = b["attn"](h, h, h, need_weights=False)
            x = x + a
            x = x + b["ffn"](b["ln2"](x))

        pooled = x.mean(dim=1)                    # (N, latent)
        out = self.out(pooled)                    # (N, hidden)
        out = out.reshape(*lead, self.hidden_size)
        return out.squeeze(0) if squeeze_bank and out.dim() > 1 else out
