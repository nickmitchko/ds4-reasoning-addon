"""Inject reasoning-head latents into the DeepSeek-V4 backbone.

Turns a source-layer hidden state into an injectable hidden-space vector and
delivers it into the backbone. Two delivery paths:

- **Hook path (default, pristine fork):** overwrite the embed_tokens output at
  chosen positions via a worker forward hook (``hook_inject``). Token ids flow
  normally, so DeepSeek-V4's hash-based MoE routing (which requires ``input_ids``)
  keeps working. This is the path that actually serves on this fork.
- **prompt_embeds path:** ``build_embeds_prompt`` assembles vLLM's native
  mixed-mode request. Kept for reference / non-hash-MoE models, but on this fork
  ``--enable-prompt-embeds`` nulls ``input_ids`` and crashes hash-MoE, so prefer
  the hook path here.

Pipeline (matches training, see modules/reasoning_head.py::colar_loss):
    h_src  --layer_norm-->  head -> mu  (latent, latent_dim)
    mu     --layer_norm-->  decoder      -> hidden-space vector (hidden_size)

The decoder output lives in the *layer-normed* hidden space the autoencoder was
trained on.
"""

import functools
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from .checkpoint import ReasoningBundle, load_bundle
from . import hook_inject

logger = logging.getLogger(__name__)


class ReasoningInjector:
    """Runs the head->decoder pipeline and builds mixed-mode prompt_embeds."""

    def __init__(self, bundle: ReasoningBundle, device: str = "cpu"):
        self.bundle = bundle
        self.device = device
        bundle.reasoning_head.to(device)
        if bundle.decoder is not None:
            bundle.decoder.to(device)
        if bundle.target_proj is not None:
            bundle.target_proj.to(device)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ReasoningInjector":
        return cls(load_bundle(path, map_location=device), device=device)

    @property
    def hidden_size(self) -> int:
        return self.bundle.hidden_size

    @torch.no_grad()
    def latent_from_hidden(self, h_src: torch.Tensor) -> torch.Tensor:
        """Source-layer hidden (..., hidden_size) -> predicted latent mu (..., latent_dim).

        Applies the same layer-norm the head was trained under.
        """
        h_src = h_src.to(self.device).float()
        h_src = F.layer_norm(h_src, (self.hidden_size,))
        mu, _ = self.bundle.reasoning_head(h_src)
        return mu

    @torch.no_grad()
    def decode_latent(self, mu: torch.Tensor) -> torch.Tensor:
        """Latent mu (..., latent_dim) -> injectable hidden vector (..., hidden_size).

        Layer-norms the latent (matching the autoencoder's input) before decoding.
        Raises if the bundle has no decoder.
        """
        if self.bundle.decoder is None:
            raise RuntimeError(
                "This bundle has no decoder; the latent cannot be mapped back "
                "to hidden space. Retrain with train_decoder=True."
            )
        mu = mu.to(self.device).float()
        mu = F.layer_norm(mu, (self.bundle.latent_dim,))
        return self.bundle.decoder(mu)

    @torch.no_grad()
    def hidden_to_injectable(self, h_src: torch.Tensor) -> torch.Tensor:
        """Convenience: source hidden -> injectable hidden (head then decoder)."""
        return self.decode_latent(self.latent_from_hidden(h_src))

    # --- hook-based injection (pristine-fork path) ---------------------------

    def set_hook_injection(self, engine, positions, embeds) -> None:
        """Install/refresh the embed_tokens overwrite hook in every TP worker.

        Args:
            engine: a vLLM ``LLM`` (or subclass) exposing ``apply_model``.
            positions: list[int] sequence positions to overwrite.
            embeds: (len(positions), hidden_size) decoded vectors (CPU ok).
        """
        embeds_cpu = embeds.detach().to("cpu", copy=True).float()
        pos = list(positions)
        engine.apply_model(
            functools.partial(hook_inject.set_injection, positions=pos, embeds=embeds_cpu)
        )

    def clear_hook_injection(self, engine) -> None:
        """Disable injection in every worker (hook stays installed but inert)."""
        engine.apply_model(hook_inject.clear_injection)

    def remove_hook_injection(self, engine) -> None:
        """Remove the injection hook entirely in every worker."""
        engine.apply_model(hook_inject.remove_injection)

    def build_embeds_prompt(
        self,
        prompt_token_ids: list[int],
        inject_embeds: torch.Tensor,
        inject_positions: Optional[list[int]] = None,
        placeholder_token_id: int = 0,
    ) -> dict:
        """Assemble a mixed-mode EmbedsPrompt for vLLM ``LLM.generate``.

        Builds a full-length ``prompt_embeds`` tensor (seq_len, hidden_size) and a
        ``prompt_is_token_ids`` mask. Token positions carry zeros in the embed
        tensor (the runner overwrites them from the token id via ``torch.where``);
        injected positions carry rows from ``inject_embeds`` and are masked False.

        Args:
            prompt_token_ids: Token ids for the ordinary (text) positions plus one
                placeholder id per injected position.
            inject_embeds: (n_inject, hidden_size) injectable hidden vectors, e.g.
                from ``hidden_to_injectable``.
            inject_positions: Indices into the final sequence that are latent
                positions. Defaults to the last ``n_inject`` positions.
            placeholder_token_id: Token id to sit at injected positions (masked
                out; only needs to be in-vocab).

        Returns:
            A dict suitable as a vLLM prompt (EmbedsPrompt schema).
        """
        if inject_embeds.ndim == 1:
            inject_embeds = inject_embeds.unsqueeze(0)
        n_inject = inject_embeds.shape[0]
        if inject_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"inject_embeds hidden dim {inject_embeds.shape[-1]} != "
                f"bundle hidden_size {self.hidden_size}"
            )

        seq_len = len(prompt_token_ids)
        if inject_positions is None:
            inject_positions = list(range(seq_len - n_inject, seq_len))
        if len(inject_positions) != n_inject:
            raise ValueError(
                f"{len(inject_positions)} inject_positions for {n_inject} embeds"
            )

        prompt_embeds = torch.zeros(
            seq_len, self.hidden_size, dtype=torch.float32
        )
        is_token_ids = [True] * seq_len
        token_ids = list(prompt_token_ids)
        for row, pos in enumerate(inject_positions):
            prompt_embeds[pos] = inject_embeds[row].float().cpu()
            is_token_ids[pos] = False
            token_ids[pos] = placeholder_token_id

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_token_ids": token_ids,
            "prompt_is_token_ids": is_token_ids,
        }

    def build_embeds_prompts_batch(
        self,
        prompt_token_ids: list[list[int]],
        inject_embeds: list[torch.Tensor],
        inject_positions: Optional[list[list[int]]] = None,
        placeholder_token_id: int = 0,
    ) -> list[dict]:
        """Batched ``build_embeds_prompt``: one EmbedsPrompt per request.

        Each request is self-describing (its own ``prompt_embeds`` +
        ``prompt_is_token_ids`` mask), so submitting the returned list to a
        single ``engine.generate([...])`` is batch-safe -- vLLM tracks the
        per-request offsets internally. This is the injection counterpart to
        ``HookedVLLMEngine.get_capture_batch`` on the capture side.

        Args:
            prompt_token_ids: per-request token id lists.
            inject_embeds: per-request injectable vectors; entry i is
                ``(n_inject_i, hidden_size)`` (or 1D for a single position).
            inject_positions: per-request position lists; defaults per request
                to the trailing ``n_inject_i`` positions (see
                ``build_embeds_prompt``).
            placeholder_token_id: in-vocab id parked at injected positions.

        Returns:
            list of EmbedsPrompt dicts aligned with the inputs.
        """
        n = len(prompt_token_ids)
        if len(inject_embeds) != n:
            raise ValueError(
                f"{len(inject_embeds)} inject_embeds for {n} prompts"
            )
        if inject_positions is not None and len(inject_positions) != n:
            raise ValueError(
                f"{len(inject_positions)} inject_positions lists for {n} prompts"
            )
        return [
            self.build_embeds_prompt(
                prompt_token_ids[i],
                inject_embeds[i],
                inject_positions=None if inject_positions is None
                else inject_positions[i],
                placeholder_token_id=placeholder_token_id,
            )
            for i in range(n)
        ]
