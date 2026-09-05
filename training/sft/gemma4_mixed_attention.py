"""Route Gemma-4's sliding-window layers to FlashAttention and its global layers to SDPA.

Gemma-4 declares ``head_dim: 256`` for sliding-window layers but
``global_head_dim: 512`` for full-attention layers. FlashAttention supports a head
dimension of at most 256, so a whole-model ``flash_attention_*`` request fails on the
global layers even though every sliding layer runs fine.

The sliding layers are where the quadratic cost lives: under SDPA each one computes
the full ``L x L`` score matrix and then discards everything outside a 512-wide band.
Sending only those layers to FlashAttention makes them genuinely banded and leaves the
global layers on SDPA, which is the only backend that accepts their 512-wide heads.

Transformers dispatches attention by a single string, so this module registers one
implementation that dispatches internally on the layer type.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

ATTENTION_IMPLEMENTATION = "gemma4_mixed"

# FlashAttention refuses a head dimension beyond this; Gemma-4's global layers are 512.
_FLASH_MAX_HEAD_DIM = 256

# transformers resolves the concrete kernel from ``config._attn_implementation``, which
# under this implementation is our own name. The sliding branch borrows the real one.
FLASH_IMPLEMENTATION = FLASH_ATTN_KERNEL_FALLBACK["flash_attention_3"]


def _is_sliding(module: torch.nn.Module, sliding_window: int | None) -> bool:
    """A layer is sliding when it carries a window; fall back to the module's type."""
    if sliding_window is not None:
        return True
    return getattr(module, "layer_type", None) == "sliding_attention"


def mixed_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    head_dim = query.shape[-1]
    if _is_sliding(module, sliding_window) and head_dim <= _FLASH_MAX_HEAD_DIM:
        # ``flash_attention_forward`` selects its kernel from the module's configured
        # implementation, so point that at the real FlashAttention repo for this call.
        config = module.config
        previous = config._attn_implementation
        config._attn_implementation = FLASH_IMPLEMENTATION
        try:
            return flash_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                sliding_window=sliding_window,
                **kwargs,
            )
        finally:
            config._attn_implementation = previous
    # Global layers (head_dim 512) and any sliding layer too wide for FlashAttention.
    kwargs.pop("softcap", None)
    kwargs.pop("s_aux", None)
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )


def mixed_attention_mask(*args: Any, **kwargs: Any) -> torch.Tensor | None:
    """Sliding layers get no mask; global layers get the ordinary SDPA mask.

    ``create_sliding_window_causal_mask`` passes ``local_size=sliding_window`` while
    ``create_causal_mask`` leaves it ``None``, which is how the two calls are told apart.

    For a sliding layer the window is applied by the FlashAttention kernel itself, so the
    mask must be ``None`` -- ``sdpa_mask`` would otherwise materialize an ``L x L`` band
    that the kernel then ignores. That is safe only when nothing else needs masking, so
    padding is rejected loudly rather than silently attended to.
    """
    if kwargs.get("local_size") is None:
        # Global layer: SDPA handles whatever mask is required (often None when the
        # batch is a single unpadded causal sequence).
        return sdpa_mask(*args, **kwargs)

    padding_mask = kwargs.get("attention_mask")
    if padding_mask is not None and not bool(padding_mask.all()):
        raise ValueError(
            f"{ATTENTION_IMPLEMENTATION} cannot mask padded positions on sliding-window "
            "layers: FlashAttention applies the window itself and would ignore the mask. "
            "Use attn_implementation: sdpa for padded or packed batches."
        )
    return None


def register() -> None:
    """Install the implementation under both the attention and mask registries."""
    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, mixed_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, mixed_attention_mask)
