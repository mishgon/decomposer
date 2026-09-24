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
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    flash_attention_mask,
    sdpa_mask,
)
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
    """Give each layer type the mask its own backend expects.

    ``create_sliding_window_causal_mask`` passes ``local_size=sliding_window`` while
    ``create_causal_mask`` leaves it ``None``, which is how the two calls are told apart.
    The model keeps the results in separate entries of its mask mapping, so the sliding
    and global layers can be served different mask formats.

    Sliding layers go to FlashAttention, which applies the window itself and wants either
    ``None`` or the 2D padding mask it unpads with -- a materialized 4D band would simply
    be ignored. Global layers go to SDPA and take the ordinary 4D mask.
    """
    if kwargs.get("local_size") is None:
        return sdpa_mask(*args, **kwargs)
    return flash_attention_mask(*args, **kwargs)


def register() -> None:
    """Install the implementation under both the attention and mask registries."""
    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, mixed_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, mixed_attention_mask)
