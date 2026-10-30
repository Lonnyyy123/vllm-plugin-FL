# SPDX-License-Identifier: Apache-2.0
"""MetaX sampling candidate: retain original reductions, fuse final mask/scatter.

GPU correctness/performance acceptance is deliberately deferred to the unified
module comparison. Public sort still returns int64 indices; this controlled
consumer uses the paired FlagGems radix implementation's private int32 result.
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.sort import radix_sort
from flag_gems.runtime import torch_device_fn


@triton.jit
def _mask_scatter_kernel(
    sorted_logits, indices, cumulative_probs, top_p, output,
    N: tl.constexpr, P_STRIDE: tl.constexpr, HAS_P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = column < N
    value = tl.load(sorted_logits + row * N + column, valid, other=0)
    index = tl.load(indices + row * N + column, valid, other=0)
    if HAS_P:
        prefix = tl.load(cumulative_probs + row * N + column, valid, other=0)
        p = tl.load(top_p + row * P_STRIDE)
        discard = (prefix <= 1.0 - p) & (column != N - 1)
        value = tl.where(discard, -float("inf"), value)
    # Stable radix indices are a permutation, so each destination is written
    # exactly once. No atomics, cleared output, or int64 index tensor is needed.
    tl.store(output + row * N + index, value, valid)


def supports_int32_filter(logits, k, p):
    if (logits.ndim != 2 or not logits.is_cuda or logits.dtype != torch.float32
            or not logits.is_contiguous() or logits.shape[1] != 130560
            or not 1 <= logits.shape[0] <= 64):
        return False
    for parameter in [k, p]:
        if parameter is not None and (parameter.ndim != 1
                or parameter.shape[0] != logits.shape[0]
                or parameter.device != logits.device):
            return False
        if (parameter is not None and parameter.untyped_storage().data_ptr()
                == logits.untyped_storage().data_ptr()):
            # A fused scatter must not overwrite a per-row threshold before a
            # different CTA has read it. The original unfused function is safe.
            return False
    if p is not None and p.dtype != torch.float32:
        return False
    return k is not None or p is not None


def apply_top_k_top_p_int32(logits, k, p):
    """Use only after supports_int32_filter; mutate/return the original logits."""
    if k is None and p is None:
        return logits
    assert supports_int32_filter(logits, k, p)
    with torch_device_fn.device(logits.device):
        sorted_logits, indices = radix_sort(
            logits, 4, descending=False, _return_int32_indices=True
        )
        if k is not None:
            threshold_index = sorted_logits.size(1) - k.to(torch.long)
            threshold = sorted_logits.gather(1, threshold_index.unsqueeze(dim=1))
            sorted_logits.masked_fill_(sorted_logits < threshold, -float("inf"))
        prefix = None
        if p is not None:
            # Preserve the original reduction implementations and order. Moving
            # these into another kernel could move a top-p boundary by one token.
            probabilities = sorted_logits.softmax(dim=-1)
            prefix = torch.cumsum(probabilities, dim=-1, out=probabilities)
        _mask_scatter_kernel[(triton.cdiv(logits.shape[1], 1024), logits.shape[0])](
            sorted_logits, indices, prefix, p, logits,
            logits.shape[1], p.stride(0) if p is not None else 0, p is not None,
            BLOCK=1024,
        )
    return logits
