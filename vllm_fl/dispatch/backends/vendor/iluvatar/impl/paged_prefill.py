# Copyright (c) 2026 BAAI. All rights reserved.
"""Causal paged prefill with K/V reuse across several GQA query positions.

The generic decode-oriented kernel covers only two query positions per block
for GQA=8. This kernel packs more query positions into each matrix product,
reuses the loaded K/V tile, and omits features outside the guarded fast path.
It keeps FP32 online softmax and accumulation, with BF16 probabilities for PV.
"""

from functools import wraps

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import resolve_seq_and_query_len


@triton.jit
def paged_prefill_kernel(
    Q,
    K,
    V,
    O,
    CU_Q,
    KV_LEN,
    PAGES,
    scale,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    GROUP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    BQ: tl.constexpr = BM // GROUP
    seq, local_block, q_start, q_len, kv_len = resolve_seq_and_query_len(
        CU_Q, KV_LEN, tl.program_id(0), NUM_SEQS, BQ
    )
    if local_block * BQ >= q_len:
        return
    kv_head = tl.program_id(1)
    rows = tl.arange(0, BM)
    dims = tl.arange(0, D)
    cols = tl.arange(0, BN)
    q_pos = local_block * BQ + rows // GROUP
    q_head = kv_head * GROUP + rows % GROUP
    q = tl.load(
        Q + (q_start + q_pos[:, None]) * Q0 + q_head[:, None] * Q1 + dims[None, :],
        mask=q_pos[:, None] < q_len,
        other=0,
    )
    context = kv_len - q_len
    # Do not read future KV pages for early causal query tiles.
    stop = tl.minimum(kv_len, context + (local_block + 1) * BQ)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    for start in range(0, tl.cdiv(stop, BN)):
        pos = start * BN + cols
        page = tl.load(
            PAGES + seq * PAGE_STRIDE + pos // PAGE_SIZE,
            mask=pos < kv_len,
            other=0,
        ).to(tl.int64)
        key = tl.load(
            K
            + page[None, :] * K0
            + (pos[None, :] % PAGE_SIZE) * K1
            + kv_head * K2
            + dims[:, None],
            mask=pos[None, :] < kv_len,
            other=0,
        )
        value = tl.load(
            V
            + page[:, None] * V0
            + (pos[:, None] % PAGE_SIZE) * V1
            + kv_head * V2
            + dims[None, :],
            mask=pos[:, None] < kv_len,
            other=0,
        )
        score = tl.dot(q, key) * scale
        score = tl.where(
            (pos[None, :] < kv_len) & (pos[None, :] <= context + q_pos[:, None]),
            score,
            -float("inf"),
        )
        new_maximum = tl.maximum(maximum, tl.max(score, 1))
        probability = tl.exp(score - new_maximum[:, None])
        correction = tl.exp(maximum - new_maximum)
        denominator = denominator * correction + tl.sum(probability, 1)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probability.to(value.dtype), value)
        maximum = new_maximum
    result = accumulator / denominator[:, None]
    tl.store(
        O + (q_start + q_pos[:, None]) * O0 + q_head[:, None] * O1 + dims[None, :],
        result,
        mask=q_pos[:, None] < q_len,
    )


_OPTION_DEFAULTS = {
    "alibi_slopes": None,
    "output_scale": None,
    "qq_bias": None,
    "sinks": None,
    "mm_prefix_range": None,
    "use_alibi_sqrt": False,
    "kv_quant_mode": 0,
    "k_scale_cache": None,
    "v_scale_cache": None,
    "chunk_lookback": -1,
    "use_td": False,
}
_PREFILL_UNUSED = {
    "seq_threshold_3D",
    "num_par_softmax_segments",
    "softmax_segm_output",
    "softmax_segm_max",
    "softmax_segm_expsum",
}


def make_unified_attention(original):
    """Wrap the existing backend, retaining its complete unsupported-feature path."""

    @wraps(original)
    def attention(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        *args,
        **kwargs,
    ):
        supported = (
            not args
            and max_seqlen_q >= 16
            and causal is True
            and tuple(window_size) == (-1, -1)
            and softcap == 0
            # vLLM passes K/V descale tensors even with BF16 caches. Like
            # the baseline kernel, ignore them for unquantized BF16.
            and q_descale is None
            and q.dtype == k.dtype == v.dtype == out.dtype == torch.bfloat16
            and q.ndim == out.ndim == 3
            and k.ndim == v.ndim == 4
            and q.shape[-1] == k.shape[-1] == v.shape[-1] == out.shape[-1] == 128
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == out.stride(-1) == 1
            and k.shape[2] == v.shape[2]
            and k.shape[2] > 0
            and q.shape[1] % k.shape[2] == 0
            and q.shape[1] // k.shape[2] in (1, 2, 4, 8)
            and all(
                name in _PREFILL_UNUSED
                or (name in _OPTION_DEFAULTS and value == _OPTION_DEFAULTS[name])
                for name, value in kwargs.items()
            )
        )
        if not supported:
            return original(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                max_seqlen_q,
                seqused_k,
                max_seqlen_k,
                softmax_scale,
                causal,
                window_size,
                block_table,
                softcap,
                q_descale,
                k_descale,
                v_descale,
                *args,
                **kwargs,
            )
        group = q.shape[1] // k.shape[2]
        bm, bn = 64, 32
        bq = bm // group
        num_seqs = seqused_k.numel()
        grid = (q.shape[0] // bq + num_seqs, k.shape[2])
        paged_prefill_kernel[grid](
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            seqused_k,
            block_table,
            softmax_scale,
            q.stride(0),
            q.stride(1),
            *k.stride()[:3],
            *v.stride()[:3],
            out.stride(0),
            out.stride(1),
            block_table.stride(0),
            k.shape[1],
            num_seqs,
            group,
            bm,
            bn,
            128,
            num_warps=4,
            num_stages=1,
        )

    attention._iluvatar_paged_prefill = True
    return attention


def register_paged_prefill():
    """Install within Iluvatar's existing TRITON_ATTN implementation."""
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.ops import triton_unified_attention

    if triton_unified_attention.is_batch_invariant:
        return False
    original = triton_unified_attention.unified_attention
    if getattr(original, "_iluvatar_paged_prefill", False):
        return True
    optimized = make_unified_attention(original)
    triton_unified_attention.unified_attention = optimized
    triton_attn.unified_attention = optimized
    return True
