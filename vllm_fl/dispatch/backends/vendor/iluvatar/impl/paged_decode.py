# Copyright (c) 2026 BAAI. All rights reserved.
"""Direct sequence-indexed split-K decode for paged BF16 GQA.

Each CTA handles one sequence, KV head, and KV segment. The query has one
position, so no binary search or generic flattened-query grid is needed.
Scratch storage and the number of segments come from the existing backend.
"""

from functools import wraps

import torch

from vllm.triton_utils import tl, triton

from .paged_prefill import _OPTION_DEFAULTS, _PREFILL_UNUSED


@triton.jit
def paged_decode_kernel(
    Q,
    K,
    V,
    CU,
    LENGTH,
    PAGES,
    PARTIAL,
    MAXIMUM,
    DENOMINATOR,
    scale,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    M0: tl.constexpr,
    M1: tl.constexpr,
    M2: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    L2: tl.constexpr,
    TABLE0: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    SEGMENTS: tl.constexpr,
    BN: tl.constexpr,
):
    seq, kv_head, segment = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    query_index = tl.load(CU + seq)
    query_end = tl.load(CU + seq + 1)
    if query_end == query_index:
        return
    length = tl.load(LENGTH + seq)
    rows = tl.arange(0, 16)
    dims = tl.arange(0, 128)
    columns = tl.arange(0, BN)
    heads = kv_head * GROUP + rows
    query = tl.load(
        Q + query_index * Q0 + heads[:, None] * Q1 + dims[None, :],
        mask=rows[:, None] < GROUP,
        other=0,
    )
    tiles = tl.cdiv(length, SEGMENTS * BN)
    first = segment * tiles
    last = tl.minimum((segment + 1) * tiles, tl.cdiv(length, BN))
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, 128), tl.float32)
    for tile in range(first, last):
        pos = tile * BN + columns
        page = tl.load(
            PAGES + seq * TABLE0 + pos // PAGE, mask=pos < length, other=0
        ).to(tl.int64)
        key = tl.load(
            K
            + page[None, :] * K0
            + (pos[None, :] % PAGE) * K1
            + kv_head * K2
            + dims[:, None],
            mask=pos[None, :] < length,
            other=0,
        )
        value = tl.load(
            V
            + page[:, None] * V0
            + (pos[:, None] % PAGE) * V1
            + kv_head * V2
            + dims[None, :],
            mask=pos[:, None] < length,
            other=0,
        )
        scores = tl.dot(query, key) * scale
        scores = tl.where(pos[None, :] < length, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        probabilities = tl.exp(scores - new_maximum[:, None])
        correction = tl.exp(maximum - new_maximum)
        denominator = denominator * correction + tl.sum(probabilities, 1)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(value.dtype), value)
        maximum = new_maximum
    tl.store(
        PARTIAL + query_index * P0 + heads[:, None] * P1 + segment * P2 + dims[None, :],
        accumulator,
        mask=rows[:, None] < GROUP,
    )
    tl.store(
        MAXIMUM + query_index * M0 + heads * M1 + segment * M2,
        maximum,
        mask=rows < GROUP,
    )
    tl.store(
        DENOMINATOR + query_index * L0 + heads * L1 + segment * L2,
        denominator,
        mask=rows < GROUP,
    )


@triton.jit
def merge_decode_segments(
    CU,
    PARTIAL,
    MAXIMUM,
    DENOMINATOR,
    OUT,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    M0: tl.constexpr,
    M1: tl.constexpr,
    M2: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    L2: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    SEGMENTS: tl.constexpr,
):
    seq, head = tl.program_id(0), tl.program_id(1)
    query_index = tl.load(CU + seq)
    if tl.load(CU + seq + 1) == query_index:
        return
    segments = tl.arange(0, SEGMENTS)
    dims = tl.arange(0, 128)
    maxima = tl.load(MAXIMUM + query_index * M0 + head * M1 + segments * M2)
    normalizers = tl.load(DENOMINATOR + query_index * L0 + head * L1 + segments * L2)
    maximum = tl.max(maxima, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(maxima - maximum)
    normalizer = tl.sum(normalizers * weights, 0)
    partial = tl.load(
        PARTIAL + query_index * P0 + head * P1 + segments[:, None] * P2 + dims[None, :]
    )
    output = tl.sum(partial * weights[:, None], 0)
    output = tl.where(normalizer > 0.0, output / normalizer, 0.0)
    tl.store(OUT + query_index * O0 + head * O1 + dims, output)


def make_decode_attention(original, *, tile_size=16):
    if tile_size not in (16, 32, 64, 128):
        raise ValueError("unsupported decode KV tile")

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
        partial = kwargs.get("softmax_segm_output")
        maxima = kwargs.get("softmax_segm_max")
        denominators = kwargs.get("softmax_segm_expsum")
        segments = kwargs.get("num_par_softmax_segments")
        threshold = kwargs.get("seq_threshold_3D")
        supported = (
            max_seqlen_q == 1
            and not args
            and causal is True
            and tuple(window_size) == (-1, -1)
            and softcap == 0
            and q_descale is None
            and q.dtype == k.dtype == v.dtype == out.dtype == torch.bfloat16
            and q.ndim == out.ndim == 3
            and k.ndim == v.ndim == 4
            and q.shape[-1] == k.shape[-1] == v.shape[-1] == out.shape[-1] == 128
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == out.stride(-1) == 1
            and k.shape[2] == v.shape[2]
            and q.shape[1] == 8 * k.shape[2]
            and segments == 16
            and threshold is not None
            and seqused_k.numel() <= threshold
            and all(
                name in _PREFILL_UNUSED
                or (name in _OPTION_DEFAULTS and value == _OPTION_DEFAULTS[name])
                for name, value in kwargs.items()
            )
            and all(
                tensor is not None
                and tensor.dtype == torch.float32
                and tensor.ndim >= 3
                and tensor.shape[0] >= q.shape[0]
                and tensor.shape[1] == q.shape[1]
                and tensor.shape[2] == segments
                for tensor in (partial, maxima, denominators)
            )
            and partial.ndim == 4
            and partial.shape[3] == 128
            and partial.stride(3) == 1
            and maxima.ndim == denominators.ndim == 3
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
        strides = (*partial.stride()[:3], *maxima.stride(), *denominators.stride())
        paged_decode_kernel[(seqused_k.numel(), k.shape[2], segments)](
            q,
            k,
            v,
            cu_seqlens_q,
            seqused_k,
            block_table,
            partial,
            maxima,
            denominators,
            softmax_scale,
            q.stride(0),
            q.stride(1),
            *k.stride()[:3],
            *v.stride()[:3],
            *strides,
            block_table.stride(0),
            k.shape[1],
            8,
            segments,
            tile_size,
            num_warps=4,
            num_stages=1,
        )
        merge_decode_segments[(seqused_k.numel(), q.shape[1])](
            cu_seqlens_q,
            partial,
            maxima,
            denominators,
            out,
            *strides,
            out.stride(0),
            out.stride(1),
            segments,
            num_warps=4,
        )

    attention._iluvatar_paged_decode = True
    return attention


def register_paged_decode():
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.ops import triton_unified_attention

    if triton_unified_attention.is_batch_invariant:
        return False
    original = triton_unified_attention.unified_attention
    if getattr(original, "_iluvatar_paged_decode", False):
        return True
    optimized = make_decode_attention(original)
    triton_unified_attention.unified_attention = optimized
    triton_attn.unified_attention = optimized
    return True
