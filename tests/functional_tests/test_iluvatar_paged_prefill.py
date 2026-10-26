# Copyright (c) 2026 BAAI. All rights reserved.
"""Check paged GQA against a float32 oracle, including mixed chunk lengths."""

import math

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def attention():
    if not torch.cuda.is_available() or "BI-V150" not in torch.cuda.get_device_name():
        pytest.skip("BI-V150 kernel test")
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar  # noqa: F401
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.paged_prefill import (
        make_unified_attention,
    )

    return make_unified_attention(unified_attention)


def make_case(query_lengths, kv_lengths, group=8, strided=False, page=16):
    torch.manual_seed(71)
    heads, dim = 2 * group, 128
    sizes = [math.ceil(length / page) for length in kv_lengths]
    pages = sum(sizes)
    k = torch.randn(pages, page, 2, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    permutation = torch.randperm(pages, device="cuda", dtype=torch.int32)
    table = torch.full(
        (len(sizes), max(sizes) + 2), -1, device="cuda", dtype=torch.int32
    )
    offset = 0
    for seq, size in enumerate(sizes):
        table[seq, :size] = permutation[offset : offset + size]
        offset += size
    q = torch.randn(
        sum(query_lengths),
        heads + (2 if strided else 0),
        dim,
        device="cuda",
        dtype=k.dtype,
    )[:, :heads]
    out = (
        torch.empty_like(q)
        if not strided
        else torch.empty(
            sum(query_lengths), heads + 3, dim, device="cuda", dtype=k.dtype
        )[:, :heads]
    )
    out.fill_(float("nan"))
    cu = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    lengths = torch.tensor(kv_lengths, device="cuda", dtype=torch.int32)
    return dict(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu,
        max_seqlen_q=max(query_lengths),
        seqused_k=lengths,
        max_seqlen_k=max(kv_lengths),
        softmax_scale=dim**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )


def reference(case, query_lengths, kv_lengths):
    q, k, v = case["q"], case["k"], case["v"]
    group = q.shape[1] // k.shape[2]
    result = torch.empty_like(q)
    offset = 0
    for seq, (qlen, klen) in enumerate(zip(query_lengths, kv_lengths)):
        if not qlen:
            continue
        blocks = case["block_table"][seq, : math.ceil(klen / k.shape[1])].long()
        key = k[blocks].reshape(-1, 2, 128)[:klen].float()
        value = v[blocks].reshape(-1, 2, 128)[:klen].float()
        causal = torch.arange(klen, device="cuda")[None, :] <= (
            klen - qlen + torch.arange(qlen, device="cuda")[:, None]
        )
        for head in range(q.shape[1]):
            score = q[offset : offset + qlen, head].float() @ key[:, head // group].T
            score *= case["softmax_scale"]
            score.masked_fill_(~causal, -float("inf"))
            result[offset : offset + qlen, head] = (
                score.softmax(-1) @ value[:, head // group]
            )
        offset += qlen
    return result


@pytest.mark.parametrize("group", [1, 2, 4, 8])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [
        ([17], [17]),
        ([0, 1, 3, 129], [1, 8193, 37, 4099]),
        ([33], [16387]),
    ],
)
def test_paged_prefill(attention, group, strided, qlens, klens):
    case = make_case(qlens, klens, group, strided)
    attention(**case)
    torch.testing.assert_close(
        case["out"], reference(case, qlens, klens), atol=0.016, rtol=0.016
    )


def test_graph_replay_and_input_updates(attention):
    qlens, klens = [31, 1], [129, 49]
    case = make_case(qlens, klens)
    attention(**case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attention(**case)
    case["q"].mul_(0.5)
    graph.replay()
    torch.testing.assert_close(
        case["out"], reference(case, qlens, klens), atol=0.016, rtol=0.016
    )


def test_bf16_ignores_unused_descale_tensors(attention):
    qlens, klens = [17, 3], [49, 35]
    case = make_case(qlens, klens)
    # TritonAttentionImpl passes expanded layer scales for BF16 too;
    # these must have no effect unless the cache itself is quantized.
    case["k_descale"] = torch.full((1,), 2.0, device="cuda").expand(2, 2)
    case["v_descale"] = torch.full((1,), 3.0, device="cuda").expand(2, 2)
    attention(**case)
    torch.testing.assert_close(
        case["out"], reference(case, qlens, klens), atol=0.016, rtol=0.016
    )


@pytest.mark.parametrize("page", [16, 32, 64])
def test_first_prefill_chunk(attention, page):
    case = make_case([2048], [2048], strided=True, page=page)
    attention(**case)
    torch.testing.assert_close(
        case["out"], reference(case, [2048], [2048]), atol=0.016, rtol=0.016
    )


@pytest.mark.slow
def test_kv_addresses_beyond_signed_int32(attention):
    # Only the final two pages are read. The large allocation ensures that
    # physical page * stride must be evaluated with a 64-bit integer.
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 9 * 1024**3:
        pytest.skip("requires 9 GiB free for a large-address regression")
    pages = 2**31 // (16 * 2 * 128) + 2
    case = make_case([17], [17])
    case["k"] = torch.empty(pages, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    case["v"] = torch.empty_like(case["k"])
    case["k"][-2:].normal_()
    case["v"][-2:].normal_()
    case["block_table"][0, :2] = torch.tensor(
        [pages - 1, pages - 2], device="cuda", dtype=torch.int32
    )
    attention(**case)
    torch.testing.assert_close(
        case["out"], reference(case, [17], [17]), atol=0.016, rtol=0.016
    )


def test_registration_is_idempotent_and_respects_batch_invariance(
    attention, monkeypatch
):
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.ops import triton_unified_attention

    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.paged_prefill import (
        register_paged_prefill,
    )

    original = triton_unified_attention.unified_attention
    monkeypatch.setattr(triton_unified_attention, "unified_attention", original)
    monkeypatch.setattr(triton_attn, "unified_attention", original)
    monkeypatch.setattr(triton_unified_attention, "is_batch_invariant", True)
    assert not register_paged_prefill()
    assert triton_attn.unified_attention is original
    monkeypatch.setattr(triton_unified_attention, "is_batch_invariant", False)
    assert register_paged_prefill()
    patched = triton_attn.unified_attention
    assert register_paged_prefill()
    assert triton_unified_attention.unified_attention is patched
    assert triton_attn.unified_attention is patched


@pytest.mark.parametrize(
    "change",
    [
        {"max_seqlen_q": 1},
        {"causal": False},
        {"window_size": (127, 0)},
        {"softcap": 10.0},
        {"output_scale": 1.0},
        {"sinks": "unsupported"},
        {"kv_quant_mode": 1},
        {"chunk_lookback": 0},
        {"use_td": True},
        {"future_feature": True},
    ],
)
def test_unsupported_features_fall_back(attention, change):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.paged_prefill import (
        make_unified_attention,
    )

    calls = []

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return "fallback"

    case = make_case([17], [33])
    case.update(change)
    assert make_unified_attention(fallback)(**case) == "fallback"
    assert len(calls) == 1
