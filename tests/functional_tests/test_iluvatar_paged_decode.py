# Copyright (c) 2026 BAAI. All rights reserved.
"""Numerical and graph checks for direct-indexed split-K paged decode."""

import pytest
import torch

from tests.functional_tests.test_iluvatar_paged_prefill import make_case, reference

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def attention():
    if not torch.cuda.is_available() or "BI-V150" not in torch.cuda.get_device_name():
        pytest.skip("BI-V150 kernel test")
    import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar  # noqa: F401
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.paged_decode import (
        make_decode_attention,
    )

    def unsupported(*args, **kwargs):
        raise AssertionError("expected decode fast path")

    return make_decode_attention(unsupported)


def scratch(case):
    case.update(
        seq_threshold_3D=64,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.full(
            (64, 16, 16, 128), float("nan"), device="cuda", dtype=torch.float32
        ),
        softmax_segm_max=torch.full(
            (64, 16, 16), float("nan"), device="cuda", dtype=torch.float32
        ),
        softmax_segm_expsum=torch.full(
            (64, 16, 16), float("nan"), device="cuda", dtype=torch.float32
        ),
    )
    return case


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [
        ([1], [1]),
        ([1], [17]),
        ([1], [257]),
        ([1] * 64, [4099] * 64),
        ([1] * 31, [16387] * 31),
        ([0, 1, 0, 1, 1], [1, 4097, 1, 17, 16385]),
    ],
)
def test_decode(attention, strided, qlens, klens):
    case = scratch(make_case(qlens, klens, strided=strided))
    attention(**case)
    torch.testing.assert_close(
        case["out"], reference(case, qlens, klens), atol=0.016, rtol=0.016
    )


@pytest.mark.parametrize("batch", [4, 32])
def test_graph_replay_updates_lengths_and_padding(attention, batch):
    qlens = [1] * batch
    klens = [4097, 33, 1, 257] + [17] * (batch - 4)
    case = scratch(make_case(qlens, klens, strided=True))
    # Reserve query slots with one initially inactive sequence. Empty
    # sequences must neither consume a query slot nor overwrite another.
    cumulative = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    cumulative[3:] -= 1
    case["cu_seqlens_q"].copy_(cumulative)
    attention(**case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attention(**case)
    case["cu_seqlens_q"].copy_(
        torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    )
    case["seqused_k"][0] = 1025
    case["q"].mul_(0.5)
    graph.replay()
    torch.testing.assert_close(
        case["out"],
        reference(case, qlens, [1025, *klens[1:]]),
        atol=0.016,
        rtol=0.016,
    )


def test_bf16_ignores_unused_descale_tensors(attention):
    case = scratch(make_case([1, 1], [17, 257]))
    case["k_descale"] = torch.full((1,), 2.0, device="cuda").expand(2, 2)
    case["v_descale"] = torch.full((1,), 3.0, device="cuda").expand(2, 2)
    attention(**case)
    torch.testing.assert_close(
        case["out"], reference(case, [1, 1], [17, 257]), atol=0.016, rtol=0.016
    )


@pytest.mark.parametrize("decode_first", [False, True])
def test_combined_registration_and_dispatch(attention, monkeypatch, decode_first):
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.ops import triton_unified_attention

    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.paged_decode import (
        register_paged_decode,
    )
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.paged_prefill import (
        register_paged_prefill,
    )

    def fallback(*args, **kwargs):
        raise AssertionError("expected prefill or decode fast path")

    monkeypatch.setattr(triton_unified_attention, "unified_attention", fallback)
    monkeypatch.setattr(triton_attn, "unified_attention", fallback)
    monkeypatch.setattr(triton_unified_attention, "is_batch_invariant", True)
    assert not register_paged_decode()
    assert triton_attn.unified_attention is fallback
    monkeypatch.setattr(triton_unified_attention, "is_batch_invariant", False)
    registrars = [register_paged_prefill, register_paged_decode]
    if decode_first:
        registrars.reverse()
    for register in registrars:
        assert register()
    patched = triton_attn.unified_attention
    for register in registrars:
        assert register()
    assert triton_unified_attention.unified_attention is patched
    assert triton_attn.unified_attention is patched
    for qlens, klens in [([1, 1], [17, 257]), ([17, 1], [33, 257])]:
        case = scratch(make_case(qlens, klens))
        patched(**case)
        torch.testing.assert_close(
            case["out"], reference(case, qlens, klens), atol=0.016, rtol=0.016
        )


@pytest.mark.parametrize(
    "change",
    [
        {"max_seqlen_q": 17},
        {"causal": False},
        {"window_size": (127, 0)},
        {"softcap": 10.0},
        {"kv_quant_mode": 1},
        {"num_par_softmax_segments": 8},
        {"softmax_segm_output": None},
        {"seq_threshold_3D": 0},
        {"future_feature": True},
    ],
)
def test_unsupported_falls_back(attention, change):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.paged_decode import (
        make_decode_attention,
    )

    called = []

    def fallback(*args, **kwargs):
        called.append((args, kwargs))
        return "fallback"

    case = scratch(make_case([1], [17]))
    case.update(change)
    assert make_decode_attention(fallback)(**case) == "fallback"
    assert len(called) == 1
