# SPDX-License-Identifier: Apache-2.0
"""Routing must preserve top-k, small batches, layouts, RNG and logprobs."""

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.impl.top_p import make_top_p_filter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")


@pytest.mark.parametrize(
    "unsupported",
    ["top_k", "no_p", "small_batch", "small_vocab", "dtype", "stride", "p_dtype"],
)
def test_unsupported_path_calls_original(unsupported):
    rows, vocab = (1, 65536) if unsupported == "small_batch" else (8, 65536)
    if unsupported == "small_vocab":
        vocab = 128
    dtype = torch.float16 if unsupported == "dtype" else torch.float32
    x = torch.empty(
        rows, vocab * (2 if unsupported == "stride" else 1), device="cuda", dtype=dtype
    )
    if unsupported == "stride":
        x = x[:, ::2]
    p = None if unsupported == "no_p" else torch.full((rows,), 0.95, device="cuda")
    if unsupported == "p_dtype":
        p = p.half()
    k = torch.full((rows,), 10, device="cuda") if unsupported == "top_k" else None
    calls = []

    def original(logits, top_k, top_p):
        assert logits is x and top_k is k and top_p is p
        calls.append(True)
        return logits

    def forbidden(*args):
        raise AssertionError("unsupported input selected optimized path")

    assert make_top_p_filter(original, forbidden)(x, k, p) is x
    assert calls == [True]


def test_eligible_does_not_replace_random_sampler():
    from vllm.v1.sample.ops import topk_topp_sampler as sampler

    x = torch.empty(8, 65536, device="cuda")
    p = torch.ones(16, device="cuda")[::2]
    random_sample, forward_native = (
        sampler.random_sample,
        sampler.TopKTopPSampler.forward_native,
    )
    calls = []

    def optimized(logits, levels):
        assert logits is x and levels is p
        calls.append(True)
        return logits

    def forbidden(*args):
        raise AssertionError("eligible input did not select optimized path")

    assert make_top_p_filter(forbidden, optimized)(x, None, p) is x
    assert calls == [True]
    assert sampler.random_sample is random_sample
    assert sampler.TopKTopPSampler.forward_native is forward_native


def test_registration_is_idempotent(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_sampler as sampler

    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.top_p import (
        register_row_local_top_p,
    )

    if "BI-V150" not in torch.cuda.get_device_name():
        pytest.skip("BI-V150 registration")
    original = sampler.apply_top_k_top_p
    monkeypatch.setattr(sampler, "apply_top_k_top_p", original)
    assert register_row_local_top_p()
    registered = sampler.apply_top_k_top_p
    assert getattr(registered, "_fl_row_local_top_p", False)
    assert registered.__wrapped__ is original
    assert register_row_local_top_p()
    assert sampler.apply_top_k_top_p is registered


@pytest.mark.parametrize("logprobs_mode", ["processed_logits", "processed_logprobs"])
@pytest.mark.parametrize("fp64", [False, True])
def test_seeded_sampling_and_processed_outputs(monkeypatch, logprobs_mode, fp64):
    from flag_gems.fused.row_local_top_p import row_local_top_p_

    from vllm.v1.sample.ops import topk_topp_sampler as sampler

    torch.manual_seed(971)
    source = torch.randn(8, 65536, device="cuda") * 4
    p = torch.full((8,), 0.95, device="cuda")
    layer = sampler.TopKTopPSampler(logprobs_mode=logprobs_mode, use_fp64_gumbel=fp64)

    def generators():
        return {
            i: torch.Generator(device="cuda").manual_seed(100 + i) for i in range(8)
        }

    monkeypatch.setattr(sampler, "apply_top_k_top_p", sampler.apply_top_k_top_p_pytorch)
    expected_ids, expected_processed = layer.forward_native(
        source.clone(), generators(), None, p
    )
    monkeypatch.setattr(
        sampler,
        "apply_top_k_top_p",
        make_top_p_filter(sampler.apply_top_k_top_p_pytorch, row_local_top_p_),
    )
    actual_ids, actual_processed = layer.forward_native(
        source.clone(), generators(), None, p
    )
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(
        actual_processed, expected_processed, rtol=1e-6, atol=1e-6
    )
