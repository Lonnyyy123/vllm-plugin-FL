# SPDX-License-Identifier: Apache-2.0
"""Use row-resident FlagGems nucleus filtering on BI-V150.

Only the top-p-only filter changes. vLLM retains its temperature/penalties,
top-k handling, processed logprobs and original random-sampling implementation.
"""

from functools import wraps

import torch


def make_top_p_filter(original, row_local_top_p):
    @wraps(original)
    def filtered(logits, k, p):
        if (
            k is None
            and p is not None
            and logits.ndim == 2
            and logits.is_cuda
            and logits.dtype == torch.float32
            and logits.shape[0] >= 8
            and 65536 <= logits.shape[1] <= 131072
            and logits.stride(1) == 1
            and logits.stride(0) >= logits.shape[1]
            and p.shape == (logits.shape[0],)
            and p.dtype == torch.float32
            and p.device == logits.device
        ):
            return row_local_top_p(logits, p)
        return original(logits, k, p)

    filtered._fl_row_local_top_p = True
    return filtered


def register_row_local_top_p():
    """Called after device initialization; preserve unsupported deployments."""
    from vllm.platforms import current_platform
    from vllm.v1.sample.ops import topk_topp_sampler

    if getattr(current_platform, "vendor_name", "") != "iluvatar":
        return False
    if not torch.cuda.is_available() or "BI-V150" not in torch.cuda.get_device_name():
        return False
    original = topk_topp_sampler.apply_top_k_top_p
    if getattr(original, "_fl_row_local_top_p", False):
        return True
    try:
        from flag_gems.fused.row_local_top_p import row_local_top_p_
    except ImportError:
        return False
    topk_topp_sampler.apply_top_k_top_p = make_top_p_filter(original, row_local_top_p_)
    return True
