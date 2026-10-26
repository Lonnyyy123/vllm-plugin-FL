# BI-V150 sequence-indexed paged decode

The existing Triton decode variant maps a flattened query-block grid back to
sequences. For single-token GQA decode this implementation instead launches
one CTA per sequence, KV head, and existing KV segment. Each CTA loads the
query index directly from cumulative sequence lengths and computes all eight
query heads sharing that KV head. This removes the sequence lookup and generic
query-block mapping while preserving the backend's 16-way segmented softmax.

Each segment writes an unnormalized FP32 accumulator, running maximum, and
denominator into the existing backend scratch tensors. A separate reduction
combines segments using their maxima and denominators. Empty segments write
zero accumulators/denominators and negative-infinite maxima, avoiding stale
scratch reads. Empty query sequences perform no writes; active queries are
indexed by cumulative lengths rather than sequence IDs. Physical KV page
offset arithmetic uses 64-bit integers.

## Dispatch boundary

The worker registers this wrapper after the paged-prefill wrapper, retaining
the existing `TRITON_ATTN` backend and vendor dispatch. The decode path requires
BF16 Q/K/V/output, head dimension 128, GQA ratio 8, contiguous inner dimensions,
plain causal attention, maximum query length 1, and the existing FP32 scratch
buffers with 16 segments and an eligible 3D sequence threshold. Unsupported
features and shapes call the original implementation with unchanged arguments.
BF16 ignores K/V descale tensors, consistent with the original backend.
Registration is idempotent and disabled under vLLM batch invariance.

This does not change attention precision, KV allocation, serving parameters,
benchmark inputs, or sampling. The prefill and decode kernels remain separate.

## Tile selection and measured operator results

The default KV tile has 16 columns. An offline comparison on BI-V150 tested
16/32/64/128 columns, with two KV allocations, CUDA Graph replay, graph-external
events, 12 timing samples, and an independent FP32 attention reference. Wider
tiles were not uniformly faster: 64 columns provided almost no gain in the
long-context cases, and 128 columns regressed. Sixteen columns improved both
short and long cases without selecting benchmark-specific lengths or batches.

| Queries | KV length | Original median | New median, tile 16 |
|---:|---:|---:|---:|
| 1 | 17 | 0.01577 ms | 0.01164 ms |
| 8 | 257 | 0.03241 ms | 0.02508 ms |
| 64 | 4096 | 1.68885 ms | 1.21737 ms |
| 64 | 4609 | 1.87810 ms | 1.36121 ms |
| 31 | 16384 | 3.09884 ms | 2.31139 ms |
| 31 | 16897 | 3.17243 ms | 2.35336 ms |

These are operator measurements, not official model throughput or accuracy.
Full official throughput, TTFT comparison, and MATH-500 Level 3 evaluation
must pass before treating this as a validated contest submission.

## Validation and rollback

Run both suites together to cover numerical accuracy, tail pages, strided
queries/outputs, CUDA Graph metadata changes, BF16 descales, feature fallback,
and combined prefill/decode registration in either order:

```sh
VLLM_PLUGINS=fl python -m pytest \
  tests/functional_tests/test_iluvatar_paged_prefill.py \
  tests/functional_tests/test_iluvatar_paged_decode.py -v -rA
```

Install from the validated source tree using the image's existing dependencies:

```sh
VLLM_VENDOR= python -m pip install --no-deps --no-build-isolation .
```

Verify installed worker and kernel hashes against the source and confirm both
worker registration messages before measuring. To roll back only decode,
revert its independent commit and reinstall; the prefill wrapper retains its
original decode fallback. To restore the complete contest baseline, install
revision `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` from a separate checkout.
Restart the server after either change with identical official serving args.
