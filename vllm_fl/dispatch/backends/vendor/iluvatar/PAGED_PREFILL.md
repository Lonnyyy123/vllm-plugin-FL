# BI-V150 causal paged prefill

The generic Triton attention path packs GQA heads into matrix rows. For eight
query heads per KV head its 16-row tile covers only two query positions.
Long prefills consequently reload the same KV data for many tiny query tiles.

`impl/paged_prefill.py` implements a separate causal paged-attention kernel with
64 query/head rows and 32 KV columns. With GQA=8, each tile covers eight query
positions. It reuses K/V across those positions, bounds each causal tile's KV
loop by the final visible query position, masks page-table tail loads, and
keeps running maxima, normalizers, and accumulators in FP32. The probability
matrix is converted to BF16 for the PV matrix product. No weights or KV cache
are quantized.

The worker registers this implementation on BI-V150 after selecting its
device. The existing `attention_backend` dispatch and `TRITON_ATTN` backend
remain in place. No serving or benchmark option is added or changed.

## Fast-path boundary

- Unquantized BF16 Q/K/V/output, head dimension 128, contiguous inner dimension.
- GQA ratios 1, 2, 4, or 8; at least one query length of 16 or more.
- Plain causal attention without sliding windows, sinks, softcaps, ALiBi,
  prefix ranges, attention bias, output scales, or tensor descriptors.
- Mixed query lengths, existing KV context, permuted physical pages, and
  non-contiguous query/output token strides are supported.

Other cases call the original implementation with the original arguments.
This wrapper leaves pure decode to its fallback. The separate
[`PAGED_DECODE.md`](PAGED_DECODE.md) wrapper may specialize that path. Registration is
idempotent and is skipped when vLLM batch invariance is enabled. BF16 ignores
the K/V descale tensors passed by vLLM, as does the original BF16 path.

## Validation

On a BI-V150 with the contest image:

```sh
VLLM_PLUGINS=fl python -m pytest tests/functional_tests/test_iluvatar_paged_prefill.py
```

The tests compare with independent FP32 paged causal attention and cover GQA
ratios, mixed/zero query lengths, long KV tails, permuted pages, strided
query/output tensors, CUDA Graph replay with updated inputs, BF16 descales,
first prefill chunks, page sizes 16/32/64, and fallback/registration behavior.
Tolerance is `atol=0.016, rtol=0.016` for BF16 probability products.

Initial operator measurements on BI-V150, Q=2048, Hq=16/Hkv=2/D=128, BF16,
page size 16, two distinct KV allocations, CUDA Graph replay with timing
events outside the graph (12 samples):

| Full KV length | Original median | New median | Operator speedup |
|---|---:|---:|---:|
| 2048 | 5.302 ms | 1.385 ms | 3.83x |
| 4096 | 15.215 ms | 3.835 ms | 3.97x |
| 16384 | 75.496 ms | 18.101 ms | 4.17x |

Mixed query lengths `[1,17,129]` with KV lengths `[8193,4099,4099]` measured
2.229 ms versus 1.151 ms (1.94x). These are operator results, not contest
throughput. Full official throughput and MATH-500 Level 3 validation are
pending; this implementation is not yet a validated contest submission.

The mathematical reference matters: in the first-prefill synthetic test the
original implementation's first-token output was half the expected V vector
(maximum absolute error 1.53125). The new kernel's maximum absolute error
against the FP32 reference was 0.015625 for KV=2048, 0.000977 for KV=4096, and
0.000488 for KV=16384. This observation does not establish the original
kernel/compiler defect's root cause. Full-model accuracy must be measured,
not inferred from agreement or disagreement with the original kernel.

For deployment, install the modified plugin with the image's existing build
dependencies, without upgrading vLLM or FlagGems:

```sh
VLLM_VENDOR= python -m pip install --no-deps --no-build-isolation .
```

Check the installed source hashes and the worker registration log before
measuring. A non-editable install does not pick up working-tree edits merely
because scripts run successfully from the repository directory. To roll back,
install the contest baseline revision into the same environment and restart
the server; keep the official serving command identical for both runs.
