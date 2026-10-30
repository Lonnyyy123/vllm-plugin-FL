# BI-V150 sequence-indexed paged decode

The existing Triton decode variant maps a flattened query-block grid back to
sequences. For single-token GQA decode this implementation instead launches
one CTA per sequence, KV head, and existing KV segment. Each CTA loads the
query index directly from cumulative sequence lengths and computes all eight
query heads sharing that KV head. This removes the sequence lookup and generic
query-block mapping while retaining the backend's segmented-softmax interface
and scratch allocation.

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

## Original tile selection and measured operator results

The initial implementation used a 16-column KV tile. An offline comparison on BI-V150 tested
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

## Delayed V staging and the 32-column tile

The current kernel loads V after the QK product and online-softmax update.
This shortens the live overlap of K and V staging before the PV product; the
softmax arithmetic and FP32 accumulator are unchanged. With this ordering,
the default KV tile is now 32 columns. The measured compiled kernel uses
12 KiB of shared memory, 95 registers, and no spills on BI-V150.

On 2026-10-30, with plugin reference `e297acf` and FlagGems `e303ca56`, CUDA
Graph microbenchmarks used two independent KV allocations with model Q/K/V
strides and 12 interleaved timing samples per version:

| Queries | KV length | Previous median | Delayed V, tile 32 |
|---:|---:|---:|---:|
| 1 | 17 | 0.01173 ms | 0.01321 ms |
| 8 | 257 | 0.02548 ms | 0.02799 ms |
| 31 | 4096 | 0.60126 ms | 0.53992 ms |
| 64 | 4096 | 1.21508 ms | 1.09551 ms |
| 64 | 4609 | 1.35840 ms | 1.22065 ms |

The change favors the larger decode workload; the short-context, small-batch
cases above regress. All 25 decode tests passed, including strided tensors,
tail pages, empty sequences, updated CUDA Graph inputs and fallback routing.

One fresh-server 4k wave per version, 64 requests with 4096 input and 1024
output tokens at concurrency 64, measured 1754.52 -> 1782.75 total tok/s
(+1.61%). Mean TTFT changed from 25671.71 to 25915.06 ms (+0.948%); median
and p99 TTFT also remained within +1%. Both versions completed all requests.
Only this decode source differed; serving and benchmark parameters matched.
An earlier screen on the image FlagGems baseline measured +3.25% throughput.

These single-wave screens do not establish a stable gain or replace the
official multi-round 4k/16k and MATH acceptance run. Keep the prior revision
available until the combined candidate passes those gates. Reverting the
independent delayed-V commit restores the previous load ordering and tile.

## Internal segment count and KV caching

The backend still supplies 16-segment scratch buffers and its existing
threshold. With at least 32 sequence/KV-head pairs, the wrapper now computes
four active segments and merges only those four. The scratch allocation and
strides do not change; unused capacity is never read. Both KV loads use the
`.cg` cache hint on this path. Smaller grids retain 16 segments and the
previous cache policy. This keeps at least 128 CTAs in the compact grid on
BI-V150, which reports 16 multiprocessors.

An unconditional four-segment version regressed at eight queries with 16k
KV, so the grid-size guard is part of the implementation. In a nine-shape
CUDA Graph confirmation, the guarded version preserved the small-grid cases,
sped up 64 queries with 4k KV by about 6.2%, and improved the measured large
16k cases by about 1.5-2.3%. FP32 softmax and accumulation are retained;
partitioning changes floating-point reduction grouping. All 26 decode tests
passed, including updated CUDA Graph metadata at 4 and 32 sequences.

One 4k candidate wave measured 1812.01 total tok/s against the immediately
preceding delayed-V wave's 1782.75 (+1.64%). Mean/median/p99 TTFT decreased.
The reference was reused to avoid another identical control run; both waves
had 64 successful requests with unchanged serving and benchmark parameters.
The two changes together measured +3.28% over the earlier 1754.52 control,
with all three TTFT values within +1%. These remain single-wave screening
results, pending the combined official throughput and MATH gates.

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
