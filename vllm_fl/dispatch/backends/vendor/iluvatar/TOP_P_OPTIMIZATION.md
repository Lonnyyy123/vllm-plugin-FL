# BI-V150 row-resident top-p filtering

This change depends on the independently developed FlagGems
`row_local_top_p_` operator. It preserves the existing vLLM sampler and changes
only the top-p-only filtering call on BI-V150: FP32 GPU rows, batch >= 8,
vocabulary 65536–131072, nonoverlapping rows with unit inner stride, and
matching FP32 per-row top-p metadata. Other inputs call the original dispatcher.
If the new FlagGems operator is absent, registration leaves sampling unchanged.

The kernel retains each row's logit and weight vectors while searching the
probability boundary, avoiding the original tiled candidate-buffer traffic.
It uses runtime top-p values, without a fixed candidate count or assumed logit
distribution. The adapter does not change temperature, top-k settings, token
counts, penalties, softmax/RNG, per-request generators, processed logprobs,
model weights, attention, or the official serve command. It includes no RoPE
fusion changes.

Boundary ties retain the lowest token IDs first. PyTorch's unstable vocabulary
sort may choose other IDs with exactly equal boundary logits. Therefore mask
identity or identical generated text is not promised for tied inputs. The
nucleus probability boundary, retained count, and original finite logits must
remain correct within FP32 rounding. Accuracy still requires the full official
MATH-500 Level 3 evaluation; unit tests cannot establish that gate.

BI-V150 operator measurement on captured MiniCPM `[64, 130560]` logits:
6.733 ms original, 3.032 ms row-resident (2.22x), with identical reset copies
included in CUDA Graph timings. Retained counts match PyTorch in all rows;
non-boundary membership differences are zero. Small batches regress in the
prototype, so they retain vLLM's original path. End-to-end benefit is pending.

Related tests cover supported/fallback routing, idempotent registration,
seeded sampling with FP32/FP64 exponential noise, and processed logits/logprobs.
Raw experiment data belongs outside the source commit, under the project's
`baseline/results/20261027-iluvatar-sampling/` directory.

For rollback, stop the candidate's server, reinstall the saved plugin wheel
`vllm_plugin_fl-0.3.0rc0+g3b6f38f49-py3-none-any.whl` and original FlagGems
`flag_gems-5.3.5-py3-none-any.whl` using `pip install --no-deps --no-index
--force-reinstall --no-compile`, then restart with the same serve command.
The attention-only source branch remains `opt/iluvatar-paged-prefill`.
