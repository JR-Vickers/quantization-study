# Quantization Study: DeepSeek-Coder-V2-Lite-Instruct

This project studies post-training quantization tradeoffs for
`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, a 16B-parameter
Mixture-of-Experts coding model, on local Apple Silicon hardware.

The goal is not just to produce a small model. The goal is to show a
defensible experimental workflow: establish baselines, measure deployment
tradeoffs, identify sensitive components, design a mixed-precision policy,
and validate that policy with a strict same-harness comparison.

## Summary

Final canonical mixed policy:

- Default precision: `Q4_K_M`
- Protected precision: `Q8_0`
- Protected layers: `3, 11, 14, 26`
- Policy id: `aggressive_q4_k_m_default_q8_protected_layers_3_11_14_26`
- Validation artifact:
  `results/mixed_precision/runs/aggressive_q4_q8p_03_11_14_26_perfrefresh_2026-05-27.validation.json`

Final strict validation result:

| Model | Size (GiB) | HumanEval pass@1 | Passed | Tok/s | Latency (s) |
|---|---:|---:|---:|---:|---:|
| FP16 GGUF | 29.27 | 73.17% | 120/164 | 69.60 | 1.985 |
| Q8_0 GGUF | 15.56 | 71.95% | 118/164 | 102.64 | 1.354 |
| Q4_K_M GGUF | 9.66 | 56.71% | 93/164 | 124.26 | 1.099 |
| Mixed `3,11,14,26` | 10.46 | 63.41% | 104/164 | 112.09 | 1.228 |

All rows above come from the same Notebook 09 run with:

- `quality_source = fresh_09_humaneval`
- `perf_source = fresh_09_perf`
- status `complete`
- `IMPORT_LEGACY_09_AGGREGATE = False`

The mixed policy recovered **+6.71 pass@1 points** over Q4_K_M while
remaining **~1.61x faster than FP16** in the short generation benchmark. It
did not preserve full Q4_K_M throughput, so the final result is a quality
recovery tradeoff, not a free deployment win.

## What Was Tested

The notebooks progress from basic inspection to validated mixed precision:

1. Model loading and architecture exploration
2. Weight statistics and distribution visualization
3. FP16 HumanEval baseline
4. Manual INT8 quantization math
5. GGUF exports with `llama.cpp`
6. Deployment benchmarking for FP16, Q8_0, and Q4_K_M
7. Layer-level sensitivity analysis
8. Mixed-precision policy synthesis
9. Mixed-precision GGUF validation and policy iteration

The project uses `llama.cpp` / GGUF for deployment-style quantization and
HumanEval pass@1 for code-quality evaluation.

## Key Results

### Baseline And Quantized Artifacts

From `results/05_ptq_artifacts_summary.json`:

| Artifact | Size (GiB) | Relative size |
|---|---:|---:|
| FP16 GGUF | 29.27 | 100.0% |
| Q8_0 GGUF | 15.56 | 53.2% |
| Q4_K_M GGUF | 9.66 | 33.0% |

Notebook 06 established the deployment payoff of lower precision: Q8_0 and
Q4_K_M were substantially faster and smaller than FP16, but whole-model Q4_K_M
lost too much HumanEval quality to be an acceptable final answer by itself.

### Layer Sensitivity

Notebook 07 uses targeted PyTorch INT4 ablations as component-level
sensitivity estimates. These are not deployment measurements; they are used
to decide which components deserve protection in a mixed-precision GGUF.

Important Notebook 07 findings:

- Full PyTorch baseline: **70.12%** HumanEval pass@1 (`115/164`)
- Top screened sensitive layers: `3, 26, 11, 14`
- Full follow-up protected group: `3, 11, 14, 26`
- Mean full INT4 drop for protected group: **13.11 pass@1 points**
- Low-sensitivity controls: `0, 2, 15, 17`
- Mean full delta for control group: **-3.51 pass@1 points**

Negative control deltas are treated as "no observed harm" in that run, not as
evidence that quantization improves quality.

### Mixed Policy Iterations

Notebook 09 validates concrete mixed GGUF artifacts. Starting with the final
strict protocol, only same-session HumanEval and same-session perf runs are
used for decisions.

| Policy | Protected layers | Pass@1 | Tok/s | Decision |
|---|---:|---:|---:|---|
| 3-layer variant | `11,14,26` | 59.15% | 104.19 | Rejected: worse quality and speed than canonical |
| Canonical | `3,11,14,26` | 63.41% | 112.09 | Selected |
| +2 variant | `2,3,11,14,26` | 62.20% | 102.50 | Rejected: lower quality and speed than canonical |

The canonical policy was selected because it gave the strongest observed
quality/speed tradeoff among tested layer-level policies.

## Decision Rule

Policy search stopped after bounded, targeted iteration.

The acceptance rule was:

- Compare only strict same-session runs.
- Require full HumanEval completion (`164/164`) for all compared GGUFs.
- Require fresh Notebook 09 perf for all compared GGUFs.
- Prefer policies that recover meaningful quality over Q4_K_M without
  eliminating most of Q4_K_M's throughput advantage.
- Stop when targeted variants fail to materially improve the current best.

The `11,14,26` and `2,3,11,14,26` variants both underperformed the
`3,11,14,26` candidate, so further layer toggling was treated as diminishing
return rather than useful search.

## Scope Boundary: MoE Experts

DeepSeek-Coder-V2-Lite-Instruct is a Mixture-of-Experts model, and expert-level
sensitivity is a real next question. This phase intentionally stops at
layer-level mixed precision.

That boundary is explicit:

- The current project identifies sensitive layers and validates layer-level
  GGUF mixed precision.
- It does not claim to identify the most sensitive experts inside each MoE
  layer.
- Expert-level policy search is the natural phase-2 extension.

A bounded expert-level extension would be:

1. Select 3-6 layers from the sensitive set.
2. Run a short per-expert screening pass within those layers.
3. Promote only strong expert signals to full HumanEval validation.
4. Validate any expert-targeted GGUF policy under the same strict Notebook 09
   protocol used here.

This was left out of the final scope to avoid rushing a noisy secondary study
after the layer-level evidence had already reached a defensible endpoint.

## Reproducibility

This project uses `uv`.

```bash
uv sync
```

Run notebooks in order from `notebooks/01...ipynb` through
`notebooks/09...ipynb`.

For strict Notebook 09 validation runs, use:

```python
RUN_FULL_HUMANEVAL = True
RUN_FRESH_PERF_BENCHMARKS = True
REEVALUATE_EXISTING = True
IMPORT_LEGACY_09_AGGREGATE = False

HUMANEVAL_MODEL_KEYS_OVERRIDE = [
    "fp16",
    "q8_0",
    "q4_k_m",
    "<candidate_policy_id>",
]
PERF_MODEL_KEYS_OVERRIDE = HUMANEVAL_MODEL_KEYS_OVERRIDE
```

Interpret a run only if every compared model has:

- `quality_source == "fresh_09_humaneval"`
- `perf_source == "fresh_09_perf"`
- status `complete`

## Repository Structure

- `notebooks/` - step-by-step learning pipeline
- `results/` - JSON outputs and validation artifacts
- `utils/` - shared helper functions
- `exports/` - local GGUF artifacts, not intended for source control
- `PLAN.md` - methodology and project roadmap

## Caveats

- HumanEval pass@1 is one benchmark, not a complete code-generation evaluation.
- The short perf benchmark is useful for local comparison, but production
  deployment would need a broader prompt suite and repeated runs.
- Notebook 07 sensitivity tests are PyTorch component ablations, not final
  GGUF deployment results.
- This phase does not optimize per-expert precision inside MoE layers.

## References

- Model: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- Quantization backend: https://github.com/ggml-org/llama.cpp
- Evaluation: HumanEval
