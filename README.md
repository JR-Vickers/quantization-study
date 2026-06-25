# Quantization Study: DeepSeek-Coder-V2-Lite-Instruct

This project studies post-training quantization tradeoffs for `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, a 16B-parameter Mixture-of-Experts coding model, on local Apple Silicon hardware.

I chose this model for a few reasons.  I wanted to begin with a coding model because code generations lend themselves well to benchmarking - pass/fail criteria is just "Did the generated code output the correct result?"  I went with a smaller 16B model due to hardware constraints - I used a Macbook Pro M4 Max with 64 GB of RAM.  Models much larger than this would slow down the iteration speed too much.  Additionally, this model has a fairly sophisticated architecture that opens up several avenues for discovery.  In particular, it's a mixture-of-experts (MoE) model, which permits per-expert sensitivity analysis and quantization.  As of this writing, this project implements per-layer mixed-precision quantization.  Per-expert quantization is reserved for a future update.

This project has several nested goals.  Ultimately, it produces a model that uses selective quantization to improve model performance with minimal performance degradation.  Along the way, it establishes baselines, measures deployment tradeoffs, identifies sensitive components, designs a mixed-precision policy, and validates that policy with a strict same-harness comparison.

## Phase 1 Summary

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

The mixed policy recovered **+6.71 pass@1 points** over Q4_K_M while remaining **~1.61x faster than FP16** in the short generation benchmark. It did not preserve full Q4_K_M throughput, so the final result presents a quality recovery tradeoff.

## What Was Tested

The notebooks were designed to break the mixed-precision quantization and validation process into small steps that are easy to follow and replicate.

Notebook 01:  Model loading and architecture exploration
Notebook 02:  Weight statistics and distribution visualization
Notebook 03:  FP16 HumanEval baseline
Notebook 04:  Manual INT8 quantization math
Notebook 05:  GGUF exports with `llama.cpp`
Notebook 06:  Deployment benchmarking for FP16, Q8_0, and Q4_K_M
Notebook 07:  Layer-level sensitivity analysis
Notebook 08:  Mixed-precision policy synthesis
Notebook 09:  Mixed-precision GGUF validation and policy iteration

For deployment-style quantization, I used `llama.cpp` to generate new GGUF files.  For benchmarking, I chose to use **HumanEval pass@1** to evaluate code quality.

## Key Results

### Baseline And Quantized Artifacts

From `results/05_ptq_artifacts_summary.json`:

| Artifact | Size (GiB) | Relative size |
|---|---:|---:|
| FP16 GGUF | 29.27 | 100.0% |
| Q8_0 GGUF | 15.56 | 53.2% |
| Q4_K_M GGUF | 9.66 | 33.0% |

In Notebook 06, I measured the deployment payoff of lower precision.  Q8_0 and Q4_K_M were substantially smaller and faster than FP16, but naive, whole-model Q4_K_M lost far too much performance to be an acceptable final output by itself.  The rest of the notebooks were dedicated to identifying and shielding sensitive layers from aggressive Q4_K_M quantization, while continuing aggressive quantization on the less-sensitive layers.

### Layer Sensitivity

In Notebook 07, I measured layer sensitivity by performing INT4 virtual quantization on the layers individually.  Meaning: 

1. Quantize layer 0, leaving the other layers intact
2. Run evals and record the results
3. Reverse quantization on layer 0
4. Repeat steps 1-3 on every other layer

Due to time constraints, I ran only the first 30 HumanEval problems on each ablation - full runs would have taken days to complete.  This is sufficient for ranking the respective sensitivities of each layer.  This information was then used to decide which components to protect in the final mixed-precision GGUF.

One consequence of this approach is that the HumanEval scores for this section were higher than normal.  This is because HumanEval places easy problems at the beginning of the list, and harder problems towards the end.  As a result, the ablations for less-sensitive layers would return HumanEval scores that were higher than the baseline FP16 runs.  These results are interpreted as "no observed harm," not as evidence that quantization improved quality.

Important Notebook 07 findings:

- Full PyTorch baseline: **70.12%** HumanEval pass@1 (`115/164`)
- Top screened sensitive layers: `3, 11, 14, 26`
- Full follow-up protected group: `3, 11, 14, 26`
- Mean full INT4 drop for protected group: **13.11 pass@1 points**
- Low-sensitivity controls: `0, 2, 15, 17`
- Mean full delta for control group: **-3.51 pass@1 points**

### Mixed Policy Iterations

In Notebook 09, I compared the HumanEval and performance metrics of the different GGUF artifacts.  All comparisons were made in the same session, which eliminates certain forms of performance variation due to differing hardware state conditions.

| Policy | Protected layers | Pass@1 | Tok/s | Decision |
|---|---:|---:|---:|---|
| 3-layer variant | `11,14,26` | 59.15% | 104.19 | Rejected: worse quality and speed than canonical |
| Canonical | `3,11,14,26` | 63.41% | 112.09 | Selected |
| +2 variant | `2,3,11,14,26` | 62.20% | 102.50 | Rejected: lower quality and speed than canonical |

I ran several iterations using different layer protection policies.  After multiple runs, the best-performing version was selected as the canonical policy.

## Decision Rule

Policy search stopped after bounded, targeted iteration.

The acceptance rule was:

- Compare only strict same-session runs.
- Require full HumanEval completion (`164/164`) for all compared GGUFs.
- Require fresh Notebook 09 perf for all compared GGUFs.
- Prefer policies that recover meaningful quality over Q4_K_M without eliminating most of Q4_K_M's throughput advantage.
- Stop when targeted variants fail to materially improve the current best.

The `11,14,26` and `2,3,11,14,26` variants both underperformed the
`3,11,14,26` candidate, so further layer toggling was treated as diminishing
return rather than useful search.

## Phase 2: MoE Experts and Future Work

The current project identifies sensitive layers and validates layer-level GGUF mixed precision.  The next logical step for this project is to run per-expert sensitivity analysis and experiment with various per-expert quantization policies.  The idea would be to identify specific experts with high sensitivity to quantization and shield them accordingly, while aggressively quantizing the less-sensitive experts.

This is a significant increase in scope over Phase 1, and is reserved for future notebook contributions.  An expert-level exetnsion will follow this formula:

1. Select 3-6 layers from the sensitive set.
2. Run a short per-expert screening pass within those layers.
3. Promote strong expert signals to full HumanEval validation.
4. Validate any expert-targeted GGUF policy under the same strict Notebook 09 protocol used here.

The intended output of this process is an additional GGUF artifact that continues to improve performance metrics via quantization while maintaining acceptable HumanEval performance.

## Reproducibility

This project uses `uv`.

```bash
uv sync
```

To quickly reproduce the headlines results, from the .json artifacts, run:

```bash
uv run reproduce
```

Add the `--check-artifacts` flag to verify if required .gguf files are present:

```bash
uv run reproduce --check-artifacts
```

For a full repduction, add the `--full-validation` flag.
WARNING: this involves running eval pipelines locally, which will consume **significant** computing resources and time.  **Do not use this unless you know what you are doing.**

```bash
uv run reproduce --full-validation
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
- The short perf benchmark is useful for local comparison, but production deployment would need a broader prompt suite and repeated runs.
- Notebook 07 sensitivity tests are PyTorch component ablations, not final GGUF deployment results.
- This phase does not optimize per-expert precision inside MoE layers.

## Metadata

- Model: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- Quantization backend: https://github.com/ggml-org/llama.cpp
- Evaluation: HumanEval
