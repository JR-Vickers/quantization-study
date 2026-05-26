# Quantization Study: DeepSeek-Coder-V2-Lite-Instruct

This project is a study of quantization effects on DeepSeek-Coder-V2-Lite-Instruct, a 16B-parameter mixture-of-experts coding model.  This model was chosen for several reasons:

- Low parameter count enables fast iteration in local environment
- Coding outputs are easy to verify, permitting accurate performance tests

## What this project is

This repo investigates how reduced precision (FP16 -> Q8_0 -> Q4_K_M) impacts:
- code quality (HumanEval pass@1)
- inference speed (tokens/sec, latency)
- memory footprint
- sensitivity by layer/expert (MoE-specific)

Model: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`  
Hardware target: Apple M4 Max (64GB RAM)
Quantization backend: `llama.cpp` / GGUF

## Current status

Completed notebooks:
- `01_model_loading_and_exploration.ipynb`
- `02_weight_inspection.ipynb`
- `03_baseline_evaluation.ipynb`
- `04_manual_quantization.ipynb`
- `05_ptq_with_tooling.ipynb`
- `06_benchmarking.ipynb`
- `07_sensitivity_analysis.ipynb`
- `08_mixed_precision.ipynb`
- `09_mixed_precision_validation.ipynb`

Current work:
- Stabilization pass. Notebook 07 has been tightened around a symmetric full-follow-up design; Notebook 08 is being aligned to consume that completed evidence before Notebook 09 is revisited.

Notebook 09 was added after the original eight-notebook plan because mixed-precision policy design and mixed-artifact validation became separate concerns. Notebook 08 should explain and emit policy candidates; Notebook 09 should validate concrete candidates and record each iteration clearly.

## Repository structure

- `notebooks/` — step-by-step learning pipeline
- `results/` — JSON outputs from each notebook
- `utils/` — shared helper functions
- `PLAN.md` — project roadmap and methodology
- `CLAUDE.md` — project-specific working conventions

## Key results so far

### 1) Baseline quality (Notebook 03, transformers run)
From `results/03_baseline_evaluation.json`:
- HumanEval pass@1: **60.37%** (99/164)

### 2) Quantized artifact sizes (Notebook 05)
From `results/05_ptq_artifacts_summary.json`:
- FP16 GGUF: **29.27 GiB**
- Q8_0 GGUF: **15.56 GiB** (~53.15% of FP16 size)
- Q4_K_M GGUF: **9.66 GiB** (~32.99% of FP16 size)

### 3) Performance snapshot (Notebook 06, llama.cpp parity harness)
From `results/06_benchmarking_perf_snapshot.json`:

| Precision | Tokens/sec (mean) | TTFT (s) | Total latency (s) | Peak RSS (GiB) |
|---|---:|---:|---:|---:|
| FP16 | 75.31 | 0.0547 | 1.7632 | 29.51 |
| Q8_0 | 108.70 | 0.0516 | 1.2357 | 32.28 |
| Q4_K_M | 126.85 | 0.0426 | 1.0569 | 9.46 |

Derived speedups vs FP16:
- Q8_0: **1.44x**
- Q4_K_M: **1.68x**

### 4) Sensitivity analysis (Notebook 07)
From `results/07_layer_int4_screening_summary.json` and `results/07_selected_layer_int4_full_followup_runs.json`:

- Notebook 07 uses targeted PyTorch INT4 ablations as component-level sensitivity estimates, not deployment measurements.
- The full PyTorch baseline in Notebook 07 is **70.12%** HumanEval pass@1 (115/164).
- The four strongest 30-problem screen hits were layers **3**, **26**, **11**, and **14**.
- Full HumanEval follow-up is complete for the top four screen hits and bottom four screen controls.
- The protected group **3, 11, 14, 26** had a mean full INT4 drop of **13.11** pass@1 points.
- The low-sensitivity control group **0, 2, 15, 17** had no observed full-run harm; its mean full delta was **-3.51** points. Negative deltas are treated as no-harm evidence, not proof that INT4 improves quality.

Notebook 08 should now protect layers `3, 11, 14, 26` and treat layers `0, 2, 15, 17` as the best-supported lower-precision candidates.

### 5) Mixed-precision policy and validation (Notebooks 08-09)
Notebook 08 is being restabilized around the completed Notebook 07 evidence: Q4_K_M default with Q8_0 overrides for protected layers `3, 11, 14, 26`. Notebook 09 then validates concrete mixed GGUF candidates.

Current strongest saved validation from `results/09_mixed_precision_validation.json`:

| Model | Size (GiB) | HumanEval pass@1 | Passed | Tok/s |
|---|---:|---:|---:|---:|
| FP16 GGUF | 29.27 | 73.17% | 120/164 | 65.90 |
| Q8_0 GGUF | 15.56 | 71.95% | 118/164 | 109.23 |
| Q4_K_M GGUF | 9.66 | 56.71% | 93/164 | 127.77 |
| Mixed v2 GGUF | 11.33 | 66.46% | 109/164 | 104.53 |

The mixed v2 candidate protects layers `1, 3, 6, 8, 10, 11, 14, 26` at Q8_0 while using Q4_K_M as the default. It recovers substantial quality versus Q4_K_M, but it is slower than Q4_K_M and slightly slower than Q8_0 in the current short benchmark. Treat it as the best saved validation run so far, not as the final canonical policy. The next canonical candidate should use the restabilized Notebook 08 policy.

### 6) Manual quantization insight (Notebook 04)
From `results/04_manual_quantization.json`:
- For sampled projection matrices, per-channel INT8 quantization reduced average MAE dramatically vs per-tensor quantization:
  - `down_proj`: ~**90.28%** MAE improvement
  - `up_proj`: ~**86.90%** MAE improvement
  - `gate_proj`: ~**81.31%** MAE improvement

This is the core intuition behind why smarter quantization schemes outperform naive global scaling.

## Notebook progression

1. Model loading and architecture exploration
2. Weight statistics and distribution visualization
3. FP16 quality baseline on HumanEval
4. Manual INT8 math (single-layer + layerwise error analysis)
5. PTQ exports to GGUF (Q8_0, Q4_K_M)
6. End-to-end perf benchmarking across precisions
7. Layer/expert sensitivity analysis
8. Mixed-precision policy design
9. Mixed-precision artifact validation and policy iteration

## Reproducibility

This project uses `uv`.

1) Create environment and install dependencies:

```bash
uv sync
```

2) Start Jupyter:

```bash
uv run jupyter lab
```

3) Run notebooks in order (`01` -> `09`).

Results are saved into `results/` as JSON after each notebook.

## Data and artifacts

- Source model weights are loaded from the Hugging Face cache (not stored in this repo):
  - `~/.cache/huggingface/hub/`
- Quantized GGUF exports live under `exports/` (large files, typically multi-GB).

## Caveats

- Notebook 03 baseline (`transformers`) and later Notebook 07 parity runs (`llama.cpp`) may use different harnesses; do not compare them directly without accounting for backend differences.
- Notebook 07 sensitivity ablations are component-level PyTorch estimates, not final deployment measurements.
- Notebook 08 policy artifacts are recommendations until Notebook 09 validates a concrete mixed GGUF.
- Notebook 09 is currently in iteration mode. Protected layers, policy IDs, and artifact paths should be made more reusable before further experiments.

## References

- Model: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- Evaluation datasets: HumanEval, MBPP
- Quantization backend: llama.cpp / GGUF
