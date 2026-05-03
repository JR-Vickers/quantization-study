# Quantization Study: DeepSeek-Coder-V2-Lite-Instruct

A learning-first, notebook-driven study of quantization effects on a 16B Mixture-of-Experts coding model.

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
- `07_sensitivity_analysis.ipynb` (in progress for ablation sweep)

Planned:
- `08_mixed_precision.ipynb`

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
| FP16 | 74.74 | 0.0564 | 1.7785 | 29.45 |
| Q8_0 | 114.01 | 0.0505 | 1.1794 | 38.82 |
| Q4_K_M | 137.56 | 0.0415 | 0.9769 | 9.44 |

Derived speedups vs FP16:
- Q8_0: **1.53x**
- Q4_K_M: **1.84x**

### 4) Parity quality runs (Notebook 07, llama.cpp backend)
From `results/07_parity_baseline_summary.json`:
- FP16 pass@1: **72.56%**
- Q8_0 pass@1: **71.95%**
- Q4_K_M pass@1: **75.00%**

Note: these are same-harness parity runs and are the correct baseline for Notebook 07 comparisons.

### 5) Manual quantization insight (Notebook 04)
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
7. Layer/expert sensitivity ablations (ongoing)
8. Mixed-precision policy design (planned)

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

3) Run notebooks in order (`01` -> `08`).

Results are saved into `results/` as JSON after each notebook.

## Data and artifacts

- Source model weights are loaded from the Hugging Face cache (not stored in this repo):
  - `~/.cache/huggingface/hub/`
- Quantized GGUF exports live under `exports/` (large files, typically multi-GB).

## Caveats

- Notebook 03 baseline (`transformers`) and Notebook 07 parity runs (`llama.cpp`) use different harnesses; do not compare them directly without accounting for backend differences.
- Sensitivity sweep in Notebook 07 is still running; final per-layer/per-expert ranking is not complete yet.

## References

- Model: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- Evaluation datasets: HumanEval, MBPP
- Quantization backend: llama.cpp / GGUF
