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

In progress:
- `07_sensitivity_analysis.ipynb` (reset for a cleaner per-layer/per-expert sensitivity workflow)

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

### 4) Sensitivity analysis (Notebook 07)
Notebook 07 has been reset so the sensitivity workflow can be rebuilt clearly.

The notebook should distinguish two related but different tasks:
- Whole-model parity checks compare complete FP16, Q8_0, and Q4_K_M artifacts under the same evaluation backend.
- Sensitivity ablations quantize one selected layer group or expert while holding the rest of the model constant, then measure the quality delta.

The next milestone is one valid end-to-end ablation run. After that works, the notebook can grow into a manifest-driven sweep and produce layer/expert rankings.

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
7. Layer/expert sensitivity analysis (reset and being rebuilt)
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

- Notebook 03 baseline (`transformers`) and later Notebook 07 parity runs (`llama.cpp`) may use different harnesses; do not compare them directly without accounting for backend differences.
- Notebook 07 should not treat a manifest entry as a completed ablation unless there is a real candidate model artifact or runtime mechanism that applies the targeted quantization.

## References

- Model: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- Evaluation datasets: HumanEval, MBPP
- Quantization backend: llama.cpp / GGUF
