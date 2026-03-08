# Quantization Study: DeepSeek-Coder-V2-Lite-Instruct

## Objective

Perform a systematic quantization study on DeepSeek-Coder-V2-Lite-Instruct (16B parameters, MoE architecture) to understand how reduced numerical precision affects code generation quality and inference performance. The goal is to produce a rigorous benchmark comparing FP16, INT8, and INT4 precision levels, including per-layer and per-expert sensitivity analysis.

## Why This Model

- **DeepSeek-Coder-V2-Lite-Instruct** (Hugging Face ID: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`)
- 16B total parameters, ~2.4B active per forward pass (Mixture of Experts)
- Instruction-tuned, so it responds well to HumanEval-style prompts out of the box
- 204k downloads, well-benchmarked by the community — important for validating our baseline
- MoE architecture makes quantization analysis more interesting: different experts may have different sensitivity to precision loss

## Hardware

- Apple M4 Max, 64GB RAM
- Model fits comfortably at FP16; INT8 and INT4 give progressively more headroom
- Apple Silicon Neural Engine and AMX blocks handle low-precision matrix multiplies natively

## Project Structure

```
quantization-study/
├── notebooks/
│   ├── 01_model_loading_and_exploration.ipynb
│   ├── 02_weight_inspection.ipynb
│   ├── 03_baseline_evaluation.ipynb
│   ├── 04_manual_quantization.ipynb
│   ├── 05_ptq_with_tooling.ipynb
│   ├── 06_benchmarking.ipynb
│   ├── 07_sensitivity_analysis.ipynb
│   └── 08_mixed_precision.ipynb
├── utils/                    # shared helper functions extracted from notebooks
├── results/                  # all benchmark outputs, JSON format
├── exports/                  # quantized model artifacts (INT8, INT4, mixed)
├── .gitignore
├── README.md
├── PLAN.md
├── CLAUDE.md
└── requirements.txt
```

### Learning-Oriented Design

This project is optimized for deep understanding, not speed of completion. The notebook progression is sequential and each notebook introduces new concepts that build on the previous one. Do not skip ahead.

**Notebook Progression:**

1. **01 — Model Loading & Exploration:** Load the model, inspect its architecture, understand the MoE routing structure. How many experts? How does the router decide which experts activate? Print the model's module tree, count parameters per component.

2. **02 — Weight Inspection:** Before touching anything, look at what the weights actually are. Visualize weight distributions per layer. What's the range? Are they normally distributed? Are some layers tighter than others? Build intuition for why some layers will tolerate quantization better.

3. **03 — Baseline Evaluation:** Run HumanEval at FP16 and establish ground truth. Understand the evaluation pipeline — how does generated code get executed and tested? Measure inference speed and memory. These numbers are the reference point for everything that follows.

4. **04 — Manual Quantization:** Before using any library, implement naive quantization by hand on a single layer. Convert FP16 weights to INT8 using basic scale-and-round math. Visualize the before/after weight distribution. Run inference and see what happens to output quality. This notebook is about understanding the math, not producing good results.

5. **05 — PTQ With Tooling:** Now use proper tools (llama.cpp / GGUF, auto-gptq, etc.) to quantize the full model to INT8 and INT4. Compare the results to your manual attempt. What are these tools doing differently? What is calibration actually computing?

6. **06 — Benchmarking:** Systematic measurement across all precision levels. Quality (HumanEval, MBPP), speed (tokens/sec), memory. Produce clean comparison tables and charts.

7. **07 — Sensitivity Analysis:** Per-layer and per-expert analysis. Which components of the model are most affected by quantization? Heatmaps and visualizations. This is the core intellectual contribution.

8. **08 — Mixed Precision:** Based on sensitivity findings, construct an optimized mixed-precision configuration. Benchmark it. Write up conclusions.

**Important:** Original model weights are NOT stored in this project. They live in the default Hugging Face cache (`~/.cache/huggingface/hub/`). Load them by Hugging Face ID:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
```

The `exports/` and `results/` directories should be gitignored if they contain large quantized model files. Only code, configs, and summary results should be committed.

## Key Libraries

- `transformers` — model loading and inference
- `huggingface_hub` — model download and cache management
- `auto-gptq` or `llama.cpp` (via `llama-cpp-python`) — quantization backends
- `bitsandbytes` — INT8/INT4 quantization (may have limited Apple Silicon support; verify)
- `datasets` — loading HumanEval and MBPP benchmarks
- `time`, `psutil`, `torch.profiler` — performance measurement

Check Apple Silicon compatibility for each library before committing to a stack. `llama.cpp` has strong Metal support and may be the path of least resistance for M4 Max.

## Methodology

### Phase 1: Baseline

1. Load the model at full precision (FP16)
2. Run HumanEval (164 problems) and record pass@1 scores
3. Measure inference speed (tokens/second) and memory footprint
4. Compare baseline results against published benchmarks to validate the evaluation pipeline

### Phase 2: Quantization

1. Quantize to INT8 using post-training quantization (PTQ)
2. Quantize to INT4 using PTQ
3. For each precision level:
   - Run calibration on a small representative code dataset
   - Export the quantized model to `exports/`
   - Record the quantization parameters (scale factors, zero points)

### Phase 3: Benchmarking

For each precision level (FP16, INT8, INT4), measure:
- **Quality:** HumanEval pass@1, MBPP pass@1
- **Speed:** tokens/second (prompt processing and generation separately)
- **Memory:** peak RAM usage during inference
- **Latency:** time-to-first-token, total generation time for fixed-length outputs

Store all results as structured JSON in `results/`.

### Phase 4: Sensitivity Analysis

This is where the project goes from "I ran a script" to "I understand the architecture."

1. **Per-layer analysis:** Quantize individual layers (or groups of layers) to INT4 while keeping the rest at FP16. Measure HumanEval degradation per layer to identify which layers are most sensitive.
2. **Per-expert analysis (MoE-specific):** Since this is a Mixture of Experts model, analyze whether certain experts degrade more under quantization than others. This could reveal which experts encode more critical information.
3. **Mixed-precision experiment:** Based on sensitivity results, construct a mixed-precision configuration — INT4 for robust layers/experts, INT8 for sensitive ones — and benchmark it. The goal is to find a configuration that approaches INT4 memory/speed while retaining closer to INT8 quality.

### Phase 5: Writeup

Produce a clear README.md with:
- Methodology description
- Results tables (quality, speed, memory across precision levels)
- Sensitivity analysis visualizations (heatmaps of per-layer/per-expert degradation)
- Key findings and interpretation
- Reproduction instructions

## Evaluation Details

### HumanEval
- 164 Python programming problems
- Each problem has a function signature and docstring; the model generates the function body
- Evaluated by running the generated code against a test suite
- Metric: pass@1 (percentage of problems solved on first attempt)

### MBPP (Mostly Basic Python Problems)
- 974 crowd-sourced Python problems
- Simpler than HumanEval on average; provides a second data point
- Same pass@k evaluation methodology

### Performance Metrics
- Tokens/second: measure over 100+ generations and report mean and standard deviation
- Memory: peak RSS during inference, measured via `psutil` or system tools
- Report all measurements with variance — single-run numbers are not credible

## Notes

- This model uses a Mixture of Experts architecture. Be aware that standard quantization tools may not handle expert routing layers correctly. Verify that the router weights are preserved at full precision or handled appropriately.
- Start with `llama.cpp` / GGUF format for quantization if `bitsandbytes` or `auto-gptq` have compatibility issues on Apple Silicon.
- If the V2-Lite-Instruct MoE architecture creates too many complications for a first pass, consider starting with **DeepSeek-Coder V1 6.7B** (dense architecture) to build the pipeline, then porting it to the MoE model.
- The quantized model exports in `exports/` may be 5-15GB each. Plan disk space accordingly.