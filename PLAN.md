# Quantization Study: DeepSeek-Coder-V2-Lite-Instruct

## Objective

Perform a systematic quantization study on DeepSeek-Coder-V2-Lite-Instruct (16B parameters, MoE architecture) to understand how reduced numerical precision affects code generation quality and inference performance. The completed phase produces a rigorous benchmark comparing FP16, Q8_0, Q4_K_M, and layer-level mixed precision. Per-expert sensitivity remains an explicitly scoped follow-up, not a completed claim in this phase.

The main deliverable is the analysis: measured tradeoffs, clear methodology, sensitivity findings, and a defensible mixed-precision policy. GGUF artifacts are important supporting artifacts for deployment-style benchmarking, and the final policy is validated as a concrete mixed-precision GGUF under the same llama.cpp-based harness used for FP16, Q8_0, and Q4_K_M references.

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
│   ├── 08_mixed_precision.ipynb
│   └── 09_mixed_precision_validation.ipynb
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

7. **07 — Sensitivity Analysis:** Layer-level sensitivity analysis. Which layers are most affected by quantization? Use controlled ablations to estimate component sensitivity, then produce rankings and heatmaps. Expert-level analysis is documented as a MoE-specific follow-up rather than mixed into the final layer-level policy claim.

8. **08 — Mixed Precision Policy:** Based on sensitivity findings, design a mixed-precision policy. The minimum output is a clear recommendation for which components should remain higher precision and which can tolerate lower precision. If the tooling supports targeted mixed-precision artifacts, prepare a reproducible tensor-type manifest and optional build path; otherwise, document the policy as an evidence-backed design and identify artifact generation as future work.

9. **09 — Mixed Precision Validation:** Validate concrete mixed-precision GGUF candidates produced from Notebook 08 policies. Compare each candidate against FP16, Q8_0, and Q4_K_M GGUF references under one llama.cpp-based harness. Treat each policy iteration as its own run with explicit metadata so the best-performing policy can later be promoted to canonical.

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

Notebook 06 focuses on deployment-style behavior of complete GGUF artifacts. These results establish the practical payoff of quantization: smaller files, lower memory pressure, and faster inference. They do not, by themselves, explain which internal components are responsible for any quality change.

### Phase 4: Sensitivity Analysis

This is where the project goes from "I ran a script" to "I understand the architecture."

1. **Per-layer analysis:** Quantize individual layers (or groups of layers) while keeping the rest of the model at the reference precision. Measure HumanEval degradation per layer to identify which layers are most sensitive.
2. **MoE scope boundary:** Since this is a Mixture-of-Experts model, per-expert sensitivity is a natural extension. This phase does not claim expert-level optimization; it stops at layer-level policy because that evidence was complete and validated under deployment-style GGUF runs.
3. **Methodology distinction:** GGUF artifacts are the right tool for whole-model deployment benchmarking. PyTorch/runtime ablations may be the right tool for targeted sensitivity analysis because they expose individual tensors and modules directly. If Notebook 07 uses simulated quantization in PyTorch, label the results as component-level sensitivity estimates, not final deployment measurements.
4. **Mixed-precision policy:** Based on layer sensitivity results, recommend a policy such as Q4_K_M by default with Q8_0 protection for sensitive layers. The goal is to explain how one would approach INT4-like memory/speed while protecting components that appear quality-critical.

### Phase 5: Mixed Precision Synthesis

Notebook 08 synthesizes the deployment and sensitivity evidence:

- Use Notebook 06 to quantify the payoff of lower precision in real GGUF artifacts.
- Use Notebook 07 to identify which components appear fragile under targeted quantization.
- Produce a mixed-precision policy that is explicit about assumptions, expected benefits, and validation limits.
- If the available tooling supports tensor-level or layer-level mixed-precision GGUF generation, prepare a candidate artifact plan and optionally build a candidate.
- If the tooling does not support that cleanly, document the policy and the missing implementation step rather than forcing an artifact that the methodology cannot justify.

### Phase 6: Mixed Precision Validation

Notebook 09 validates concrete policy candidates:

- Load one explicit policy candidate at a time.
- Build or locate the corresponding mixed GGUF artifact.
- Verify that protected tensors use the requested precision.
- Run fresh HumanEval quality results for FP16, Q8_0, Q4_K_M, and mixed GGUF artifacts when making final policy decisions.
- Run fresh same-harness performance benchmarks for every compared artifact when making final policy decisions.
- Save each policy iteration as a distinct validation artifact, then compare policy candidates.

This phase exists because policy design and artifact validation are different claims. Notebook 08 can justify a candidate policy; Notebook 09 determines whether a particular artifact actually preserves enough quality while retaining enough of the Q4_K_M deployment payoff.

The final canonical policy is `aggressive_q4_k_m_default_q8_protected_layers_3_11_14_26`: Q4_K_M by default with Q8_0 overrides for layers 3, 11, 14, and 26. It was selected after strict apples-to-apples Notebook 09 validation and bounded policy iteration. The 3-layer `11,14,26` variant and the 5-layer `2,3,11,14,26` variant both underperformed the canonical policy.

### Phase 7: Writeup

Produce a clear README.md with:
- Methodology description
- Results tables (quality, speed, memory across precision levels)
- Sensitivity analysis visualizations (heatmaps of per-layer degradation)
- Key findings and interpretation
- Mixed-precision policy recommendation, validation results, and any tooling limitations
- Explicit MoE expert-analysis scope boundary and follow-up protocol
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
- Tokens/second: report mean and standard deviation for the active harness. Notebook 09 uses a short same-session prompt benchmark for policy comparison; production-grade deployment claims would require a broader repeated prompt suite.
- Memory: peak RSS during inference, measured via `psutil` or system tools
- Report measurement source and freshness. Mixed-source performance comparisons are not used for final policy decisions.

## Notes

- This model uses a Mixture of Experts architecture. Be aware that standard quantization tools may not handle expert routing layers correctly. Verify that the router weights are preserved at full precision or handled appropriately.
- Start with `llama.cpp` / GGUF format for quantization if `bitsandbytes` or `auto-gptq` have compatibility issues on Apple Silicon.
- If the V2-Lite-Instruct MoE architecture creates too many complications for a first pass, consider starting with **DeepSeek-Coder V1 6.7B** (dense architecture) to build the pipeline, then porting it to the MoE model.
- The quantized model exports in `exports/` may be 5-15GB each. Plan disk space accordingly.
