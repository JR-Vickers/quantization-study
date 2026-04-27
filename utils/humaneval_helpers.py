import ast
import re
import subprocess
import sys
import textwrap


def format_humaneval_prompt(problem, tokenizer):
    instruction = (
        "Complete the following Python function. Return raw Python code only with correct "
        "newlines and indentation. Do not"
        " use markdown fences or explanations. "
        "You may return either the full function or just the function body.\n\n"
        f"{problem['prompt']}"
    )
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_completion(problem, model, tokenizer, max_new_tokens=512):
    import torch

    formatted = format_humaneval_prompt(problem, tokenizer)
    inputs = tokenizer(formatted, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    generated = generated.replace("\r\n", "\n").replace("\r", "\n")
    generated = generated.replace("Ġ", " ").replace("Ċ", "\n").replace("č", "\n")
    # Keep leading indentation (important for body-only HumanEval completions).
    return generated.rstrip()


def has_newlines(text):
    return "\n" in text.strip()


def reconstruct_newlines(code):
    code = re.sub(r"(```python\s*)", "", code)
    code = re.sub(r"(```\s*$)", "", code)
    code = code.strip()

    if has_newlines(code):
        return code

    code = re.sub(
        r"(?<=[^\s])(\s{4,})(?=def |class |return |if |elif |else:|for |while |try:|except |finally:|with |import |from |raise |assert |pass|break|continue)",
        r"\n\1",
        code,
    )
    code = re.sub(r"(?<=:)(    )", r"\n    ", code)
    return code


def strip_code_fences(code):
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", code, re.DOTALL | re.IGNORECASE)
    if blocks:
        # Preserve leading spaces inside fenced code; only trim surrounding newlines.
        code = blocks[0].strip("\n")

    code = re.sub(r"^\s*python(?=from |import |def |class |\n)", "", code, flags=re.IGNORECASE)
    return code.strip("\n")


def _stitch_with_prompt(problem_prompt: str, candidate: str):
    """Try multiple prompt+candidate stitches while preserving indentation."""
    variants = []

    # 1) Raw concatenation, preserving left whitespace from candidate.
    variants.append(problem_prompt + candidate)

    # 2) Ensure newline boundary if prompt doesn't end with one.
    if not problem_prompt.endswith("\n"):
        variants.append(problem_prompt + "\n" + candidate.lstrip("\n"))

    # 3) Force body-style indentation for snippets that are likely function bodies.
    body_like = not re.match(r"\s*(def |class |from |import )", candidate)
    if body_like:
        body = candidate.lstrip("\n")
        indented = textwrap.indent(body, "    ")
        if not problem_prompt.endswith("\n"):
            variants.append(problem_prompt + "\n" + indented)
        else:
            variants.append(problem_prompt + indented)

    seen = set()
    deduped = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def reflow_compact_code(code):
    if "\n" in code:
        return code

    code = re.sub(r"(?<=[A-Za-z0-9_\]\)])(?=(?:def|class)\s)", "\n", code)
    code = re.sub(r"(?<!^)(?=\b(?:def|class)\b)", "\n", code)
    code = re.sub(r":(\s{4,})(?=\S)", r":\n\1", code)
    code = re.sub(r"(?<=[^\s])(\s{4,})(?=\S)", r"\n\1", code)
    return code


def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_function_body(candidate, entry_point):
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None

    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            fn = node
            break

    if fn is None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                fn = node
                break

    if fn is None or not fn.body:
        return None

    lines = candidate.splitlines()
    start = fn.body[0].lineno - 1
    end = fn.body[-1].end_lineno
    return "\n".join(lines[start:end]).rstrip()


def salvage_body_from_compact(code):
    compact = code.strip()
    if "def " in compact and ":" in compact:
        compact = compact.split(":", 1)[1].strip()

    if not compact:
        return "    pass"

    parts = [p.strip() for p in re.split(r"\s{4,}", compact) if p.strip()]
    if not parts:
        return "    pass"

    return "\n".join(f"    {p}" for p in parts)


def clean_and_extract(generated, problem):
    base = strip_code_fences(generated)

    candidates = [base]
    if base and not has_newlines(base):
        candidates.append(reconstruct_newlines(base))
        candidates.append(reflow_compact_code(base))

    seen = set()
    for candidate in candidates:
        # Keep leading spaces (indentation). Only trim trailing whitespace.
        candidate = candidate.rstrip()
        if not candidate.strip() or candidate in seen:
            continue
        seen.add(candidate)

        if re.match(r"\s*(from |import |class )", candidate) and is_valid_python(candidate):
            return candidate

        if re.match(r"\s*def ", candidate) and is_valid_python(candidate):
            body = extract_function_body(candidate, problem["entry_point"])
            if body:
                stitched = problem["prompt"] + body
                if is_valid_python(stitched):
                    return stitched
            return candidate

        for with_prompt in _stitch_with_prompt(problem["prompt"], candidate):
            if is_valid_python(with_prompt):
                return with_prompt

    if re.match(r"\s*(from |import )", base) and is_valid_python(base):
        return base

    if re.match(r"\s*def ", base) and is_valid_python(base):
        body = extract_function_body(base, problem["entry_point"])
        if body:
            stitched = problem["prompt"] + body
            if is_valid_python(stitched):
                return stitched
        return base

    for with_prompt in _stitch_with_prompt(problem["prompt"], base):
        if is_valid_python(with_prompt):
            return with_prompt

    salvaged = problem["prompt"] + salvage_body_from_compact(base)
    if is_valid_python(salvaged):
        return salvaged

    # Last-resort fallback.
    return problem["prompt"] + base


def execute_with_timeout(code, timeout=10):
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return "passed"
        stderr = result.stderr.strip()
        if "AssertionError" in stderr:
            return "failed"
        return f"error: {stderr.split(chr(10))[-1]}"
    except subprocess.TimeoutExpired:
        return "error: timeout"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def evaluate_problem(problem, model, tokenizer, timeout=10, max_new_tokens=512):
    generated = generate_completion(
        problem,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    code = clean_and_extract(generated, problem)
    full_code = code + "\n\n" + problem["test"] + f"\ncheck({problem['entry_point']})"
    result = execute_with_timeout(full_code, timeout=timeout)
    return {"task_id": problem["task_id"], "result": result, "generated": generated}