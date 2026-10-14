"""
mbpp_main_llama3.py
基于 mbpp_main.py 修改，适配 Llama-3.1-8B-Instruct
改动点：
  1. MODEL_PATH
  2. OUTPUT_FILES 加 _llama3 后缀，避免覆盖 Qwen 结果
  3. tokenizer padding side（Llama3 需要 left padding）
  4. eos_token_id 显式传入，防止 Llama3 不停生成
"""

import os
import torch
import json
import ast
import re
import signal as signal_module
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 配置 ==========
MODEL_PATH      = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
MBPP_PATH       = os.environ["MBPP_PATH"]
SPLIT           = "test"
TEMPERATURE     = 0.8
MAX_NEW_TOKENS  = 512
TIMEOUT_SECONDS = 5

# ========== 加载模型 ==========
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Llama3 没有默认 pad_token，用 eos_token 代替
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # decoder-only 模型推理时用 left padding

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded.")

# Llama3 的 eos token id（用于 generate 的终止条件）
EOS_TOKEN_ID = tokenizer.eos_token_id


# ========== 工具函数 ==========

def extract_expected_func_name(test_list: list) -> str | None:
    for test in test_list:
        m = re.match(r'\s*assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', test)
        if m:
            return m.group(1)
    return None


# ========== 代码生成 + 信号提取 ==========

def generate_with_signals(task_text: str, expected_func_name: str | None, test_list: list):
    func_hint = (
        f"The function MUST be named exactly `{expected_func_name}`."
        if expected_func_name else ""
    )
    system_msg = (
        "You are a Python coding assistant. "
        "Write a complete Python function that solves the task described. "
        "Output ONLY the function definition with proper indentation. "
        "No explanation, no markdown, no extra text. "
        + func_hint
    )
    user_msg = (
        f"# Task: {task_text}\n"
        + (f"# The function must be named: {expected_func_name}\n" if expected_func_name else "")
        + "# Now write the function:"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

    # Llama3 的 chat template 和 Qwen 格式不同，但 apply_chat_template 自动处理
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=EOS_TOKEN_ID,      # Llama3 需要显式指定，否则可能不停生成
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs.sequences[0][input_len:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    log_probs_list = []
    logit_margins  = []
    for score, token_id in zip(outputs.scores, generated_ids):
        lp = torch.log_softmax(score[0], dim=-1)
        log_probs_list.append(lp[token_id].item())
        top2   = torch.topk(lp, 2)
        margin = (top2.values[0] - top2.values[1]).item()
        logit_margins.append(max(margin, -10.0))

    ht1 = -float(np.mean(log_probs_list)) if log_probs_list else 0.0
    ht2 =  float(-np.min(logit_margins))  if logit_margins  else 0.0
    if logit_margins:
        ht3 = float(np.mean(np.exp(-np.clip(logit_margins, -5, None))))
    else:
        ht3 = 0.0

    cost_token_len  = float(input_len / MAX_NEW_TOKENS)
    cost_test_count = float(len(test_list) / 10.0)
    cost_output_len = float(len(generated_ids) / MAX_NEW_TOKENS)
    return raw_output, ht1, ht2, ht3, cost_token_len, cost_test_count, cost_output_len


# ========== 代码提取 ==========

def extract_function(raw_output: str, expected_func_name: str | None = None) -> str:
    text = re.sub(r"```(?:python)?\s*", "", raw_output)
    text = re.sub(r"```\s*", "", text)
    lines = text.split("\n")

    def_indices = [i for i, l in enumerate(lines) if l.strip().startswith("def ")]
    if not def_indices:
        return text.strip()

    target_idx = def_indices[0]
    if expected_func_name:
        for i in def_indices:
            if re.match(rf'\s*def\s+{re.escape(expected_func_name)}\s*\(', lines[i]):
                target_idx = i
                break

    result_lines = [lines[target_idx]]
    for line in lines[target_idx + 1:]:
        if not line.strip():
            result_lines.append(line)
            continue
        if not line[0].isspace():
            break
        result_lines.append(line)
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    extracted = "\n".join(result_lines)

    if expected_func_name:
        m = re.match(r'\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', extracted)
        if m and m.group(1) != expected_func_name:
            extracted = re.sub(
                rf'\bdef\s+{re.escape(m.group(1))}\b',
                f'def {expected_func_name}',
                extracted,
                count=1,
            )

    return extracted


# ========== 测试执行 ==========

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def run_tests(code: str, test_list: list, test_setup_code: str = "") -> bool:
    namespace = {}
    if test_setup_code and test_setup_code.strip():
        try:
            exec(compile(test_setup_code, "<setup>", "exec"), namespace)
        except Exception:
            return False
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
    except Exception:
        return False
    for test_case in test_list:
        if not test_case.strip():
            continue
        try:
            signal_module.signal(signal_module.SIGALRM, _timeout_handler)
            signal_module.alarm(TIMEOUT_SECONDS)
            try:
                exec(compile(test_case, "<test>", "exec"), namespace)
            finally:
                signal_module.alarm(0)
        except (AssertionError, TimeoutError, Exception):
            return False
    return True


# ========== 主流程 ==========

def main():
    dataset  = load_from_disk(MBPP_PATH)
    problems = dataset[SPLIT]

    print(f"Loaded MBPP '{SPLIT}' split: {len(problems)} problems")
    print("=" * 60)

    all_samples, all_signals, all_eval = [], [], []

    for idx, problem in enumerate(problems):
        task_id    = problem["task_id"]
        task_text  = problem["text"]
        test_list  = problem["test_list"]
        test_setup = problem.get("test_setup_code", "") or ""

        expected_func = extract_expected_func_name(test_list)
        print(f"[{idx+1}/{len(problems)}] task_id={task_id}  expected_func={expected_func}")

        raw, ht1, ht2, ht3, cost_tok, cost_test, cost_out = generate_with_signals(
            task_text, expected_func, test_list
        )
        code = extract_function(raw, expected_func)

        try:
            ast.parse(code)
        except SyntaxError:
            sig  = f"def {expected_func}(*args, **kwargs)" if expected_func else "def solution()"
            code = f"{sig}:\n    pass"

        passed = run_tests(code, test_list, test_setup)

        signals = {
            "ht1_neg_logprob":    float(ht1),
            "ht2_neg_min_margin": float(ht2),
            "ht3_logprob_var":    float(ht3),
            "cost_token_len":     float(cost_tok),
            "cost_test_count":    float(cost_test),
            "cost_output_len":    float(cost_out),
        }

        all_samples.append({"task_id": task_id, "completion": code})
        all_signals.append({"task_id": task_id, "signals": signals})
        all_eval.append({"task_id": task_id, "passed": passed})

        print(f"  passed={passed} | ht1={ht1:.3f} | ht2={ht2:.3f} | ht3={ht3:.3f}")

    # 输出文件加 _llama3 后缀，避免覆盖 Qwen 结果
    for path, data, is_jsonl in [
        ("mbpp_samples_llama3.jsonl",       all_samples, True),
        ("mbpp_signal_results_llama3.json", all_signals, False),
        ("mbpp_eval_results_llama3.jsonl",  all_eval,    True),
    ]:
        with open(path, "w") as f:
            if is_jsonl:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            else:
                json.dump(data, f, indent=2)
        print(f"Saved {path}")

    total  = len(all_eval)
    passed = sum(1 for e in all_eval if e["passed"])
    print(f"\nPass@1 on '{SPLIT}': {passed}/{total} = {passed/total:.4f}")


if __name__ == "__main__":
    main()
