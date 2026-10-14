import json, re, ast, signal as signal_module, numpy as np, time, os
from openai import OpenAI
from datasets import load_from_disk
# ========== 配置 ==========
API_KEY   = os.environ["OPENAI_API_KEY"]
API_BASE  = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
PROXY     = None  # 国内如需代理: {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
MODEL     = "gpt-4o-mini"   # 或 gpt-4o-mini / claude-3-5-sonnet 等
TEMPERATURE = 0.8
MAX_TOKENS  = 512
MBPP_PATH   = os.environ["MBPP_PATH"]
SPLIT       = "test"
TIMEOUT     = 5


# ========== 客户端 ==========
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# 测试连通性
import requests
print(f"Testing chat at {API_BASE} with model={MODEL}...")
test_resp = requests.post(
    f"{API_BASE}/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": MODEL, "messages": [{"role": "user", "content": "say hi"}],
          "max_tokens": 10, "temperature": 0},
    timeout=30, proxies=PROXY,
)
print(f"Status: {test_resp.status_code}")
if test_resp.status_code == 200:
    data = test_resp.json()
    print(f"OK: {data['choices'][0]['message']['content']}")
elif test_resp.status_code == 404:
    print("Model not found, trying gpt-4o-mini...")
    MODEL = "gpt-4o-mini"
    test_resp2 = requests.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 10, "temperature": 0},
        timeout=30, proxies=PROXY,
    )
    print(f"Status: {test_resp2.status_code}")
    if test_resp2.status_code == 200:
        print(f"OK with {MODEL}")
    else:
        print(f"Failed: {test_resp2.text[:200]}")
        exit(1)
else:
    print(f"Failed: {test_resp.text[:200]}")
    exit(1)
print()

# ========== 工具函数 ==========

def extract_expected_func_name(test_list: list) -> str | None:
    for test in test_list:
        m = re.match(r'\s*assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', test)
        if m:
            return m.group(1)
    return None


def generate_with_signals(task_text: str, expected_func_name: str | None, test_list: list):
    func_hint = f"The function MUST be named exactly `{expected_func_name}`." if expected_func_name else ""
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

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": 5,
    }
    r = requests.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=120, proxies=PROXY,
    )
    data = r.json()

    choice = data["choices"][0]
    raw_output = choice["message"]["content"] or ""

    # 信号提取：API 返回的 logprobs
    log_probs_list = []
    logit_margins  = []
    logprob_content = choice.get("logprobs", {}).get("content") or []
    for token_info in logprob_content:
        lp = token_info["logprob"]
        log_probs_list.append(lp)
        top = token_info.get("top_logprobs") or []
        if len(top) >= 2:
            margin = top[0]["logprob"] - top[1]["logprob"]
        elif len(top) == 1:
            margin = top[0]["logprob"] - (-10.0)
        else:
            margin = 0.0
        logit_margins.append(max(margin, -10.0))

    ht1 = -float(np.mean(log_probs_list)) if log_probs_list else 0.0
    ht2 =  float(-np.min(logit_margins))  if logit_margins  else 0.0
    if logit_margins:
        ht3 = float(np.mean(np.exp(-np.clip(logit_margins, -5, None))))
    else:
        ht3 = 0.0

    usage = data.get("usage", {})
    cost_input  = usage.get("prompt_tokens", 0)
    cost_output = usage.get("completion_tokens", 0)

    return raw_output, ht1, ht2, ht3, cost_input, cost_output, len(test_list)


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
            extracted = re.sub(rf'\bdef\s+{re.escape(m.group(1))}\b',
                               f'def {expected_func_name}', extracted, count=1)
    return extracted


# ========== 测试执行（同 MBPP_Qwen3.py）==========

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
            signal_module.alarm(TIMEOUT)
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
    print(f"Model: {MODEL}  Temperature: {TEMPERATURE}")
    print("=" * 60)

    all_signals, all_eval = [], []
    model_slug = MODEL.replace(".", "-").replace("/", "-")

    for idx, problem in enumerate(problems):
        task_id    = problem["task_id"]
        task_text  = problem["text"]
        test_list  = problem["test_list"]
        test_setup = problem.get("test_setup_code", "") or ""
        expected_func = extract_expected_func_name(test_list)

        print(f"[{idx+1}/{len(problems)}] task_id={task_id}")

        try:
            raw, ht1, ht2, ht3, cost_in, cost_out, cost_test = generate_with_signals(
                task_text, expected_func, test_list)
        except Exception as e:
            print(f"  API error: {e}, sleeping 5s...")
            time.sleep(5)
            try:
                raw, ht1, ht2, ht3, cost_in, cost_out, cost_test = generate_with_signals(
                    task_text, expected_func, test_list)
            except Exception as e2:
                print(f"  Retry failed: {e2}, skipping")
                continue

        code = extract_function(raw, expected_func)
        try:
            ast.parse(code)
        except SyntaxError:
            sig = f"def {expected_func}(*args, **kwargs)" if expected_func else "def solution()"
            code = f"{sig}:\n    pass"

        passed = run_tests(code, test_list, test_setup)

        signals = {
            "ht1_neg_logprob":    float(ht1),
            "ht2_neg_min_margin": float(ht2),
            "ht3_logprob_var":    float(ht3),
            "cost_token_len":     float(cost_in),
            "cost_test_count":    float(cost_test / 10.0),
            "cost_output_len":    float(cost_out),
        }
        all_signals.append({"task_id": task_id, "signals": signals})
        all_eval.append({"task_id": task_id, "passed": passed})

        print(f"  passed={passed} | ht1={ht1:.3f} ht2={ht2:.3f} ht3={ht3:.3f}"
              f" | cost_in={cost_in} cost_out={cost_out}")

    # 保存
    for path, data, is_jsonl in [
        (f"mbpp_signal_results_{model_slug}.json", all_signals, False),
        (f"mbpp_eval_results_{model_slug}.jsonl",  all_eval,    True),
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
    print(f"\nPass@1: {passed}/{total} = {passed/total:.4f}")

if __name__ == "__main__":
    main()
