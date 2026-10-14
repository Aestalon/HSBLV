"""
MATH_Qwen3_simple.py — 单卡精简版
用 CUDA_VISIBLE_DEVICES 控制卡号，数据分片手动传参

用法:
    CUDA_VISIBLE_DEVICES=0 python MATH_Qwen3_simple.py --total 1 --rank 0
    CUDA_VISIBLE_DEVICES=0 python MATH_Qwen3_simple.py --total 2 --rank 0 &
    CUDA_VISIBLE_DEVICES=1 python MATH_Qwen3_simple.py --total 2 --rank 1 &
"""

import torch
import json
import re
import os
import glob
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 配置 ==========
MODEL_PATH      = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
DATA_PATH       = os.environ["MATH_PATH"]
TEMPERATURE     = 0.8
MAX_NEW_TOKENS  = 1536

def extract_boxed_answer(text: str) -> str | None:
    m = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if m:
        return m.group(1).strip()
    matches = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    return None


def extract_gold_answer(solution_text: str) -> str:
    ans = extract_boxed_answer(solution_text)
    if ans:
        return ans.strip()
    lines = solution_text.strip().split('\n')
    return lines[-1].strip()


def normalize_math_answer(ans: str) -> str:
    ans = ans.strip().replace(" ", "")
    ans = re.sub(r'\.$', '', ans)
    ans = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', ans)
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=1)   # 总共几张卡
    parser.add_argument("--rank",  type=int, default=0)   # 当前是第几张（0-indexed）
    args = parser.parse_args()

    # 数据分片
    print(f"Loading MATH test data from {DATA_PATH}/test/...")
    dataset = []
    test_dir = os.path.join(DATA_PATH, "test")
    for json_file in sorted(glob.glob(os.path.join(test_dir, "**/*.json"), recursive=True)):
        with open(json_file) as f:
            data = json.load(f)
            dataset.append(data)

    # 取自己的分片
    my_data = dataset[args.rank::args.total]
    print(f"Total={len(dataset)}  Rank={args.rank}/{args.total}  My chunk={len(my_data)}")

    # 加载模型（只看 CUDA_VISIBLE_DEVICES 指定的卡）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",  # "auto" 但只有一张可见卡 → 全在一张上
        trust_remote_code=True,
    )
    model.eval()

    suffix = f"_r{args.rank}" if args.total > 1 else ""
    all_samples, all_signals, all_eval = [], [], []

    for idx, problem in enumerate(my_data):
        task_id = f"math_{args.rank}_{idx}"
        question = problem["problem"]
        gold_solution = problem["solution"]
        raw_level = problem.get("level", "3")
        level = str(raw_level).replace("Level ", "").strip()
        problem_type = problem.get("type", "unknown")
        gold_answer = extract_gold_answer(gold_solution)

        system_msg = (
            "You are a math reasoning assistant. "
            "Solve the problem step by step. "
            "End your solution with the final answer in \\boxed{...}."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"Problem: {question}\n\nLet's solve this step by step."},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=(TEMPERATURE > 0),
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0][input_len:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        log_probs_list, logit_margins = [], []
        for score, token_id in zip(outputs.scores, generated_ids):
            lp = torch.log_softmax(score[0], dim=-1)
            log_probs_list.append(lp[token_id].item())
            top2 = torch.topk(lp, 2)
            margin = (top2.values[0] - top2.values[1]).item()
            logit_margins.append(max(margin, -10.0))

        ht1 = -float(np.mean(log_probs_list)) if log_probs_list else 0.0
        ht2 =  float(-np.min(logit_margins))  if logit_margins  else 0.0
        ht3 = (float(np.mean(np.exp(-np.clip(logit_margins, -5, None))))
               if logit_margins else 0.0)

        pred_answer = extract_boxed_answer(raw_output)
        passed = False
        if pred_answer is not None and gold_answer:
            passed = (normalize_math_answer(pred_answer)
                      == normalize_math_answer(gold_answer))

        signals = {
            "ht1_neg_logprob":    float(ht1),
            "ht2_neg_min_margin": float(ht2),
            "ht3_logprob_var":    float(ht3),
            "cost_token_len":     float(input_len / MAX_NEW_TOKENS),
            "cost_output_len":    float(len(generated_ids) / MAX_NEW_TOKENS),
            "cost_level":         float((int(level) - 1) / 4.0),
        }

        all_samples.append({"task_id": task_id, "completion": raw_output})
        all_signals.append({"task_id": task_id, "signals": signals})
        all_eval.append({"task_id": task_id, "passed": passed,
                          "level": level, "type": problem_type})

        print(f"[{idx+1}/{len(my_data)}] {task_id}  level={level}  "
              f"passed={passed} | ht1={ht1:.3f} ht2={ht2:.3f} ht3={ht3:.3f}")

    # 保存
    for path, data, is_jsonl in [
        (f"math_signal_results_qwen3{suffix}.json", all_signals, False),
        (f"math_eval_results_qwen3{suffix}.jsonl",  all_eval,    True),
    ]:
        with open(path, "w") as f:
            if is_jsonl:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            else:
                json.dump(data, f, indent=2)

    total  = len(all_eval)
    n_pass = sum(1 for e in all_eval if e["passed"])
    print(f"Done. Pass@1: {n_pass}/{total} = {n_pass/total:.4f}")


if __name__ == "__main__":
    main()
