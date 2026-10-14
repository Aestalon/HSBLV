"""
main.py — 主实验入口

用法:
    python main.py
    python main.py --signal signal_results_humaneval_llama3.json \
                   --result  samples_humaneval.jsonl_results.jsonl
    python main.py --n-seeds 10   # 多种子 bootstrap CI
"""
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Ensure src/ directory is on the path when run from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODE, NOISE_STD, FLIP_PROB,
    load_data, build_features, FeedbackOracle,
)
from signals import pretrain_theta, check_cost_signal_orthogonality, signal_correlation_report
from policies import (
    baseline_random, baseline_threshold_online,
    baseline_threshold_stratified, baseline_rule_system,
    baseline_conformal_stratified,
    mirror_prox_static_lambda, mirror_prox_adaptive_lambda,
    mirror_prox_stratified, oracle_policy,
    mirror_prox_adaptive_signal,
)
from evaluation import evaluate_hit_rate, evaluate_corrected_accuracy, bootstrap_ci
from analysis import (
    run_mismatch_sensitivity,
    run_memory_ablation,
    run_memory_signal_analysis,
    run_stratification_ablation,
    run_clean_feedback_ablation,
    run_quartile_analysis,
    run_oracle_signal_experiment,
    run_signal_strength_sweep,
)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="mbpp", choices=["mbpp", "math", "gpt"])
    parser.add_argument("--model",  default="qwen3", choices=["qwen3", "llama3", "gpt-4o-mini"])
    parser.add_argument("--signal", default="")
    parser.add_argument("--result", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--mode",   default=MODE, choices=["pretrain", "uniform"])
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of seeds for bootstrap CI (default: 1 = single run)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Base random seed (default: 42)")
    args = parser.parse_args()
    if args.model == "gpt-4o-mini" and args.data == "mbpp":
        args.data = "gpt"
    dataset = "math" if args.data == "math" else "mbpp"
    model = "gpt-4o-mini" if args.data == "gpt" else args.model
    suffix = "_4096" if dataset == "math" else ""
    args.signal = args.signal or str(ROOT / "data" / f"{dataset}_signal_results_{model}{suffix}.json")
    args.result = args.result or str(ROOT / "data" / f"{dataset}_eval_results_{model}{suffix}.jsonl")
    args.output = args.output or str(ROOT / "outputs" / f"{dataset}_experiment_results_{model}.json")
    return args


def load_math_data(model: str, seed: int = 42, signal_path=None, result_path=None):
    """加载 MATH 数据，返回与 build_features 一致的格式。"""
    with open(signal_path or ROOT / "data" / f"math_signal_results_{model}_4096.json") as f:
        signals = json.load(f)
    evals = []
    with open(result_path or ROOT / "data" / f"math_eval_results_{model}_4096.jsonl") as f:
        for line in f:
            evals.append(json.loads(line))

    task_ids = [str(i) for i in range(len(signals))]
    passed_map = {str(i): float(evals[i]["passed"]) for i in range(len(evals))}

    N = len(signals)
    rng = np.random.default_rng(seed)

    def normalize(x):
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    ht1_r = np.array([s["signals"]["ht1_neg_logprob"]    for s in signals])
    ht2_r = np.array([s["signals"]["ht2_neg_min_margin"] for s in signals])
    ht3_r = np.array([s["signals"]["ht3_logprob_var"]    for s in signals])

    ht1_n = normalize(normalize(ht1_r) + NOISE_STD * rng.standard_normal(N))
    ht2_n = normalize(normalize(ht2_r) + NOISE_STD * rng.standard_normal(N))
    ht3_n = normalize(normalize(ht3_r) + NOISE_STD * rng.standard_normal(N))

    cost_out = np.array([s["signals"]["cost_output_len"] for s in signals])
    cost_in  = np.array([s["signals"]["cost_token_len"]  for s in signals])

    H = np.stack([ht1_n, ht2_n, ht3_n], axis=1)
    costs_hetero   = 0.1 + 0.9 * normalize(cost_out)
    costs_decision = 0.1 + 0.9 * normalize(cost_in)
    costs_homo     = np.ones(N) * 0.5

    y_passed = np.array([e["passed"] for e in evals], dtype=float)
    y_error  = 1.0 - y_passed

    rng_fb = np.random.default_rng(seed + 1)
    flip_mask     = rng_fb.random(N) < FLIP_PROB
    y_error_noisy = np.where(flip_mask, 1.0 - y_error, y_error)

    raw_signals    = dict(ht1_raw=normalize(ht1_r), ht2_raw=normalize(ht2_r), ht3_raw=normalize(ht3_r))
    noisy_signals  = dict(ht1_n=ht1_n, ht2_n=ht2_n, ht3_n=ht3_n,
                          out_n=normalize(cost_out), cost_tok_n=normalize(cost_in))

    return task_ids, passed_map, H, costs_hetero, costs_decision, costs_homo, \
           y_passed, y_error, y_error_noisy, raw_signals, noisy_signals


# ──────────────────────────────────────────────
# 单种子实验
# ──────────────────────────────────────────────
def _run_single_experiment(args, seed: int, verbose: bool = True) -> dict:
    """Run the full experiment once with a given seed. Returns results dict."""
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "result", exist_ok=True)

    # 1. 数据加载与特征构造
    if args.data == "math":
        task_ids, passed_map, H, costs_hetero, costs_decision, costs_homo, \
           y_passed, y_error, y_error_noisy, raw_signals, noisy_signals = load_math_data(args.model, seed, args.signal, args.result)
        args.output = args.output or f"result/math_experiment_results_{args.model}.json"
        feedback = FeedbackOracle(task_ids, passed_map, y_error_noisy)
    elif args.data == "gpt":
        args.signal = args.signal or "result/mbpp_signal_results_gpt-4o-mini.json"
        args.result = args.result or "result/mbpp_eval_results_gpt-4o-mini.jsonl"
        args.output = args.output or "result/mbpp_experiment_results_gpt-4o-mini.json"
        task_ids, signal_map, passed_map = load_data(args.signal, args.result)
        (H, costs_hetero, costs_decision, costs_homo,
         y_passed, y_error, y_error_noisy,
         raw_signals, noisy_signals) = build_features(task_ids, signal_map, passed_map, seed)
        feedback = FeedbackOracle(task_ids, passed_map, y_error_noisy)
    else:
        task_ids, signal_map, passed_map = load_data(args.signal, args.result)
        (H, costs_hetero, costs_decision, costs_homo,
         y_passed, y_error, y_error_noisy,
         raw_signals, noisy_signals) = build_features(task_ids, signal_map, passed_map, seed)
        feedback = FeedbackOracle(task_ids, passed_map, y_error_noisy)

    N  = len(task_ids)
    T0 = min(50, N // 5)

    ht1_n      = noisy_signals["ht1_n"]
    ht2_n      = noisy_signals["ht2_n"]
    ht3_n      = noisy_signals["ht3_n"]
    out_n      = noisy_signals["out_n"]

    if verbose:
        print(f"Seed: {seed}")
        print(f"Signal noise std: {NOISE_STD}")
        print(f"Feedback flip prob: {FLIP_PROB}")
        print(f"H dimensions: {H.shape}  (ht1, ht2, ht3)")
        for label, c in [("costs_eval(output_len)", costs_hetero),
                          ("costs_act(input_len)",   costs_decision)]:
            print(f"{label}: min={c.min():.3f}, max={c.max():.3f}, "
                  f"std={c.std():.3f}, mean={c.mean():.3f}")

    # 2. 预训练 θ
    if verbose:
        print(f"\n冷启动预训练 θ（前 {T0} 题）...")
    if args.mode == "pretrain":
        theta_init = pretrain_theta(H, y_error_noisy, T0)
    else:
        theta_init = np.ones(H.shape[1]) / H.shape[1]

    ut = H @ theta_init

    # 3. 正交性检验 (only verbose)
    if verbose:
        check_cost_signal_orthogonality(H, ut, y_error, costs_hetero, costs_decision)

        # 4. Mismatch sweep
        run_mismatch_sensitivity(N, H, ut, y_error, out_n, costs_hetero)

    # 5. 主实验
    budget_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    no_audit   = float(np.mean(y_passed))
    error_rate = float(np.mean(y_error))

    STRATEGIES = ["random", "threshold", "threshold_strat", "cst_mem",
                  "rule_system", "conformal", "mp_static", "mp_adapt", "mp_strat", "oracle"]

    all_results: dict = {}

    for cost_mode, costs_eval, costs_act in [
        ("homogeneous",   costs_homo,   costs_homo),
        ("heterogeneous", costs_hetero, costs_decision),
    ]:
        if verbose:
            print()
            print("=" * 120)
            print(f"[{cost_mode.upper()}]  No audit (pass@1): {no_audit:.4f} | "
                  f"Error rate: {error_rate:.4f} | N={N}")
            print("=" * 120)
            print(f"\n【方案A - {cost_mode}】命中率")
            _print_hr_header()

        all_results[cost_mode] = {}
        print_b_rows = []

        for br in budget_ratios:
            if verbose:
                print(f"  Budget={br:.0%}")

            m_rand    = baseline_random(N, br, costs_eval, seed=seed)
            m_th      = baseline_threshold_online(N, br, ut, costs_eval, costs_act)
            m_cst     = baseline_threshold_stratified(
                N, br, ut, H, costs_eval, costs_act,
                task_ids, feedback, use_memory=False)
            m_cst_mem = baseline_threshold_stratified(
                N, br, ut, H, costs_eval, costs_act,
                task_ids, feedback, use_memory=True,
                memory_weight=0.3, top_k=5)
            m_rule    = baseline_rule_system(N, br, ht1_n, ht2_n, ht3_n, costs_eval)
            m_conf    = baseline_conformal_stratified(
                N, br, ut, H, costs_eval, costs_act,
                task_ids, feedback)
            m_mps     = mirror_prox_static_lambda(
                N, br, H, theta_init, costs_eval, costs_act,
                task_ids, feedback, T0=T0, seed=seed)
            m_mpa     = mirror_prox_adaptive_lambda(
                N, br, H, theta_init, costs_eval, costs_act,
                task_ids, feedback, T0=T0, seed=seed)
            m_strat   = mirror_prox_stratified(
                N, br, H, theta_init, costs_eval, costs_act,
                task_ids, feedback, T0=T0, seed=seed)
            m_ora     = oracle_policy(N, br, costs_eval, y_error)
            m_mpa_sig = mirror_prox_adaptive_signal(
                N, br, H, theta_init, costs_eval, costs_act,
                task_ids, feedback, T0=T0, seed=seed)
            masks = [m_rand, m_th, m_cst, m_cst_mem,
                     m_rule, m_conf, m_mps, m_mpa, m_strat, m_ora, m_mpa_sig]
            hr    = [evaluate_hit_rate(m, y_error)           for m in masks]
            ca    = [evaluate_corrected_accuracy(m, y_passed) for m in masks]

            all_results[cost_mode][br] = {
                s: {"hit_rate": hr[i], "corrected_acc": ca[i]}
                for i, s in enumerate(STRATEGIES)
            }

            if verbose:
                print(f"{br:>8.0%} | "
                      f"{hr[0]['hit_rate']:>8.4f} | {hr[1]['hit_rate']:>10.4f} | "
                      f"{hr[2]['hit_rate']:>8.4f} | {hr[3]['hit_rate']:>9.4f} | "
                      f"{hr[6]['hit_rate']:>9.4f} | {hr[7]['hit_rate']:>9.4f} | "
                      f"{hr[8]['hit_rate']:>8.4f}")
                print_b_rows.append((br, ca))

        if verbose:
            print(f"\n【方案B - {cost_mode}】纠正后正确率")
            _print_hr_header()
            for br, ca in print_b_rows:
                print(f"{br:>8.0%} | "
                      f"{ca[0]:>8.4f} | {ca[1]:>10.4f} | "
                      f"{ca[2]:>8.4f} | {ca[3]:>9.4f} | "
                      f"{ca[6]:>9.4f} | {ca[7]:>9.4f} | "
                      f"{ca[8]:>8.4f}")

    # 6. 信号诊断 (only verbose)
    if verbose:
        signal_correlation_report(
            H, theta_init, y_error, y_error_noisy,
            {**{k: v for k, v in raw_signals.items()},
             **{"out_n": out_n}})

        # 7. 消融 & 分析实验
        run_memory_ablation(N, H, ut, y_error, costs_hetero, costs_decision,
                            task_ids, feedback)
        run_memory_signal_analysis(N, H, theta_init, y_error, costs_hetero, costs_decision)
        run_stratification_ablation(N, H, ut, y_error, costs_hetero, costs_decision,
                                     task_ids, feedback)

        clean_fb = feedback.with_clean_labels(y_error)
        run_clean_feedback_ablation(N, H, theta_init, y_error,
                                      costs_hetero, costs_decision,
                                      task_ids, clean_fb, T0=T0)

        run_quartile_analysis(N, H, ut, y_error, y_passed,
                               costs_hetero, costs_decision,
                               task_ids, feedback, theta_init, T0)
        run_oracle_signal_experiment(N, H, ut, y_error,
                                       costs_hetero, costs_decision,
                                       task_ids, feedback, theta_init, T0)
        run_signal_strength_sweep(N, H, ut, y_error,
                                   costs_hetero, costs_decision,
                                   task_ids, feedback, T0)

    return {
        "no_audit": no_audit,
        "error_rate": error_rate,
        "theta_init": theta_init.tolist(),
        "N": N,
        "T0": T0,
        "results": all_results,
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    if args.n_seeds <= 1:
        # 单种子模式：保留完整 verbose 输出
        result = _run_single_experiment(args, args.seed, verbose=True)
        _save_results_single(args.output, result, args.seed)
    else:
        # 多种子 bootstrap 模式
        print(f"Running {args.n_seeds} seeds for bootstrap CI (base seed={args.seed})...")
        all_seeds = []
        for i in range(args.n_seeds):
            s = args.seed + i * 100
            print(f"  Seed {i+1}/{args.n_seeds} (seed={s})...", end=" ", flush=True)
            result = _run_single_experiment(args, s, verbose=False)
            all_seeds.append(result)
            print(f"done (no_audit={result['no_audit']:.4f}, error_rate={result['error_rate']:.4f})")

        # 聚合 CIs
        aggregated = _aggregate_seeds(all_seeds)
        _print_aggregated_results(aggregated, args.n_seeds)
        _save_results_multi(args.output, all_seeds, aggregated, args)


# ──────────────────────────────────────────────
# 多种子聚合
# ──────────────────────────────────────────────
def _aggregate_seeds(all_seeds: list) -> dict:
    """将多种子结果聚合为 mean ± CI 结构."""
    aggregated: dict = {}
    # 标量字段
    for key in ["no_audit", "error_rate"]:
        vals = [s[key] for s in all_seeds]
        aggregated[key] = bootstrap_ci(vals)

    # theta_init: 每维聚合
    theta_lists = [s["theta_init"] for s in all_seeds]
    aggregated["theta_init"] = [
        bootstrap_ci([t[i] for t in theta_lists])
        for i in range(len(theta_lists[0]))
    ]

    # results: nested {cost_mode: {br: {strategy: {hit_rate: ..., corrected_acc: ...}}}}}
    cost_modes = list(all_seeds[0]["results"].keys())
    budget_ratios = list(all_seeds[0]["results"][cost_modes[0]].keys())
    strategies = list(all_seeds[0]["results"][cost_modes[0]][budget_ratios[0]].keys())

    aggregated["results"] = {}
    for cm in cost_modes:
        aggregated["results"][cm] = {}
        for br in budget_ratios:
            aggregated["results"][cm][br] = {}
            for st in strategies:
                hr_vals = [s["results"][cm][br][st]["hit_rate"]["hit_rate"]
                          for s in all_seeds]
                ca_vals = [s["results"][cm][br][st]["corrected_acc"]
                          for s in all_seeds]
                aggregated["results"][cm][br][st] = {
                    "hit_rate": bootstrap_ci(hr_vals),
                    "corrected_acc": bootstrap_ci(ca_vals),
                }

    return aggregated


# ──────────────────────────────────────────────
# 打印
# ──────────────────────────────────────────────
def _print_hr_header():
    print(f"{'Budget':>8} | {'Random':>8} | {'Threshold':>10} | {'CST':>8} | "
          f"{'CST+Mem':>9} | {'MP-Adapt':>9} | {'MP-Strat':>9} | {'Oracle':>8}")
    print("-" * 100)


def _fmt(val, width: int = 8) -> str:
    """Format a value or CI dict for printing."""
    if isinstance(val, dict) and "mean" in val:
        return f"{val['mean']:{width}.4f}"
    elif isinstance(val, (int, float)):
        return f"{val:{width}.4f}"
    return f"{str(val):>{width}}"


def _fmt_ci(val, width: int = 8) -> str:
    """Format mean ± half-CI-width for compact table display."""
    if isinstance(val, dict) and "mean" in val and "std" in val:
        mean = val["mean"]
        half = (val["ci_upper"] - val["ci_lower"]) / 2
        return f"{mean:{width}.4f}"
    return _fmt(val, width)


def _print_aggregated_results(agg: dict, n_seeds: int):
    """Print aggregated multi-seed results with CI ranges."""
    print()
    print("=" * 120)
    print(f"AGGREGATED RESULTS  (n_seeds={n_seeds}, mean ± 95% CI)")
    print(f"  no_audit:  {_fmt_ci(agg['no_audit'])} ± {_fmt_half(agg['no_audit'])}")
    print(f"  error_rate: {_fmt_ci(agg['error_rate'])} ± {_fmt_half(agg['error_rate'])}")
    print("=" * 120)

    cost_modes = list(agg["results"].keys())
    for cm in cost_modes:
        budget_ratios = list(agg["results"][cm].keys())
        print(f"\n--- [{cm.upper()}] Hit Rate (mean ± 95% CI half-width) ---")
        print(f"{'Budget':>8} | {'Random':>16} | {'Threshold':>18} | "
              f"{'CST':>16} | {'CST+Mem':>18} | "
              f"{'MP-Adapt':>18} | {'MP-Strat':>18} | {'Oracle':>16}")
        print("-" * 135)

        for br in budget_ratios:
            vals = {}
            for st in ["random", "threshold", "threshold_strat", "cst_mem",
                       "mp_adapt", "mp_strat", "oracle"]:
                ci = agg["results"][cm][br][st]["hit_rate"]
                half = (ci["ci_upper"] - ci["ci_lower"]) / 2
                vals[st] = f"{ci['mean']:.4f} ± {half:.4f}"

            print(f"{float(br):>8.0%} | "
                  f"{vals['random']:>16} | {vals['threshold']:>18} | "
                  f"{vals['threshold_strat']:>16} | {vals['cst_mem']:>18} | "
                  f"{vals['mp_adapt']:>18} | {vals['mp_strat']:>18} | "
                  f"{vals['oracle']:>16}")

        # Corrected accuracy table
        print(f"\n--- [{cm.upper()}] Corrected Accuracy (mean ± 95% CI half-width) ---")
        print(f"{'Budget':>8} | {'Random':>16} | {'Threshold':>18} | "
              f"{'CST':>16} | {'CST+Mem':>18} | "
              f"{'MP-Adapt':>18} | {'MP-Strat':>18} | {'Oracle':>16}")
        print("-" * 135)

        for br in budget_ratios:
            vals = {}
            for st in ["random", "threshold", "threshold_strat", "cst_mem",
                       "mp_adapt", "mp_strat", "oracle"]:
                ci = agg["results"][cm][br][st]["corrected_acc"]
                half = (ci["ci_upper"] - ci["ci_lower"]) / 2
                vals[st] = f"{ci['mean']:.4f} ± {half:.4f}"

            print(f"{float(br):>8.0%} | "
                  f"{vals['random']:>16} | {vals['threshold']:>18} | "
                  f"{vals['threshold_strat']:>16} | {vals['cst_mem']:>18} | "
                  f"{vals['mp_adapt']:>18} | {vals['mp_strat']:>18} | "
                  f"{vals['oracle']:>16}")


def _fmt_half(ci_dict: dict) -> str:
    """CI half-width as string."""
    half = (ci_dict["ci_upper"] - ci_dict["ci_lower"]) / 2
    return f"{half:.4f}"


# ──────────────────────────────────────────────
# 保存
# ──────────────────────────────────────────────
def _save_results_single(path, result, seed):
    with open(path, "w") as f:
        json.dump({
            "no_audit":      result["no_audit"],
            "error_rate":    result["error_rate"],
            "theta_init":    result["theta_init"],
            "budget_ratios": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            "noise_std":     NOISE_STD,
            "flip_prob":     FLIP_PROB,
            "H_dim":         3,
            "N":             result["N"],
            "T0":            result["T0"],
            "seed":          seed,
            "n_seeds":       1,
            "results": {
                cm: {str(k): v for k, v in res.items()}
                for cm, res in result["results"].items()
            },
        }, f, indent=2)
    print(f"\nSaved {path}")


def _save_results_multi(path, all_seeds, aggregated, args):
    """Save raw per-seed data plus aggregated CI info."""
    save_path = path.replace(".json", "_bootstrap.json") if path else "result/bootstrap_results.json"

    def _serialize(obj):
        if isinstance(obj, dict) and "mean" in obj and "ci_lower" in obj:
            return obj
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    output = {
        "n_seeds":   args.n_seeds,
        "base_seed": args.seed,
        "noise_std": NOISE_STD,
        "flip_prob": FLIP_PROB,
        "H_dim":     3,
        "budget_ratios": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        "aggregated": aggregated,
        "per_seed":  all_seeds,
    }
    with open(save_path, "w") as f:
        json.dump(_serialize(output), f, indent=2)
    print(f"\nSaved bootstrap results to {save_path}")


if __name__ == "__main__":
    main()
