import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import FLIP_PROB, NOISE_STD, build_features, load_data
from evaluation import evaluate_hit_rate
from policies import (
    baseline_random,
    baseline_threshold_online,
    baseline_threshold_stratified,
    mirror_prox_adaptive_lambda,
    mirror_prox_stratified,
    oracle_policy,
)
from signals import pretrain_theta


RESULTS = ROOT / "outputs"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
OUTER = DATA
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
CHEAP_MODELS = {
    "Qwen3": {
        "signal": OUTER / "mbpp_signal_results_qwen3.json",
        "eval": OUTER / "mbpp_eval_results_qwen3.jsonl",
    },
    "LLaMA3": {
        "signal": OUTER / "mbpp_signal_results_llama3.json",
        "eval": OUTER / "mbpp_eval_results_llama3.jsonl",
    },
}
EXPENSIVE = {
    "name": "GPT-4o-mini",
    "signal": DATA / "mbpp_signal_results_gpt-4o-mini.json",
    "eval": DATA / "mbpp_eval_results_gpt-4o-mini.jsonl",
}


class RoutingFeedback:
    def __init__(self, task_ids, y_benefit_noisy):
        self.index = {tid: i for i, tid in enumerate(task_ids)}
        self.y_benefit_noisy = y_benefit_noisy

    def __call__(self, task_id, noisy=True):
        y = self.y_benefit_noisy[self.index[task_id]]
        return 1.0 - y


def read_eval(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return {r["task_id"]: float(r["passed"]) for r in rows}


def normalize(x):
    x = np.asarray(x, dtype=float)
    if np.nanmax(x) == np.nanmin(x):
        return np.zeros_like(x)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def aggregate(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def run_one(cheap_name, files, seed):
    task_ids, signal_map, cheap_passed = load_data(str(files["signal"]), str(files["eval"]))
    expensive_passed = read_eval(EXPENSIVE["eval"])
    task_ids = [tid for tid in task_ids if tid in expensive_passed]

    H, _costs_hetero, costs_decision, _costs_homo, _y_passed, _y_error, _y_error_noisy, _raw, noisy = build_features(
        task_ids, signal_map, cheap_passed, seed
    )

    cheap_ok = np.array([cheap_passed[tid] for tid in task_ids], dtype=float)
    expensive_ok = np.array([expensive_passed[tid] for tid in task_ids], dtype=float)
    y_benefit = ((cheap_ok == 0.0) & (expensive_ok == 1.0)).astype(float)

    rng_fb = np.random.default_rng(seed + 1001)
    flip_mask = rng_fb.random(len(task_ids)) < FLIP_PROB
    y_benefit_noisy = np.where(flip_mask, 1.0 - y_benefit, y_benefit)
    feedback = RoutingFeedback(task_ids, y_benefit_noisy)

    T0 = min(50, len(task_ids) // 5)
    with contextlib.redirect_stdout(io.StringIO()):
        theta_init = pretrain_theta(H, y_benefit_noisy, T0)
    score = H @ theta_init

    # Routing cost is the ex-ante prompt cost of invoking the stronger model.
    routing_cost = 0.1 + 0.9 * noisy["cost_tok_n"]

    results = {}
    for br in BUDGETS:
        with contextlib.redirect_stdout(io.StringIO()):
            masks = {
                "Random": baseline_random(len(task_ids), br, routing_cost, seed=seed),
                "Threshold": baseline_threshold_online(len(task_ids), br, score, routing_cost, routing_cost),
                "CST": baseline_threshold_stratified(
                    len(task_ids), br, score, H, routing_cost, routing_cost, task_ids, feedback, use_memory=False
                ),
                "MP-Adapt": mirror_prox_adaptive_lambda(
                    len(task_ids), br, H, theta_init, routing_cost, routing_cost, task_ids, feedback, T0=T0, seed=seed
                ),
                "MP-Strat": mirror_prox_stratified(
                    len(task_ids), br, H, theta_init, routing_cost, routing_cost, task_ids, feedback, T0=T0, seed=seed
                ),
                "Oracle": oracle_policy(len(task_ids), br, routing_cost, y_benefit),
            }

        results[str(br)] = {
            name: {
                **evaluate_hit_rate(mask, y_benefit),
                "selected": int(np.sum(mask)),
                "budget_used": float(np.sum(routing_cost[mask]) / (len(task_ids) * br * float(np.mean(routing_cost)))),
            }
            for name, mask in masks.items()
        }

    return {
        "cheap_model": cheap_name,
        "expensive_model": EXPENSIVE["name"],
        "n": len(task_ids),
        "cheap_accuracy": float(np.mean(cheap_ok)),
        "expensive_accuracy": float(np.mean(expensive_ok)),
        "routing_benefit_rate": float(np.mean(y_benefit)),
        "results": results,
    }


def main():
    all_runs = {}
    seeds = list(range(42, 52))
    for cheap_name, files in CHEAP_MODELS.items():
        runs = [run_one(cheap_name, files, seed) for seed in seeds]
        summary = {
            "cheap_model": cheap_name,
            "expensive_model": EXPENSIVE["name"],
            "n": runs[0]["n"],
            "cheap_accuracy": runs[0]["cheap_accuracy"],
            "expensive_accuracy": runs[0]["expensive_accuracy"],
            "routing_benefit_rate": runs[0]["routing_benefit_rate"],
            "seeds": seeds,
            "results": {},
        }
        for br in BUDGETS:
            bkey = str(br)
            summary["results"][bkey] = {}
            methods = runs[0]["results"][bkey].keys()
            for method in methods:
                summary["results"][bkey][method] = {
                    metric: aggregate([r["results"][bkey][method][metric] for r in runs])
                    for metric in ["hit_rate", "error_coverage", "info_gain", "budget_used", "selected"]
                }
        all_runs[cheap_name] = summary

    out_json = RESULTS / "mbpp_routing_reanalysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    out_tex = RESULTS / "mbpp_routing_reanalysis_table.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llccccc}\n")
        f.write("\\toprule\n")
        f.write("Route & $\\beta$ & Random & Threshold & MP-Adapt & MP-Strat & CST \\\\\n")
        f.write("\\midrule\n")
        for cheap_name, summary in all_runs.items():
            br = "0.2"
            route = f"{cheap_name}$\\rightarrow$GPT-4o-mini"
            vals = summary["results"][br]
            row_methods = ["Random", "Threshold", "MP-Adapt", "MP-Strat", "CST"]
            row_values = {m: vals[m]["hit_rate"]["mean"] for m in row_methods}
            best = max(row_values, key=row_values.get)
            formatted = {
                m: f"\\textbf{{{row_values[m]:.3f}}}" if m == best else f"{row_values[m]:.3f}"
                for m in row_methods
            }
            f.write(
                f"{route} & 20\\% & "
                f"{formatted['Random']} & "
                f"{formatted['Threshold']} & "
                f"{formatted['MP-Adapt']} & "
                f"{formatted['MP-Strat']} & "
                f"{formatted['CST']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_tex}")
    for cheap_name, summary in all_runs.items():
        vals = summary["results"]["0.2"]
        print(f"\n{cheap_name} -> GPT-4o-mini, benefit rate={summary['routing_benefit_rate']:.3f}")
        for method in ["Random", "Threshold", "MP-Adapt", "MP-Strat", "CST", "Oracle"]:
            print(f"  {method:10s} HR@20%={vals[method]['hit_rate']['mean']:.3f}")


if __name__ == "__main__":
    main()
