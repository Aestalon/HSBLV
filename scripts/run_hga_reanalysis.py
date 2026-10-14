import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import FeedbackOracle, build_features, load_data
from evaluation import evaluate_hit_rate
from policies import (
    baseline_threshold_online,
    baseline_threshold_stratified,
    heterogeneity_gated_allocation,
)
from signals import pretrain_theta


RESULTS = ROOT / "outputs"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
CODE_ROOT = DATA
SEEDS = list(range(42, 52))
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

SETTINGS = [
    {
        "name": "MBPP-Qwen3",
        "signal": CODE_ROOT / "mbpp_signal_results_qwen3.json",
        "eval": CODE_ROOT / "mbpp_eval_results_qwen3.jsonl",
    },
    {
        "name": "MBPP-LLaMA3",
        "signal": CODE_ROOT / "mbpp_signal_results_llama3.json",
        "eval": CODE_ROOT / "mbpp_eval_results_llama3.jsonl",
    },
    {
        "name": "MBPP-GPT-4o-mini",
        "signal": DATA / "mbpp_signal_results_gpt-4o-mini.json",
        "eval": DATA / "mbpp_eval_results_gpt-4o-mini.jsonl",
    },
    {
        "name": "MATH-Qwen3",
        "signal": DATA / "math_signal_results_qwen3_4096.json",
        "eval": DATA / "math_eval_results_qwen3_4096.jsonl",
        "math": True,
    },
    {
        "name": "MATH-LLaMA3",
        "signal": DATA / "math_signal_results_llama3_4096.json",
        "eval": DATA / "math_eval_results_llama3_4096.jsonl",
        "math": True,
    },
]


def load_math_like(signal_path, eval_path):
    with open(signal_path, encoding="utf-8") as f:
        signals = json.load(f)
    evals = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                evals.append(json.loads(line))
    task_ids = [str(i) for i in range(len(signals))]
    signal_map = {str(i): s["signals"] for i, s in enumerate(signals)}
    passed_map = {str(i): float(evals[i]["passed"]) for i in range(len(evals))}
    return task_ids, signal_map, passed_map


def aggregate(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def run_setting(setting, seed):
    if setting.get("math"):
        task_ids, signal_map, passed_map = load_math_like(setting["signal"], setting["eval"])
    else:
        task_ids, signal_map, passed_map = load_data(str(setting["signal"]), str(setting["eval"]))

    H, costs_hetero, costs_decision, _costs_homo, y_passed, y_error, y_error_noisy, _raw, _noisy = build_features(
        task_ids, signal_map, passed_map, seed
    )
    feedback = FeedbackOracle(task_ids, passed_map, y_error_noisy)
    score_T0 = min(50, len(task_ids))
    gate_T0 = max(score_T0, int(0.5 * len(task_ids)))
    with contextlib.redirect_stdout(io.StringIO()):
        theta_init = pretrain_theta(H, y_error_noisy, score_T0)
    score = H @ theta_init

    out = {}
    for br in BUDGETS:
        with contextlib.redirect_stdout(io.StringIO()):
            masks = {
                "threshold": baseline_threshold_online(
                    len(task_ids), br, score, costs_hetero, costs_decision
                ),
                "cst": baseline_threshold_stratified(
                    len(task_ids), br, score, H, costs_hetero, costs_decision,
                    task_ids, feedback, use_memory=False
                ),
            }
        with contextlib.redirect_stdout(io.StringIO()):
            masks["hga"] = heterogeneity_gated_allocation(
                len(task_ids), br, score, H, costs_hetero, costs_decision,
                task_ids, feedback, T0=gate_T0
            )
        out[str(br)] = {
            name: {
                **evaluate_hit_rate(mask, y_error),
                "selected": int(np.sum(mask)),
                "budget_used": float(np.sum(costs_hetero[mask]) /
                                     (len(task_ids) * br * float(np.mean(costs_hetero)))),
            }
            for name, mask in masks.items()
        }
    return {
        "N": len(task_ids),
        "no_audit": float(np.mean(y_passed)),
        "error_rate": float(np.mean(y_error)),
        "results": out,
    }


def main():
    output = {}
    for setting in SETTINGS:
        print(f"Running {setting['name']}...")
        runs = [run_setting(setting, seed) for seed in SEEDS]
        summary = {
            "N": runs[0]["N"],
            "no_audit": runs[0]["no_audit"],
            "error_rate": runs[0]["error_rate"],
            "seeds": SEEDS,
            "results": {},
        }
        for br in BUDGETS:
            bkey = str(br)
            summary["results"][bkey] = {}
            for method in runs[0]["results"][bkey].keys():
                summary["results"][bkey][method] = {
                    metric: aggregate([r["results"][bkey][method][metric] for r in runs])
                    for metric in ["hit_rate", "error_coverage", "info_gain", "budget_used", "selected"]
                }
        output[setting["name"]] = summary

    out_json = RESULTS / "hga_reanalysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    out_tex = RESULTS / "hga_reanalysis_table.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("Setting & $\\beta$ & Threshold & CST & HGA \\\\\n")
        f.write("\\midrule\n")
        for name, summary in output.items():
            for br in ["0.1", "0.2", "0.3", "0.5"]:
                vals = summary["results"][br]
                methods = ["threshold", "cst", "hga"]
                means = {m: vals[m]["hit_rate"]["mean"] for m in methods}
                best = max(means, key=means.get)
                cells = []
                for m in methods:
                    cell = f"{means[m]:.3f}"
                    if m == best:
                        cell = f"\\textbf{{{cell}}}"
                    cells.append(cell)
                f.write(f"{name} & {float(br):.0%} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_tex}")
    print("\nHit rate at beta=20%, heterogeneous cost:")
    for name, summary in output.items():
        vals = summary["results"]["0.2"]
        print(
            f"{name}: "
            f"Threshold={vals['threshold']['hit_rate']['mean']:.3f}, "
            f"CST={vals['cst']['hit_rate']['mean']:.3f}, "
            f"HGA={vals['hga']['hit_rate']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
