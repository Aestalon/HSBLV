"""
config.py — 全局配置、数据加载、特征构造
"""

import json
import numpy as np

# ──────────────────────────────────────────────
# 超参
# ──────────────────────────────────────────────
MODE       = "pretrain"   # "pretrain" | "uniform"
NOISE_SEED = 42
NOISE_STD  = 0.1
FLIP_PROB  = 0.10

# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────
def load_data(
    signal_file: str = "mbpp_signal_results_qwen3.json",
    result_file: str = "mbpp_eval_results_qwen3.jsonl",
):
    with open(signal_file) as f:
        results = json.load(f)
    signal_map = {r["task_id"]: r["signals"] for r in results}

    passed_map: dict[str, float] = {}
    with open(result_file) as f:
        for line in f:
            r = json.loads(line)
            passed_map[r["task_id"]] = float(r["passed"])

    task_ids = [tid for tid in signal_map if tid in passed_map]
    return task_ids, signal_map, passed_map


# ──────────────────────────────────────────────
# 特征构造（归一化 + 噪声注入）
# ──────────────────────────────────────────────
def normalize(x: np.ndarray) -> np.ndarray:
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def build_features(task_ids, signal_map, passed_map, seed=None):
    N = len(task_ids)
    s = seed if seed is not None else NOISE_SEED
    rng = np.random.default_rng(s)

    # 原始信号
    ht1 = np.array([signal_map[tid]["ht1_neg_logprob"]    for tid in task_ids])
    ht2 = np.array([signal_map[tid]["ht2_neg_min_margin"] for tid in task_ids])
    ht3 = np.array([signal_map[tid]["ht3_logprob_var"]    for tid in task_ids])

    finite_min = np.nanmin(ht2[np.isfinite(ht2)]) if np.any(np.isfinite(ht2)) else 0.0
    ht2 = np.where(np.isneginf(ht2), finite_min, ht2)

    cost_tok = np.array([signal_map[tid].get("cost_token_len",  0.5) for tid in task_ids])
    out_len  = np.array([signal_map[tid].get("cost_output_len", 0.5) for tid in task_ids])

    # 归一化原始值（用于消融对比）
    ht1_raw = normalize(ht1)
    ht2_raw = normalize(ht2)
    ht3_raw = normalize(ht3)

    # 加噪归一化
    ht1_n = normalize(ht1_raw + NOISE_STD * rng.standard_normal(N))
    ht2_n = normalize(ht2_raw + NOISE_STD * rng.standard_normal(N))
    ht3_n = normalize(ht3_raw + NOISE_STD * rng.standard_normal(N))
    out_n      = normalize(out_len)
    cost_tok_n = normalize(cost_tok)

    H = np.stack([ht1_n, ht2_n, ht3_n], axis=1)

    # 成本向量
    costs_hetero   = 0.1 + 0.9 * out_n        # evaluation cost (output len)
    costs_decision = 0.1 + 0.9 * cost_tok_n   # action cost (input len)
    costs_homo     = np.ones(N) * 0.5

    # 标签
    y_passed = np.array([passed_map[tid] for tid in task_ids])
    y_error  = 1.0 - y_passed

    # noisy feedback
    rng_fb = np.random.default_rng(s + 1)
    flip_mask     = rng_fb.random(N) < FLIP_PROB
    y_error_noisy = np.where(flip_mask, 1.0 - y_error, y_error)

    raw_signals = dict(ht1_raw=ht1_raw, ht2_raw=ht2_raw, ht3_raw=ht3_raw)
    noisy_signals = dict(ht1_n=ht1_n, ht2_n=ht2_n, ht3_n=ht3_n,
                         out_n=out_n, cost_tok_n=cost_tok_n)

    return H, costs_hetero, costs_decision, costs_homo, \
           y_passed, y_error, y_error_noisy, raw_signals, noisy_signals


# ──────────────────────────────────────────────
# Noisy feedback 访问器（模块间共享状态注入）
# ──────────────────────────────────────────────
class FeedbackOracle:
    """封装 slow_feedback，避免全局变量."""

    def __init__(self, task_ids, passed_map, y_error_noisy):
        self._task_ids      = task_ids
        self._passed_map    = passed_map
        self._y_error_noisy = y_error_noisy

    def __call__(self, task_id: str, noisy: bool = True) -> float:
        idx = self._task_ids.index(task_id)
        if noisy:
            return 1.0 - self._y_error_noisy[idx]
        return self._passed_map[task_id]

    def with_clean_labels(self, y_error):
        """返回无噪声版本（消融实验用）."""
        return FeedbackOracle(self._task_ids, self._passed_map, y_error)
