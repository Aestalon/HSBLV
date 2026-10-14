"""
memory.py — MemoryBank：基于余弦相似度的 top-k 加权记忆检索
"""

import numpy as np


class MemoryBank:
    """
    存储历史审计样本，按余弦相似度检索 top-k 邻居，
    支持按 cost 分层过滤。
    """

    def __init__(self, top_k: int = 5, min_confidence: float = 0.0, min_size: int = 20):
        self.signals: list[np.ndarray] = []
        self.labels:  list[float]      = []
        self.costs:   list[float]      = []
        self.scores:  list[float]      = []

        self.top_k          = top_k
        self.min_confidence = min_confidence
        self.min_size       = min_size
        self._global_error_rate = 0.0

    # ──────────────────────────────────────────
    # 写入
    # ──────────────────────────────────────────
    def write(self, h_t: np.ndarray, y_t: float, c_t: float,
              score_t: float, stratum=None) -> None:
        if score_t >= self.min_confidence:
            self.signals.append(h_t.copy())
            self.labels.append(float(y_t))
            self.costs.append(float(c_t))
            self.scores.append(float(score_t))
            self._global_error_rate = float(np.mean(self.labels))

    # ──────────────────────────────────────────
    # 检索
    # ──────────────────────────────────────────
    def retrieve(self, h_t: np.ndarray,
                 cost_stratum=None, q_bounds=None) -> float | None:
        if len(self.labels) == 0:
            return None
        if len(self.signals) < self.min_size:
            return self._global_error_rate

        mem    = np.array(self.signals)
        labels = np.array(self.labels)

        # 分层过滤
        if cost_stratum is not None and q_bounds is not None:
            costs_arr    = np.array(self.costs)
            stratum_mask = self._get_stratum_mask(costs_arr, cost_stratum, q_bounds)
            if stratum_mask.sum() >= 3:
                mem    = mem[stratum_mask]
                labels = labels[stratum_mask]

        # 余弦相似度
        norm_mem = np.linalg.norm(mem, axis=1)
        norm_q   = np.linalg.norm(h_t)
        denom    = norm_mem * norm_q + 1e-9
        sims     = mem @ h_t / denom

        k       = min(self.top_k, len(mem))
        topk_idx = np.argsort(sims)[-k:]
        weights  = np.maximum(sims[topk_idx], 0)

        if weights.sum() < 1e-9:
            return float(np.mean(labels))

        return float(np.dot(weights, labels[topk_idx]) / weights.sum())

    # ──────────────────────────────────────────
    # 辅助
    # ──────────────────────────────────────────
    def _get_stratum_mask(self, costs_arr: np.ndarray,
                          stratum: int, q_bounds: np.ndarray) -> np.ndarray:
        if stratum == 3:
            return costs_arr >= q_bounds[3]
        return (costs_arr >= q_bounds[stratum]) & (costs_arr < q_bounds[stratum + 1])

    def size(self) -> int:
        return len(self.signals)