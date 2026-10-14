"""
policies.py — 审计策略集合

核心修改（相对原版）：
  1. mirror_prox_adaptive_lambda:
       - θ 改为仅在审计时更新（真正的部分反馈/bandit设定）
       - λ 更新改为对称（去掉 no_audit_factor，审计/不审计用同一 eta_lam）
       - 追赶机制改为基于累积预算偏差，而非逐步乘以衰减系数
       - 学习率衰减从 0.01 改为 0.002（更慢衰减，后期保留学习能力）

  2. mirror_prox_stratified:
       - θ_k 同样改为仅在审计时更新
       - λ 更新改为对称
       - 追赶机制统一

  3. 新增 mirror_prox_adaptive_signal（核心新策略）：
       - 在线估计各区间信号质量 ρ_k
       - 学习率与 ρ_k 成正比（importance weighting）
       - 仅审计时更新 θ_k（部分反馈）
       - λ 对称更新 + 累积预算追赶

其余策略（baseline_random / threshold / rule_system / oracle / oracle_signal /
mp_with_synthetic_signal）保持不变。
"""

import numpy as np
from scipy.stats import pearsonr

from memory import MemoryBank


# ══════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════

def _eg_update(theta: np.ndarray, grad: np.ndarray,
               eta: float, entropy_reg: float = 0.05) -> np.ndarray:
    theta_new = theta * np.exp(eta * grad)
    theta_new = theta_new ** (1 - entropy_reg)
    theta_new = np.maximum(theta_new, 1e-8)
    return theta_new / theta_new.sum()


def _compute_grad_theta(theta, lam, h_t, y_t, cost_t):
    utility = np.dot(h_t, theta) - lam * cost_t
    p       = 1 / (1 + np.exp(-utility))
    return (y_t - p) * h_t


def _init_lambda(scores_T0: np.ndarray, costs_eval_T0: np.ndarray,
                 budget_ratio: float) -> float:
    mean_score      = float(np.mean(scores_T0))
    mean_cost       = float(np.mean(costs_eval_T0))
    normalized_cost = max(mean_cost, 0.5)
    base_ratio      = mean_score / (normalized_cost + 1e-9)
    scale           = np.log(1.0 / (budget_ratio + 1e-9)) / np.log(20.0)
    lam             = base_ratio * (0.5 + scale)
    return float(np.clip(lam, 0.1, 3.0))


def _update_lambda(lam: float, eta_lam: float,
                   ema_spent: float, target_budget_cost: float) -> float:
    return max(lam + eta_lam * (ema_spent - target_budget_cost), 0.0)


def _get_stratum_fn(q_bounds: np.ndarray):
    homogeneous = np.all(np.diff(q_bounds) < 1e-6)
    n = len(q_bounds) - 1
    if homogeneous:
        return lambda c: 0, 1
    def get_k(c):
        for k in range(n - 1):
            if c < q_bounds[k + 1]:
                return k
        return n - 1
    return get_k, n


def _estimate_rho(scores: np.ndarray, labels: np.ndarray) -> float:
    """在线估计 score-error Pearson 相关，不足 5 个样本返回 0."""
    if len(scores) < 5:
        return 0.0
    try:
        r, _ = pearsonr(scores, labels)
        return float(np.clip(r, 0.0, 1.0))   # 只保留正相关
    except Exception:
        return 0.0


# ══════════════════════════════════════════════
# 基线策略（不变）
# ══════════════════════════════════════════════

def baseline_random(N: int, budget_ratio: float,
                    costs_eval: np.ndarray, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    B_total = N * budget_ratio * float(np.mean(costs_eval))
    mask    = np.zeros(N, dtype=bool)
    spent   = 0.0
    for i in np.random.permutation(N):
        if spent + costs_eval[i] <= B_total:
            mask[i]  = True
            spent   += costs_eval[i]
    return mask


def baseline_threshold_online(N: int, budget_ratio: float, signal: np.ndarray,
                               costs_eval: np.ndarray,
                               costs_act: np.ndarray) -> np.ndarray:
    B_total       = N * budget_ratio * float(np.mean(costs_eval))
    mask          = np.zeros(N, dtype=bool)
    spent         = 0.0
    history_ratio: list[float] = []

    for t in range(N):
        ratio_t = signal[t] / (costs_act[t] + 1e-9)
        if len(history_ratio) > 10 and spent + costs_eval[t] <= B_total:
            threshold = np.quantile(history_ratio, 1 - budget_ratio)
            if ratio_t > threshold:
                mask[t]  = True
                spent   += costs_eval[t]
        history_ratio.append(ratio_t)
    return mask


def baseline_threshold_stratified(N: int, budget_ratio: float,
                                   signal: np.ndarray, H: np.ndarray,
                                   costs_eval: np.ndarray, costs_act: np.ndarray,
                                   task_ids: list, feedback_fn,
                                   use_memory: bool = False,
                                   memory_weight: float = 0.3,
                                   top_k: int = 5,
                                   min_size: int = 20,
                                   n_bins: int = 4) -> np.ndarray:
    B_total  = N * budget_ratio * float(np.mean(costs_eval))
    mask     = np.zeros(N, dtype=bool)
    spent    = 0.0
    q_bounds = np.percentile(costs_act, np.linspace(0, 100, n_bins + 1))
    get_k, _n = _get_stratum_fn(q_bounds)

    history_ratio_k = [[] for _ in range(n_bins)]
    memory = MemoryBank(top_k=top_k, min_size=min_size) if use_memory else None

    for t in range(N):
        k       = get_k(costs_act[t])
        ratio_t = signal[t] / (costs_act[t] + 1e-9)

        if memory is not None:
            h4 = memory.retrieve(H[t], cost_stratum=k, q_bounds=q_bounds)
            if h4 is not None:
                memory_ratio = h4 / (costs_act[t] + 1e-9)
                ratio_t = (1 - memory_weight) * ratio_t + memory_weight * memory_ratio

        if len(history_ratio_k[k]) > 10 and spent + costs_eval[t] <= B_total:
            threshold_k = np.quantile(history_ratio_k[k], 1 - budget_ratio)
            if ratio_t > threshold_k:
                mask[t]  = True
                spent   += costs_eval[t]
                if memory is not None:
                    y_t = 1.0 - feedback_fn(task_ids[t], noisy=True)
                    memory.write(H[t], y_t, costs_eval[t], ratio_t, k)

        history_ratio_k[k].append(ratio_t)

    if memory is not None:
        print(f"    [CST+Memory] 记忆库大小: {memory.size()}  "
              f"写入率: {memory.size()/N:.3f}  "
              f"记忆错误率: {memory._global_error_rate:.3f}")
    return mask


def baseline_rule_system(N: int, budget_ratio: float,
                          ht1_n: np.ndarray, ht2_n: np.ndarray, ht3_n: np.ndarray,
                          costs_eval: np.ndarray) -> np.ndarray:
    B_total = N * budget_ratio * float(np.mean(costs_eval))
    mask    = np.zeros(N, dtype=bool)
    spent   = 0.0
    h1_hist, h2_hist, h3_hist = [], [], []

    for t in range(N):
        h1_hist.append(ht1_n[t]); h2_hist.append(ht2_n[t]); h3_hist.append(ht3_n[t])
        if len(h1_hist) > 1 and spent + costs_eval[t] <= B_total:
            th1 = np.mean(h1_hist[:-1]) + 0.5 * np.std(h1_hist[:-1])
            th2 = np.mean(h2_hist[:-1]) + 0.5 * np.std(h2_hist[:-1])
            th3 = np.mean(h3_hist[:-1]) + 0.5 * np.std(h3_hist[:-1])
            if (ht1_n[t] > th1) or (ht2_n[t] > th2) or (ht3_n[t] > th3):
                mask[t]  = True
                spent   += costs_eval[t]
    return mask


# ══════════════════════════════════════════════
# Group-conditional conformal calibration baseline
# ══════════════════════════════════════════════

def baseline_conformal_stratified(N: int, budget_ratio: float,
                                   signal: np.ndarray, H: np.ndarray,
                                   costs_eval: np.ndarray, costs_act: np.ndarray,
                                   task_ids: list, feedback_fn,
                                   n_bins: int = 4, calib_frac: float = 0.5) -> np.ndarray:
    """
    Per-stratum calibrated thresholding (split-conformal style).
    The first calib_frac × N samples serve as a within-stratum calibration set;
    fixed per-stratum quantile thresholds are computed from calibration data
    and applied to the remaining samples. This tests whether CST's online
    quantile updating provides benefit over a one-shot calibrated threshold.
    """
    B_total = N * budget_ratio * float(np.mean(costs_eval))
    mask     = np.zeros(N, dtype=bool)
    spent    = 0.0
    q_bounds = np.percentile(costs_act, np.linspace(0, 100, n_bins + 1))
    get_k, _ = _get_stratum_fn(q_bounds)

    calib_cutoff = int(N * calib_frac)

    # Collect calibration data per stratum
    calib_ratios = {k: [] for k in range(n_bins)}
    for t in range(calib_cutoff):
        k = get_k(costs_act[t])
        calib_ratios[k].append(signal[t] / (costs_act[t] + 1e-9))

    # Compute per-stratum fixed thresholds
    thresholds = {}
    for k in range(n_bins):
        if len(calib_ratios[k]) >= 10:
            thresholds[k] = np.quantile(calib_ratios[k], 1 - budget_ratio)
        else:
            thresholds[k] = 0.0

    # Apply fixed thresholds online
    for t in range(N):
        k = get_k(costs_act[t])
        if k in thresholds and spent + costs_eval[t] <= B_total:
            ratio_t = signal[t] / (costs_act[t] + 1e-9)
            if ratio_t > thresholds[k]:
                mask[t]  = True
                spent   += costs_eval[t]

    return mask


# ══════════════════════════════════════════════
# Mirror Prox 策略族
# ══════════════════════════════════════════════

def comet_group_calibrated_allocation(N: int, budget_ratio: float,
                                      signal: np.ndarray, H: np.ndarray,
                                      theta_init: np.ndarray,
                                      costs_eval: np.ndarray, costs_act: np.ndarray,
                                      task_ids: list, feedback_fn,
                                      T0: int = 50, k_init: int = 8,
                                      min_group_hist: int = 10,
                                      min_calib_bin: int = 4,
                                      merge_tol: float = 0.18,
                                      rho_floor: float = 0.05,
                                      seed: int = 42) -> np.ndarray:
    """Comparability-aware group-calibrated allocation (COMET/GCA).

    Fixed CST uses pre-defined cost strata. COMET starts with finer ex-ante
    cost bins, merges adjacent bins with similar cold-start calibration
    profiles, and only trusts group-specific thresholds when the group signal
    is reliable. Otherwise it falls back to the global threshold.
    """
    del theta_init, seed

    B_total = N * budget_ratio * float(np.mean(costs_eval))
    mask = np.zeros(N, dtype=bool)
    spent = 0.0

    score = np.asarray(signal, dtype=float)
    ratio = score / (costs_act + 1e-9)
    calib_n = max(min(T0, N), min(k_init * min_calib_bin, N))
    q_bounds = np.percentile(costs_act, np.linspace(0, 100, k_init + 1))
    fine_bin, n_bins = _get_stratum_fn(q_bounds)

    y_calib = np.array([1.0 - feedback_fn(task_ids[t], noisy=True)
                        for t in range(calib_n)], dtype=float)
    global_rho = _estimate_rho(score[:calib_n], y_calib)
    global_base = float(np.mean(y_calib)) if calib_n else 0.0

    profiles = []
    for k in range(n_bins):
        idx = np.array([t for t in range(calib_n)
                        if fine_bin(costs_act[t]) == k], dtype=int)
        if len(idx) >= min_calib_bin:
            base = float(np.mean(y_calib[idx]))
            rho = _estimate_rho(score[idx], y_calib[idx])
        else:
            base = global_base
            rho = global_rho
        profiles.append({"base": base, "rho": max(rho, 0.0), "n": int(len(idx))})

    groups = []
    current = [0]
    for k in range(1, n_bins):
        prev = profiles[current[-1]]
        cur = profiles[k]
        distance = abs(prev["base"] - cur["base"]) + abs(prev["rho"] - cur["rho"])
        sparse = prev["n"] < min_calib_bin or cur["n"] < min_calib_bin
        if sparse or distance <= merge_tol:
            current.append(k)
        else:
            groups.append(current)
            current = [k]
    groups.append(current)

    bin_to_group = {}
    group_profiles = []
    for gid, bins in enumerate(groups):
        for b in bins:
            bin_to_group[b] = gid
        idx = np.array([t for t in range(calib_n)
                        if fine_bin(costs_act[t]) in bins], dtype=int)
        if len(idx) >= min_calib_bin:
            base = float(np.mean(y_calib[idx]))
            rho = _estimate_rho(score[idx], y_calib[idx])
        else:
            base = global_base
            rho = global_rho
        reliable = (rho >= max(rho_floor, 0.5 * global_rho)) or (base >= global_base + 0.05)
        group_profiles.append({"base": base, "rho": max(rho, 0.0), "reliable": reliable})

    global_history = []
    group_history = [[] for _ in groups]

    for t in range(N):
        gid = bin_to_group[fine_bin(costs_act[t])]
        r_t = ratio[t]

        use_group = group_profiles[gid]["reliable"] and len(group_history[gid]) >= min_group_hist
        if use_group:
            threshold = np.quantile(group_history[gid], 1 - budget_ratio)
        elif len(global_history) > 10:
            threshold = np.quantile(global_history, 1 - budget_ratio)
        else:
            threshold = np.inf

        if spent + costs_eval[t] <= B_total and r_t > threshold:
            mask[t] = True
            spent += costs_eval[t]

        global_history.append(r_t)
        group_history[gid].append(r_t)

    group_desc = ",".join("-".join(str(b) for b in g) for g in groups)
    reliable_count = sum(1 for p in group_profiles if p["reliable"])
    print(f"    [COMET/GCA] groups={group_desc} reliable={reliable_count}/{len(groups)} "
          f"budget_used={spent/B_total:.3f}")
    return mask


def comet_validation_selected_allocation(N: int, budget_ratio: float,
                                         signal: np.ndarray, H: np.ndarray,
                                         theta_init: np.ndarray,
                                         costs_eval: np.ndarray, costs_act: np.ndarray,
                                         task_ids: list, feedback_fn,
                                         T0: int = 50, k_init: int = 8,
                                         min_group_hist: int = 10,
                                         min_calib_bin: int = 4,
                                         merge_tol: float = 0.18,
                                         seed: int = 42) -> np.ndarray:
    """COMET with cold-start validation over grouping granularity.

    Candidate granularities are global, fixed cost quartiles, and learned
    comparability groups. The cold-start set chooses the granularity with the
    best budgeted hit rate, then the chosen groups are used online.
    """
    del H, theta_init, seed

    score = np.asarray(signal, dtype=float)
    ratio = score / (costs_act + 1e-9)
    calib_n = max(min(T0, N), min(k_init * min_calib_bin, N))
    y_calib = np.array([1.0 - feedback_fn(task_ids[t], noisy=True)
                        for t in range(calib_n)], dtype=float)

    def make_ids(bounds):
        get_k, _ = _get_stratum_fn(bounds)
        return np.array([get_k(c) for c in costs_act], dtype=int)

    global_ids = np.zeros(N, dtype=int)
    cst_ids = make_ids(np.percentile(costs_act, [0, 25, 50, 75, 100]))

    fine_bounds = np.percentile(costs_act, np.linspace(0, 100, k_init + 1))
    fine_get, n_bins = _get_stratum_fn(fine_bounds)
    fine_ids = np.array([fine_get(c) for c in costs_act], dtype=int)

    profiles = []
    global_base = float(np.mean(y_calib)) if calib_n else 0.0
    global_rho = _estimate_rho(score[:calib_n], y_calib)
    for k in range(n_bins):
        idx = np.where(fine_ids[:calib_n] == k)[0]
        if len(idx) >= min_calib_bin:
            base = float(np.mean(y_calib[idx]))
            rho = _estimate_rho(score[idx], y_calib[idx])
        else:
            base = global_base
            rho = global_rho
        profiles.append({"base": base, "rho": max(rho, 0.0), "n": int(len(idx))})

    groups = []
    current = [0]
    for k in range(1, n_bins):
        prev = profiles[current[-1]]
        cur = profiles[k]
        distance = abs(prev["base"] - cur["base"]) + abs(prev["rho"] - cur["rho"])
        sparse = prev["n"] < min_calib_bin or cur["n"] < min_calib_bin
        if sparse or distance <= merge_tol:
            current.append(k)
        else:
            groups.append(current)
            current = [k]
    groups.append(current)
    fine_to_group = {b: gid for gid, bins in enumerate(groups) for b in bins}
    learned_ids = np.array([fine_to_group[b] for b in fine_ids], dtype=int)

    def run_online(group_ids, end, y_eval=None):
        B_total = end * budget_ratio * float(np.mean(costs_eval[:end]))
        mask = np.zeros(end, dtype=bool)
        spent = 0.0
        histories = {int(g): [] for g in np.unique(group_ids[:end])}
        for t in range(end):
            gid = int(group_ids[t])
            hist = histories[gid]
            if len(hist) >= min_group_hist:
                threshold = np.quantile(hist, 1 - budget_ratio)
            else:
                all_hist = [v for h in histories.values() for v in h]
                threshold = np.quantile(all_hist, 1 - budget_ratio) if len(all_hist) > 10 else np.inf
            if spent + costs_eval[t] <= B_total and ratio[t] > threshold:
                mask[t] = True
                spent += costs_eval[t]
            hist.append(ratio[t])
        if y_eval is None:
            return mask
        selected = int(np.sum(mask))
        if selected == 0:
            return 0.0
        return float(np.sum(mask & (y_eval[:end] == 1)) / selected)

    candidates = {
        "global": global_ids,
        "cost4": cst_ids,
        "learned": learned_ids,
    }
    calib_scores = {name: run_online(ids, calib_n, y_calib) for name, ids in candidates.items()}
    chosen_name = max(calib_scores, key=calib_scores.get)
    chosen_ids = candidates[chosen_name]
    mask = run_online(chosen_ids, N)

    spent = float(np.sum(costs_eval[mask]))
    B_total = N * budget_ratio * float(np.mean(costs_eval))
    print(f"    [COMET-Select] choice={chosen_name} calib={calib_scores} "
          f"groups={len(np.unique(chosen_ids))} budget_used={spent/B_total:.3f}")
    return mask


def heterogeneity_gated_allocation(N: int, budget_ratio: float,
                                   signal: np.ndarray, H: np.ndarray,
                                   costs_eval: np.ndarray, costs_act: np.ndarray,
                                   task_ids: list, feedback_fn,
                                   T0: int = 50,
                                   min_gain: float = 0.0,
                                   min_heterogeneity: float = 0.002,
                                   min_alignment: float = 0.03) -> np.ndarray:
    """HGA: choose global or cost-stratified allocation from a warm-up diagnostic.

    HGA treats CST as a conditional intervention. It estimates, on warm-up
    feedback only, whether cost strata change signal value enough to justify
    per-stratum thresholds. If not, it falls back to the global threshold.
    """
    del H

    warm_n = max(min(T0, N), min(int(0.3 * N), N))
    y_warm = np.array([1.0 - feedback_fn(task_ids[t], noisy=True)
                       for t in range(warm_n)], dtype=float)

    q_bounds = np.percentile(costs_act[:warm_n], [0, 25, 50, 75, 100])
    get_k, n_bins = _get_stratum_fn(q_bounds)

    rho_vals = []
    base_vals = []
    for k in range(n_bins):
        idx = np.array([t for t in range(warm_n) if get_k(costs_act[t]) == k], dtype=int)
        if len(idx) >= 5:
            rho_vals.append(_estimate_rho(signal[idx], y_warm[idx]))
            base_vals.append(float(np.mean(y_warm[idx])))
        else:
            rho_vals.append(0.0)
            base_vals.append(float(np.mean(y_warm)) if warm_n else 0.0)
    heterogeneity = float(np.var(rho_vals))
    alignment = float(max(base_vals) - min(base_vals))

    def simulate_threshold(end):
        B_total = end * budget_ratio * float(np.mean(costs_eval[:end]))
        mask = np.zeros(end, dtype=bool)
        spent = 0.0
        hist = []
        for t in range(end):
            ratio_t = signal[t] / (costs_act[t] + 1e-9)
            if len(hist) > 10 and spent + costs_eval[t] <= B_total:
                if ratio_t > np.quantile(hist, 1 - budget_ratio):
                    mask[t] = True
                    spent += costs_eval[t]
            hist.append(ratio_t)
        return mask

    def simulate_cst(end):
        B_total = end * budget_ratio * float(np.mean(costs_eval[:end]))
        mask = np.zeros(end, dtype=bool)
        spent = 0.0
        histories = [[] for _ in range(n_bins)]
        for t in range(end):
            k = get_k(costs_act[t])
            ratio_t = signal[t] / (costs_act[t] + 1e-9)
            if len(histories[k]) > 10 and spent + costs_eval[t] <= B_total:
                if ratio_t > np.quantile(histories[k], 1 - budget_ratio):
                    mask[t] = True
                    spent += costs_eval[t]
            histories[k].append(ratio_t)
        return mask

    def warm_hit(mask):
        selected = int(np.sum(mask))
        if selected == 0:
            return 0.0
        return float(np.sum(mask & (y_warm[:len(mask)] == 1)) / selected)

    th_warm = warm_hit(simulate_threshold(warm_n))
    cst_warm = warm_hit(simulate_cst(warm_n))
    gate_open = (
        cst_warm >= th_warm + min_gain
        and heterogeneity >= min_heterogeneity
        and alignment >= min_alignment
    )

    if gate_open:
        mask = simulate_cst(N)
        choice = "CST"
    else:
        mask = simulate_threshold(N)
        choice = "Threshold"

    spent = float(np.sum(costs_eval[mask]))
    B_total = N * budget_ratio * float(np.mean(costs_eval))
    print(f"    [HGA] choice={choice} warm_thr={th_warm:.3f} warm_cst={cst_warm:.3f} "
          f"H={heterogeneity:.4f} align={alignment:.3f} budget_used={spent/B_total:.3f}")
    return mask


def mirror_prox_static_lambda(N: int, budget_ratio: float,
                               H: np.ndarray, theta_init: np.ndarray,
                               costs_eval: np.ndarray, costs_act: np.ndarray,
                               task_ids: list, feedback_fn,
                               eta_theta: float = 0.5,
                               T0: int = 50, seed: int = 42) -> np.ndarray:
    """MP-Static：λ 固定，θ 仅在审计时更新（部分反馈）。"""
    np.random.seed(seed)
    B_total        = N * budget_ratio * float(np.mean(costs_eval))
    theta          = theta_init.copy()
    mean_score     = float(np.mean(H @ theta))
    mean_cost_eval = float(np.mean(costs_eval))
    lam = (mean_score - np.log(budget_ratio / (1 - budget_ratio + 1e-9))) / (mean_cost_eval + 1e-9)
    lam = float(np.clip(lam, 0.0, 5.0))
    print(f"    [MP-Static] λ_init={lam:.3f}")

    mask  = np.zeros(N, dtype=bool)
    spent = 0.0

    for t in range(N):
        tid     = task_ids[t]
        score_t = np.dot(H[t], theta)
        utility = score_t - lam * costs_act[t]

        remaining_budget = B_total - spent
        remaining_tasks  = N - t
        avg_cost_remain  = float(np.mean(costs_eval[t:]))

        if remaining_budget <= 0:
            do_audit = False
        elif remaining_budget >= remaining_tasks * avg_cost_remain:
            do_audit = True
        else:
            prob     = 1 / (1 + np.exp(-utility))
            do_audit = (np.random.random() < prob)

        if do_audit and spent + costs_eval[t] <= B_total:
            mask[t]  = True
            spent   += costs_eval[t]
            # 仅审计时获取反馈并更新 θ
            y_t   = 1.0 - feedback_fn(tid, noisy=True)
            eta_t = eta_theta / (1 + 0.002 * t)
            grad1 = _compute_grad_theta(theta, lam, H[t], y_t, costs_act[t])
            theta_mid = _eg_update(theta, grad1, eta_t)
            grad2 = _compute_grad_theta(theta_mid, lam, H[t], y_t, costs_act[t])
            theta = _eg_update(theta, grad2, eta_t)

    print(f"    [MP-Static] 实际预算消耗: {spent/B_total:.3f} (target=1.000)")
    print(f"    [MP-Static] θ_final={theta.round(4)}")
    return mask


def mirror_prox_adaptive_lambda(N: int, budget_ratio: float,
                                 H: np.ndarray, theta_init: np.ndarray,
                                 costs_eval: np.ndarray, costs_act: np.ndarray,
                                 task_ids: list, feedback_fn,
                                 eta_theta: float = 0.5, eta_lam: float = None,
                                 T0: int = 50, seed: int = 42) -> np.ndarray:
    """
    MP-Adapt（修复版）：
      - θ 仅在审计时更新（真正的部分反馈）
      - λ 对称更新：审计/不审计均使用同一 eta_lam，消除系统性偏高
      - 追赶机制：基于累积预算利用率直接设置 λ，快速响应预算不足
      - 学习率慢衰减（0.002 代替 0.01）
    """
    LOW_BUDGET = 0.15
    if eta_lam is None:
        if budget_ratio < LOW_BUDGET:
            eta_lam = 0.3 / (budget_ratio + 0.05)
        else:
            eta_lam = 0.5 / (budget_ratio + 0.05)

    print(f"    [MP-Adapt] eta_lam={eta_lam:.3f} "
          f"(low_budget_mode={budget_ratio < LOW_BUDGET})")

    theta   = theta_init.copy()
    B_total = N * budget_ratio * float(np.mean(costs_eval))

    scores_T0 = H[:T0] @ theta
    lam       = _init_lambda(scores_T0, costs_eval[:T0], budget_ratio)
    if budget_ratio < LOW_BUDGET:
        lam *= 0.5
    print(f"    [MP-Adapt] λ_init={lam:.3f}")

    target_budget_cost = budget_ratio * float(np.mean(costs_eval))
    # 每步平均允许消耗
    per_step_target    = B_total / N

    mask   = np.zeros(N, dtype=bool)
    spent  = 0.0
    lam_up = lam_dn = 0

    for t in range(N):
        tid     = task_ids[t]
        score_t = np.dot(H[t], theta)
        utility = score_t - lam * costs_act[t]

        do_audit = False if spent + costs_eval[t] > B_total else (utility > 0)

        if do_audit:
            mask[t]  = True
            spent   += costs_eval[t]

            # ── 仅审计时获取反馈并更新 θ ──
            y_t   = 1.0 - feedback_fn(tid, noisy=True)
            eta_t = eta_theta / (1 + 0.002 * t)   # 慢衰减
            grad1 = _compute_grad_theta(theta, lam, H[t], y_t, costs_act[t])
            theta_mid = _eg_update(theta, grad1, eta_t)
            grad2 = _compute_grad_theta(theta_mid, lam, H[t], y_t, costs_act[t])
            theta = _eg_update(theta, grad2, eta_t)

            # ── λ 对称更新：审计时用实际成本 ──
            lam = _update_lambda(lam, eta_lam, costs_eval[t], target_budget_cost)
            lam_up += 1
        else:
            # ── λ 对称更新：不审计时成本为 0，eta_lam 相同 ──
            lam = _update_lambda(lam, eta_lam, 0.0, target_budget_cost)
            lam_dn += 1

        # ── 累积预算追赶：若利用率严重不足则直接压低 λ ──
        remaining_budget = B_total - spent
        remaining_steps  = N - t - 1
        if remaining_steps > 0:
            needed_rate = remaining_budget / (remaining_steps * float(np.mean(costs_eval[t+1:])) + 1e-9)
            if needed_rate > 1.2:
                # 剩余预算明显超出剩余容量，大幅压低 λ
                lam = max(lam * 0.90, 0.0)
            elif needed_rate > 0.9:
                # 轻微超出，温和压低
                lam = max(lam * 0.97, 0.0)

    print(f"    [MP-Adapt] 实际预算消耗: {spent/B_total:.3f} (target=1.000)")
    print(f"    [MP-Adapt] λ 变化: 上升{lam_up}次, 下降{lam_dn}次")
    print(f"    [MP-Adapt] θ_final={theta.round(4)}  lam_final={lam:.4f}")
    return mask


def mirror_prox_stratified(N: int, budget_ratio: float,
                            H: np.ndarray, theta_init: np.ndarray,
                            costs_eval: np.ndarray, costs_act: np.ndarray,
                            task_ids: list, feedback_fn,
                            eta_theta: float = 0.5, eta_lam: float = None,
                            momentum: float = 0.9,
                            T0: int = 50, seed: int = 42) -> np.ndarray:
    """
    MP-Strat（per-stratum λ_k 版）：
      - θ_k 仅在审计时更新
      - λ_k per stratum，各自独立对称更新
      - 累积预算追赶（全局）
      - momentum>0 时使用 teacher-student EMA 稳定决策
    """
    if eta_lam is None:
        eta_lam = 0.3 / (budget_ratio + 0.05)
    mom_str = f"  momentum={momentum}" if momentum > 0 else ""
    print(f"    [MP-Strat] eta_lam={eta_lam:.3f}{mom_str}")

    q_bounds      = np.percentile(costs_act, [0, 25, 50, 75, 100])
    get_k, n_bins = _get_stratum_fn(q_bounds)
    theta_k       = np.stack([theta_init.copy() for _ in range(n_bins)], axis=0)
    # teacher θ for decision (only used when momentum>0)
    theta_teacher_k = np.stack([theta_init.copy() for _ in range(n_bins)], axis=0) if momentum > 0 else theta_k

    B_total   = N * budget_ratio * float(np.mean(costs_eval))
    target_budget_cost = budget_ratio * float(np.mean(costs_eval))

    # ── per-stratum λ_k 初始化 ──
    lam_k = np.full(n_bins, np.nan)
    for k in range(n_bins):
        idx_k = [t for t in range(min(T0, N)) if get_k(costs_act[t]) == k]
        if len(idx_k) >= 3:
            scores_k = np.array([np.dot(H[t], theta_k[k]) for t in idx_k])
            costs_k  = costs_eval[idx_k]
            lam_k[k] = _init_lambda(scores_k, costs_k, budget_ratio)
        else:
            lam_k[k] = 0.5
    lam_k[lam_k != lam_k] = 0.5  # NaN → default
    lam_k = np.clip(lam_k, 0.05, 5.0)
    lam_str = ", ".join([f"{v:.3f}" for v in lam_k])
    print(f"    [MP-Strat] λ_k=[{lam_str}]  n_bins={n_bins}")

    lam_floor = 0.1  # 防止 λ_k 降到 0 导致单个 bin 独占预算

    mask  = np.zeros(N, dtype=bool)
    spent = 0.0

    for t in range(N):
        tid     = task_ids[t]
        k       = get_k(costs_act[t])
        # 决策用 teacher（momentum>0）或直接 θ_k（momentum=0）
        theta_dec = theta_teacher_k[k] if momentum > 0 else theta_k[k]
        score_t = np.dot(H[t], theta_dec)
        utility = score_t - lam_k[k] * costs_act[t]

        do_audit = False if spent + costs_eval[t] > B_total else (utility > 0)

        if do_audit:
            mask[t]  = True
            spent   += costs_eval[t]

            # 仅审计时更新 student θ_k
            y_t   = 1.0 - feedback_fn(tid, noisy=True)
            eta_t = eta_theta / (1 + 0.002 * t)
            grad1 = _compute_grad_theta(theta_k[k], lam_k[k], H[t], y_t, costs_act[t])
            theta_mid = _eg_update(theta_k[k], grad1, eta_t)
            grad2 = _compute_grad_theta(theta_mid, lam_k[k], H[t], y_t, costs_act[t])
            theta_k[k] = _eg_update(theta_k[k], grad2, eta_t)

            # Teacher 动量跟随 student
            if momentum > 0:
                theta_teacher_k[k] = (momentum * theta_teacher_k[k]
                                      + (1 - momentum) * theta_k[k])
                theta_teacher_k[k] = theta_teacher_k[k] / theta_teacher_k[k].sum()

            lam_k[k] = max(_update_lambda(lam_k[k], eta_lam, costs_eval[t], target_budget_cost), lam_floor)
        else:
            lam_k[k] = max(_update_lambda(lam_k[k], eta_lam, 0.0, target_budget_cost), lam_floor)

        # 累积预算追赶（全局，作用到所有 bin）
        remaining_budget = B_total - spent
        remaining_steps  = N - t - 1
        if remaining_steps > 0:
            needed_rate = remaining_budget / (remaining_steps * float(np.mean(costs_eval[t+1:])) + 1e-9)
            if needed_rate > 1.2:
                lam_k = np.maximum(lam_k * 0.90, lam_floor)
            elif needed_rate > 0.9:
                lam_k = np.maximum(lam_k * 0.97, lam_floor)

    for k in range(n_bins):
        n_k = sum(1 for t in range(N) if get_k(costs_act[t]) == k)
        th = theta_teacher_k[k] if momentum > 0 else theta_k[k]
        print(f"    [MP-Strat] θ_k[{k}]={th.round(4)}  λ_k={lam_k[k]:.4f}  (n={n_k})")
    print(f"    [MP-Strat] 实际预算消耗: {spent/B_total:.3f} (target=1.000)")
    return mask


def mirror_prox_adaptive_signal(N: int, budget_ratio: float,
                                 H: np.ndarray, theta_init: np.ndarray,
                                 costs_eval: np.ndarray, costs_act: np.ndarray,
                                 task_ids: list, feedback_fn,
                                 eta_theta: float = 0.5, eta_lam: float = None,
                                 momentum: float = 0.9,
                                 T0: int = 50, rho_min: float = 0.05,
                                 seed: int = 42) -> np.ndarray:
    """
    MP-AdaptSig：分层自适应信号质量 Mirror-Prox + Momentum teacher-student
    ────────────────────────────────────────────────────────────────────
      - ρ_k 加权学习率（高信号 bin 学得快）
      - per-stratum λ_k
      - teacher-student momentum 稳定决策
    """
    if eta_lam is None:
        if budget_ratio < 0.15:
            eta_lam = 0.3 / (budget_ratio + 0.05)
        else:
            eta_lam = 0.5 / (budget_ratio + 0.05)

    mom_str = f"  momentum={momentum}" if momentum > 0 else ""
    print(f"    [MP-AdaptSig] eta_lam={eta_lam:.3f}{mom_str}")

    q_bounds      = np.percentile(costs_act, [0, 25, 50, 75, 100])
    get_k, n_bins = _get_stratum_fn(q_bounds)

    # student θ_k 用于学习，teacher 用于决策
    theta_k = np.stack([theta_init.copy() for _ in range(n_bins)], axis=0)
    theta_teacher_k = np.stack([theta_init.copy() for _ in range(n_bins)], axis=0) if momentum > 0 else theta_k

    # 在线 ρ_k 估计缓冲（存审计样本的 score 和 label）
    rho_buf_scores = [[] for _ in range(n_bins)]
    rho_buf_labels = [[] for _ in range(n_bins)]
    rho_k          = np.full(n_bins, rho_min)   # 初始保守估计
    rho_update_interval = 20                    # 每 20 次审计重估一次

    B_total   = N * budget_ratio * float(np.mean(costs_eval))
    target_budget_cost = budget_ratio * float(np.mean(costs_eval))

    # ── per-stratum λ_k 初始化 ──
    lam_k = np.full(n_bins, np.nan)
    for kk in range(n_bins):
        idx_k = [t for t in range(min(T0, N)) if get_k(costs_act[t]) == kk]
        if len(idx_k) >= 3:
            scores_k = np.array([np.dot(H[t], theta_k[kk]) for t in idx_k])
            costs_k  = costs_eval[idx_k]
            lam_k[kk] = _init_lambda(scores_k, costs_k, budget_ratio)
        else:
            lam_k[kk] = 0.5
    lam_k[lam_k != lam_k] = 0.5
    if budget_ratio < 0.15:
        lam_k *= 0.5
    lam_k = np.clip(lam_k, 0.05, 5.0)

    lam_floor = 0.1  # 防止 λ_k 降到 0 导致单个 bin 独占预算

    mask        = np.zeros(N, dtype=bool)
    spent       = 0.0
    audit_count = 0

    lam_str = ", ".join([f"{v:.3f}" for v in lam_k])
    print(f"    [MP-AdaptSig] eta_lam={eta_lam:.3f}  λ_k=[{lam_str}]  n_bins={n_bins}")

    for t in range(N):
        tid = task_ids[t]
        k   = get_k(costs_act[t])

        theta_dec = theta_teacher_k[k] if momentum > 0 else theta_k[k]
        score_t = np.dot(H[t], theta_dec)
        utility = score_t - lam_k[k] * costs_act[t]

        do_audit = False if spent + costs_eval[t] > B_total else (utility > 0)

        if do_audit:
            mask[t]  = True
            spent   += costs_eval[t]
            audit_count += 1

            y_t = 1.0 - feedback_fn(tid, noisy=True)

            # ── 更新 ρ_k 缓冲 ──
            rho_buf_scores[k].append(score_t)
            rho_buf_labels[k].append(y_t)

            if audit_count % rho_update_interval == 0:
                for kk in range(n_bins):
                    rho_k[kk] = max(
                        _estimate_rho(
                            np.array(rho_buf_scores[kk]),
                            np.array(rho_buf_labels[kk])
                        ),
                        rho_min
                    )

            # ── 自适应学习率 ──
            max_rho = max(float(np.max(rho_k)), rho_min)
            eta_k   = eta_theta * (rho_k[k] / max_rho)
            eta_t   = eta_k / (1 + 0.002 * t)

            # ── Mirror-Prox 更新 student θ_k ──
            grad1     = _compute_grad_theta(theta_k[k], lam_k[k], H[t], y_t, costs_act[t])
            theta_mid = _eg_update(theta_k[k], grad1, eta_t)
            grad2     = _compute_grad_theta(theta_mid, lam_k[k], H[t], y_t, costs_act[t])
            theta_k[k] = _eg_update(theta_k[k], grad2, eta_t)

            # Teacher 动量跟随 student
            if momentum > 0:
                theta_teacher_k[k] = (momentum * theta_teacher_k[k]
                                      + (1 - momentum) * theta_k[k])
                theta_teacher_k[k] = theta_teacher_k[k] / theta_teacher_k[k].sum()

            lam_k[k] = max(_update_lambda(lam_k[k], eta_lam, costs_eval[t], target_budget_cost), lam_floor)
        else:
            lam_k[k] = max(_update_lambda(lam_k[k], eta_lam, 0.0, target_budget_cost), lam_floor)

        # ── 累积预算追赶（全局） ──
        remaining_budget = B_total - spent
        remaining_steps  = N - t - 1
        if remaining_steps > 0:
            needed_rate = remaining_budget / (
                remaining_steps * float(np.mean(costs_eval[t+1:])) + 1e-9
            )
            if needed_rate > 1.2:
                lam_k = np.maximum(lam_k * 0.90, lam_floor)
            elif needed_rate > 0.9:
                lam_k = np.maximum(lam_k * 0.97, lam_floor)

    # ── 输出诊断 ──
    for k in range(n_bins):
        n_k = sum(1 for t in range(N) if get_k(costs_act[t]) == k)
        th = theta_teacher_k[k] if momentum > 0 else theta_k[k]
        print(f"    [MP-AdaptSig] θ_k[{k}]={th.round(4)}"
              f"  λ_k={lam_k[k]:.4f}  ρ_k={rho_k[k]:.3f}  (n={n_k})")
    print(f"    [MP-AdaptSig] 实际预算消耗: {spent/B_total:.3f} (target=1.000)")
    print(f"    [MP-AdaptSig] audit_count={audit_count}")
    return mask


# ══════════════════════════════════════════════
# Oracle
# ══════════════════════════════════════════════

def oracle_policy(N: int, budget_ratio: float,
                  costs_eval: np.ndarray, y_error: np.ndarray) -> np.ndarray:
    B_total    = N * budget_ratio * float(np.mean(costs_eval))
    error_idx  = np.where(y_error == 1)[0]
    error_by_c = error_idx[np.argsort(costs_eval[error_idx])]
    correct    = np.where(y_error == 0)[0]
    mask  = np.zeros(N, dtype=bool)
    spent = 0.0
    for i in list(error_by_c) + list(correct):
        if spent + costs_eval[i] <= B_total:
            mask[i]  = True
            spent   += costs_eval[i]
    return mask


# ══════════════════════════════════════════════
# 实验性：oracle signal / synthetic signal（不变）
# ══════════════════════════════════════════════

def mirror_prox_oracle_signal(N: int, budget_ratio: float,
                               y_error: np.ndarray,
                               costs_eval: np.ndarray, costs_act: np.ndarray,
                               task_ids: list, feedback_fn,
                               eta_theta: float = 0.5, eta_lam: float = None,
                               T0: int = 50, seed: int = 42) -> np.ndarray:
    if eta_lam is None:
        eta_lam = 0.3 / (budget_ratio + 0.05)

    rng_o    = np.random.default_rng(0)
    H_oracle = np.stack([
        y_error + 0.01 * rng_o.standard_normal(N),
        y_error + 0.01 * rng_o.standard_normal(N),
        y_error + 0.01 * rng_o.standard_normal(N),
    ], axis=1)

    theta    = np.ones(3) / 3
    B_total  = N * budget_ratio * float(np.mean(costs_eval))
    lam      = _init_lambda(H_oracle[:T0] @ theta, costs_eval[:T0], budget_ratio)
    target   = budget_ratio * float(np.mean(costs_eval))
    mask     = np.zeros(N, dtype=bool)
    spent    = 0.0

    for t in range(N):
        score_t = np.dot(H_oracle[t], theta)
        utility = score_t - lam * costs_act[t]
        do_audit = False if spent + costs_eval[t] > B_total else (utility > 0)

        y_t   = y_error[t]
        eta_t = eta_theta / (1 + 0.01 * t)
        grad1 = (y_t - 1/(1+np.exp(-utility))) * H_oracle[t]
        theta_mid = np.maximum(theta * np.exp(eta_t * grad1), 1e-8)
        theta_mid = theta_mid / theta_mid.sum()
        utility_m = np.dot(H_oracle[t], theta_mid) - lam * costs_act[t]
        grad2 = (y_t - 1/(1+np.exp(-utility_m))) * H_oracle[t]
        theta = np.maximum(theta_mid * np.exp(eta_t * grad2), 1e-8)
        theta = theta / theta.sum()

        if do_audit:
            mask[t]  = True
            spent   += costs_eval[t]
            lam = _update_lambda(lam, eta_lam, costs_eval[t], target)
        else:
            lam = _update_lambda(lam, eta_lam * 0.2, 0.0, target)

        remaining_budget = B_total - spent
        remaining_steps  = N - t - 1
        if remaining_steps > 0:
            avg_remain = float(np.mean(costs_eval[t+1:]))
            if remaining_budget > 0.8 * remaining_steps * avg_remain:
                lam = max(lam * 0.95, 0.0)

    print(f"    [MP-Oracle] budget={budget_ratio:.0%}  "
          f"spent={spent/B_total:.3f}  θ={theta.round(4)}  lam={lam:.4f}")
    return mask


def mp_with_synthetic_signal(N: int, budget_ratio: float, target_r: float,
                              y_error: np.ndarray,
                              costs_eval: np.ndarray, costs_act: np.ndarray,
                              task_ids: list, feedback_fn,
                              eta_theta: float = 0.5, eta_lam: float = None,
                              T0: int = 50, seed: int = 42):
    if eta_lam is None:
        eta_lam = 0.3 / (budget_ratio + 0.05)

    def standardize(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    rng_s    = np.random.default_rng(seed)
    y_base   = standardize(y_error.astype(float))
    noise    = standardize(rng_s.standard_normal(N))
    syn      = (target_r * y_base + np.sqrt(max(1 - target_r**2, 0)) * noise
                if abs(target_r) < 1.0 else y_base.copy())
    actual_r = pearsonr(syn, y_error)[0]

    rng_p  = np.random.default_rng(seed + 1)
    H_syn  = np.stack([
        standardize(syn + 0.01 * rng_p.standard_normal(N)),
        standardize(syn + 0.01 * rng_p.standard_normal(N)),
        standardize(syn + 0.01 * rng_p.standard_normal(N)),
    ], axis=1)

    theta   = np.ones(3) / 3
    B_total = N * budget_ratio * float(np.mean(costs_eval))
    lam     = _init_lambda(H_syn[:T0] @ theta, costs_eval[:T0], budget_ratio)
    target  = budget_ratio * float(np.mean(costs_eval))
    mask    = np.zeros(N, dtype=bool)
    spent   = 0.0

    for t in range(N):
        score_t = np.dot(H_syn[t], theta)
        utility = score_t - lam * costs_act[t]
        do_audit = False if spent + costs_eval[t] > B_total else (utility > 0)

        y_t   = y_error[t]
        eta_t = eta_theta / (1 + 0.01 * t)
        grad1 = (y_t - 1/(1+np.exp(-utility))) * H_syn[t]
        theta_mid = np.maximum(theta * np.exp(eta_t * grad1), 1e-8)
        theta_mid = theta_mid / theta_mid.sum()
        utility_m = np.dot(H_syn[t], theta_mid) - lam * costs_act[t]
        grad2 = (y_t - 1/(1+np.exp(-utility_m))) * H_syn[t]
        theta = np.maximum(theta_mid * np.exp(eta_t * grad2), 1e-8)
        theta = theta / theta.sum()

        if do_audit:
            mask[t]  = True
            spent   += costs_eval[t]
            lam = _update_lambda(lam, eta_lam, costs_eval[t], target)
        else:
            lam = _update_lambda(lam, eta_lam * 0.2, 0.0, target)

        remaining_budget = B_total - spent
        remaining_steps  = N - t - 1
        if remaining_steps > 0:
            avg_remain = float(np.mean(costs_eval[t+1:]))
            if remaining_budget > 0.8 * remaining_steps * avg_remain:
                lam = max(lam * 0.95, 0.0)

    return mask, actual_r
