"""
analysis.py — 消融实验与诊断分析

所有函数通过参数接收数据，不依赖全局变量。
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import pearsonr

from memory import MemoryBank
from policies import (
    baseline_random, baseline_threshold_online,
    baseline_threshold_stratified, mirror_prox_adaptive_lambda,
    mirror_prox_oracle_signal, mp_with_synthetic_signal,
    oracle_policy, _init_lambda, _update_lambda, _eg_update,
)
from evaluation import evaluate_hit_rate, quartile_hit_rates, audit_concentration


# ══════════════════════════════════════════════
# Mismatch sensitivity sweep
# ══════════════════════════════════════════════

def run_mismatch_sensitivity(N, H, ut, y_error, out_n,
                              costs_hetero, budget_ratios=(0.05, 0.20, 0.50)):
    print("\n=== Threshold sensitivity to cost mismatch ===")
    print(f"{'alpha':>8} | {'corr(proxy,true)':>16} | "
          f"{'hr_5%':>8} | {'hr_20%':>8} | {'hr_50%':>8}")
    print("-" * 60)

    rng_sweep   = np.random.default_rng(99)
    noise_sweep = rng_sweep.random(N)

    def normalize(x):
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    for alpha in [1.0, 0.75, 0.5, 0.25, 0.0, -0.5]:
        proxy = alpha * out_n + (1 - abs(alpha)) * normalize(noise_sweep)
        if alpha < 0:
            proxy = normalize(-out_n + 0.5 * noise_sweep)
        proxy_cost = 0.1 + 0.9 * normalize(proxy)
        corr = pearsonr(proxy_cost, costs_hetero)[0]

        rows = {}
        for br in budget_ratios:
            m = baseline_threshold_online(N, br, ut, costs_hetero, proxy_cost)
            rows[br] = evaluate_hit_rate(m, y_error)["hit_rate"]

        print(f"{alpha:>8.2f} | {corr:>16.3f} | "
              f"{rows[0.05]:>8.4f} | {rows[0.20]:>8.4f} | {rows[0.50]:>8.4f}")


# ══════════════════════════════════════════════
# Memory ablation
# ══════════════════════════════════════════════

def run_memory_ablation(N, H, ut, y_error,
                         costs_hetero, costs_decision,
                         task_ids, feedback_fn,
                         budget_ratio: float = 0.20):
    print("\n" + "=" * 80)
    print("MEMORY ABLATION EXPERIMENT  (budget=20%, heterogeneous cost)")
    print("=" * 80)

    m_cst_base = baseline_threshold_stratified(
        N, budget_ratio, ut, H, costs_hetero, costs_decision,
        task_ids, feedback_fn, use_memory=False)
    hr_cst = evaluate_hit_rate(m_cst_base, y_error)["hit_rate"]

    # ── 消融1：memory_weight 敏感性 ──
    print("\n--- 消融1：memory_weight 敏感性（min_confidence=0.0, top_k=5）---")
    print(f"  {'weight':>8} | {'CST+Mem HR':>12} | {'vs CST':>8}")
    print("  " + "-" * 36)
    for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
        m = baseline_threshold_stratified(
            N, budget_ratio, ut, H, costs_hetero, costs_decision,
            task_ids, feedback_fn, use_memory=True, memory_weight=w, top_k=5)
        hr = evaluate_hit_rate(m, y_error)["hit_rate"]
        print(f"  {w:>8.1f} | {hr:>12.4f} | {hr-hr_cst:>+8.4f}")

    # ── 消融2：分层分位数门控 ──
    print("\n--- 消融2：分层分位数门控效果验证 ---")
    print(f"  {'mode':>20} | {'CST+Mem HR':>12} | {'vs CST':>8}")
    print("  " + "-" * 44)
    for label, mk in [("stratified-gate(default)", 20), ("no-gate(min_size=0)", 0)]:
        m = baseline_threshold_stratified(
            N, budget_ratio, ut, H, costs_hetero, costs_decision,
            task_ids, feedback_fn, use_memory=True, memory_weight=0.3,
            top_k=5, min_size=mk)
        hr = evaluate_hit_rate(m, y_error)["hit_rate"]
        print(f"  {label:>20} | {hr:>12.4f} | {hr-hr_cst:>+8.4f}")

    # ── 消融3：top_k 敏感性 ──
    print("\n--- 消融3：top_k 敏感性（memory_weight=0.3）---")
    print(f"  {'top_k':>8} | {'CST+Mem HR':>12} | {'vs CST':>8}")
    print("  " + "-" * 36)
    for k in [1, 3, 5, 10, 20]:
        m = baseline_threshold_stratified(
            N, budget_ratio, ut, H, costs_hetero, costs_decision,
            task_ids, feedback_fn, use_memory=True, memory_weight=0.3, top_k=k)
        hr = evaluate_hit_rate(m, y_error)["hit_rate"]
        print(f"  {k:>8d} | {hr:>12.4f} | {hr-hr_cst:>+8.4f}")

    print(f"\n  基准 CST (no memory) HR = {hr_cst:.4f}")


# ══════════════════════════════════════════════
# Memory signal analysis
# ══════════════════════════════════════════════

def run_memory_signal_analysis(N, H, theta_init, y_error,
                                costs_hetero, costs_decision):
    print("\n" + "=" * 80)
    print("MEMORY SIGNAL ANALYSIS  (h_t^(4) correlation with y_error)")
    print("=" * 80)

    T_split = int(N * 0.7)
    mem     = MemoryBank(top_k=5, min_size=5)

    q_bounds_mem = np.percentile(costs_hetero[:T_split], [0, 25, 50, 75, 100])

    def get_k_mem(c):
        for k in range(3):
            if c < q_bounds_mem[k + 1]:
                return k
        return 3

    for t in range(T_split):
        k = get_k_mem(costs_hetero[t])
        mem.write(H[t], y_error[t], costs_hetero[t], 1.0, k)

    q_bounds    = np.percentile(costs_decision, [0, 25, 50, 75, 100])
    homogeneous = np.all(np.diff(q_bounds) < 1e-6)

    def get_k(c):
        if homogeneous:
            return 0
        for k in range(3):
            if c < q_bounds[k + 1]:
                return k
        return 3

    test_indices = list(range(T_split, N))
    h4_vals = []
    for t in test_indices:
        k  = get_k(costs_decision[t])
        h4 = mem.retrieve(H[t], cost_stratum=k, q_bounds=q_bounds)
        h4_vals.append(h4 if h4 is not None else y_error[:T_split].mean())

    h4_arr   = np.array(h4_vals)
    y_test   = y_error[test_indices]
    phi_test = (H @ theta_init)[test_indices]

    r_phi, p_phi = pearsonr(phi_test, y_test)
    r_h4,  p_h4  = pearsonr(h4_arr,  y_test)
    phi_aug      = 0.7 * phi_test + 0.3 * h4_arr
    r_aug, p_aug = pearsonr(phi_aug, y_test)

    print(f"\n  测试集: t={T_split}~{N-1}  (N={len(test_indices)})")
    print(f"  原始信号 φ(h_t):    r={r_phi:.4f}, p={p_phi:.4f}")
    print(f"  记忆信号 h_t^(4):   r={r_h4:.4f},  p={p_h4:.4f}")
    print(f"  融合信号 (0.7+0.3): r={r_aug:.4f}, p={p_aug:.4f}")

    print(f"\n  分层相关性（按 ex-post cost 四分位）:")
    q_eval = np.percentile(costs_hetero, [0, 25, 50, 75, 100])
    print(f"  {'Quartile':>10} | {'r(φ,err)':>10} | {'r(h4,err)':>10} | {'r(aug,err)':>11}")
    print("  " + "-" * 50)
    for q in range(4):
        lo, hi = q_eval[q], q_eval[q + 1]
        in_q = np.array([
            (costs_hetero[t] >= lo and
             (costs_hetero[t] <= hi if q == 3 else costs_hetero[t] < hi))
            for t in test_indices
        ])
        if in_q.sum() < 5:
            continue
        r1 = pearsonr(phi_test[in_q], y_test[in_q])[0]
        r2 = pearsonr(h4_arr[in_q],   y_test[in_q])[0]
        r3 = pearsonr(phi_aug[in_q],  y_test[in_q])[0]
        print(f"  {'Q'+str(q+1):>10} | {r1:>10.4f} | {r2:>10.4f} | {r3:>11.4f}")

    return r_h4, r_aug


# ══════════════════════════════════════════════
# Stratification ablation
# ══════════════════════════════════════════════

def run_stratification_ablation(N, H, ut, y_error,
                                  costs_hetero, costs_decision,
                                  task_ids, feedback_fn,
                                  budget_ratio: float = 0.20):
    print("\n" + "=" * 80)
    print("STRATIFICATION ABLATION  (budget=20%, heterogeneous cost)")
    print("=" * 80)

    B_total = N * budget_ratio * float(np.mean(costs_hetero))

    m_th  = baseline_threshold_online(N, budget_ratio, ut, costs_hetero, costs_decision)
    hr_th = evaluate_hit_rate(m_th, y_error)["hit_rate"]

    m_cst  = baseline_threshold_stratified(
        N, budget_ratio, ut, H, costs_hetero, costs_decision,
        task_ids, feedback_fn, use_memory=False)
    hr_cst = evaluate_hit_rate(m_cst, y_error)["hit_rate"]

    # 随机分层
    rng_strat = np.random.default_rng(123)
    random_strata = rng_strat.integers(0, 4, size=N)
    hist_rand = [[] for _ in range(4)]
    mask_rand_strat = np.zeros(N, dtype=bool)
    spent = 0.0
    for t in range(N):
        k       = random_strata[t]
        ratio_t = ut[t] / (costs_decision[t] + 1e-9)
        if len(hist_rand[k]) > 10 and spent + costs_hetero[t] <= B_total:
            thr = np.quantile(hist_rand[k], 1 - budget_ratio)
            if ratio_t > thr:
                mask_rand_strat[t] = True
                spent += costs_hetero[t]
        hist_rand[k].append(ratio_t)
    hr_rand_strat = evaluate_hit_rate(mask_rand_strat, y_error)["hit_rate"]

    # 按信号强度分层
    signal_q = np.percentile(ut, [0, 25, 50, 75, 100])
    def get_signal_k(s):
        for k in range(3):
            if s < signal_q[k + 1]:
                return k
        return 3
    hist_sig = [[] for _ in range(4)]
    mask_sig_strat = np.zeros(N, dtype=bool)
    spent = 0.0
    for t in range(N):
        k       = get_signal_k(ut[t])
        ratio_t = ut[t] / (costs_decision[t] + 1e-9)
        if len(hist_sig[k]) > 10 and spent + costs_hetero[t] <= B_total:
            thr = np.quantile(hist_sig[k], 1 - budget_ratio)
            if ratio_t > thr:
                mask_sig_strat[t] = True
                spent += costs_hetero[t]
        hist_sig[k].append(ratio_t)
    hr_sig_strat = evaluate_hit_rate(mask_sig_strat, y_error)["hit_rate"]

    # Oracle 分层
    q_bounds = np.percentile(costs_decision, [0, 25, 50, 75, 100])
    mask_ora_strat = np.zeros(N, dtype=bool)
    for q in range(4):
        lo, hi = q_bounds[q], q_bounds[q + 1]
        in_q = ((costs_decision >= lo) & (costs_decision <= hi)
                if q == 3 else (costs_decision >= lo) & (costs_decision < hi))
        idx_q  = np.where(in_q)[0]
        B_q    = len(idx_q) * budget_ratio * float(np.mean(costs_hetero[idx_q]))
        err_q  = idx_q[y_error[idx_q] == 1]
        err_sorted = err_q[np.argsort(costs_hetero[err_q])]
        cor_q  = idx_q[y_error[idx_q] == 0]
        spent_q = 0.0
        for i in list(err_sorted) + list(cor_q):
            if spent_q + costs_hetero[i] <= B_q:
                mask_ora_strat[i] = True
                spent_q += costs_hetero[i]
    hr_ora_strat = evaluate_hit_rate(mask_ora_strat, y_error)["hit_rate"]

    m_ora  = oracle_policy(N, budget_ratio, costs_hetero, y_error)
    hr_ora = evaluate_hit_rate(m_ora, y_error)["hit_rate"]

    print(f"\n{'分层准则':>25} | {'Hit Rate':>10} | {'vs Threshold':>13}")
    print("-" * 55)
    for label, hr in [
        ("全局（无分层）",   hr_th),
        ("随机分层",         hr_rand_strat),
        ("按信号强度分层",   hr_sig_strat),
        ("按cost分层 (CST)", hr_cst),
        ("Oracle分层",       hr_ora_strat),
    ]:
        diff = f"{hr-hr_th:>+13.4f}" if label != "全局（无分层）" else f"{'---':>13}"
        print(f"  {label:>23} | {hr:>10.4f} | {diff}")

    print(f"\n{'策略':>20} | {'Hit Rate':>10} | {'vs 全局Oracle':>14}")
    print("-" * 50)
    for label, hr in [
        ("全局Oracle", hr_ora),
        ("Oracle分层", hr_ora_strat),
        ("CST",        hr_cst),
        ("Threshold",  hr_th),
    ]:
        diff = f"{'---':>14}" if label == "全局Oracle" else f"{hr-hr_ora:>+14.4f}"
        print(f"  {label:>18} | {hr:>10.4f} | {diff}")

    print(f"\n  CST 达到 Oracle 分层上界的: {hr_cst/hr_ora_strat*100:.1f}%")


# ══════════════════════════════════════════════
# Clean feedback ablation
# ══════════════════════════════════════════════

def run_clean_feedback_ablation(N, H, theta_init, y_error,
                                  costs_hetero, costs_decision,
                                  task_ids, oracle_fb,
                                  T0: int = 50,
                                  noisy_ref: float = 0.6726,
                                  cst_ref: float = 0.8454):
    """
    oracle_fb: FeedbackOracle 的 clean 版本（flip=0）
    """
    print("\n" + "=" * 60)
    print("CLEAN FEEDBACK (flip=0) with REAL signals")
    print("=" * 60)

    m_mpa = mirror_prox_adaptive_lambda(
        N, 0.20, H, theta_init, costs_hetero, costs_decision,
        task_ids, oracle_fb, T0=T0)
    hr = evaluate_hit_rate(m_mpa, y_error)["hit_rate"]
    print(f"MP-Adapt (clean feedback): HR={hr:.4f}")
    print(f"vs noisy MP-Adapt ({noisy_ref}): diff={hr - noisy_ref:>+.4f}")
    print(f"vs CST ({cst_ref}): diff={hr - cst_ref:>+.4f}")


# ══════════════════════════════════════════════
# Quartile analysis
# ══════════════════════════════════════════════

def run_quartile_analysis(N, H, ut, y_error, y_passed,
                           costs_hetero, costs_decision,
                           task_ids, feedback_fn, theta_init, T0):
    from evaluation import correlation_by_quartile

    print("\n" + "=" * 80)
    print("COST QUARTILE STRATIFIED ANALYSIS")
    print("=" * 80)

    # 1. 基线 error_rate
    print("\n=== 1. cost 四分位的 error_rate ===")
    q_bounds = np.percentile(costs_hetero, [0, 25, 50, 75, 100])
    print(f"  cost 分位点: {[f'{v:.3f}' for v in q_bounds]}")
    print(f"  {'Quartile':>10} | {'cost range':>16} | {'N':>5} | {'error_rate':>11}")
    print("  " + "-" * 52)
    for q in range(4):
        lo, hi = q_bounds[q], q_bounds[q + 1]
        in_q = ((costs_hetero >= lo) & (costs_hetero <= hi)
                if q == 3 else (costs_hetero >= lo) & (costs_hetero < hi))
        print(f"  {'Q'+str(q+1):>10} | [{lo:.3f}, {hi:.3f}] | "
              f"{np.sum(in_q):>5} | {y_error[in_q].mean():>11.4f}")

    overall_r, overall_p = pearsonr(costs_hetero, y_error)
    print(f"\n  overall corr(cost_eval, y_error): r={overall_r:.4f}, p={overall_p:.4f}")

    # 2. budget=20% 各策略
    br = 0.20
    print(f"\n=== 2. budget={br:.0%}，各策略按 cost 四分位的 hit_rate ===")

    m_rand   = baseline_random(N, br, costs_hetero)
    m_th     = baseline_threshold_online(N, br, ut, costs_hetero, costs_decision)
    m_cst    = baseline_threshold_stratified(
        N, br, ut, H, costs_hetero, costs_decision,
        task_ids, feedback_fn, use_memory=False)
    m_cst_mg = baseline_threshold_stratified(
        N, br, ut, H, costs_hetero, costs_decision,
        task_ids, feedback_fn, use_memory=True, memory_weight=0.3)
    m_mpa    = mirror_prox_adaptive_lambda(
        N, br, H, theta_init, costs_hetero, costs_decision,
        task_ids, feedback_fn, T0=T0)

    for name, mask in [("random",    m_rand),
                        ("threshold", m_th),
                        ("cst",       m_cst),
                        ("cst+mem",   m_cst_mg),
                        ("mp_adapt",  m_mpa)]:
        quartile_hit_rates(mask, costs_hetero, y_error, label=name)
        audit_concentration(mask, costs_hetero, label=name)

    # 3. score-error 相关性异质性检验
    print(f"\n=== 3. score-error 相关性的 cost 异质性检验 ===")
    correlation_by_quartile(costs_hetero, y_error, ut,
                             label="score vs error by cost quartile")

    # 4. threshold 偏向检验
    print(f"\n=== 4. threshold 选择偏向检验 ===")
    th_cost   = costs_hetero[m_th]
    rand_cost = costs_hetero[m_rand]
    print(f"  threshold audited cost mean: {th_cost.mean():.4f}")
    print(f"  random    audited cost mean: {rand_cost.mean():.4f}")
    print(f"  差值（threshold - random）: {th_cost.mean()-rand_cost.mean():+.4f}")

    th_top   = int(np.sum(m_th   & (costs_hetero >= q_bounds[3])))
    rand_top = int(np.sum(m_rand & (costs_hetero >= q_bounds[3])))
    print(f"\n  top-quartile(Q4) 被审计数:")
    print(f"    threshold: {th_top}  ({th_top/(np.sum(m_th)+1e-9):.2%} of audited)")
    print(f"    random:    {rand_top}  ({rand_top/(np.sum(m_rand)+1e-9):.2%} of audited)")

    # 5. 跨 budget 四分位趋势
    print(f"\n=== 5. 跨 budget 的四分位 hit_rate 趋势 ===")
    print(f"  {'Budget':>8} | {'Q1_th':>7} | {'Q2_th':>7} | "
          f"{'Q3_th':>7} | {'Q4_th':>7} | {'Q4_rand':>8}")
    print("  " + "-" * 60)
    for b in [0.05, 0.10, 0.20, 0.30, 0.50]:
        m_th_b   = baseline_threshold_online(N, b, ut, costs_hetero, costs_decision)
        m_rand_b = baseline_random(N, b, costs_hetero)
        qr_th    = quartile_hit_rates(m_th_b,   costs_hetero, y_error)
        qr_rand  = quartile_hit_rates(m_rand_b, costs_hetero, y_error)
        q_hr     = [qr_th.get(f"Q{q+1}", {}).get("hit_rate", float("nan"))
                    for q in range(4)]
        q4_rand  = qr_rand.get("Q4", {}).get("hit_rate", float("nan"))
        print(f"  {b:>8.0%} | {q_hr[0]:>7.4f} | {q_hr[1]:>7.4f} | "
              f"{q_hr[2]:>7.4f} | {q_hr[3]:>7.4f} | {q4_rand:>8.4f}")

    # 6. 结论
    print(f"\n=== 6. 诊断结论 ===")
    print(f"  cost-error corr  r={overall_r:.3f}")
    print(f"  threshold 相比 random 偏向高cost: "
          f"{th_cost.mean()-rand_cost.mean():+.4f}")


# ══════════════════════════════════════════════
# Oracle signal experiment
# ══════════════════════════════════════════════

def run_oracle_signal_experiment(N, H, ut, y_error,
                                   costs_hetero, costs_decision,
                                   task_ids, feedback_fn, theta_init, T0,
                                   budget_ratios=(0.05, 0.10, 0.20, 0.30, 0.50)):
    print("\n" + "=" * 80)
    print("ORACLE SIGNAL EXPERIMENT")
    print("=" * 80)

    print(f"\n{'Budget':>8} | {'Random':>8} | {'Threshold':>10} | "
          f"{'MP-Adapt':>9} | {'MP-Oracle':>10} | {'Oracle-Policy':>13}")
    print("-" * 70)

    for br in budget_ratios:
        m_rand       = baseline_random(N, br, costs_hetero)
        m_th         = baseline_threshold_online(N, br, ut, costs_hetero, costs_decision)
        m_mpa        = mirror_prox_adaptive_lambda(
            N, br, H, theta_init, costs_hetero, costs_decision,
            task_ids, feedback_fn, T0=T0)
        m_oracle_sig = mirror_prox_oracle_signal(
            N, br, y_error, costs_hetero, costs_decision,
            task_ids, feedback_fn, T0=T0)
        m_ora        = oracle_policy(N, br, costs_hetero, y_error)

        print(f"{br:>8.0%} | "
              f"{evaluate_hit_rate(m_rand,       y_error)['hit_rate']:>8.4f} | "
              f"{evaluate_hit_rate(m_th,         y_error)['hit_rate']:>10.4f} | "
              f"{evaluate_hit_rate(m_mpa,        y_error)['hit_rate']:>9.4f} | "
              f"{evaluate_hit_rate(m_oracle_sig, y_error)['hit_rate']:>10.4f} | "
              f"{evaluate_hit_rate(m_ora,        y_error)['hit_rate']:>13.4f}")


# ══════════════════════════════════════════════
# Signal strength sweep
# ══════════════════════════════════════════════

def run_signal_strength_sweep(N, H, ut, y_error,
                               costs_hetero, costs_decision,
                               task_ids, feedback_fn, T0,
                               target_rs=None,
                               budget_ratios=(0.10, 0.20, 0.30)):
    if target_rs is None:
        target_rs = [0.0, 0.1, 0.2, 0.29, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    print("\n" + "=" * 80)
    print("SIGNAL STRENGTH SWEEP")
    print("=" * 80)

    for br in budget_ratios:
        m_th   = baseline_threshold_online(N, br, ut, costs_hetero, costs_decision)
        m_rand = baseline_random(N, br, costs_hetero)
        hr_th   = evaluate_hit_rate(m_th,   y_error)["hit_rate"]
        hr_rand = evaluate_hit_rate(m_rand, y_error)["hit_rate"]

        print(f"\n  budget={br:.0%}  threshold={hr_th:.4f}  random={hr_rand:.4f}")
        print(f"  {'target_r':>9} | {'actual_r':>9} | {'MP-Syn':>8} | "
              f"{'vs threshold':>13} | {'vs random':>10}")
        print("  " + "-" * 58)

        for tr in target_rs:
            mask_syn, actual_r = mp_with_synthetic_signal(
                N, br, tr, y_error, costs_hetero, costs_decision,
                task_ids, feedback_fn, T0=T0)
            hr_syn    = evaluate_hit_rate(mask_syn, y_error)["hit_rate"]
            diff_th   = hr_syn - hr_th
            diff_rand = hr_syn - hr_rand
            marker    = " <--" if diff_th > 0.02 else ""
            print(f"  {tr:>9.2f} | {actual_r:>9.4f} | {hr_syn:>8.4f} | "
                  f"{diff_th:>+13.4f} | {diff_rand:>+10.4f}{marker}")

    print("\n  <--标记表示 MP 比 threshold 高出 >2%，即开始有实质性优势")