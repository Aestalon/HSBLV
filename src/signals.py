"""
signals.py — θ 预训练、信号正交性检验
"""

import numpy as np
from scipy.stats import pearsonr


# ──────────────────────────────────────────────
# 冷启动预训练
# ──────────────────────────────────────────────
def pretrain_theta(H: np.ndarray, y_error_noisy: np.ndarray,
                   T0: int = 50, temp: float = 2.0) -> np.ndarray:
    """
    用前 T0 个样本拟合 θ：按各信号与噪声标签的正相关系数做 softmax 加权。
    """
    X = H[:T0]
    y = y_error_noisy[:T0]
    D = H.shape[1]
    names = [f"ht{i+1}" for i in range(D)]

    if y.sum() == 0 or y.sum() == T0:
        print("  [pretrain] 冷启动期全对或全错，使用等权 θ")
        return np.ones(D) / D

    corrs = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(D)])
    corrs_clipped = np.maximum(corrs, 0)

    if corrs_clipped.sum() < 1e-9:
        print("  [pretrain] 全负相关，使用等权 θ")
        return np.ones(D) / D

    exp_corrs = np.exp(corrs_clipped * temp)
    theta = exp_corrs / exp_corrs.sum()

    theta_str = ", ".join([f"{v:.3f}" for v in theta])
    print(f"  [pretrain] θ = [{theta_str}]  ({'+'.join(names)})")
    print(f"  [pretrain] 冷启动期错误率(noisy): {y.mean():.3f}")
    print(f"  [pretrain] 各信号与错误的相关系数: {[f'{c:.3f}' for c in corrs]}")
    return theta


# ──────────────────────────────────────────────
# 正交性验证
# ──────────────────────────────────────────────
def check_cost_signal_orthogonality(H: np.ndarray, scores: np.ndarray,
                                    y_error: np.ndarray,
                                    costs_eval: np.ndarray,
                                    costs_act: np.ndarray) -> None:
    """打印 score / 各信号 与 cost 的 Pearson 相关，验证正交假设."""
    print("\n=== Cost-Signal 正交性验证 ===")
    for label, costs in [("costs_eval(output_len)", costs_eval),
                          ("costs_act(input_len)",   costs_act)]:
        r, p = pearsonr(scores, costs)
        flag = "[OK] orthogonal" if abs(r) < 0.2 else "[WARN] correlated"
        print(f"  score vs {label}: r={r:.3f}, p={p:.4f}  {flag}")

        for i, name in enumerate([f"ht{j+1}" for j in range(H.shape[1])]):
            r2, p2 = pearsonr(H[:, i], costs)
            print(f"    {name} vs {label}: r={r2:.3f}, p={p2:.4f}")

        r3, p3 = pearsonr(costs, y_error)
        print(f"  {label} vs y_error: r={r3:.3f}, p={p3:.4f}")
        print(f"  {label} 均值: 错误题={costs[y_error==1].mean():.3f}, "
              f"正确题={costs[y_error==0].mean():.3f}")


# ──────────────────────────────────────────────
# 信号诊断
# ──────────────────────────────────────────────
def signal_correlation_report(H: np.ndarray, theta: np.ndarray,
                               y_error: np.ndarray, y_error_noisy: np.ndarray,
                               extra_signals: dict[str, np.ndarray]) -> None:
    """打印各信号与 y_error 的相关系数，以及 noisy feedback 翻转统计。"""
    phi = H @ theta
    N   = len(y_error)

    print("\n=== 信号与错误的相关性检查 ===")
    print(f"φ(ht) 均值: 错误题={phi[y_error==1].mean():.4f}, "
          f"正确题={phi[y_error==0].mean():.4f}")

    for name, sig in {**extra_signals, "φ(ht)": phi}.items():
        r, p = pearsonr(sig, y_error)
        print(f"  {name} vs y_error: r={r:.4f}, p={p:.4f}")

    flip_actual = int(np.sum(y_error != y_error_noisy))
    from config import FLIP_PROB
    print(f"\nNoisy feedback: {flip_actual}/{N} 条翻转 "
          f"({flip_actual/N:.1%}，目标={FLIP_PROB:.1%})")