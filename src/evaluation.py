"""
evaluation.py — 审计策略评估指标
"""

import numpy as np
from scipy import stats


def bootstrap_ci(values: list, ci: float = 0.95) -> dict:
    """Compute mean, std, and confidence interval from a list of metric values."""
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        return {"mean": float(arr[0]) if n == 1 else np.nan,
                "std": 0.0, "ci_lower": float(arr[0]) if n == 1 else np.nan,
                "ci_upper": float(arr[0]) if n == 1 else np.nan, "n": n}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n)
    alpha = 1 - ci
    t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return {"mean": mean, "std": std,
            "ci_lower": mean - t_val * se,
            "ci_upper": mean + t_val * se, "n": n}


def aggregate_multi_seed(all_seeds: list, ci: float = 0.95) -> dict:
    """Aggregate results from multiple seeds into mean ± CI.

    all_seeds: list of result dicts, each from one seed run.
    Returns a dict with the same structure but values replaced by CI dicts.
    """
    if len(all_seeds) < 2:
        return all_seeds[0] if all_seeds else {}

    def _collect(path_parts, data, collector):
        """Recursively collect scalar values at leaf nodes."""
        if isinstance(data, dict):
            for k, v in data.items():
                _collect(path_parts + [k], v, collector)
        else:
            collector.setdefault(tuple(path_parts), []).append(data)

    collector: dict = {}
    for seed_result in all_seeds:
        _collect([], seed_result, collector)

    result: dict = {}
    for path, vals in collector.items():
        ci_info = bootstrap_ci(vals, ci=ci)
        d = result
        for part in path[:-1]:
            d = d.setdefault(part, {})
        d[path[-1]] = ci_info

    return result


def evaluate_hit_rate(audit_mask: np.ndarray,
                      y_error: np.ndarray) -> dict:
    """
    命中率、错误覆盖率、信息增益。

    Returns
    -------
    dict with keys: hit_rate, error_coverage, info_gain, random_hit_rate
    """
    B = int(np.sum(audit_mask))
    random_hit_rate = float(np.mean(y_error))

    if B == 0:
        return {"hit_rate": 0.0, "error_coverage": 0.0,
                "info_gain": 0.0, "random_hit_rate": random_hit_rate}

    hits           = int(np.sum(audit_mask & (y_error == 1)))
    hit_rate       = hits / B
    total_errors   = int(np.sum(y_error == 1))
    error_coverage = hits / (total_errors + 1e-9)
    info_gain      = (hit_rate - random_hit_rate) / (random_hit_rate + 1e-9)

    return {
        "hit_rate":        float(hit_rate),
        "error_coverage":  float(error_coverage),
        "info_gain":       float(info_gain),
        "random_hit_rate": random_hit_rate,
    }


def evaluate_corrected_accuracy(audit_mask: np.ndarray,
                                 y_passed: np.ndarray) -> float:
    """
    假设审计到的错误全部被纠正后的正确率。
    """
    not_audited_correct = int(np.sum(~audit_mask & (y_passed == 1)))
    audited_correct     = int(np.sum( audit_mask & (y_passed == 1)))
    audited_fixed       = int(np.sum( audit_mask & (y_passed == 0)))
    return float((not_audited_correct + audited_correct + audited_fixed) / len(y_passed))


def quartile_hit_rates(mask: np.ndarray, cost_arr: np.ndarray,
                       y_error: np.ndarray, label: str = "") -> dict:
    """
    按 cost 四分位统计：每组的 error_rate / audit_rate / hit_rate。
    """
    q_boundaries = np.percentile(cost_arr, [0, 25, 50, 75, 100])
    results_q: dict = {}

    if label:
        print(f"\n  [{label}]  cost 四分位分析")
        print(f"  {'Quartile':>10} | {'cost range':>16} | "
              f"{'N':>5} | {'error_rate':>11} | "
              f"{'audit_rate':>11} | {'hit_rate':>9}")
        print("  " + "-" * 75)

    for q in range(4):
        lo, hi = q_boundaries[q], q_boundaries[q + 1]
        in_q = ((cost_arr >= lo) & (cost_arr <= hi)
                if q == 3 else (cost_arr >= lo) & (cost_arr < hi))

        n_q        = int(np.sum(in_q))
        err_rate   = float(np.mean(y_error[in_q])) if n_q > 0 else float("nan")
        audit_rate = float(np.mean(mask[in_q]))    if n_q > 0 else float("nan")
        hit_rate   = float(np.sum(mask & in_q & (y_error == 1)) /
                           (np.sum(mask & in_q) + 1e-9))

        results_q[f"Q{q+1}"] = {
            "cost_lo": float(lo), "cost_hi": float(hi),
            "n": n_q, "error_rate": err_rate,
            "audit_rate": audit_rate, "hit_rate": hit_rate,
        }

        if label:
            print(f"  {'Q'+str(q+1):>10} | [{lo:.3f}, {hi:.3f}] | "
                  f"{n_q:>5} | {err_rate:>11.4f} | "
                  f"{audit_rate:>11.4f} | {hit_rate:>9.4f}")

    return results_q


def audit_concentration(mask: np.ndarray, cost_arr: np.ndarray,
                         label: str = "") -> None:
    """对比审计 vs 未审计样本的 cost 分布，以及两者的相关性。"""
    from scipy.stats import pearsonr

    audited   = cost_arr[mask]
    unaudited = cost_arr[~mask]

    print(f"\n  [{label}]  审计样本 vs 未审计样本的 cost 分布")
    print(f"    audited   mean={audited.mean():.4f}  "
          f"median={np.median(audited):.4f}  std={audited.std():.4f}")
    print(f"    unaudited mean={unaudited.mean():.4f}  "
          f"median={np.median(unaudited):.4f}  std={unaudited.std():.4f}")

    r, p = pearsonr(cost_arr.astype(float), mask.astype(float))
    print(f"    corr(cost, audited): r={r:.4f}, p={p:.4f}")


def correlation_by_quartile(cost_arr: np.ndarray, y_error: np.ndarray,
                             score_arr: np.ndarray, label: str = "") -> None:
    """在每个 cost 四分位组内分别计算 score-error / cost-error 相关性。"""
    from scipy.stats import pearsonr

    q_boundaries = np.percentile(cost_arr, [0, 25, 50, 75, 100])
    print(f"\n  [{label}]  各四分位组的 score-error / cost-error 相关性")
    print(f"  {'Quartile':>10} | {'r(score,err)':>13} | {'r(cost,err)':>12} | "
          f"{'p(score)':>10} | {'p(cost)':>9}")
    print("  " + "-" * 65)

    for q in range(4):
        lo, hi = q_boundaries[q], q_boundaries[q + 1]
        in_q = ((cost_arr >= lo) & (cost_arr <= hi)
                if q == 3 else (cost_arr >= lo) & (cost_arr < hi))
        if np.sum(in_q) < 10:
            continue
        r_se, p_se = pearsonr(score_arr[in_q], y_error[in_q])
        r_ce, p_ce = pearsonr(cost_arr[in_q],  y_error[in_q])
        print(f"  {'Q'+str(q+1):>10} | {r_se:>13.4f} | {r_ce:>12.4f} | "
              f"{p_se:>10.4f} | {p_ce:>9.4f}")