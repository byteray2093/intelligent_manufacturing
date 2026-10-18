"""奇异点与可操作度分析。

概念与公式（报告中必须与本节一致）：

对几何 Jacobian J(q) ∈ R^{6×6}（v = J q̇）做奇异值分解 J = U Σ Vᵀ，
σ1 ≥ σ2 ≥ ... ≥ σ6 ≥ 0。定义：

- 条件数   cond(J) = σ1 / σ6      （σ6 → 0 时发散，奇异位形处无穷大）
- 数值条件 κ(J)   = 1 / σ6         （工程上更常用，避免除零，越大越接近奇异）
- 可操作度 w(J)   = σ1 σ2 ... σ6 = |det J| = sqrt(det(J Jᵀ))
                    （Yoshikawa 1985，单位随行单位混合，只用于同一机器人的相对比较）
- 各向同性 iso   = σ6 / σ1 ∈ (0, 1]（1 表示全方向运动能力一致）

物理意义：σ6 小 => 某些末端运动方向需要极大的关节速度（|q̇| = |J⁻¹ v| ~ |v|/σ6），
数值 IK 的迭代矩阵病态、收敛变慢或发散。因此"数值 IK 失败"与"数学不可达"必须分开：
- 数学不可达：目标点超出最大伸展、或姿态根本无解（与算法无关）
- 数值失败：目标可达，但 σ6 很小 / 初值差 / 迭代步长限制导致未收敛

另有一个常见的"腕部奇异"：UR 系列在 q5 = 0 时第 4、6 轴共线，J 秩下降（det → 0）。
用参考模型默认零位 q = 0 即可复现该现象，适合作为课堂示例。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from src.analysis.linalg import geometric_jacobian, DEFAULT_EPS


@dataclass
class SingularityMetrics:
    """单个构型的奇异点指标。"""

    sigma_min: float
    sigma_max: float
    condition_number: float          # inf 表示数值上已奇异
    numerical_condition: float       # 1 / sigma_min
    manipulability_yoshikawa: float  # prod(sigma)
    isotropy: float                  # sigma_min / sigma_max
    rank: int
    is_singular: bool                # sigma_min < 阈值
    sigma_min_threshold: float

    def to_row(self) -> dict:
        row = asdict(self)
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = np.inf
        return row


def singular_values(J: np.ndarray) -> np.ndarray:
    """J 的奇异值，降序。要求 J 为 (6,6)。"""
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError("J must be a 2D matrix")
    return np.linalg.svd(J, compute_uv=False)


def singularity_metrics(J: np.ndarray, sigma_min_threshold: float = 1e-3) -> SingularityMetrics:
    """由 Jacobian 计算奇异点指标。

    sigma_min_threshold：判定"奇异"的阈值，默认 1e-3（无量纲，因为 J 的混合单位下
    σ 的量纲不统一，该阈值是**约定值**，报告中必须写明并保持全局一致）。
    """
    J = np.asarray(J, dtype=np.float64)
    if J.shape != (6, 6):
        raise ValueError("J must be (6, 6) for this analysis")
    s = singular_values(J)
    s_min = float(s[-1])
    s_max = float(s[0])
    if s_min <= 1e-12:
        cond = float("inf")
        num_cond = float("inf")
    else:
        cond = s_max / s_min
        num_cond = 1.0 / s_min
    # 数值秩：相对阈值取 max(M,N) * eps * s_max
    rank = int(np.sum(s > max(J.shape) * np.finfo(float).eps * max(s_max, 1e-300)))
    return SingularityMetrics(
        sigma_min=s_min,
        sigma_max=s_max,
        condition_number=cond,
        numerical_condition=num_cond,
        manipulability_yoshikawa=float(np.prod(s)),
        isotropy=float(s_min / s_max) if s_max > 0 else 0.0,
        rank=rank,
        is_singular=bool(s_min < sigma_min_threshold),
        sigma_min_threshold=float(sigma_min_threshold),
    )


def analyze_trajectory_singularity(
    time_s: np.ndarray,
    q_rad: np.ndarray,
    fk,
    eps: float = DEFAULT_EPS,
    sigma_min_threshold: float = 1e-3,
    jac_fn=None,
) -> dict:
    """对整条轨迹逐采样点做奇异点分析。

    返回字典，包含逐点数组（可写 CSV）与汇总统计（可写图表标题/报告）。

    注意：每点 6 次中心差分 = 每点 12 次 FK 调用；N=300 时 3600 次 FK，
    纯 NumPy FK 约 0.1 s 量级，可接受。若 FK 换成 MuJoCo 且开销大，
    可降低采样率（例如每 5 个点取 1 个）——必须在报告中注明降采样。

    参数 ``jac_fn``：可选，``jac_fn(q) -> (6,6)`` 直接给出解析 Jacobian
    （例如 MuJoCo 的 ``mj_jacSite``）。给了就走解析路径，不再数值微分；
    **必须与世界系角速度约定一致**，此时 ``fk`` 可传 None。
    """
    t = np.asarray(time_s, dtype=np.float64)
    q = np.asarray(q_rad, dtype=np.float64)
    if t.ndim != 1 or q.shape != (len(t), 6):
        raise ValueError("time_s (N,) and q_rad (N,6) required")

    n = len(t)
    sigma_min = np.zeros(n)
    sigma_max = np.zeros(n)
    cond = np.zeros(n)
    manip = np.zeros(n)
    iso = np.zeros(n)
    rank = np.zeros(n, dtype=int)
    per_point: list[SingularityMetrics] = []

    for i in range(n):
        if jac_fn is not None:
            J = np.asarray(jac_fn(q[i]), dtype=np.float64)
        else:
            J = geometric_jacobian(fk, q[i], eps=eps)
        m = singularity_metrics(J, sigma_min_threshold=sigma_min_threshold)
        per_point.append(m)
        sigma_min[i] = m.sigma_min
        sigma_max[i] = m.sigma_max
        cond[i] = m.condition_number
        manip[i] = m.manipulability_yoshikawa
        iso[i] = m.isotropy
        rank[i] = m.rank

    worst = int(np.argmin(sigma_min))
    return {
        "time_s": t,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": cond,
        "manipulability": manip,
        "isotropy": iso,
        "rank": rank,
        "worst_index": worst,
        "worst_time_s": float(t[worst]),
        "worst_q_rad": q[worst].copy(),
        "n_singular_samples": int(np.sum(sigma_min < sigma_min_threshold)),
        "min_sigma_min": float(sigma_min[worst]),
        "summary": per_point[worst],
        "sigma_min_threshold": float(sigma_min_threshold),
    }


def classify_reachability(
    fk,
    target_position: np.ndarray,
    q_init: np.ndarray | None = None,
    ik_solver=None,
    max_reach_m: float | None = None,
    position_tol_m: float = 1e-3,
    min_reach_m: float | None = None,
    center_m: np.ndarray | None = None,
) -> dict:
    """区分"数学不可达"与"数值 IK 失败"（验收要求）。

    判定流程（顺序不可颠倒，否则会把算法问题说成数学问题）：
    1. 若给定 max_reach_m 且 ||p_target - center|| > max_reach_m： -> unreachable_geometry
       （``center_m`` 默认世界原点。UR5e 的臂长从肩部枢轴量起，判几何可达应传
        ``UR5eModel.reach_center()`` 并配合"到肩部的最大/最小伸展"，否则"基座正
        下方"这类点会被误判为可达）
    1b.若给定 min_reach_m 且 ||p_target - center|| < min_reach_m： -> unreachable_geometry
       （工作空间内腔由采样/优化得到的经验边界给出，需在报告中注明口径）
    2. 否则若提供 ik_solver 且返回 success： -> reachable
    3. 否则： -> numerical_failure（可达区域内但算法未收敛，需要附 sigma_min、
       iteration、初值等证据）

    返回 dict：status ∈ {reachable, unreachable_geometry, numerical_failure}、原因文字说明。
    """
    p = np.asarray(target_position, dtype=np.float64).reshape(3)
    center = np.zeros(3) if center_m is None else np.asarray(center_m, dtype=np.float64).reshape(3)
    radius = float(np.linalg.norm(p - center))
    out = {
        "target_position_m": p.tolist(),
        "center_m": center.tolist(),
        "target_radius_m": radius,
        "max_reach_m": None if max_reach_m is None else float(max_reach_m),
        "status": None,
        "reason": "",
        "ik_success": None,
        "ik_position_error_m": None,
        "ik_iterations": None,
        "ik_message": None,
    }
    if max_reach_m is not None and radius > float(max_reach_m):
        out["status"] = "unreachable_geometry"
        out["reason"] = (
            f"目标到可达球心距离 {radius:.4f} m 超过最大伸展 {float(max_reach_m):.4f} m；"
            "任何算法都不可能到达（与 IK 实现无关）"
        )
        return out

    if min_reach_m is not None and radius < float(min_reach_m):
        out["status"] = "unreachable_geometry"
        out["reason"] = (
            f"目标半径 {radius:.4f} m 小于实测最小伸展 {float(min_reach_m):.4f} m；"
            "落在工作空间内腔之外（内腔由采样得到的经验边界给出，非解析证明）"
        )
        return out

    if ik_solver is None:
        out["status"] = "unknown"
        out["reason"] = "目标在伸展范围内但未提供 IK 求解器，无法进一步判定"
        return out

    q0 = None if q_init is None else np.asarray(q_init, dtype=np.float64)
    result = ik_solver(p, q0) if q0 is not None else ik_solver(p)
    success = bool(getattr(result, "success", bool(result)))
    out["ik_success"] = success
    out["ik_position_error_m"] = float(getattr(result, "position_error_m", np.nan))
    out["ik_iterations"] = int(getattr(result, "iterations", -1))
    out["ik_message"] = str(getattr(result, "message", ""))
    if success and out["ik_position_error_m"] <= position_tol_m:
        out["status"] = "reachable"
        out["reason"] = f"IK 收敛且位置误差 {out['ik_position_error_m']:.3e} m ≤ {position_tol_m:g} m"
    elif success:
        out["status"] = "reachable_but_imprecise"
        out["reason"] = (
            f"IK 报告成功但位置误差 {out['ik_position_error_m']:.3e} m 超过容差 {position_tol_m:g} m，"
            "视为精度不足，需检查收敛判据"
        )
    else:
        out["status"] = "numerical_failure"
        out["reason"] = (
            "目标在最大伸展范围内，但数值 IK 未收敛；属于算法/初值/奇异问题，"
            "不是数学不可达，报告中必须给出 sigma_min 或初值敏感性证据"
        )
    return out
