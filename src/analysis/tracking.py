"""规划 vs 实际执行：末端误差 / 路径长度 / 速度加速度 / 误差来源分解。

数据流
======
::

    F 组 exec_log_latest.csv ──exec_log.load_exec_log()──> ExecLog
                                                          ├─ q_cmd    (N,6) 规划侧
                                                          └─ q_actual (N,6) 执行侧
                                                                   │
                    mujoco_backend.UR5eModel.fk_batch() ───────────┤
                                                                   ▼
                                           fk(q_cmd) / fk(q_actual) = 规划/实际 TCP 位姿
                                                                   ▼
                                                       本模块的指标与分解

三条口径（报告中必须分开写，不能混为一谈）
==========================================
1. **关节跟踪误差** `q_actual - q_cmd`（rad）：控制器的直接误差，单位 rad。
2. **末端跟踪误差** `fk(q_actual) - fk(q_cmd)`（m / rad）：任务关心的误差。
   它 ≈ J(q_cmd)·(q_actual-q_cmd)，本模块逐点验证这个一阶关系
   （`jacobian_decomposition`），残差是二阶项，量级见结果 CSV。
3. **末端绝对误差 / 放置误差**：只对抓放点有意义（F 的 `place_dxy_mm`），
   G 不做抓取力学结论，只用 TCP 位置差。

**为什么不用 F 日志里的 `ee_*` 列直接算误差**：实测该列比 `q_actual` 滞后
一个采样（0.002 s，见 `detect_pose_lag`），直接用会凭空多出最大 1.26 mm 的
"假误差"。G 一律用 `fk(q_actual)` 回代；`ee_*` 只用于交叉核对。

单位约定：关节 rad、位置 m、姿态 rad、时间 s；输出 CSV 里列名自带单位。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .exec_log import ExecLog, SegmentRun, quat_wxyz_to_matrix
from .linalg import geometric_jacobian, so3_log


# ======================================================================
# 姿态误差（批量）
# ======================================================================
def batch_pose_errors(T_a: np.ndarray, T_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """批量位姿误差：(N,4,4) vs (N,4,4) → (位置误差 m, 姿态误差 rad)，各 (N,)。"""
    A = np.asarray(T_a, dtype=np.float64)
    B = np.asarray(T_b, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 3:
        raise ValueError("T_a / T_b must share shape (N,4,4)")
    pos = np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1)
    n = len(A)
    rot = np.empty(n, dtype=np.float64)
    for i in range(n):
        rot[i] = np.linalg.norm(so3_log(A[i, :3, :3], B[i, :3, :3]))
    return pos, rot


# ======================================================================
# 数据质量：F 日志 ee_* 列的采样滞后
# ======================================================================
def detect_pose_lag(
    model,
    log: ExecLog,
    max_lag: int = 5,
) -> dict:
    """检测 F 日志 `ee_*` 列相对 `q_actual` 的采样滞后（单位：采样点）。

    F 的写法是"先 `mj_step`（更新 qpos）后读 `site_xpos`"，而 `site_xpos` 要等
    下一次 `mj_forward` 才刷新，所以日志里的末端位姿实际对应**上一步**的关节角。

    返回 dict：`lag_samples`（最优滞后）、`rmse_m`（对齐后残差）、`best_mean_m`、
    `naive_max_m`（不校正时的最大偏差）。lag 为正是"ee 落后于 q_actual"。
    """
    T = model.fk_batch(log.q_actual)
    P = T[:, :3, 3]
    ee = log.ee_pos
    n = len(P)
    scores: dict[int, tuple[float, float]] = {}
    lo, hi = -max_lag, max_lag
    for k in range(lo, hi + 1):
        if k >= 0:
            a, b = P[k:], ee[: n - k] if k else ee
        else:
            a, b = P[:k], ee[-k:]
        e = np.linalg.norm(a - b, axis=1)
        scores[k] = (float(e.mean()), float(e.max()))
    best = min(scores, key=lambda k: scores[k][0])
    return {
        "lag_samples": int(best),
        "lag_time_s": float(best * log.dt_s),
        "best_mean_m": scores[best][0],
        "best_max_m": scores[best][1],
        "rmse_m": float(np.sqrt(scores[best][0] ** 2 + (scores[best][1] / 3.0) ** 2)),
        "naive_max_m": scores[0][1],
        "scan": {int(k): v for k, v in scores.items()},
        "interpretation": (
            f"日志 ee_* 比 q_actual 滞后 {-best} 个采样（{-best * log.dt_s * 1000:.1f} ms）；"
            f"对齐后残差均值 {scores[best][0] * 1000:.4f} mm（≈CSV 6 位小数舍入），"
            f"不校正时最大偏差 {scores[0][1] * 1000:.4f} mm。"
            "G 的所有指标因此统一用 fk(q_actual) 回代，不使用 ee_* 列。"
        ),
    }


# ======================================================================
# 单段指标
# ======================================================================
@dataclass
class SegmentMetrics:
    """一段流程（cycle + segment）的规划-执行对比指标。字段名自带单位。"""

    cycle: int
    segment: str
    n_samples: int
    duration_s: float
    # ---- 关节跟踪误差 ----
    joint_err_max_rad: float
    joint_err_mean_rad: float
    joint_err_rms_rad: float
    joint_err_max_deg: float
    joint_err_max_joint: int          # 误差最大的关节序号 1..6
    # ---- 末端跟踪误差（fk(q_actual) vs fk(q_cmd)）----
    ee_pos_err_max_m: float
    ee_pos_err_mean_m: float
    ee_pos_err_rms_m: float
    ee_ori_err_max_rad: float
    ee_ori_err_mean_rad: float
    # ---- 路径长度 ----
    joint_path_planned_rad: float
    joint_path_actual_rad: float
    ee_path_planned_m: float
    ee_path_actual_m: float
    ee_path_excess_m: float
    ee_path_excess_ratio: float
    ee_straightness_planned: float
    ee_straightness_actual: float
    ee_max_dev_from_chord_planned_m: float
    ee_max_dev_from_chord_actual_m: float
    # ---- 速度 / 加速度（规划 vs 实际）----
    joint_speed_max_planned_rad_s: float
    joint_speed_max_actual_rad_s: float
    joint_speed_rms_planned_rad_s: float
    joint_speed_rms_actual_rad_s: float
    ee_speed_max_planned_m_s: float
    ee_speed_max_actual_m_s: float
    joint_accel_max_planned_rad_s2: float
    joint_accel_max_actual_rad_s2: float
    joint_accel_rms_planned_rad_s2: float
    joint_accel_rms_actual_rad_s2: float
    # ---- 末端误差的 Jacobian 一阶分解 ----
    jac_pred_err_max_m: float
    jac_residual_max_m: float
    jac_residual_rel: float
    jac_r2: float
    # ---- 与机构状态的关系 ----
    sigma_min_min: float
    sigma_min_mean: float
    condition_number_max: float
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = dict(self.__dict__)
        extra = d.pop("extra", {}) or {}
        for k, v in extra.items():
            d[f"extra_{k}"] = v
        return d


def _sub_slice(run: SegmentRun, step: int) -> slice:
    return slice(run.i0, run.i1, max(1, int(step)))


def analyze_segment(
    model,
    log: ExecLog,
    run: SegmentRun,
    jac_step: int = 1,
    sigma_min_threshold: float = 0.02,
) -> SegmentMetrics:
    """对一段执行数据算全部指标。

    参数
    ----
    jac_step : Jacobian 的降采样步长。F 日志 0.002 s 一步，全程 2.5 万点，
               逐点 `mj_jacSite` 约 3~5 s；默认 1（不降采样），
               若只要趋势可设 5 并**在报告里注明降采样**。
    """
    sl = slice(run.i0, run.i1)
    t = log.t[sl]
    q_cmd = log.q_cmd[sl]
    q_act = log.q_actual[sl]
    if len(t) < 2:
        raise ValueError(f"段 {run} 样本不足")

    dq = q_act - q_cmd

    # ---- FK：规划侧与实际侧 ----
    T_cmd = model.fk_batch(q_cmd)
    T_act = model.fk_batch(q_act)
    pos_err, ori_err = batch_pose_errors(T_act, T_cmd)

    # ---- 路径长度 ----
    jp_plan = metrics.joint_path_length_rad(q_cmd)
    jp_act = metrics.joint_path_length_rad(q_act)
    ee_plan = metrics.cartesian_path_length_m(T_cmd, "arc")
    ee_act = metrics.cartesian_path_length_m(T_act, "arc")

    # ---- 速度 / 加速度（同一差分口径）----
    v_plan = metrics.joint_velocity_rad_s(t, q_cmd)
    v_act = metrics.joint_velocity_rad_s(t, q_act)
    a_plan = metrics.joint_acceleration_rad_s2(t, q_cmd)
    a_act = metrics.joint_acceleration_rad_s2(t, q_act)
    ee_v_plan = metrics.cartesian_velocity_m_s(t, T_cmd)
    ee_v_act = metrics.cartesian_velocity_m_s(t, T_act)

    # ---- Jacobian 一阶分解：Δp ≈ J(q_cmd) Δq ----
    idx = np.arange(run.i0, run.i1, max(1, int(jac_step)))
    pred = np.zeros((len(idx), 3))
    actual = np.zeros((len(idx), 3))
    sigmas = np.zeros(len(idx))
    conds = np.zeros(len(idx))
    for j, i in enumerate(idx):
        J = model.jacobian_tcp(log.q_cmd[i])
        d = log.q_actual[i] - log.q_cmd[i]
        pred[j] = J[:3, :] @ d
        actual[j] = T_act[i - run.i0, :3, 3] - T_cmd[i - run.i0, :3, 3]
        s = np.linalg.svd(J, compute_uv=False)
        sigmas[j] = s[-1]
        conds[j] = s[0] / max(s[-1], 1e-16)

    resid = np.linalg.norm(pred - actual, axis=1)
    act_norm = np.linalg.norm(actual, axis=1)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((act_norm - act_norm.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    per_joint = np.abs(dq).max(axis=0)
    return SegmentMetrics(
        cycle=run.cycle,
        segment=run.segment,
        n_samples=int(len(t)),
        duration_s=float(t[-1] - t[0]),
        joint_err_max_rad=float(np.abs(dq).max()),
        joint_err_mean_rad=float(np.mean(np.abs(dq))),
        joint_err_rms_rad=float(np.sqrt(np.mean(dq ** 2))),
        joint_err_max_deg=float(np.degrees(np.abs(dq).max())),
        joint_err_max_joint=int(np.argmax(per_joint)) + 1,
        ee_pos_err_max_m=float(pos_err.max()),
        ee_pos_err_mean_m=float(pos_err.mean()),
        ee_pos_err_rms_m=float(np.sqrt(np.mean(pos_err ** 2))),
        ee_ori_err_max_rad=float(ori_err.max()),
        ee_ori_err_mean_rad=float(ori_err.mean()),
        joint_path_planned_rad=jp_plan,
        joint_path_actual_rad=jp_act,
        ee_path_planned_m=ee_plan,
        ee_path_actual_m=ee_act,
        ee_path_excess_m=ee_act - ee_plan,
        ee_path_excess_ratio=(ee_act / ee_plan) if ee_plan > 0 else float("nan"),
        ee_straightness_planned=metrics.straightness_ratio(T_cmd),
        ee_straightness_actual=metrics.straightness_ratio(T_act),
        ee_max_dev_from_chord_planned_m=metrics.max_endpoint_deviation_m(T_cmd),
        ee_max_dev_from_chord_actual_m=metrics.max_endpoint_deviation_m(T_act),
        joint_speed_max_planned_rad_s=float(np.abs(v_plan).max()),
        joint_speed_max_actual_rad_s=float(np.abs(v_act).max()),
        joint_speed_rms_planned_rad_s=float(np.sqrt(np.mean(v_plan ** 2))),
        joint_speed_rms_actual_rad_s=float(np.sqrt(np.mean(v_act ** 2))),
        ee_speed_max_planned_m_s=float(ee_v_plan.max()),
        ee_speed_max_actual_m_s=float(ee_v_act.max()),
        joint_accel_max_planned_rad_s2=float(np.abs(a_plan).max()),
        joint_accel_max_actual_rad_s2=float(np.abs(a_act).max()),
        joint_accel_rms_planned_rad_s2=float(np.sqrt(np.mean(a_plan ** 2))),
        joint_accel_rms_actual_rad_s2=float(np.sqrt(np.mean(a_act ** 2))),
        jac_pred_err_max_m=float(np.linalg.norm(pred, axis=1).max()),
        jac_residual_max_m=float(resid.max()),
        jac_residual_rel=float(resid.max() / max(act_norm.max(), 1e-12)),
        jac_r2=r2,
        sigma_min_min=float(sigmas.min()),
        sigma_min_mean=float(sigmas.mean()),
        condition_number_max=float(conds.max()),
        extra={
            "jac_samples": int(len(idx)),
            "jac_step": int(max(1, int(jac_step))),
            "sigma_min_threshold": float(sigma_min_threshold),
            "per_joint_max_deg": [float(np.degrees(v)) for v in per_joint],
        },
    )


# ======================================================================
# 全日志分析
# ======================================================================
def analyze_exec_log(
    model,
    log: ExecLog,
    jac_step: int = 1,
    sigma_min_threshold: float = 0.02,
) -> dict:
    """把整份日志按 39 段逐段分析，并汇总到循环级与全局。

    返回 dict：

    ``segments``  list[SegmentMetrics]      —— 逐段（CSV 一行一段）
    ``cycles``    list[dict]                —— 每次搬运循环汇总
    ``overall``   dict                      —— 全日志汇总
    ``series``    dict[str, np.ndarray]     —— 逐采样点时间序列（画图用）
    """
    runs = log.segments()
    seg_metrics = [analyze_segment(model, log, r, jac_step=jac_step,
                                   sigma_min_threshold=sigma_min_threshold)
                   for r in runs]

    # ---- 逐采样点序列（全日志一次算完，避免重复 FK）----
    T_cmd = model.fk_batch(log.q_cmd)
    T_act = model.fk_batch(log.q_actual)
    pos_err, ori_err = batch_pose_errors(T_act, T_cmd)
    dq = log.q_actual - log.q_cmd
    series = {
        "t": log.t,
        "joint_err_max_deg": np.degrees(np.abs(dq).max(axis=1)),
        "joint_err_rms_deg": np.degrees(np.sqrt(np.mean(dq ** 2, axis=1))),
        "ee_pos_err_mm": pos_err * 1000.0,
        "ee_ori_err_deg": np.degrees(ori_err),
        "ee_speed_planned_m_s": metrics.cartesian_velocity_m_s(log.t, T_cmd),
        "ee_speed_actual_m_s": metrics.cartesian_velocity_m_s(log.t, T_act),
        "ee_pos_planned": T_cmd[:, :3, 3],
        "ee_pos_actual": T_act[:, :3, 3],
        "ee_quat_planned_wxyz": _rotmat_to_quat(T_cmd[:, :3, :3]),
        "cycle": log.cycle,
        "segment": log.segment,
    }

    # ---- 循环级汇总 ----
    cycles: list[dict] = []
    for c in log.cycles:
        rs = [m for m in seg_metrics if m.cycle == c]
        sl = log.cycle_slice(c)
        cycles.append({
            "cycle": c,
            "duration_s": float(log.t[sl][-1] - log.t[sl][0]),
            "n_samples": int(sl.stop - sl.start),
            "joint_err_max_deg": max(m.joint_err_max_deg for m in rs),
            "ee_pos_err_max_mm": max(m.ee_pos_err_max_m for m in rs) * 1000.0,
            "ee_ori_err_max_deg": float(np.degrees(max(m.ee_ori_err_max_rad for m in rs))),
            "ee_path_planned_m": sum(m.ee_path_planned_m for m in rs),
            "ee_path_actual_m": sum(m.ee_path_actual_m for m in rs),
            "joint_path_planned_rad": sum(m.joint_path_planned_rad for m in rs),
            "joint_path_actual_rad": sum(m.joint_path_actual_rad for m in rs),
            "sigma_min_min": min(m.sigma_min_min for m in rs),
            "worst_segment": max(rs, key=lambda m: m.ee_pos_err_max_m).segment,
        })

    # ---- 全局汇总 ----
    overall = {
        "n_cycles": len(log.cycles),
        "duration_s": float(log.t[-1] - log.t[0]),
        "joint_err_max_deg": float(np.degrees(np.abs(dq).max())),
        "joint_err_rms_deg": float(np.degrees(np.sqrt(np.mean(dq ** 2)))),
        "ee_pos_err_max_mm": float(pos_err.max() * 1000),
        "ee_pos_err_rms_mm": float(np.sqrt(np.mean(pos_err ** 2)) * 1000),
        "ee_ori_err_max_deg": float(np.degrees(ori_err.max())),
        "ee_path_planned_m": float(sum(m.ee_path_planned_m for m in seg_metrics if m.cycle == log.cycles[0])),
        "ee_path_actual_m": float(sum(m.ee_path_actual_m for m in seg_metrics if m.cycle == log.cycles[0])),
        "sigma_min_min": float(min(m.sigma_min_min for m in seg_metrics)),
        "segments_analyzed": len(seg_metrics),
    }
    # 三循环重复性：逐段指标应完全一致（同设置、确定性仿真）
    repeat = _repeatability(seg_metrics)
    overall["repeatability"] = repeat

    return {
        "segments": seg_metrics,
        "cycles": cycles,
        "overall": overall,
        "series": series,
        "runs": runs,
    }


def _repeatability(seg_metrics: list[SegmentMetrics]) -> dict:
    """检查不同循环的同名段指标是否一致（确定性仿真的合理性检查）。"""
    by_name: dict[str, list[float]] = {}
    for m in seg_metrics:
        by_name.setdefault(m.segment, []).append(m.ee_pos_err_max_m)
    worst = ""
    worst_spread = 0.0
    for name, vals in by_name.items():
        if len(vals) < 2:
            continue
        spread = float(max(vals) - min(vals))
        if spread > worst_spread:
            worst, worst_spread = name, spread
    return {
        "checked_segments": sorted(by_name),
        "max_ee_err_spread_m": worst_spread,
        "max_ee_err_spread_mm": worst_spread * 1000,
        "worst_segment": worst,
        "deterministic": worst_spread < 1e-9,
    }


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(N,3,3) 旋转矩阵 → (N,4) 四元数 (w,x,y,z)，只用于画图/落盘。"""
    R = np.asarray(R, dtype=np.float64)
    n = len(R)
    q = np.empty((n, 4))
    for i in range(n):
        M = R[i]
        tr = np.trace(M)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            q[i] = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q[i] = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q[i] = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q[i] = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
        if q[i, 0] < 0:
            q[i] = -q[i]
    return q
