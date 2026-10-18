"""G 组指标定义（第一周"分析定义"阶段的核心交付）。

所有指标在此文件中**唯一定义**，脚本与报告只引用这里，避免出现两套口径。
单位：位置 m、时间 s、角度 rad、速度 rad/s、加速度 rad/s²、线速度 m/s。

指标分层（对应 INTERFACES.md 的要求："区分规划误差、执行跟踪误差、末端误差"）

A. 规划层（planning）——只依赖 JointTrajectory，不需要 MuJoCo
   1. joint_path_length_rad            关节空间路径长度
   2. joint_velocity_rad_s / accel     有限差分速度、加速度（含峰值）
   3. cartesian_path_length_m          TCP 轨迹弧长（对采样序列求和）
   4. straightness_ratio               直线度 = 起终点直线距离 / 实际弧长（1.0 表示完美直线）
   5. cartesian_velocity_m_s           末端线速度（有限差分）
   6. rest_endpoint_*                  端点是否零速/零加速度（五次多项式的一个重要性质）

B. 执行层（tracking）——需要 ExecutionLog
   7. joint_tracking_rmse_rad / max    指令 vs 实测的关节角误差
   8. ee_position/rotation_error        逐采样点的末端位置/姿态偏差

C. 目标层（task）
   9. endpoint_position_error_m / endpoint_rotation_error_rad  终末 TCP 相对目标位姿的误差
  10. workspace envelope（min/max xyz、半径范围）——用于说明采样覆盖范围

约定细节（写报告时必须照抄，否则结论无法复现）：
- 路径长度用相邻采样点欧氏距离求和；采样间隔改变时数值会变，报告中必须给出 dt。
- "直线度"只在笛卡尔直线轨迹的验收里用；对关节空间轨迹，该指标不构成"缺陷"，
  因为关节空间插值的笛卡尔路径本来就不是直线。
- 差分速度/加速度使用 np.gradient（二阶中心差分，端点一阶），单位为 rad/s，
  与 sampling dt 一致；不做滤波，因此高频抖动会体现在峰值里，这是真实特性。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from src.analysis.linalg import pose_errors


# --------------------------------------------------------------------------------------
# 基础指标
# --------------------------------------------------------------------------------------
def joint_path_length_rad(q_rad: np.ndarray) -> float:
    """关节空间路径长度（rad）：sum_i ||q_{i+1} - q_i||_2。"""
    q = np.asarray(q_rad, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("q_rad must be (N, 6)")
    if len(q) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))


def cartesian_path_length_m(T_seq: np.ndarray, method: str = "arc") -> float:
    """TCP 位置路径长度（m）。

    method="arc"      相邻采样点欧氏距离之和（对稠密采样收敛到真实弧长，默认）
    method="chord"    起终点直线距离（只用于对照，不能当作路径长度结论）
    """
    T = np.asarray(T_seq, dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError("T_seq must be (N, 4, 4)")
    p = T[:, :3, 3]
    if method == "arc":
        if len(p) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    if method == "chord":
        return float(np.linalg.norm(p[-1] - p[0]))
    raise ValueError("method must be 'arc' or 'chord'")


def straightness_ratio(T_seq: np.ndarray) -> float:
    """直线度 = 起终点直线距离 / 实际路径弧长，属于 [0, 1]。

    说明：这是**相对**指标，不含单位。理想笛卡尔直线轨迹 -> 1.0（受采样密度影响，
    严格小于 1）。用它比较"关节空间五次多项式"与"笛卡尔直线逐点 IK"两条轨迹的
    末端偏离程度，但不能用它替代末端逐点偏差（见 max_endpoint_deviation_m）。
    """
    arc = cartesian_path_length_m(T_seq, "arc")
    if arc <= 0:
        return float("nan")
    return float(cartesian_path_length_m(T_seq, "chord") / arc)


def max_endpoint_deviation_m(T_seq: np.ndarray) -> float:
    """末端轨迹相对"起终点直线"的最大垂距（m）。

    这是笛卡尔直线路径质量的直接度量：对关节空间轨迹，它通常明显大于 0；
    对笛卡尔直线逐点 IK，它应接近 0（仅剩 IK 残差与采样离散误差）。
    """
    T = np.asarray(T_seq, dtype=np.float64)
    p = T[:, :3, 3]
    if len(p) < 3:
        return 0.0
    a, b = p[0], p[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-12:
        return float(np.max(np.linalg.norm(p - a, axis=1)))
    # 逐点到直线 ab 的距离
    t = ((p - a) @ ab) / (n * n)
    proj = a + np.outer(t, ab)
    return float(np.max(np.linalg.norm(p - proj, axis=1)))


def joint_velocity_rad_s(time_s: np.ndarray, q_rad: np.ndarray) -> np.ndarray:
    """关节速度（rad/s）随时间的变化，shape (N, 6)。

    时间必须严格递增（JointTrajectory 已保证）。使用 np.gradient，时间间隔不要求均匀。
    """
    t = np.asarray(time_s, dtype=np.float64)
    q = np.asarray(q_rad, dtype=np.float64)
    if t.ndim != 1 or q.shape != (len(t), 6):
        raise ValueError("time_s (N,) and q_rad (N,6) required")
    return np.gradient(q, t, axis=0)


def joint_acceleration_rad_s2(time_s: np.ndarray, q_rad: np.ndarray) -> np.ndarray:
    """关节加速度（rad/s²），对速度再做一次时间差分。"""
    v = joint_velocity_rad_s(time_s, q_rad)
    t = np.asarray(time_s, dtype=np.float64)
    return np.gradient(v, t, axis=0)


def cartesian_velocity_m_s(time_s: np.ndarray, T_seq: np.ndarray) -> np.ndarray:
    """末端线速度（m/s），shape (N,)。姿态角速度本次不计入（报告中说明）。"""
    t = np.asarray(time_s, dtype=np.float64)
    T = np.asarray(T_seq, dtype=np.float64)
    p = T[:, :3, 3]
    return np.linalg.norm(np.gradient(p, t, axis=0), axis=1)


def joint_tracking_errors(q_command_rad: np.ndarray, q_measured_rad: np.ndarray) -> dict:
    """执行跟踪误差（rad）：整体 RMSE、峰值、逐关节 RMSE。"""
    qc = np.asarray(q_command_rad, dtype=np.float64)
    qm = np.asarray(q_measured_rad, dtype=np.float64)
    if qc.shape != qm.shape or qc.ndim != 2 or qc.shape[1] != 6:
        raise ValueError("q_command_rad and q_measured_rad must both be (N, 6) and equal shape")
    err = qm - qc
    return {
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean(err**2))),
        "joint_tracking_max_abs_rad": float(np.max(np.abs(err))),
        "joint_tracking_rmse_per_joint_rad": [float(v) for v in np.sqrt(np.mean(err**2, axis=0))],
    }


# --------------------------------------------------------------------------------------
# 工作空间包络：**只依赖 public 契约里的数据**，不要求任何新接口
# --------------------------------------------------------------------------------------
def trajectory_envelope_m(time_s: np.ndarray, T_seq: np.ndarray) -> dict:
    """一条轨迹的 TCP 包络尺寸（m），以及**时间窗口**内的每轴行程。

    参数
    ----
    time_s : (N,) 时间（s）
    T_seq  : (N,4,4) TCP 位姿序列

    返回包含：
      span_m          : 各轴行程 (dx, dy, dz)（m）——包络"箱子"的边长
      diagonal_m      : 包络对角线长度 sqrt(dx²+dy²+dz²)（m）
      max_axis_travel_m : 单轴最大行程（m）
      min/max_m       : 各轴上下界（m）
      t_argmax_per_axis_s : 每个轴达到最大值/最小值的时间（s）

    为什么需要"包围盒边长"而不是只报"包络"这个模糊说法：
    验收要求指标必须有明确单位与定义。这里是世界系轴对齐包围盒（AABB），
    单位 m，随采样密度不变（只取决于极值点），可跨机器复现。
    """
    t = np.asarray(time_s, dtype=np.float64)
    T = np.asarray(T_seq, dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError("T_seq must be (N, 4, 4)")
    if t.shape != (len(T),):
        raise ValueError("time_s must be (N,) matching T_seq")
    p = T[:, :3, 3]
    lo = np.min(p, axis=0)
    hi = np.max(p, axis=0)
    span = hi - lo
    i_hi = np.argmax(p, axis=0)
    i_lo = np.argmin(p, axis=0)
    return {
        "n_samples": int(len(t)),
        "span_m": tuple(float(v) for v in span),
        "diagonal_m": float(np.linalg.norm(span)),
        "max_axis_travel_m": float(np.max(span)),
        "min_m": tuple(float(v) for v in lo),
        "max_m": tuple(float(v) for v in hi),
        "t_at_max_s": tuple(float(t[i]) for i in i_hi),
        "t_at_min_s": tuple(float(t[i]) for i in i_lo),
    }


def poses_from_joint_trajectory(time_s: np.ndarray, q_rad: np.ndarray, fk) -> np.ndarray:
    """用 C 组公共接口 `fk(q) -> T` 把关节轨迹转成 TCP 位姿序列 (N,4,4)。

    这是 G 组与上游之间的**唯一**几何依赖：只要 C 组提供 `fk`，G 组就能算出
    末端路径、包络、误差，不需要任何人新增接口。
    """
    t = np.asarray(time_s, dtype=np.float64)
    q = np.asarray(q_rad, dtype=np.float64)
    if q.shape != (len(t), 6):
        raise ValueError("time_s (N,) and q_rad (N,6) required")
    out = np.empty((len(q), 4, 4), dtype=np.float64)
    for i, qi in enumerate(q):
        T = np.asarray(fk(qi), dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("fk must return a 4x4 matrix")
        out[i] = T
    return out


# --------------------------------------------------------------------------------------
# 汇总结构
# --------------------------------------------------------------------------------------
@dataclass
class TrajectoryMetrics:
    """一条轨迹的完整指标集合。字段名即 CSV 列名，单位写在字段名里。"""

    label: str
    n_samples: int
    duration_s: float
    dt_mean_s: float

    # 规划层
    joint_path_length_rad: float
    joint_peak_velocity_rad_s: float
    joint_peak_acceleration_rad_s2: float
    cartesian_path_length_m: float
    straightness_ratio: float
    max_endpoint_deviation_m: float
    cartesian_peak_velocity_m_s: float
    start_rest: bool
    end_rest: bool

    # 目标层（可选，需要目标位姿）
    endpoint_position_error_m: float | None = None
    endpoint_rotation_error_rad: float | None = None

    # 执行层（可选，需要 ExecutionLog）
    joint_tracking_rmse_rad: float | None = None
    joint_tracking_max_abs_rad: float | None = None

    # 工作空间包络（可选）
    workspace_span_m: tuple[float, float, float] | None = None
    workspace_min_m: tuple[float, float, float] | None = None
    workspace_max_m: tuple[float, float, float] | None = None

    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """展开成标量字典，供 pandas / csv 使用（元组与列表字段转成可读字符串）。"""
        row = asdict(self)
        row.pop("extra", None)
        for key, value in list(row.items()):
            if isinstance(value, (list, tuple)):
                row[key] = ";".join(f"{v:.6g}" if isinstance(v, (int, float)) else str(v) for v in value)
        row.update({f"extra_{k}": v for k, v in self.extra.items()})
        return row


def _rest_flags(time_s: np.ndarray, q_rad: np.ndarray, tol: float = 1e-3) -> tuple[bool, bool]:
    """判断轨迹起终点是否近似零速（用于验证五次多项式的边界条件）。

    阈值说明：五次多项式在端点的**解析**速度为 0，但用有限差分在端点只能用单侧
    差分估计，误差为 O(q⁗·dt) 量级，实测约 1e-4 ~ 1e-3 rad/s（dt = 0.005~0.01 s）。
    因此阈值取 1e-3 rad/s，表示"低于测量分辨率"，而不是物理上非零速度。
    报告里必须写明该阈值，避免把数值下限说成轨迹缺陷。
    """
    v = joint_velocity_rad_s(time_s, q_rad)
    return bool(np.max(np.abs(v[0])) <= tol), bool(np.max(np.abs(v[-1])) <= tol)


def analyze_joint_trajectory(
    label: str,
    time_s: np.ndarray,
    q_rad: np.ndarray,
    fk,
    target_pose: np.ndarray | None = None,
    q_measured_rad: np.ndarray | None = None,
) -> tuple[TrajectoryMetrics, np.ndarray]:
    """对一条关节轨迹算出全部规划层指标，并返回 TCP 位姿序列 T_seq (N,4,4)。

    参数
    ----
    fk : Callable[[(6,)], (4,4)]  C 组的正运动学（或用参考模型占位）
    target_pose : (4,4) 规划目标位姿；给定则计算终末误差
    q_measured_rad : (N,6) F 组实测关节角；给定则计算跟踪误差
    """
    t = np.asarray(time_s, dtype=np.float64)
    q = np.asarray(q_rad, dtype=np.float64)
    if t.ndim != 1 or q.shape != (len(t), 6):
        raise ValueError("time_s (N,) and q_rad (N,6) required")

    T_seq = np.array([np.asarray(fk(qi), dtype=np.float64) for qi in q])
    v = joint_velocity_rad_s(t, q)
    a = joint_acceleration_rad_s2(t, q)
    vc = cartesian_velocity_m_s(t, T_seq)
    start_rest, end_rest = _rest_flags(t, q)
    env = trajectory_envelope_m(t, T_seq)

    metrics = TrajectoryMetrics(
        label=label,
        n_samples=int(len(t)),
        duration_s=float(t[-1] - t[0]),
        dt_mean_s=float(np.mean(np.diff(t))),
        joint_path_length_rad=joint_path_length_rad(q),
        joint_peak_velocity_rad_s=float(np.max(np.abs(v))),
        joint_peak_acceleration_rad_s2=float(np.max(np.abs(a))),
        cartesian_path_length_m=cartesian_path_length_m(T_seq, "arc"),
        straightness_ratio=straightness_ratio(T_seq),
        max_endpoint_deviation_m=max_endpoint_deviation_m(T_seq),
        cartesian_peak_velocity_m_s=float(np.max(vc)),
        start_rest=start_rest,
        end_rest=end_rest,
        workspace_min_m=tuple(np.min(T_seq[:, :3, 3], axis=0)),
        workspace_max_m=tuple(np.max(T_seq[:, :3, 3], axis=0)),
        workspace_span_m=tuple(np.ptp(T_seq[:, :3, 3], axis=0)),
        extra={
            "envelope_diagonal_m": env["diagonal_m"],
            "envelope_max_axis_travel_m": env["max_axis_travel_m"],
            "envelope_t_at_max_s": env["t_at_max_s"],
        },
    )
    if target_pose is not None:
        pe, re = pose_errors(T_seq[-1], np.asarray(target_pose, dtype=np.float64))
        metrics.endpoint_position_error_m = pe
        metrics.endpoint_rotation_error_rad = re
    if q_measured_rad is not None:
        track = joint_tracking_errors(q, q_measured_rad)
        metrics.joint_tracking_rmse_rad = track["joint_tracking_rmse_rad"]
        metrics.joint_tracking_max_abs_rad = track["joint_tracking_max_abs_rad"]
        metrics.extra["joint_tracking_rmse_per_joint_rad"] = track["joint_tracking_rmse_per_joint_rad"]
    return metrics, T_seq


def compare_metrics(metrics_list) -> list[dict]:
    """把多个 TrajectoryMetrics 转成可直接写 CSV 的行列表。"""
    return [m.to_row() for m in metrics_list]


def numeric_delta(a: float | None, b: float | None) -> float | None:
    """安全差值（任一为 None 时返回 None），用于"两种轨迹差异"列。"""
    if a is None or b is None:
        return None
    return float(a - b)
