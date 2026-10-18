"""轨迹生成参考实现（**仅用于 G 组第一周自闭环**）。

为什么需要它：G 组第一周必须能独立跑通"指标 + 图表 + CSV"的完整管线，
不能等 E 组交轨迹。因此这里提供一个最小实现，函数签名与 docs/INTERFACES.md 中的
公共接口一致：

    E: plan_joint(q_start, q_goal, duration, dt)       -> JointTrajectory
    E: plan_cartesian(T_start, T_goal, duration, dt, q_seed, ik_solver) -> JointTrajectory

E 组交付后，把本文件替换为 `src.planning.trajectory` 即可，G 组的分析代码
不需要任何修改（因为只依赖 time_s / q_rad 两个字段）。

同时它也是"两种轨迹方案比较"的最小对照：
- 关节空间五次多项式：q(τ) = q0 + (q1-q0)(10τ³ - 15τ⁴ + 6τ⁵)，端点零速零加速度，
  但末端笛卡尔路径一般**不是直线**；
- 笛卡尔空间直线：位置线性插值 + 逐点调用 IK，末端走直线，但可能出现不可解点、
  速度/加速度不连续（本实现对每个采样点固定插值，不做时间最优缩放，故末端速度
  在端点会有跳变——这个真实缺陷要如实写进报告，不能藏起来）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RefJointTrajectory:
    """与 src/contracts.JointTrajectory 字段一致的最小结构（time_s, q_rad）。"""

    time_s: np.ndarray
    q_rad: np.ndarray

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, dtype=np.float64)
        self.q_rad = np.asarray(self.q_rad, dtype=np.float64)
        if self.time_s.ndim != 1 or self.q_rad.shape != (len(self.time_s), 6):
            raise ValueError("time_s (N,) and q_rad (N,6) required")
        if not np.isclose(self.time_s[0], 0.0) or np.any(np.diff(self.time_s) <= 0):
            raise ValueError("time_s must start at 0 and strictly increase")

    @property
    def n(self) -> int:
        return len(self.time_s)


def _time_grid(duration: float, dt: float) -> np.ndarray:
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    n = int(round(duration / dt)) + 1
    if n < 2:
        raise ValueError("duration/dt too small; need at least 2 samples")
    t = np.linspace(0.0, duration, n)
    return t


# --------------------------------------------------------------------------------------
# 旋转工具：四元数（避免接近 180° 时的轴翻转问题）
# 放在这里而不是 src/analysis/linalg.py，是因为它们只服务于本参考轨迹生成器；
# 正式接入时 E 组应使用自己的实现或 scipy.spatial.transform.Rotation。
# --------------------------------------------------------------------------------------
def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """用 SVD 把矩阵投影到 SO(3)，消除 FK 累积的数值误差。"""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 单位四元数 [x, y, z, w]（Shepperd 分支法，数值稳定）。"""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quintic_joint_trajectory(q_start: np.ndarray, q_goal: np.ndarray, duration: float, dt: float) -> RefJointTrajectory:
    """关节空间五次多项式轨迹（逐关节独立插值）。"""
    q0 = np.asarray(q_start, dtype=np.float64).reshape(6)
    q1 = np.asarray(q_goal, dtype=np.float64).reshape(6)
    t = _time_grid(duration, dt)
    tau = t / duration
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5  # 满足 s(0)=0,s(1)=1,s'(0)=s'(1)=0,s''(0)=s''(1)=0
    q = q0[None, :] + (q1 - q0)[None, :] * s[:, None]
    return RefJointTrajectory(t, q)


def cartesian_line_trajectory(
    T_start: np.ndarray,
    T_goal: np.ndarray,
    duration: float,
    dt: float,
    q_seed: np.ndarray,
    ik_solver,
    position_tol_m: float = 1e-3,
) -> tuple[RefJointTrajectory, list[dict], np.ndarray]:
    """笛卡尔直线路径：位置线性插值 + 姿态 SLERP + 逐点 IK。

    ik_solver 的调用约定：ik_solver(T_target, q_init) -> 对象（需有 .q, .success，
    可选 .position_error_m / .message）。为了兼容不同实现，这里同时支持
    "返回 IKResult" 与 "返回 (q, success)" 两种形式。

    返回 (trajectory, ik_records, T_actual)：
    - ik_records 逐采样点记录 success / 误差，用于 G 组的失败案例分析；
    - T_actual 是**实际达成**的位姿序列（用 FK 回代，而不是用理想目标），
      这样"执行/规划误差"才诚实。
    """
    T0 = np.asarray(T_start, dtype=np.float64)
    T1 = np.asarray(T_goal, dtype=np.float64)
    if T0.shape != (4, 4) or T1.shape != (4, 4):
        raise ValueError("T_start and T_goal must be 4x4")
    t = _time_grid(duration, dt)
    tau = t / duration

    p = T0[:3, 3][None, :] + (T1[:3, 3] - T0[:3, 3])[None, :] * tau[:, None]
    R0, R1 = _orthonormalize(T0[:3, :3]), _orthonormalize(T1[:3, :3])

    def rotvec(Ra, Rb):
        """最短路径的旋转向量：用四元数双覆盖消歧，避免接近 180° 时轴翻转。

        纯用 trace 公式在 θ→π 时轴方向对数值噪声极其敏感（trace 的导数趋于 0），
        这里改用单位四元数：先算 q = q_a^{-1} ⊗ q_b，若实部为负则取 -q
        （对应"最短旋转"），再由虚部得到轴角。这样 θ=π 时结果稳定。
        """
        qa = _matrix_to_quaternion(Ra)
        qb = _matrix_to_quaternion(Rb)
        q = _quaternion_multiply(_quaternion_conjugate(qa), qb)
        if q[3] < 0:  # 双覆盖：取最短路径
            q = -q
        v = q[:3]
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.zeros(3)
        angle = 2.0 * np.arctan2(n, float(q[3]))
        return v / n * angle

    rv = rotvec(R0, R1)

    def rodrigues(axis_angle):
        theta = np.linalg.norm(axis_angle)
        if theta < 1e-15:
            return np.eye(3)
        k = axis_angle / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    q_list, records = [], []
    q_init = np.asarray(q_seed, dtype=np.float64).reshape(6).copy()
    T_actual = np.empty((len(t), 4, 4), dtype=np.float64)
    for i in range(len(t)):
        T_target = np.eye(4)
        T_target[:3, :3] = R0 @ rodrigues(rv * tau[i])
        T_target[:3, 3] = p[i]
        out = ik_solver(T_target, q_init)
        if hasattr(out, "q"):
            q_i = np.asarray(out.q, dtype=np.float64)
            success = bool(out.success)
            err = float(getattr(out, "position_error_m", np.nan))
            msg = str(getattr(out, "message", ""))
        else:  # (q, success) 形式
            q_i = np.asarray(out[0], dtype=np.float64)
            success = bool(out[1])
            err, msg = np.nan, ""
        records.append(
            {"index": i, "time_s": float(t[i]), "success": success, "position_error_m": err, "message": msg}
        )
        q_list.append(q_i)
        q_init = q_i  # 以上一点解作为下一点初值（连续路径的自然策略）
        # 用调用方提供的 FK 回代（ik_solver 需带 .fk 属性；否则用理想目标占位）
        fk = getattr(ik_solver, "fk", None)
        T_actual[i] = np.asarray(fk(q_i), dtype=np.float64) if fk is not None else T_target

    traj = RefJointTrajectory(t, np.array(q_list))
    return traj, records, T_actual


def make_ik_solver(fk, ik_impl, ik_params: dict | None = None):
    """把任意 IK 实现包装成带 .fk 属性的可调用对象，便于回代验证。"""
    params = dict(ik_params or {})

    def _call(self, T_target, q_init=None):
        if q_init is None:
            q_init = np.zeros(6)
        return ik_impl(fk, T_target, q_init, **params)

    # 用 type() 动态建类：类体作用域读不到同名的外层变量 fk，动态建类没有这个限制
    return type("_IKSolver", (), {"fk": staticmethod(fk), "__call__": _call})()


def numerical_ik_dls(
    fk,
    T_target: np.ndarray,
    q_init: np.ndarray,
    max_iter: int = 300,
    tol_position_m: float = 1e-4,
    tol_rotation_rad: float = 1e-4,
    damping: float = 1e-3,
    step_limit_rad: float = 0.35,
    rotation_weight_m: float = 0.25,
):
    """阻尼最小二乘（DLS / Levenberg–Marquardt）数值 IK 参考实现。

    仅用于 G 组产生对照样本与失败案例；真正交付的 IK 由 D 组负责
    （`src/kinematics/ik.py`）。返回值结构与 `src/contracts.IKResult` 一致。

    实现要点（这些细节正是"数值 IK 失败"的真实来源，报告里要讲清楚）：
    1. **误差向量单位归一**：位置误差 m 与姿态误差 rad 直接拼成 6 维向量时，
       姿态项会主宰步长（本项目实测：不归一会在 300 次迭代内停在 5 cm 误差处）。
       因此姿态项乘以 `rotation_weight_m`（单位 m/rad，默认 0.25），把两者放到同一量级。
    2. **自适应阻尼**：步长让误差下降就减小 λ（加快收敛），否则增大 λ（退回梯度方向）。
    3. **步长上限**：单步 |Δq|∞ 不超过 `step_limit_rad`，避免奇异附近爆走。
    4. **保留最佳解**：返回迭代过程中误差最小的构型，而不是最后一次（避免末端跳变）。
    5. **初值敏感性**：从不同初值出发会收敛到不同的 IK 分支（肘上/肘下、腕翻转），
       这是数值 IK 的固有性质，不是 bug —— G 组要如实报告。
    """
    from src.analysis.linalg import geometric_jacobian, so3_log
    from src.contracts import IKResult

    T_target = np.asarray(T_target, dtype=np.float64)
    q = np.asarray(q_init, dtype=np.float64).reshape(6).copy()
    lo, hi = np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)

    def pose_error(qq):
        T_cur = np.asarray(fk(qq), dtype=np.float64)
        e_pos = T_target[:3, 3] - T_cur[:3, 3]
        e_rot = so3_log(T_cur[:3, :3], T_target[:3, :3])
        # 归一化后的误差向量（仅用于求步长/线搜索）
        e = np.concatenate([e_pos, e_rot * rotation_weight_m])
        return T_cur, e_pos, e_rot, e

    lam = float(damping)
    best = None
    accepted = True
    it = 0
    for it in range(1, max_iter + 1):
        _, e_pos, e_rot, e = pose_error(q)
        pos_err = float(np.linalg.norm(e_pos))
        rot_err = float(np.linalg.norm(e_rot))
        if best is None or e @ e < best[0]:
            best = (float(e @ e), q.copy(), pos_err, rot_err, it)
        if pos_err <= tol_position_m and rot_err <= tol_rotation_rad:
            return IKResult(q, True, pos_err, rot_err, it, "converged")

        J = geometric_jacobian(fk, q)
        # 与误差向量一致地缩放姿态行
        Jw = J.copy()
        Jw[3:, :] *= rotation_weight_m
        accepted = False
        for _ in range(8):  # λ 逐步放大，直到误差确实下降（线搜索/信赖域）
            A = Jw.T @ Jw + (lam**2) * np.eye(6)
            try:
                dq = np.linalg.solve(A, Jw.T @ e)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            dq = np.clip(dq, -step_limit_rad, step_limit_rad)
            q_new = np.clip(q + dq, lo, hi)
            _, e_pos_new, e_rot_new, e_new = pose_error(q_new)
            if e_new @ e_new < e @ e:
                q = q_new
                lam = max(lam * 0.7, 1e-6)
                accepted = True
                break
            lam *= 3.0
        if not accepted:
            # 无法继续下降：返回历史最佳解并说明是收敛停滞
            break

    # best 的结构：(误差平方和, q, 位置误差, 姿态误差, 迭代序号)
    _, q_best, pos_err, rot_err, it_best = best if best is not None else (np.inf, q, np.nan, np.nan, it)
    ok = pos_err <= tol_position_m and rot_err <= tol_rotation_rad
    if ok:
        msg = "converged"
    elif not accepted:
        msg = "stalled: no improving step"
    else:
        msg = "max_iter reached without convergence"
    return IKResult(q_best, ok, pos_err, rot_err, it_best, msg)
