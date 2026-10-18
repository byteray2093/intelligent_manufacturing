"""G 组（奇异点 / 可达性 / 误差分析）基础数学工具。

约定（与 docs/INTERFACES.md 对齐）：
- 单位：位置 m、时间 s、角度 rad。
- 位姿：4x4 齐次变换矩阵，`T[:3, :3]` 属于 SO(3)，最后一行为 [0,0,0,1]。
- 关节向量 q：shape (6,)，float64。
- 所有函数只依赖 NumPy，不依赖 MuJoCo / matplotlib，便于单元测试。
"""
from __future__ import annotations

import numpy as np

#: 数值微分默认步长（rad）。关节角是弧度制（约 1e-6 rad ~ 0.2 微弧度）。
DEFAULT_EPS = 1e-6


def rotation_error_rad(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """两个旋转矩阵之间的测地线夹角（rad），取值范围 [0, pi]。

    用旋转矩阵的迹计算，避免 logm / 四元数分支问题：

        theta = arccos( (trace(R_a^T R_b) - 1) / 2 )

    对数值噪声做截断，保证 arccos 的定义域。
    """
    R_a = np.asarray(R_a, dtype=np.float64)
    R_b = np.asarray(R_b, dtype=np.float64)
    if R_a.shape != (3, 3) or R_b.shape != (3, 3):
        raise ValueError("rotation_error_rad expects two 3x3 matrices")
    cos_theta = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def pose_position_error_m(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """两个位姿之间的平移误差（m）：欧氏距离。"""
    T_a = np.asarray(T_a, dtype=np.float64)
    T_b = np.asarray(T_b, dtype=np.float64)
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))


def pose_errors(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """返回 (位置误差 m, 姿态误差 rad)。两者必须分别报告，不能只报平移。"""
    return pose_position_error_m(T_a, T_b), rotation_error_rad(T_a[:3, :3], T_b[:3, :3])


def geometric_jacobian(fk, q: np.ndarray, eps: float = DEFAULT_EPS, method: str = "central") -> np.ndarray:
    """用数值微分估计几何 Jacobian J(q)，shape (6, 6)。

    行 0:3 为线速度 v = J_v qdot（m/s per rad/s，即 m/rad）；
    行 3:6 为角速度 w = J_w qdot（rad/s per rad/s，即无量纲）。

    参数
    ----
    fk : Callable[[np.ndarray], np.ndarray]
        正运动学，输入 (6,) 关节角，返回 4x4 齐次矩阵 T_world_tcp。
        这是 C 组（`src/kinematics/fk.py`）的接口，因此本函数不依赖 MuJoCo。
    q : (6,) 关节角
    eps : 差分步长（rad）
    method : "central"（二阶精度，默认）或 "forward"（一阶精度）

    注意：差分法在奇异位形附近仍然有效（J 本身连续），但用有限差分做的
    数值 IK 在奇异附近会不稳定——这正是 G 组要量化的现象，不是本函数的缺陷。
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError(f"q must have shape (6,); got {q.shape}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    base = np.asarray(fk(q), dtype=np.float64)
    if base.shape != (4, 4):
        raise ValueError("fk must return a 4x4 homogeneous matrix")

    J = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        dq = np.zeros(6)
        if method == "central":
            dq[i] = eps
            T_plus = np.asarray(fk(q + dq), dtype=np.float64)
            T_minus = np.asarray(fk(q - dq), dtype=np.float64)
            J[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2.0 * eps)
            # 角速度必须用"右乘/体坐标"增量：exp([ω]dt) = R(q)ᵀ R(q+dq)。
            # 若写成 so3_log(R(q-h), R(q+h)) 会得到 -ω（本项目实际踩过这个坑，
            # 会让数值 IK 朝反方向迭代而不收敛），因此这里显式用转置。
            J[3:, i] = so3_log(np.eye(3), T_minus[:3, :3].T @ T_plus[:3, :3]) / (2.0 * eps)
        elif method == "forward":
            dq[i] = eps
            T_plus = np.asarray(fk(q + dq), dtype=np.float64)
            J[:3, i] = (T_plus[:3, 3] - base[:3, 3]) / eps
            J[3:, i] = so3_log(np.eye(3), base[:3, :3].T @ T_plus[:3, :3]) / eps
        else:
            raise ValueError("method must be 'central' or 'forward'")
    return J


def so3_log(R_from: np.ndarray, R_to: np.ndarray) -> np.ndarray:
    """最小旋转向量（axis-angle, rad）：把 R_from 转到 R_to。

    精确公式（**不是**小角度近似）：

        theta = arccos((trace(R) - 1) / 2),  R = R_from^T R_to
        omega = theta / (2 sin theta) * (R - R^T)     （theta 既不接近 0 也不接近 pi）

    为什么不能用 0.5 (R - R^T) 近似：该近似只在小角度成立；旋转接近 180° 时会严重
    失真（本项目的数值 IK 曾因此把"姿态差 180°"误判为"已收敛"，属于真实踩过的坑），
    也会让数值 Jacobian 的角速度行在大角度差分下出错。

    theta → 0：返回 0。
    theta → pi：反对称部分趋零，改用四元数（Shepperd）分支定轴，保证稳定。
    """
    R = np.asarray(R_from, dtype=np.float64).T @ np.asarray(R_to, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-10:
        return np.zeros(3)
    if theta < np.pi - 1e-6:
        skew_part = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return skew_part * (theta / np.sin(theta))
    # 接近 180°：由对称部分反解轴，再用非对角项定符号
    w = max(0.0, (1.0 + cos_theta) / 2.0)
    axis = np.sqrt(np.clip(
        [(R[0, 0] + 1.0) / 2.0 - w, (R[1, 1] + 1.0) / 2.0 - w, (R[2, 2] + 1.0) / 2.0 - w], 0.0, None))
    if axis[0] >= max(axis[1], axis[2]):
        axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
        axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])
    elif axis[1] >= axis[2]:
        axis[0] = np.copysign(axis[0], R[0, 1] + R[1, 0])
        axis[2] = np.copysign(axis[2], R[1, 2] + R[2, 1])
    else:
        axis[0] = np.copysign(axis[0], R[0, 2] + R[2, 0])
        axis[1] = np.copysign(axis[1], R[1, 2] + R[2, 1])
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.zeros(3)
    return axis / n * theta


def _log3_direction(R_from: np.ndarray, R_to: np.ndarray) -> np.ndarray:
    """向后兼容别名，见 `so3_log`（精确 so(3) 对数映射）。"""
    return so3_log(R_from, R_to)


def jacobian_at_tcp(
    J_body: np.ndarray,
    T_world_tcp: np.ndarray,
    p_body: np.ndarray | None = None,
    frame: str = "world",
) -> np.ndarray:
    """把 MuJoCo 风格的 body Jacobian 换算到指定坐标系下的 TCP Jacobian。

    MuJoCo（`mujoco.mj_jac`）给出的是 body 原点处的 Jacobian，其线速度行是
    **body 原点速度**。要得到 TCP（site / 工具中心）处的速度，需要补上
    刚体绕 body 原点的转动贡献：

        v_tcp = v_body + w x (p_tcp - p_body)
        w_tcp = w_body                     （刚体上任意点角速度相同）

    随后可再左乘坐标系变换（`frame="world"` 时直接返回）。

    参数
    ----
    J_body : (6, 6)  行 0:3 线速度、行 3:6 角速度，列对应 6 个关节
    T_world_tcp : (4, 4)  TCP 位姿
    p_body : (3,)  body 原点在世界系的位置；None 表示与 TCP 重合（退化为恒等）
    frame : "world" | "tcp"

    返回 (6, 6)。这是一段纯函数逻辑，因此测试不需要安装 MuJoCo 也能验证。
    """
    J = np.asarray(J_body, dtype=np.float64)
    T = np.asarray(T_world_tcp, dtype=np.float64)
    if J.shape != (6, 6):
        raise ValueError("J_body must have shape (6, 6)")
    if T.shape != (4, 4):
        raise ValueError("T_world_tcp must have shape (4, 4)")
    out = J.copy()
    if p_body is not None:
        p_body = np.asarray(p_body, dtype=np.float64).reshape(3)
        r = T[:3, 3] - p_body
        # v_tcp = v_body + w x r  =>  J_v_tcp = J_v - skew(r) J_w
        out[:3, :] = J[:3, :] + np.cross(J[3:, :].T, r).T
        # 展开写法等价于: out[:3, :] = J[:3, :] - _skew(r) @ J[3:, :]
    if frame == "world":
        return out
    if frame == "tcp":
        R = T[:3, :3]
        Rt = R.T
        out[:3, :] = Rt @ out[:3, :]
        out[3:, :] = Rt @ out[3:, :]
        return out
    raise ValueError("frame must be 'world' or 'tcp'")


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """标准 DH 参数的单连杆变换（Craig 约定下的经典形式）。"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def homogeneous(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return T


def check_homogeneous(T: np.ndarray, atol: float = 1e-8) -> None:
    """校验 4x4 齐次矩阵合法性（与 src/contracts.Pose 同一标准）。"""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("T must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=atol):
        raise ValueError("invalid homogeneous bottom row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        raise ValueError("rotation block must be in SO(3)")
