"""UR5e 参考 DH 模型（仅用于 G 组在没有真实 MuJoCo 模型时的占位数据）。

**重要**：这不是团队最终使用的模型。真实基座 / TCP / 关节顺序以 B 组提交的
`docs/ROBOT_MODEL.md` 和 `scenes/pick_place.xml` 为准。B 组冻结模型后，G 组
必须把 `UR5E_REFERENCE` 换成从 MuJoCo 读取的真实参数，并在图上标注模型来源。

用途：
1. 第一周（分析定义阶段）先用参考模型把指标、图表、CSV 管线全部跑通；
2. 作为 C 组 FK 的**独立**交叉验证对象（两条独立代码路径比对）；
3. 参考模型是"数学上可达"的判定基准，用于区分数值 IK 失败与数学不可达。

参考的经典 UR5/UR5e DH 参数（ur_description 风格，单位 m/rad），
0 位姿下 TCP 约在 (0.8172, 0.1915, -0.0055) m，最大伸展约 0.9 m。
"""
from __future__ import annotations

import numpy as np

from src.analysis.linalg import dh_transform

#: 经典 UR5/UR5e DH 参数表： (theta_offset, d, a, alpha)
UR5E_REFERENCE: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.1625, 0.0, np.pi / 2),
    (0.0, 0.0, -0.425, 0.0),
    (0.0, 0.0, -0.3922, 0.0),
    (0.0, 0.1333, 0.0, np.pi / 2),
    (0.0, 0.0997, 0.0, -np.pi / 2),
    (0.0, 0.0996, 0.0, 0.0),
)

#: UR5e 官方关节限位（rad），来自 Universal Robots 数据手册；
#: **必须** 在 B 组发布真实 MJCF 限位后核对替换。
UR5E_JOINT_LIMITS_RAD: tuple[tuple[float, float], ...] = (
    (-2.0 * np.pi, 2.0 * np.pi),
    (-2.0 * np.pi, 2.0 * np.pi),
    (-2.0 * np.pi, 2.0 * np.pi),
    (-2.0 * np.pi, 2.0 * np.pi),
    (-2.0 * np.pi, 2.0 * np.pi),
    (-2.0 * np.pi, 2.0 * np.pi),
)

#: 名义最大伸展（m），用于"数学不可达"的预筛（保守值，仅作为快速判据的上界）。
UR5E_NOMINAL_MAX_REACH_M = 0.90


def fk_dh(q: np.ndarray, dh: tuple[tuple[float, float, float, float], ...] = UR5E_REFERENCE) -> np.ndarray:
    """标准 DH 正运动学：返回 T_base_tcp (4x4)。

    q 为 6 个关节角（rad）。基座固定为 world（若 B 组定义了非平凡 T_world_base，
    调用方在外部左乘即可）。
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError(f"q must have shape (6,); got {q.shape}")
    T = np.eye(4)
    for qi, (offset, d, a, alpha) in zip(q, dh):
        T = T @ dh_transform(qi + offset, d, a, alpha)
    return T


def ur5e_fk(q: np.ndarray) -> np.ndarray:
    """G 组第一周使用的占位 FK（接口与 C 组 `fk(q) -> T` 一致）。"""
    return fk_dh(q)


def ur5e_joint_limits() -> tuple[np.ndarray, np.ndarray]:
    limits = np.array(UR5E_JOINT_LIMITS_RAD, dtype=np.float64)
    return limits[:, 0].copy(), limits[:, 1].copy()


def flange_and_tcp_axes(q: np.ndarray):
    """返回 (T_base_tcp, 关节轴线段列表)，用于画机械臂简图。

    轴线段列表元素为 (p_start(3,), p_end(3,), axis_index)。便于在可达性/奇异点
    报告里画出代表性构型，便于人工检查坐标定义是否与 B 组一致。
    """
    q = np.asarray(q, dtype=np.float64)
    dh = UR5E_REFERENCE
    T = np.eye(4)
    frames = [T.copy()]
    joints = []
    for i, (offset, d, a, alpha) in enumerate(dh):
        # 关节 i 的轴线：当前 T 的 z 轴，长度取 a/d 的尺度
        z = T[:3, 2].copy()
        p = T[:3, 3].copy()
        # 轴段长度只用于绘图：相邻连杆距离的近似
        joints.append((p.copy(), (p + z * 0.07).copy(), i))
        T = T @ dh_transform(q[i] + offset, d, a, alpha)
        frames.append(T.copy())
    return frames, joints
