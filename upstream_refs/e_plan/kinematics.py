"""
UR5e 运动学（MuJoCo-free 独立实现）
====================================

与 D 模块 `ik.py` 对齐的接口，用于 E 模块的轨迹生成，同时不依赖 MuJoCo 与
ur5e.xml，可在任何环境运行验证：

    get_fk(q)                     -> 4x4 TCP(attachment_site) 位姿   [对应 D.get_fk]
    get_fk_flange(q)              -> 4x4 法兰(wrist_3_link) 位姿     [对应 C.fk]
    ik_dls(T_target, q_init, ...) -> (q, success, info_dict)          [对应 D.ik_dls]

关键坐标系说明（F 模块提醒）：
    C 的 fk(q) 返回法兰 wrist_3_link；D 的 get_fk(q) 返回工具 TCP attachment_site。
    两者相差：flange->TCP = 平移 [0,0.1,0]（沿法兰局部 y，100 mm）+ 旋转 Rx(-90°)。
    集成系统中以 D 的 get_fk（读取工程实际 ur5e.xml）为权威，本模块用标准 ur5e.xml
    的参数复现同一 TCP 定义（若工程 xml 为 99.5 mm，替换 self._tcp_pos 即可）。

模型结构来自 mujoco_menagerie 标准 universal_robots_ur5e/ur5e.xml（去掉 mesh 的
纯运动学树），body 顺序与关节轴均按 xml 原样复现，已与 MuJoCo mj_forward 逐点交叉
验证（误差 ~1e-16）。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 四元数 / 旋转工具（MuJoCo 约定 quat = [w,x,y,z]）
# ---------------------------------------------------------------------------

def _qrot(q):
    q = np.asarray(q, float) / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rot4(axis, theta):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    T = np.eye(4)
    T[:3, :3] = _qrot([np.cos(theta / 2)] + list(np.sin(theta / 2) * axis))
    return T


def _hm(pos, quat):
    T = np.eye(4)
    T[:3, :3] = _qrot(quat)
    T[:3, 3] = np.asarray(pos, float)
    return T


def rot_vector(R):
    """旋转矩阵 -> 旋转向量（theta*单位轴）。"""
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-12:
        return np.zeros(3)
    u = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return theta * u


def pose4(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def mat_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        q = (0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = ((R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S)
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S)
    else:
        S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S)
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    return _qrot(q)


def quat_slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


# ---------------------------------------------------------------------------
# UR5e 运动学
# ---------------------------------------------------------------------------

# body 树: (name, parent, pos, quat, joint_axis or None)
# 关节顺序: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
_UR5E_BODIES = [
    ("base",     None,       [0, 0, 0],         [0, 0, 0, -1], None),
    ("shoulder", "base",     [0, 0, 0.163],     [1, 0, 0, 0],  [0, 0, 1]),
    ("upper_arm","shoulder", [0, 0.138, 0],     [1, 0, 1, 0],  [0, 1, 0]),
    ("forearm",  "upper_arm",[0, -0.131, 0.425],[1, 0, 0, 0],  [0, 1, 0]),
    ("wrist1",   "forearm",  [0, 0, 0.392],     [1, 0, 1, 0],  [0, 1, 0]),
    ("wrist2",   "wrist1",   [0, 0.127, 0],     [1, 0, 0, 0],  [0, 0, 1]),
    ("wrist3",   "wrist2",   [0, 0, 0.1],       [1, 0, 0, 0],  [0, 1, 0]),
]
_JOINT_BODY_ORDER = [1, 2, 3, 4, 5, 6]          # 各关节对应的 body 下标

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


class RobotModel:
    """UR5e + TCP(attachment_site)，与 D.ik.py 接口一致。

    tcp_offset: 可选 (pos3, quat4)，默认来自标准 ur5e.xml：
        pos=[0,0.1,0], quat=[-1,1,0,0]（即 flange->TCP 平移 100mm + Rx(-90°)）。
    """

    def __init__(self, tcp_offset=None):
        self.n = 6
        if tcp_offset is None:
            tcp_offset = ([0.0, 0.1, 0.0], [-1.0, 1.0, 0.0, 0.0])
        self._tcp_pos = np.asarray(tcp_offset[0], float)
        self._tcp_quat = np.asarray(tcp_offset[1], float)
        # 关节限位（rad）：elbow(J3) 最严 ±3.1415
        self.joint_limits = [(-6.28319, 6.28319)] * 6
        self.joint_limits[2] = (-3.1415, 3.1415)

    # ------------------------------------------------------------------
    def _frames(self, q):
        """返回所有 body 位姿 dict、TCP 位姿、各关节轴(基座系)、各关节原点。"""
        q = np.asarray(q, float)
        T = {}
        for i, (name, par, pos, quat, ax) in enumerate(_UR5E_BODIES):
            if par is None:
                T[name] = _hm(pos, quat)
            else:
                t = _hm(pos, quat)
                if ax is not None:
                    k = _JOINT_BODY_ORDER.index(i)
                    t = t @ _rot4(ax, q[k])
                T[name] = T[par] @ t
        T_tcp = T["wrist3"] @ _hm(self._tcp_pos, self._tcp_quat)
        axes, origins = [], []
        for i, (name, par, pos, quat, ax) in enumerate(_UR5E_BODIES):
            if ax is not None:
                axes.append(T[name][:3, :3] @ np.asarray(ax, float))
                origins.append(T[name][:3, 3])
        return T, T_tcp, np.array(axes), np.array(origins)

    # ------------------------------------------------------------------
    def get_fk_flange(self, q):
        """法兰位姿（wrist_3_link），对应 C.fk(q)。"""
        T, _, _, _ = self._frames(q)
        return T["wrist3"].copy()

    def get_fk(self, q):
        """TCP 位姿（attachment_site），对应 D.get_fk(q)。返回 4x4。"""
        _, T_tcp, _, _ = self._frames(q)
        return T_tcp.copy()

    def fk(self, q):
        """别名：默认返回 TCP（与轨迹规划一致）。"""
        return self.get_fk(q)

    # ------------------------------------------------------------------
    def jacobian_at_tcp(self, q):
        """TCP 处 6x6 几何雅可比（线速度 + 角速度）。"""
        _, T_tcp, axes, origins = self._frames(q)
        p = T_tcp[:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            z = axes[i]
            J[:3, i] = np.cross(z, p - origins[i])
            J[3:, i] = z
        return J

    # ------------------------------------------------------------------
    def ik_dls(self, T_target, q_init, max_iter=200, lam=0.01, alpha=0.5):
        """阻尼最小二乘 IK（TCP 参考点），签名与 D.ik_dls 一致。

        返回 (q, success, info)，info = {"iterations","pos_err","rot_err"}。
        """
        q = np.asarray(q_init, float).copy()
        p_t = T_target[:3, 3]
        R_t = T_target[:3, :3]
        pos_err = rot_err = np.inf
        for it in range(max_iter):
            T = self.get_fk(q)
            e_pos = p_t - T[:3, 3]
            e_rot = rot_vector(R_t @ T[:3, :3].T)
            pos_err = np.linalg.norm(e_pos)
            rot_err = np.linalg.norm(e_rot)
            if pos_err < 1e-4 and rot_err < 1e-3:
                return q, True, {"iterations": it, "pos_err": pos_err, "rot_err": rot_err}
            J = self.jacobian_at_tcp(q)
            e = np.hstack([e_pos, e_rot])
            A = J @ J.T + lam ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(A, e)
            q = q + alpha * dq
        ok = (pos_err < 1e-4 and rot_err < 1e-3)
        return q, ok, {"iterations": max_iter, "pos_err": pos_err, "rot_err": rot_err}

    def ik_dls_limited(self, T_target, q_init, max_iter=200, lam=0.01, alpha=0.5,
                       q_min=None, q_max=None, step_cap=0.05):
        """带关节限位裁剪 + 单步限幅的 IK（F 推荐的增强版，默认 UR5e ctrlrange）。"""
        q_min = np.array([-6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319]) \
            if q_min is None else np.asarray(q_min, float)
        q_max = np.array([6.28319, 6.28319, 3.1415, 6.28319, 6.28319, 6.28319]) \
            if q_max is None else np.asarray(q_max, float)
        q = np.clip(np.asarray(q_init, float), q_min, q_max)
        p_t = T_target[:3, 3]
        R_t = T_target[:3, :3]
        pos_err = rot_err = np.inf
        for it in range(max_iter):
            T = self.get_fk(q)
            e_pos = p_t - T[:3, 3]
            e_rot = rot_vector(R_t @ T[:3, :3].T)
            pos_err = np.linalg.norm(e_pos)
            rot_err = np.linalg.norm(e_rot)
            if pos_err < 1e-4 and rot_err < 1e-3:
                return q, True, {"iterations": it, "pos_err": pos_err, "rot_err": rot_err}
            J = self.jacobian_at_tcp(q)
            e = np.hstack([e_pos, e_rot])
            dq = alpha * (J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e))
            n = np.linalg.norm(dq)
            if n > step_cap:
                dq = dq * (step_cap / n)
            q = np.clip(q + dq, q_min, q_max)
        ok = (pos_err < 1e-4 and rot_err < 1e-3)
        return q, ok, {"iterations": max_iter, "pos_err": pos_err, "rot_err": rot_err}

    def ik(self, T_target, q_init=None):
        """简写：q_init 缺省用零位。返回 (q, success, info)。"""
        if q_init is None:
            q_init = np.zeros(6)
        return self.ik_dls(T_target, q_init)

    # ------------------------------------------------------------------
    def verify_ik(self, n_samples=100, seed=0):
        """往返验证：FK(IK(pose))≈pose（TCP 参考点）。返回 (max_pos, max_rot, fails)。"""
        rng = np.random.default_rng(seed)
        max_p = max_o = 0.0
        fails = 0
        for _ in range(n_samples):
            q = np.array([rng.uniform(-2.5, 2.5) for _ in range(6)])
            q[2] = rng.uniform(-2.8, 2.8)
            T = self.get_fk(q)
            qik, ok, _ = self.ik_dls(T, q_init=q + rng.normal(0, 0.15, 6))
            T2 = self.get_fk(qik)
            ep = np.linalg.norm(T[:3, 3] - T2[:3, 3])
            eo = np.linalg.norm(rot_vector(T[:3, :3] @ T2[:3, :3].T))
            if not ok:
                fails += 1
            max_p = max(max_p, ep)
            max_o = max(max_o, eo)
        return max_p, max_o, fails

    # ------------------------------------------------------------------
    def is_reachable(self, p, tol=1e-3):
        """按 TCP 位置粗判是否在工作空间内（用于不可解点演示）。"""
        return bool(np.linalg.norm(p) <= 0.85 + tol)
