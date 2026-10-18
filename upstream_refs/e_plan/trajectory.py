"""
轨迹生成模块（E 模块交付物，对齐 F 接口）
=========================================

坐标系与接口（按 F 模块约定）：
    - 末端参考点统一为 TCP(attachment_site)：笛卡尔直线逐点调用 get_fk/ik_dls。
      （注意 C 的 fk 返回法兰 wrist_3_link，与 TCP 差 100mm + 90°，勿混用。）
    - 导出 .npz：time[N]、q_traj[N,6]、dq_traj[N,6]、ddq_traj[N,6]，
      time 为真实秒数（dt=0.01），供控制模块 --traj-file 直接加载。
    - 采样 dt=0.01 s；导出 time 反映真实总时长（非点数）。
    - 限位：J3(elbow) ±3.1415 最严，其余 ±6.28319。

内容：
    QuinticJointSegment   关节空间五次多项式（rest-to-rest，C2）
    CartesianLineSegment 笛卡尔直线（TCP），路径参数五次曲线 -> 段端速度≈0，C1 拼接
    PickAndPlaceTrajectory 上/下料搬运轨迹：home->接近->抓取->抬升->搬运->放置->退让->home
    plan_pick_and_place() 一键生成

运动学后端：
    优先 import D 的 get_fk/ik_dls（工程真实 ur5e.xml，权威 TCP）；
    否则回退到自带的 MuJoCo-free UR5e+TCP 实现（kinematics.RobotModel）。
    可通过 .backend 查看实际使用的后端。
"""
from __future__ import annotations

import os

import numpy as np

from .kinematics import (RobotModel, pose4, mat_to_quat, quat_to_mat, quat_slerp,
                         rot_vector)


# ---------------------------------------------------------------------------
# 后端解析：优先 D 的 ik.py，否则用自带 UR5e
# ---------------------------------------------------------------------------
def _load_backend():
    """返回 (get_fk, ik_dls, ik_dls_limited, robot_model, name)。"""
    try:
        from src.kinematics.ik import get_fk as _dfk, ik_dls as _dik
        try:
            from src.kinematics.ik import ik_dls_limited as _dikl
        except Exception:
            _dikl = None
        return _dfk, _dik, _dikl, None, "D.ik.py (MuJoCo+ur5e.xml)"
    except Exception:
        _r = RobotModel()
        return _r.get_fk, _r.ik_dls, _r.ik_dls_limited, _r, "bundled UR5e (MuJoCo-free)"


BACKEND_GETFK, BACKEND_IK, BACKEND_IKLIM, BACKEND_MODEL, BACKEND_NAME = _load_backend()


def _pick_ik():
    """笛卡尔逐点 IK：优先 ik_dls_limited（F 推荐，带限位裁剪+步长上限）。"""
    return BACKEND_IKLIM if BACKEND_IKLIM is not None else BACKEND_IK


def _smooth_q(Q, t):
    """逐列 Savitzky-Golay 平滑，去除逐点 IK 数值噪声（不改变轨迹形状）。"""
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return np.asarray(Q, float)
    Q = np.asarray(Q, float)
    n = Q.shape[0]
    if n < 5:
        return Q
    win = min(15, n if n % 2 == 1 else n - 1)
    if win < 3:
        return Q
    return savgol_filter(Q, win, 3, axis=0)


# ---------------------------------------------------------------------------
# 五次多项式
# ---------------------------------------------------------------------------
def quintic_coeffs(qs, qe, T):
    """端点速度、加速度为 0 的五次多项式系数（n 关节）。"""
    d = np.asarray(qe) - np.asarray(qs)
    c = np.zeros((len(d), 6))
    c[:, 0] = qs
    c[:, 3] = 10 * d / T ** 3
    c[:, 4] = -15 * d / T ** 4
    c[:, 5] = 6 * d / T ** 5
    return c


def eval_quintic(c, t):
    q = c[:, 0] + c[:, 1] * t + c[:, 2] * t ** 2 + c[:, 3] * t ** 3 + c[:, 4] * t ** 4 + c[:, 5] * t ** 5
    dq = c[:, 1] + 2 * c[:, 2] * t + 3 * c[:, 3] * t ** 2 + 4 * c[:, 4] * t ** 3 + 5 * c[:, 5] * t ** 4
    ddq = 2 * c[:, 2] + 6 * c[:, 3] * t + 12 * c[:, 4] * t ** 2 + 20 * c[:, 5] * t ** 3
    return q, dq, ddq


# ---------------------------------------------------------------------------
# 关节空间五次多项式段
# ---------------------------------------------------------------------------
class QuinticJointSegment:
    """两关节位形之间的五次多项式，端点速度/加速度为 0（C2）。"""

    def __init__(self, qs, qe, T, dt=0.01, vmax=None, amax=None, joint_limits=None):
        self.qs = np.asarray(qs, float)
        self.qe = np.asarray(qe, float)
        self.n = len(self.qs)
        self.dt = dt
        self.vmax = np.ones(self.n) * 1.0 if vmax is None else np.asarray(vmax, float)
        self.amax = np.ones(self.n) * 3.0 if amax is None else np.asarray(amax, float)
        self.joint_limits = joint_limits
        self.T = self._scale_time(T)
        self._build()

    def _sample(self, T):
        coeffs = quintic_coeffs(self.qs, self.qe, T)
        n = int(round(T / self.dt))
        t = np.linspace(0, n * self.dt, n + 1)
        Q, dQ, ddQ = np.zeros((n + 1, self.n)), np.zeros((n + 1, self.n)), np.zeros((n + 1, self.n))
        for k, tk in enumerate(t):
            Q[k], dQ[k], ddQ[k] = eval_quintic(coeffs, tk)
        return t, Q, dQ, ddQ

    def _scale_time(self, T0):
        T = T0
        for _ in range(4):
            _, _, dQ, ddQ = self._sample(T)
            Tn = T
            for j in range(self.n):
                if self.vmax[j] > 0:
                    Tn = max(Tn, T * np.abs(dQ[:, j]).max() / self.vmax[j])
                if self.amax[j] > 0:
                    Tn = max(Tn, T * np.sqrt(max(np.abs(ddQ[:, j]).max(), 0) / self.amax[j]))
            if abs(Tn - T) < 1e-12:
                break
            T = Tn
        # 对齐到整数个 dt
        return np.ceil(max(T, self.dt) / self.dt) * self.dt

    def _build(self):
        self.t, self.q, self.dq, self.ddq = self._sample(self.T)
        self.pos = None          # 关节段无固定 TCP 路径
        self.pose_seq = None


# ---------------------------------------------------------------------------
# 笛卡尔直线段（TCP）
# ---------------------------------------------------------------------------
class CartesianLineSegment:
    """TCP 沿直线：位置线性 + 姿态 slerp；路径参数 s(t) 用五次曲线（端点静止）。

    逐点调用 IK（热启动）得到关节序列；关节速度/加速度由数值微分得到。
    """

    def __init__(self, get_fk, ik_dls, T_start, T_end, T, dt=0.01,
                 vmax=None, amax=None, joint_limits=None, q_start=None):
        self.get_fk = get_fk
        self.ik_dls = ik_dls
        self.Ts = np.asarray(T_start, float)
        self.Te = np.asarray(T_end, float)
        self.dt = dt
        self.vmax = np.ones(6) * 1.0 if vmax is None else np.asarray(vmax, float)
        self.amax = np.ones(6) * 3.0 if amax is None else np.asarray(amax, float)
        self.joint_limits = joint_limits
        # 起点关节解：优先沿用上一段末端（热启动保证分支连续），否则零位
        self.q_start = (np.zeros(6) if q_start is None else np.asarray(q_start, float)).copy()
        self.T = self._scale_time(T)
        self._build()

    def _interp_pose(self, s):
        p = (1 - s) * self.Ts[:3, 3] + s * self.Te[:3, 3]
        q0 = mat_to_quat(self.Ts[:3, :3])
        q1 = mat_to_quat(self.Te[:3, :3])
        q = quat_slerp(q0, q1, s)
        return pose4(quat_to_mat(q), p)

    def _sample(self, T):
        n = int(round(T / self.dt))
        t = np.linspace(0, n * self.dt, n + 1)
        # 路径参数 s(t)：五次曲线，端点速度/加速度为 0 -> 段端静止，C1 拼接
        s_coeff = quintic_coeffs(np.array([0.0]), np.array([1.0]), T)[0]
        Q = np.zeros((n + 1, 6))
        pose_seq = np.zeros((n + 1, 4, 4))
        fail = []
        q_prev = self.q_start.copy()
        for k in range(n + 1):
            sk = float(eval_quintic(np.array([s_coeff]), t[k])[0][0])
            Td = self._interp_pose(sk)
            qk, ok, info = self.ik_dls(Td, q_init=q_prev)
            if not ok:
                fail.append(k)
            Q[k] = qk
            pose_seq[k] = self.get_fk(qk)
            q_prev = qk.copy()
        # 求导前 Savitzky-Golay 平滑：抑制逐点 IK 数值噪声在 dt 很小时被梯度放大
        Qs = _smooth_q(Q, t)
        dQ = np.gradient(Qs, t, axis=0)
        dQs = _smooth_q(dQ, t)
        ddQ = np.gradient(dQs, t, axis=0)
        return t, Qs, dQ, ddQ, pose_seq, fail

    def _scale_time(self, T0):
        T = T0
        for _ in range(4):
            _, _, dQ, ddQ, _, _ = self._sample(T)
            Tn = T
            for j in range(6):
                if self.vmax[j] > 0:
                    Tn = max(Tn, T * np.abs(dQ[:, j]).max() / self.vmax[j])
                if self.amax[j] > 0:
                    Tn = max(Tn, T * np.sqrt(max(np.abs(ddQ[:, j]).max(), 0) / self.amax[j]))
            if abs(Tn - T) < 1e-12:
                break
            T = Tn
        return np.ceil(max(T, self.dt) / self.dt) * self.dt

    def _build(self):
        self.t, self.q, self.dq, self.ddq, self.pose_seq, self.fail_idx = self._sample(self.T)
        self.pos = self.pose_seq[:, :3, 3]
        self.success = len(self.fail_idx) == 0


# ---------------------------------------------------------------------------
# 搬运轨迹（拼接）
# ---------------------------------------------------------------------------
class PickAndPlaceTrajectory:
    """上/下料搬运：home -> 接近 -> 抓取 -> 抬升 -> 搬运 -> 放置 -> 退让 -> home。"""

    def __init__(self, home, keypoints, dt=0.002, vmax=None, amax=None,
                 joint_limits=None, backend=None, phase_durations=None):
        """
        home       : 6 关节角（UR5e keyframe home）
        keypoints  : dict（F 接口口径，单位 m/rad）:
                     up_pos=(x,y)   上料盘中心
                     down_pos=(x,y) 下料盘中心
                     hover_z=0.30   安全过渡 HOVER（接近）
                     grasp_z=0.055  抓取高度
                     lift_z=0.38    搬运抬升 LIFT
                     place_z=0.065  放置点 LOWER
                     clear_z=0.12   退让 CLEAR
                     R_down: 3x3 抓取姿态（默认 TCP 朝 -Z）
        phase_durations: dict 各段时长（s，F 实测参照），缺省用 F 的：
                     home=2.0  descend=1.6  lift=1.6  transfer=3.0
                     lower=1.6  retreat=1.2  return=2.0
        dt         : 采样周期，默认 0.002（F 仿真步长，不触发重采样）
        """
        if backend is None:
            self.get_fk, self.ik_dls, self.ik_limited, self.model, self.backend = \
                BACKEND_GETFK, BACKEND_IK, BACKEND_IKLIM, BACKEND_MODEL, BACKEND_NAME
        else:
            self.get_fk, self.ik_dls, self.ik_limited, self.model, self.backend = backend
        # 笛卡尔逐点 IK：优先 ik_dls_limited（F 推荐）
        self.ik_point = self.ik_limited if self.ik_limited is not None else self.ik_dls
        self.home = np.asarray(home, float)
        kp = keypoints
        self.dt = dt
        self.vmax = vmax
        self.amax = amax
        self.joint_limits = joint_limits
        R_down = kp.get("R_down")
        if R_down is None:
            R_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])   # TCP z -> -Z
        self.phase_durations = {"home": 2.0, "descend": 1.6, "lift": 1.6,
                                "transfer": 3.0, "lower": 1.6, "retreat": 1.2,
                                "return": 2.0}
        if phase_durations is not None:
            self.phase_durations.update(phase_durations)
        upx, upy = kp["up_pos"]
        dx, dy = kp["down_pos"]
        hz, gz = kp.get("hover_z", 0.30), kp.get("grasp_z", 0.055)
        lz, pz, cz = kp.get("lift_z", 0.38), kp.get("place_z", 0.065), kp.get("clear_z", 0.12)
        self.waypoints = {
            "approach_up": pose4(R_down, [upx, upy, hz]),      # HOVER
            "grasp_up":    pose4(R_down, [upx, upy, gz]),      # 抓取
            "lift_up":     pose4(R_down, [upx, upy, lz]),      # LIFT
            "lift_down":   pose4(R_down, [dx, dy, lz]),        # 搬运到位
            "place_down":  pose4(R_down, [dx, dy, pz]),        # LOWER
            "retreat_down":pose4(R_down, [dx, dy, cz]),        # CLEAR
        }
        self._plan()
        self._check_limits()

    # ------------------------------------------------------------------
    def _q_of(self, T_pose, q_init):
        q, ok, info = self.ik_point(T_pose, q_init=q_init)
        if not ok:
            raise RuntimeError(f"端点 IK 失败: pos_err={info['pos_err']:.2e}")
        return q

    def _plan(self):
        pd = self.phase_durations
        # 1) home -> 上料 HOVER（关节空间，approach 2.0s）
        q_app = self._q_of(self.waypoints["approach_up"], self.home)
        seg_home = QuinticJointSegment(self.home, q_app, pd["home"], self.dt,
                                       self.vmax, self.amax, self.joint_limits)
        # 2) 笛卡尔直线：descend->lift->transfer->lower->retreat（逐段热启动串联）
        cart_pairs = [
            ("approach_up", "grasp_up", "descend"),
            ("grasp_up", "lift_up", "lift"),
            ("lift_up", "lift_down", "transfer"),
            ("lift_down", "place_down", "lower"),
            ("place_down", "retreat_down", "retreat"),
        ]
        seg_carts = []
        q_prev = q_app
        for a, b, ph in cart_pairs:
            seg = CartesianLineSegment(self.get_fk, self.ik_point,
                                       self.waypoints[a], self.waypoints[b],
                                       pd[ph], self.dt, self.vmax, self.amax,
                                       self.joint_limits, q_start=q_prev)
            seg_carts.append(seg)
            q_prev = seg.q[-1].copy()          # 前段末解 -> 后段热启动
        # 3) 退让 -> home（关节空间，return 2.0s）
        q_ret = seg_carts[-1].q[-1]
        seg_home2 = QuinticJointSegment(q_ret, self.home, pd["return"], self.dt,
                                        self.vmax, self.amax, self.joint_limits)

        # 拼接（各段已对齐到整数 dt；dq/ddq 按段拼接，避免跨边界微分尖峰）
        self.segments = [seg_home] + seg_carts + [seg_home2]
        self.segment_names = ["home->hover"] + [f"{a}->{b}" for a, b, _ in cart_pairs] \
            + ["retreat->home"]
        # 各段全局起始时间偏移（真实累积时长）
        durations = [s.t[-1] for s in self.segments]
        self.segment_starts = np.cumsum([0.0] + durations[:-1])
        # 拼接：后续段丢弃首点（=上一段末点，同一时刻，避免重复时间戳）
        def _cat(attr):
            first = getattr(self.segments[0], attr)
            rest = [getattr(s, attr)[1:] for s in self.segments[1:]]
            return np.vstack([first] + rest)
        offs = list(self.segment_starts)
        self.t = np.concatenate(
            [self.segments[0].t + offs[0]] +
            [s.t[1:] + offs[i] for i, s in enumerate(self.segments[1:], start=1)])
        self.q = _cat("q")
        self.dq = _cat("dq")
        self.ddq = _cat("ddq")
        # TCP 轨迹（复核）
        self.tcp_pose = np.array([self.get_fk(q) for q in self.q])
        self.pos = self.tcp_pose[:, :3, 3]
        self.T_total = float(self.t[-1])

    def _check_limits(self):
        if self.joint_limits is not None:
            ok = all(self.joint_limits[j][0] - 1e-6 <= self.q[:, j].min()
                     and self.q[:, j].max() <= self.joint_limits[j][1] + 1e-6
                     for j in range(6))
            self.within_limits = bool(ok)
        else:
            self.within_limits = True

    # ------------------------------------------------------------------
    @property
    def peak_velocity(self):
        return np.abs(self.dq).max(axis=0)

    @property
    def peak_acceleration(self):
        return np.abs(self.ddq).max(axis=0)

    def to_npz(self, path):
        """F 接口：time[N], q_traj[N,6], dq_traj[N,6], ddq_traj[N,6]。"""
        np.savez_compressed(
            path,
            time=self.t,
            q_traj=self.q,
            dq_traj=self.dq,
            ddq_traj=self.ddq,
            dt=self.dt,
            backend=self.backend,
        )
        return path

    def to_csv(self, path):
        cols = ["t"] + [f"q{i+1}" for i in range(6)] \
             + [f"dq{i+1}" for i in range(6)] + [f"ddq{i+1}" for i in range(6)]
        data = np.hstack([self.t[:, None], self.q, self.dq, self.ddq])
        np.savetxt(path, data, delimiter=",", header=",".join(cols), comments="", fmt="%.6f")
        return path


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------
def plan_pick_and_place(home, keypoints, dt=0.002, vmax=None, amax=None,
                        joint_limits=None, backend=None, phase_durations=None):
    """一键生成搬运轨迹，返回 PickAndPlaceTrajectory。"""
    return PickAndPlaceTrajectory(home, keypoints, dt, vmax, amax,
                                  joint_limits, backend, phase_durations)
