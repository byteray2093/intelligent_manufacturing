"""G 组可运行的检查脚本（不依赖 pytest 也能执行）。

背景：课程环境建议安装 pytest（见 requirements.txt），但组员机器上不一定装好。
因此这里的"测试"是纯 Python + assert，既能 `python tests/run_checks.py` 直接跑，
也能被 pytest 收集（test_* 函数）。

检查思路：**能用解析式对照的，绝不"看起来差不多就算了"**。
1. 数值 Jacobian 与平面 3R 臂的解析 Jacobian 逐元素对照（阈值 1e-7）
2. 中心差分精度优于前向差分（验证差分实现方向正确）
3. Jacobian 坐标系换算（body -> TCP）的 w×r 补偿
4. 旋转误差公式（已知旋转 30°）
5. 路径长度 / 直线度 / 端点最大垂距（几何可手算）
6. 五次多项式端点零速、零加速度（阈值体现单侧差分精度）
7. 可达性四分类（几何不可达 / 可达 / 数值失败 / 精度不足）
8. 采样可复现性与限位约束
9. 工作空间最大半径 ≤ 连杆总长（物理上界）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 允许直接 `python tests/run_checks.py` 运行（把仓库根加入 sys.path）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.linalg import (  # noqa: E402
    geometric_jacobian,
    jacobian_at_tcp,
    pose_errors,
    rotation_error_rad,
    so3_log,
)
from src.analysis.metrics import (  # noqa: E402
    analyze_joint_trajectory,
    cartesian_path_length_m,
    joint_acceleration_rad_s2,
    joint_velocity_rad_s,
    max_endpoint_deviation_m,
    poses_from_joint_trajectory,
    straightness_ratio,
    trajectory_envelope_m,
)
from src.analysis.singularity import (  # noqa: E402
    classify_reachability,
    singularity_metrics,
)
from src.analysis.workspace import sample_joint_space, sample_workspace  # noqa: E402
from src.analysis.reference_model import ur5e_fk  # noqa: E402
from src.planning_reference import quintic_joint_trajectory  # noqa: E402

# --------------------------------------------------------------------------------------
# 参考模型：平面 3R 臂（连杆长度 L1..L3，绕 y 轴转动，末端在 xz 平面内运动）
# 用 R_y(a) 的**标准**形式，保证 FK 与解析 Jacobian 在同一约定下：
#     R_y(a) = [[cos a, 0, sin a], [0, 1, 0], [-sin a, 0, cos a]]
# 位置： x = Σ Li cos(Σθ), z = Σ Li sin(Σθ)
# --------------------------------------------------------------------------------------
L = (0.4, 0.3, 0.2)


def planar3r_fk(q: np.ndarray) -> np.ndarray:
    """3R 平面臂正运动学（只使用 q 的前 3 维，返回 4x4）。

    姿态用 R_y(a)：R_y(a) = [[cos a, 0, sin a], [0, 1, 0], [-sin a, 0, cos a]]，
    与解析 Jacobian 的 J_w = [0,1,0]（绕 +y 轴）严格一致。
    """
    q = np.asarray(q, dtype=np.float64)
    th1, th2, th3 = q[0], q[1], q[2]
    x = L[0] * np.cos(th1) + L[1] * np.cos(th1 + th2) + L[2] * np.cos(th1 + th2 + th3)
    z = L[0] * np.sin(th1) + L[1] * np.sin(th1 + th2) + L[2] * np.sin(th1 + th2 + th3)
    a = th1 + th2 + th3
    T = np.eye(4)
    T[:3, :3] = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    T[:3, 3] = [x, 0.0, z]
    return T


def planar3r_jacobian_analytic(q: np.ndarray) -> np.ndarray:
    """解析几何 Jacobian，**体坐标（body frame）角速度行**，与数值差分口径一致。

    线速度（世界系）：dx/dθ_j = -Σ_{i>=j} Li sin(a_i)，dz/dθ_j = +Σ_{i>=j} Li cos(a_i)。
    角速度（末端体坐标）：ω_body = Rᵀ Ṙ，其中 R = R_y(a)。
        R_y(a)ᵀ dR_y(a)/da = [[0,0,1],[0,0,0],[-1,0,0]]  ->  ω_body = [0, 1, 0] · ȧ
    所以每个关节对 ω_body 的贡献都是 +ŷ（常数），与 `geometric_jacobian` 中
    so3_log(I, R(q-h)ᵀ R(q+h)) 的定义一致。
    """
    a1, a2, a3 = q[0], q[1], q[2]
    a12, a123 = a1 + a2, a1 + a2 + a3
    J = np.zeros((6, 6))
    J[0, 0] = -L[0] * np.sin(a1) - L[1] * np.sin(a12) - L[2] * np.sin(a123)
    J[0, 1] = -L[1] * np.sin(a12) - L[2] * np.sin(a123)
    J[0, 2] = -L[2] * np.sin(a123)
    J[2, 0] = L[0] * np.cos(a1) + L[1] * np.cos(a12) + L[2] * np.cos(a123)
    J[2, 1] = L[1] * np.cos(a12) + L[2] * np.cos(a123)
    J[2, 2] = L[2] * np.cos(a123)
    J[4, 0] = J[4, 1] = J[4, 2] = 1.0
    return J


# --------------------------------------------------------------------------------------
def test_numerical_jacobian_matches_analytic():
    """核心正确性检查：数值微分 Jacobian 必须复现手算解析 Jacobian。"""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(5):
        q = np.zeros(6)
        q[:3] = rng.uniform(-np.pi, np.pi, 3)
        J_num = geometric_jacobian(planar3r_fk, q, eps=1e-6, method="central")
        J_ana = planar3r_jacobian_analytic(q)
        worst = max(worst, float(np.max(np.abs(J_num - J_ana))))
    assert worst < 1e-7, f"数值 Jacobian 与解析值偏差过大: {worst:.3e}"
    print(f"  [ok] 数值 vs 解析 Jacobian 最大偏差 = {worst:.3e}（阈值 1e-7）")


def test_forward_difference_jacobian_accuracy_order():
    """一阶差分精度应明显差于中心差分（验证差分实现没有写反）。

    注意：解析值本身是精确的，所以这里比较的是"数值近似 vs 解析"的误差量级，
    断言用 10 倍相对关系而不是严格不等（避免浮点噪声让断言随机失败）。
    """
    q = np.array([0.3, -0.4, 0.9, 0.0, 0.0, 0.0])
    J_true = planar3r_jacobian_analytic(q)
    err_central = np.max(np.abs(geometric_jacobian(planar3r_fk, q, 1e-6, "central") - J_true))
    err_forward = np.max(np.abs(geometric_jacobian(planar3r_fk, q, 1e-6, "forward") - J_true))
    assert err_forward > 10 * err_central, (err_forward, err_central)
    print(f"  [ok] 中心差分误差 {err_central:.2e} 比前向差分误差 {err_forward:.2e} 小 1 个数量级以上")


def test_jacobian_at_tcp_translation():
    """body 原点与 TCP 沿 x 相差 r 时，v_tcp = v_body + w × r。"""
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.0]
    J = np.zeros((6, 6))
    J[3:, 0] = [0.0, 0.0, 1.0]  # 关节 0 产生绕 z 的角速度
    J_tcp = jacobian_at_tcp(J, T, p_body=np.array([0.0, 0.0, 0.0]))
    assert np.allclose(J_tcp[:3, 0], [0.0, 0.5, 0.0]), J_tcp[:3, 0]
    assert np.allclose(J_tcp[3:, 0], [0.0, 0.0, 1.0])
    # frame="tcp" 时，旋转到 TCP 系应把世界 x 方向的线速度变成 TCP 系下的 x 方向（此处单位阵）
    assert np.allclose(jacobian_at_tcp(J, T, frame="tcp")[:3, 0], 0.0)
    print("  [ok] Jacobian body->TCP 的 w×r 补偿正确")


def test_jacobian_predicts_actual_motion_ur5e():
    """回归测试（真实踩过的坑）：Jacobian 必须能预测实际位姿增量，包括**角速度符号**。

    早期实现把角速度行写成 so3_log(R(q-h), R(q+h))，等于 -ω，导致数值 IK
    朝反方向迭代、永远不收敛。平面 3R 的解析对照能发现该问题，但为了在任何模型上
    都能自动拦住它，这里用 UR5e 参考模型做一次"预测 vs 实测"的对照。
    """
    rng = np.random.default_rng(11)
    q = np.array([-0.30, -1.10, 1.20, -1.70, -1.57, 0.00])
    J = geometric_jacobian(ur5e_fk, q)
    T0 = ur5e_fk(q)
    dq = rng.normal(0, 1e-5, 6)
    T1 = ur5e_fk(q + np.clip(dq, -1e-5, 1e-5))
    dp_actual = T1[:3, 3] - T0[:3, 3]
    dp_pred = J[:3, :] @ np.clip(dq, -1e-5, 1e-5)
    dw_actual = so3_log(np.eye(3), T0[:3, :3].T @ T1[:3, :3])
    dw_pred = J[3:, :] @ np.clip(dq, -1e-5, 1e-5)
    assert np.linalg.norm(dp_actual - dp_pred) < 1e-8, (dp_actual, dp_pred)
    assert np.linalg.norm(dw_actual - dw_pred) < 1e-8, (dw_actual, dw_pred)
    print(f"  [ok] UR5e Jacobian 预测误差：线速度 {np.linalg.norm(dp_actual - dp_pred):.2e} m，"
          f"角速度 {np.linalg.norm(dw_actual - dw_pred):.2e} rad")


def test_trajectory_envelope_known_box():
    """包络指标：沿 x 走 1 m 的轨迹，包络应为 (1,0,0)，对角线 1 m。"""
    n = 51
    t = np.linspace(0.0, 2.0, n)
    T_seq = np.array([np.eye(4) for _ in range(n)])
    T_seq[:, 0, 3] = np.linspace(0.0, 1.0, n)
    env = trajectory_envelope_m(t, T_seq)
    assert abs(env["span_m"][0] - 1.0) < 1e-12
    assert abs(env["span_m"][1]) < 1e-12 and abs(env["span_m"][2]) < 1e-12
    assert abs(env["diagonal_m"] - 1.0) < 1e-12
    assert abs(env["max_axis_travel_m"] - 1.0) < 1e-12
    assert abs(env["t_at_max_s"][0] - 2.0) < 1e-12
    assert abs(env["t_at_min_s"][0] - 0.0) < 1e-12
    # 立方体对角线：三轴各 1 m -> sqrt(3)
    T_seq2 = np.array([np.eye(4) for _ in range(2)])
    T_seq2[0, :3, 3] = [0.0, 0.0, 0.0]
    T_seq2[1, :3, 3] = [1.0, 1.0, 1.0]
    env2 = trajectory_envelope_m([0.0, 1.0], T_seq2)
    assert abs(env2["diagonal_m"] - np.sqrt(3)) < 1e-12
    print(f"  [ok] 包络 AABB：单轴 1 m -> 对角线 1.000000 m；三轴 1 m -> √3 = {env2['diagonal_m']:.6f} m")


def test_poses_from_joint_trajectory_matches_fk():
    """poses_from_joint_trajectory 必须与逐点调用 fk 等价（唯一的上游几何依赖）。"""
    q = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, -0.5, 0.8, -1.2, 0.4, 0.1],
    ])
    t = np.array([0.0, 0.1])
    seq = poses_from_joint_trajectory(t, q, ur5e_fk)
    assert seq.shape == (2, 4, 4)
    assert np.allclose(seq[1], ur5e_fk(q[1]))
    print("  [ok] 关节轨迹 -> TCP 位姿序列 与 fk 逐点调用一致")


def test_so3_log_exact_at_large_angles():
    """so3_log 在大角度（含 180°）必须精确，不能用小角度近似。"""
    for deg in [30.0, 90.0, 179.9, 180.0]:
        th = np.deg2rad(deg)
        axis = np.array([0.3, -0.5, 0.81])
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        w = so3_log(np.eye(3), R)
        assert abs(np.linalg.norm(w) - th) < 1e-9, (deg, np.linalg.norm(w))
        assert np.linalg.norm(w / max(np.linalg.norm(w), 1e-15) - axis) < 1e-6, deg
    print("  [ok] so3_log 在 30°/90°/179.9°/180° 下角度与轴均精确")


def test_rotation_error_known_angles():
    """绕 z 轴旋转 30° 时，姿态误差应为 30°；位置误差 0.1 m。"""
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    assert abs(rotation_error_rad(np.eye(3), R) - th) < 1e-12
    assert abs(rotation_error_rad(R, np.eye(3)) - th) < 1e-12
    assert rotation_error_rad(R, R) == 0.0
    T_b = np.eye(4)
    T_b[:3, :3] = R
    T_b[:3, 3] = [0.1, 0.0, 0.0]
    pe, re = pose_errors(np.eye(4), T_b)
    assert abs(pe - 0.1) < 1e-12 and abs(re - th) < 1e-12
    print("  [ok] 姿态误差 = 30.0° 与位置误差 = 0.1 m 计算正确")


def test_straight_line_metrics():
    """沿 x 轴匀速直线的几何指标可与手算对照。"""
    n = 101
    t = np.linspace(0, 1, n)
    T_seq = np.array([np.eye(4) for _ in range(n)])
    T_seq[:, 0, 3] = t  # 沿 x 从 0 到 1 m
    assert abs(cartesian_path_length_m(T_seq, "arc") - 1.0) < 1e-12
    assert abs(cartesian_path_length_m(T_seq, "chord") - 1.0) < 1e-12
    assert abs(straightness_ratio(T_seq) - 1.0) < 1e-12
    assert max_endpoint_deviation_m(T_seq) < 1e-12

    # 折线 (0,0,0) -> (0,1,0) -> (1,1,0)：弧长 = 2，弦长 = √2，最大垂距 = √2/2
    P = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    T2 = np.array([np.eye(4) for _ in range(3)])
    T2[:, :3, 3] = P
    assert abs(cartesian_path_length_m(T2, "arc") - 2.0) < 1e-12
    assert abs(straightness_ratio(T2) - np.sqrt(2) / 2) < 1e-12
    assert abs(max_endpoint_deviation_m(T2) - np.sqrt(2) / 2) < 1e-12
    print("  [ok] 直线 / 折线的路径长度、直线度、最大垂距均与手算一致")


def test_quintic_endpoint_rest_and_monotone_time():
    """五次多项式：端点解析零速、零加速度；时间严格递增。

    阈值说明：端点速度只能用单侧差分估计，误差 O(q⁗·dt)，实测 1e-4 量级，
    因此断言用 1e-3 rad/s（详见 metrics._rest_flags 的说明）。
    """
    q0 = np.zeros(6)
    q1 = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.4])
    traj = quintic_joint_trajectory(q0, q1, duration=2.0, dt=0.005)
    v = joint_velocity_rad_s(traj.time_s, traj.q_rad)
    a = joint_acceleration_rad_s2(traj.time_s, traj.q_rad)
    assert np.all(np.diff(traj.time_s) > 0)
    assert abs(traj.time_s[0]) < 1e-12
    assert np.max(np.abs(v[0])) < 1e-3, np.max(np.abs(v[0]))
    assert np.max(np.abs(v[-1])) < 1e-3, np.max(np.abs(v[-1]))
    # 加速度峰值的量级检查：|q̈|max ≈ 5.77·Δq/T²（五次多项式解析结论）
    assert np.max(np.abs(a)) < 10.0
    assert np.allclose(traj.q_rad[0], q0) and np.allclose(traj.q_rad[-1], q1)
    print(f"  [ok] 五次多项式端点 q̇≈0（{np.max(np.abs(v[0])):.2e} rad/s）、q̈ 峰值 {np.max(np.abs(a)):.3f} rad/s²")


def test_metrics_are_self_consistent():
    """同一轨迹：弧长 ≥ 弦长，直线度 ≤ 1，峰值速度/加速度 > 0，端点标记为零速。"""
    q0 = np.zeros(6)
    q1 = np.array([0.8, -0.6, 1.0, 0.5, -0.4, 0.7])
    traj = quintic_joint_trajectory(q0, q1, 2.0, 0.005)
    m, T_seq = analyze_joint_trajectory("t", traj.time_s, traj.q_rad, planar3r_fk)
    assert m.cartesian_path_length_m >= cartesian_path_length_m(T_seq, "chord") - 1e-12
    assert 0.0 <= m.straightness_ratio <= 1.0 + 1e-12
    assert m.joint_peak_velocity_rad_s > 0
    assert m.joint_peak_acceleration_rad_s2 > 0
    assert m.start_rest and m.end_rest
    print(f"  [ok] 指标自洽：弧长 {m.cartesian_path_length_m:.4f} m, 直线度 {m.straightness_ratio:.4f}")


def test_singularity_metrics_on_known_matrices():
    """奇异点指标在已知矩阵上的取值（单位阵 / 秩亏矩阵）。"""
    m_eye = singularity_metrics(np.eye(6))
    assert abs(m_eye.sigma_min - 1) < 1e-12
    assert abs(m_eye.condition_number - 1) < 1e-12
    assert abs(m_eye.manipulability_yoshikawa - 1) < 1e-12
    assert not m_eye.is_singular
    J_sing = np.eye(6)
    J_sing[5, 5] = 0.0  # 秩 5 -> 奇异
    m_s = singularity_metrics(J_sing)
    assert m_s.sigma_min < 1e-12
    assert not np.isfinite(m_s.condition_number) or m_s.condition_number > 1e15
    assert m_s.rank == 5
    assert m_s.is_singular
    assert abs(m_s.manipulability_yoshikawa) < 1e-15
    # 病态但满秩：条件数应反映实际比例
    J_ill = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1e-3])
    m_ill = singularity_metrics(J_ill)
    assert abs(m_ill.condition_number - 1000.0) < 1e-6
    assert m_ill.rank == 6
    print("  [ok] sigma_min / 条件数 / 可操作度 / 秩 判定符合预期")


def test_reachability_classification_four_branches():
    """可达性四分类：超伸展 -> 几何不可达；成功 -> 可达；失败 -> 数值失败；误差超容差 -> 精度不足。"""
    class _Result:
        def __init__(self, success, err, it, msg=""):
            self.success, self.position_error_m, self.iterations, self.message = success, err, it, msg

    far = classify_reachability(planar3r_fk, np.array([5.0, 0.0, 0.0]), max_reach_m=0.9)
    assert far["status"] == "unreachable_geometry", far
    ok = classify_reachability(planar3r_fk, np.array([0.3, 0.0, 0.2]), max_reach_m=0.9,
                               ik_solver=lambda p, q0=None: _Result(True, 1e-6, 12, "converged"))
    assert ok["status"] == "reachable", ok
    fail = classify_reachability(planar3r_fk, np.array([0.3, 0.0, 0.2]), max_reach_m=0.9,
                                 ik_solver=lambda p, q0=None: _Result(False, 0.4, 200, "max iter"))
    assert fail["status"] == "numerical_failure", fail
    imprecise = classify_reachability(planar3r_fk, np.array([0.3, 0.0, 0.2]), max_reach_m=0.9,
                                      ik_solver=lambda p, q0=None: _Result(True, 0.05, 50, "stalled"))
    assert imprecise["status"] == "reachable_but_imprecise", imprecise
    # 未提供 IK 时不能硬判为可达或不可达
    unknown = classify_reachability(planar3r_fk, np.array([0.3, 0.0, 0.2]), max_reach_m=0.9)
    assert unknown["status"] == "unknown", unknown
    print("  [ok] 几何不可达 / 可达 / 数值失败 / 精度不足 / 未知 五类判定均正确")


def test_sampling_reproducible_and_bounded():
    """同 seed 结果一致；采样值必须落在限位内。"""
    lo = np.array([-1.0] * 6)
    hi = np.array([1.0] * 6)
    a = sample_joint_space(200, lo, hi, seed=42, method="latin")
    b = sample_joint_space(200, lo, hi, seed=42, method="latin")
    c = sample_joint_space(200, lo, hi, seed=43, method="latin")
    assert np.allclose(a, b) and not np.allclose(a, c)
    assert np.all(a >= lo - 1e-12) and np.all(a <= hi + 1e-12)
    print("  [ok] 采样可复现且严格落在关节限位内")


def test_workspace_radius_bounded_by_link_lengths():
    """工作空间半径上界 = 连杆总长（物理约束，验证 FK 与统计口径一致）。"""
    lo = np.array([-np.pi] * 6)
    hi = np.array([np.pi] * 6)
    res = sample_workspace(planar3r_fk, lo, hi, n_samples=500, seed=1, method="uniform", voxel_size_m=0.05)
    r_max = res.radial_stats["radius_max_m"]
    assert r_max <= sum(L) + 1e-9, (r_max, sum(L))
    assert r_max > 0.8 * sum(L), r_max  # 采样应接近上界
    assert res.voxel["occupied_voxels"] > 0
    print(f"  [ok] 采样最大半径 {r_max:.4f} m ≤ 连杆总长 {sum(L):.4f} m")


def test_ur5e_reference_geometry_sane():
    """UR5e 参考模型的两个可验证事实：零位 TCP 半径、伸直构型的最大半径。"""
    from src.analysis.reference_model import UR5E_NOMINAL_MAX_REACH_M, ur5e_fk

    r_zero = float(np.linalg.norm(ur5e_fk(np.zeros(6))[:3, 3]))
    # 该 DH 参数在零位下的 TCP = (-0.8172, -0.2389, -0.0055) m -> r = 0.8521 m
    # （由 d1 + a2 + a3 + d4 + d5 方向几何直接算出；B 组冻结真实模型后核对替换）
    assert abs(r_zero - 0.8520575626) < 1e-4, r_zero
    # 腕部奇异：q5 = 0 时 det(J) ≈ 0，σmin 远小于 1
    q_sing = np.array([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
    J = geometric_jacobian(ur5e_fk, q_sing, eps=1e-7)
    m = singularity_metrics(J, sigma_min_threshold=1e-2)
    assert m.is_singular, m
    assert abs(np.linalg.det(J)) < 1e-6, np.linalg.det(J)
    assert UR5E_NOMINAL_MAX_REACH_M > r_zero
    print(f"  [ok] UR5e 零位半径 {r_zero:.4f} m；q5=0 处 det(J) = {np.linalg.det(J):.2e}（腕部奇异复现）")


def test_csv_roundtrip_has_no_bom_leak():
    """写-读一次：CSV 必须是 UTF-8 with BOM，但读回来不能把 BOM 混进列名。

    这是唯一的"下游接口"约定：G 组 CSV 供 H 组报告与批改脚本直接读，
    用普通 utf-8 读会让第一列变成 '\\ufefflabel' 而触发 KeyError。
    用 tempfile 而不是 pytest 的 tmp_path，保证 `python tests/test_g_analysis.py`
    直接运行（不装 pytest）时这一项也能跑。
    """
    import tempfile

    from src.analysis.results_io import read_csv_rows, write_csv

    with tempfile.TemporaryDirectory() as td:
        path = write_csv(Path(td) / "roundtrip.csv",
                         [{"label": "a", "value": 1.5}, {"label": "b", "value": -2.0}])
        raw = path.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf"), "CSV 应写成 UTF-8 with BOM，Excel 打开中文列名才不乱码"
        rows = read_csv_rows(path)
        assert list(rows[0].keys()) == ["label", "value"], list(rows[0].keys())
        assert not any(k.startswith("\ufeff") for row in rows for k in row), rows
        assert rows[0]["label"] == "a" and float(rows[1]["value"]) == -2.0
    # 真实产物也走一遍同一入口
    real = ROOT / "results" / "csv" / "G4_interface_smoke.csv"
    if real.exists():
        real_rows = read_csv_rows(real)
        assert real_rows and "check" in real_rows[0], list(real_rows[0].keys())
        print(f"  [ok] CSV 编码 UTF-8 with BOM，读回无 BOM 泄漏；实测 G4 表 {len(real_rows)} 行")


def main() -> int:
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"运行 G 组检查：{len(tests)} 项（模型：平面 3R 参考臂 + UR5e 参考 DH；纯 NumPy）")
    failed = 0
    for fn in tests:
        print(f"[{fn.__name__}]")
        try:
            fn()
        except Exception:  # noqa: BLE001 - 检查脚本需要打印全部失败项
            failed += 1
            traceback.print_exc()
    print(f"\n结果：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
