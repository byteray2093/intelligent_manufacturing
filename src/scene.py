"""G 组第一周（分析定义）固定场景。

**为什么要单独一个文件**：验收明确要求"G 的误差指标须与 E/F 统一测试场景"。
把场景参数（起终点、采样频率、时长、关节限位、目标位姿）集中在这里，
任何人修改场景都能被 diff 看到，报告里的数字才能追溯到同一组参数。

注意：这里的目标位姿不是手工编的，而是用 FK 从关节角**算出来**的
（q_goal -> T_goal）。这与 E 组的"已知可达目标"做法一致，也保证目标一定可达。
B 组冻结真实模型后，需要更新 `USE_REFERENCE_MODEL` 与 FK 来源。
"""
from __future__ import annotations

import functools

import numpy as np

from src.analysis.reference_model import UR5E_NOMINAL_MAX_REACH_M, UR5E_REFERENCE, fk_dh, ur5e_joint_limits

#: 当前是否使用 G 组的参考 DH 模型占位（B 组交付后应改为 False 并接 MuJoCo FK）
USE_REFERENCE_MODEL = True

#: 场景版本号：写入所有 CSV/图的元数据，便于对照报告
SCENE_ID = "G-stage1-scene-v1"

# ---- 统一测试场景 -------------------------------------------------------------------
# 上料区接近姿态 -> 下料区放置姿态（关节角，rad）。数值取自 UR5e 常见搬运位形，
# 只要求两处构型的 TCP 都在最大伸展范围内（在脚本中会用 FK 断言检查）。
Q_START_RAD = np.array([-0.30, -1.10, 1.20, -1.70, -1.57, 0.00])
Q_GOAL_RAD = np.array([0.45, -0.85, 0.95, -1.65, -1.20, 0.60])

#: 轨迹参数（E 组的 plan_joint 接口同样参数）
DURATION_S = 3.0
DT_S = 0.005          # 200 Hz 采样，N = 601
PLAN_DT_S = 0.005

#: 工作空间采样参数（写进图注与 CSV 元数据）
WORKSPACE_N_SAMPLES = 40000
WORKSPACE_SEED = 20261015
WORKSPACE_METHOD = "latin"
WORKSPACE_VOXEL_M = 0.05

#: 关节空间采样区间：±π（连续旋转关节不绕整圈的工程区间）
SAMPLE_LIMITS_LO = np.array([-np.pi] * 6)
SAMPLE_LIMITS_HI = np.array([np.pi] * 6)

#: 判定阈值（全局唯一，报告中必须引用同一数值）
SIGMA_MIN_THRESHOLD = 0.02   # 判定"接近奇异"的 σmin 约定阈值
IK_POS_TOL_M = 1e-4
IK_ROT_TOL_RAD = 1e-4

#: 数值 IK 参考实现的参数（真值由 D 组交付，这里只用于 G 组对照样本）
DLS_PARAMS = dict(max_iter=300, damping=5e-3, step_limit_rad=0.3)


def reference_fk(q: np.ndarray) -> np.ndarray:
    """当前场景的 FK 函数（与 C 组 `fk(q) -> T (4,4)` 接口一致）。"""
    return fk_dh(q, UR5E_REFERENCE)


def scene_summary() -> dict:
    """返回场景自检信息（起终点 TCP、半径、是否在伸展范围内）。"""
    T_start = reference_fk(Q_START_RAD)
    T_goal = reference_fk(Q_GOAL_RAD)
    return {
        "scene_id": SCENE_ID,
        "q_start_rad": Q_START_RAD.tolist(),
        "q_goal_rad": Q_GOAL_RAD.tolist(),
        "T_start": T_start.tolist(),
        "T_goal": T_goal.tolist(),
        "start_radius_m": float(np.linalg.norm(T_start[:3, 3])),
        "goal_radius_m": float(np.linalg.norm(T_goal[:3, 3])),
        "chord_length_m": float(np.linalg.norm(T_goal[:3, 3] - T_start[:3, 3])),
        "max_reach_m": UR5E_NOMINAL_MAX_REACH_M,
        "duration_s": DURATION_S,
        "dt_s": DT_S,
        "n_samples": int(round(DURATION_S / DT_S)) + 1,
        "joint_limits_lo_rad": ur5e_joint_limits()[0].tolist(),
        "joint_limits_hi_rad": ur5e_joint_limits()[1].tolist(),
    }


assert scene_summary()["start_radius_m"] <= UR5E_NOMINAL_MAX_REACH_M, "起始构型超出最大伸展"
assert scene_summary()["goal_radius_m"] <= UR5E_NOMINAL_MAX_REACH_M, "目标构型超出最大伸展"


# =====================================================================================
# 第二阶段：真实 MuJoCo 模型（B 组交付的 Menagerie UR5e）+ F 组真实搬运日志
# =====================================================================================
#: 第二阶段场景版本号（写入 G5–G8 的所有 CSV/图元数据）
SCENE_ID_REAL = "G-stage2-real-model-v1"

#: B 组交付的模型相对路径（`universal_robots_ur5e/ur5e.xml`，可用环境变量 G_UR5E_XML 覆盖）
REAL_MODEL_RELPATH = "universal_robots_ur5e/ur5e.xml"

#: F 组真实执行日志（`results/logs/exec_log_latest.csv`）对应的实验标识
EXEC_LOG_SCENE = "F-pick-place-3cycles-20261017_141328"

#: 工作空间采样：关节角在**模型自己声明的** jnt_range 内均匀采样（±2π，肘部 ±π）。
#: 这与第一阶段用 ±π 的做法不同，理由见 docs/G_METRICS_SPEC.md「采样区间」一节：
#: 真实 UR5e 的 1–6 轴确实能转 ±360°，按模型声明范围采样才是它真正的可达空间。
REAL_WORKSPACE_N_SAMPLES = 200_000
REAL_WORKSPACE_SEED = 20261015
REAL_WORKSPACE_VOXEL_M = 0.05

#: 奇异点关节切片网格（每个方向多少点）
REAL_SLICE_N = 121

#: 真实模型实测最大伸展（脚本会用全采样复核，这里只作为断言上界）
REAL_MAX_REACH_M = 1.13
REAL_FLANGE_MAX_REACH_M = 1.06

#: 可达性判定：位置误差容差与 IK 参数
REAL_IK_POS_TOL_M = 1e-4
REAL_IK_ROT_TOL_RAD = 1e-4
#: 推荐给 E 组的 IK 阻尼（本组扫描结论：λ=0.005 时 seed42 十目标全收敛）
RECOMMENDED_IK_LAMBDA = 0.005

#: G5–G8 的随机种子（写进元数据，任何人可复现）
VALIDATION_SEED = 42
SLICE_SEED = 20261016


@functools.lru_cache(maxsize=1)
def real_model():
    """懒加载真实 MuJoCo 模型（`src.analysis.mujoco_backend.UR5eModel`），进程内只建一次。"""
    from src.analysis.mujoco_backend import UR5eModel

    return UR5eModel()


def real_fk(q: np.ndarray) -> np.ndarray:
    """真实模型的 TCP 正运动学（与 C 组 `fk(q) -> T (4,4)` 接口一致，但输出 TCP）。"""
    return real_model().fk_tcp(q)


def real_scene_summary(model=None) -> dict:
    """第二阶段场景自检信息（模型指纹、关节限位、home、TCP 偏移）。"""
    m = model if model is not None else real_model()
    d = m.describe()
    return {
        "scene_id": SCENE_ID_REAL,
        "exec_log_scene": EXEC_LOG_SCENE,
        "model_path": d["model_path"],
        "model_sha256_head16": d["model_sha256_head16"],
        "tcp_offset_m": d["tcp_offset_m"],
        "home_q_rad": d["home_q_rad"],
        "joint_limits_lo_rad": d["joint_limits_lo_rad"],
        "joint_limits_hi_rad": d["joint_limits_hi_rad"],
        "workspace_n_samples": REAL_WORKSPACE_N_SAMPLES,
        "workspace_seed": REAL_WORKSPACE_SEED,
        "slice_n": REAL_SLICE_N,
        "sigma_min_threshold": SIGMA_MIN_THRESHOLD,
    }


# =====================================================================================
# 第三阶段：接入 E 组规划轨迹（E 交付的整条 pick-and-place，13.386 s / 6694 点）
# =====================================================================================
#: 第三阶段场景版本号：写入 G9 的 CSV/图元数据
SCENE_ID_E = "G-stage3-e-traj-v1"

#: E 交付轨迹的标识（文件名 + 关键参数），写进元数据便于追溯
E_TRAJ_SCENE = "E-pick-place-13.386s-6694pts"

#: E 轨迹的采样步长（= F 仿真步长）；用于校验 E 声明的 dt
E_TRAJ_DT_S = 0.002

#: E 声明的关节速度/加速度上限（rad/s, rad/s²），G9 用它判是否越界
E_TRAJ_VMAX_RAD_S = 1.0
E_TRAJ_AMAX_RAD_S2 = 3.0


def scene_summary_e(model=None) -> dict:
    """第三阶段场景自述（模型指向 + E 轨迹标识 + 分段来源）。"""
    from src.analysis import e_traj as et

    base = real_scene_summary(model)
    return {
        **base,
        "scene_id": SCENE_ID_E,
        "e_traj_scene": E_TRAJ_SCENE,
        "e_traj_dt_s": E_TRAJ_DT_S,
        "e_traj_segment_names": list(et.SEGMENT_NAMES),
        "e_traj_reported_durations_s": list(et.REPORTED_DURATIONS_S),
    }
