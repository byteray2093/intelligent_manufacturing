"""G-1：两种轨迹方案的指标对比（关节空间五次多项式 vs 笛卡尔直线逐点 IK）。

运行：
    python scripts/G1_trajectory_metrics.py            # 需要仓库依赖：numpy(+matplotlib)
    python scripts/G1_trajectory_metrics.py --no-figs  # 只出 CSV，不画图

输出：
    results/csv/G1_trajectory_metrics.csv        两条轨迹的全部指标（含单位后缀列名）
    results/csv/G1_joint_profile.csv             逐采样点 q / q̇ / q̈ / σmin（关节空间轨迹）
    results/csv/G1_cartesian_ik_records.csv      笛卡尔轨迹逐点 IK 收敛记录（失败案例证据）
    results/figures/G1_trajectory_compare.png    3D 路径 + XZ 剖面 + 直线偏离
    results/figures/G1_joint_profiles.png        关节角/速度/加速度三联图
    results/csv/G1_meta.json                     可复现元数据（种子、场景、环境、命令）

对应验收：至少 3 张清晰图表；注明采样条件与单位；两种轨迹可比较。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import plots  # noqa: E402
from src.analysis.linalg import geometric_jacobian, pose_errors  # noqa: E402
from src.analysis.metrics import (  # noqa: E402
    analyze_joint_trajectory,
    compare_metrics,
    joint_acceleration_rad_s2,
    joint_velocity_rad_s,
)
from src.analysis.results_io import write_csv, write_meta_json, write_series_csv  # noqa: E402
from src.analysis.singularity import analyze_trajectory_singularity, singularity_metrics  # noqa: E402
from src.planning_reference import (  # noqa: E402
    cartesian_line_trajectory,
    make_ik_solver,
    numerical_ik_dls,
    quintic_joint_trajectory,
)
from src.scene import (  # noqa: E402
    DLS_PARAMS,
    DT_S,
    DURATION_S,
    IK_POS_TOL_M,
    Q_GOAL_RAD,
    Q_START_RAD,
    SCENE_ID,
    SIGMA_MIN_THRESHOLD,
    reference_fk,
    scene_summary,
)

CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"


def main(no_figs: bool = False) -> int:
    info = scene_summary()
    print(f"[场景] {SCENE_ID}")
    print(f"  起点 TCP = {np.asarray(info['T_start'])[:3, 3].round(4)} (r={info['start_radius_m']:.4f} m)")
    print(f"  终点 TCP = {np.asarray(info['T_goal'])[:3, 3].round(4)} (r={info['goal_radius_m']:.4f} m)")
    print(f"  直线弦长 = {info['chord_length_m']:.4f} m; N = {info['n_samples']}, dt = {DT_S:g} s, T = {DURATION_S:g} s")

    T_start = np.asarray(info["T_start"])
    T_goal = np.asarray(info["T_goal"])

    # ---- 轨迹 1：关节空间五次多项式 --------------------------------------------------
    traj_joint = quintic_joint_trajectory(Q_START_RAD, Q_GOAL_RAD, DURATION_S, DT_S)

    # ---- 轨迹 2：笛卡尔直线 + 逐点 DLS IK -------------------------------------------
    solver = make_ik_solver(reference_fk, numerical_ik_dls, DLS_PARAMS)
    traj_cart, ik_records, T_cart = cartesian_line_trajectory(
        T_start, T_goal, DURATION_S, DT_S, Q_START_RAD, solver, position_tol_m=IK_POS_TOL_M
    )
    n_fail = sum(1 for r in ik_records if not r["success"])
    print(f"[IK] 笛卡尔轨迹逐点求解 {len(ik_records)} 点，失败 {n_fail} 点")
    if n_fail:
        first = next(r for r in ik_records if not r["success"])
        print(f"     首个失败点：index={first['index']}, t={first['time_s']:.3f} s, "
              f"pos_err={first['position_error_m']:.3e} m, msg={first['message']}")

    # ---- 指标 ------------------------------------------------------------------------
    m_joint, T_joint = analyze_joint_trajectory(
        "joint_quintic", traj_joint.time_s, traj_joint.q_rad, reference_fk, target_pose=T_goal
    )
    m_cart, _ = analyze_joint_trajectory(
        "cartesian_line_dls_ik", traj_cart.time_s, traj_cart.q_rad, reference_fk, target_pose=T_goal
    )
    # 笛卡尔方案的"实际末端"用 FK 回代（而不是理想直线），同时给出两者偏差
    m_cart.extra["planned_vs_actual_max_position_gap_m"] = float(
        np.max(np.linalg.norm(T_cart[:, :3, 3] - np.array([reference_fk(q)[:3, 3] for q in traj_cart.q_rad]), axis=1))
    )

    write_csv(CSV_DIR / "G1_trajectory_metrics.csv", compare_metrics([m_joint, m_cart]))
    print("\n[指标对比]（单位见列名）")
    header = f"{'指标':<34}{'关节五次多项式':>18}{'笛卡尔直线+IK':>18}"
    print(header)
    print("-" * len(header))
    rows = [
        ("关节路径长度 (rad)", m_joint.joint_path_length_rad, m_cart.joint_path_length_rad),
        ("末端路径弧长 (m)", m_joint.cartesian_path_length_m, m_cart.cartesian_path_length_m),
        ("直线度 chord/arc", m_joint.straightness_ratio, m_cart.straightness_ratio),
        ("末端偏离直线最大值 (m)", m_joint.max_endpoint_deviation_m, m_cart.max_endpoint_deviation_m),
        ("关节峰值速度 (rad/s)", m_joint.joint_peak_velocity_rad_s, m_cart.joint_peak_velocity_rad_s),
        ("关节峰值加速度 (rad/s²)", m_joint.joint_peak_acceleration_rad_s2, m_cart.joint_peak_acceleration_rad_s2),
        ("末端峰值线速度 (m/s)", m_joint.cartesian_peak_velocity_m_s, m_cart.cartesian_peak_velocity_m_s),
        ("终点位置误差 (m)", m_joint.endpoint_position_error_m, m_cart.endpoint_position_error_m),
        ("终点姿态误差 (rad)", m_joint.endpoint_rotation_error_rad, m_cart.endpoint_rotation_error_rad),
    ]
    for name, a, b in rows:
        print(f"{name:<34}{a:>18.6g}{b:>18.6g}")
    print(f"{'起点/终点零速':<32}{str(m_joint.start_rest) + '/' + str(m_joint.end_rest):>18}"
          f"{str(m_cart.start_rest) + '/' + str(m_cart.end_rest):>18}")

    # ---- 奇异点沿轨迹演化 -------------------------------------------------------------
    sing_joint = analyze_trajectory_singularity(
        traj_joint.time_s, traj_joint.q_rad, reference_fk, sigma_min_threshold=SIGMA_MIN_THRESHOLD
    )
    v = joint_velocity_rad_s(traj_joint.time_s, traj_joint.q_rad)
    a = joint_acceleration_rad_s2(traj_joint.time_s, traj_joint.q_rad)
    write_series_csv(
        CSV_DIR / "G1_joint_profile.csv",
        {
            "time_s": traj_joint.time_s,
            "q_rad": traj_joint.q_rad,
            "qdot_rad_s": v,
            "qddot_rad_s2": a,
            "sigma_min_J": sing_joint["sigma_min"],
            "condition_number_J": sing_joint["condition_number"],
            "manipulability_J": sing_joint["manipulability"],
        },
    )
    write_csv(CSV_DIR / "G1_cartesian_ik_records.csv", ik_records)
    print(f"\n[奇异点·关节轨迹] σmin 最小值 {sing_joint['min_sigma_min']:.5f} "
          f"@ t={sing_joint['worst_time_s']:.3f} s；低于阈值 {SIGMA_MIN_THRESHOLD:g} 的采样点 "
          f"{sing_joint['n_singular_samples']}/{len(traj_joint.time_s)}")

    print(f"  条件数最大值 {np.max(sing_joint['condition_number']):.3e}")

    # ---- 图 --------------------------------------------------------------------------
    figures_ok = False
    if not no_figs:
        if not plots.available():
            print("\n[图] 跳过：当前 Python 环境没有 matplotlib。"
                  "安装后重跑即可出图：python -m pip install -r requirements.txt")
        else:
            try:
                note = (
                    f"scene={SCENE_ID}; model=UR5e reference DH (placeholder, B-group model pending); "
                    f"N={info['n_samples']}, dt={DT_S:g} s, duration={DURATION_S:g} s; unit: m/rad/s"
                )
                plots.plot_trajectory_comparison(
                    traj_joint.time_s, T_joint, traj_cart.time_s, T_cart,
                    "G-1 两种轨迹末端路径对比 | joint quintic vs cartesian line + DLS IK",
                    FIG_DIR / "G1_trajectory_compare.png", note=note,
                )
                plots.plot_joint_profiles(
                    traj_joint.time_s, traj_joint.q_rad, v, a,
                    "G-1 关节空间五次多项式：q / q̇ / q̈",
                    FIG_DIR / "G1_joint_profiles.png", note=note,
                )
                plots.plot_singularity_profile(
                    traj_joint.time_s, sing_joint,
                    "G-1 关节空间轨迹上的奇异点指标（中心差分 Jacobian，eps=1e-6 rad）",
                    FIG_DIR / "G1_singularity_profile.png", note=note,
                )
                # 两条轨迹的 σmin 分布对比
                sing_cart = analyze_trajectory_singularity(
                    traj_cart.time_s, traj_cart.q_rad, reference_fk, sigma_min_threshold=SIGMA_MIN_THRESHOLD
                )
                plots.plot_sigma_distribution(
                    sing_joint["sigma_min"], sing_cart["sigma_min"],
                    "G-1 σmin 分布：关节轨迹 vs 笛卡尔轨迹",
                    FIG_DIR / "G1_sigma_distribution.png",
                    note=f"log-scale σmin(J); threshold={SIGMA_MIN_THRESHOLD:g}",
                )
                figures_ok = True
            except Exception as exc:  # noqa: BLE001 - 画图失败不应丢掉已算出的数据
                print(f"\n[图] 生成失败（CSV 已保留）：{type(exc).__name__}: {exc}")

    meta = write_meta_json(
        CSV_DIR / "G1_meta.json",
        {
            "scene": info,
            "trajectory_params": {"duration_s": DURATION_S, "dt_s": DT_S, "plan_dt_s": DT_S},
            "ik_reference_impl": {"algo": "damped least squares (G reference only)", **DLS_PARAMS},
            "singularity": {
                "jacobian": "central finite difference on FK, eps=1e-6 rad",
                "sigma_min_threshold": SIGMA_MIN_THRESHOLD,
                "trajectory_joint_min_sigma_min": sing_joint["min_sigma_min"],
                "trajectory_joint_worst_time_s": sing_joint["worst_time_s"],
            },
            "figures": sorted(p.name for p in FIG_DIR.glob("G1_*.png")) if figures_ok else [],
            "figures_generated": figures_ok,
            "csv": sorted(p.name for p in CSV_DIR.glob("G1_*.csv")),
        },
    )
    print(f"\n[输出] CSV -> {CSV_DIR}\n       图  -> {FIG_DIR}\n       meta -> {meta}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true", help="不生成图（只出 CSV，用于无 matplotlib 环境）")
    args = ap.parse_args()
    raise SystemExit(main(no_figs=args.no_figs))
