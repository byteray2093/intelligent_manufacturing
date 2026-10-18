"""G-4：复用性检查 —— 把 G 组分析代码接到"任意 FK/轨迹来源"上。

为什么要这个脚本：验收要求"G 的误差指标须与 E/F 统一测试场景"。G 组的分析函数
只依赖两个约定：
    fk(q) -> T (4x4)          （C 组提供）
    traj.time_s (N,), traj.q_rad (N,6)   （E 组提供）
因此本脚本用**不依赖 MuJoCo 的 FK-IK 闭环**演示：只要换掉 `fk` 和 `traj`，
后面的指标、奇异点、CSV 全部复用，不需要改 G 组任何一行代码。

运行：
    python scripts/G4_interface_smoke.py

输出：
    results/csv/G4_interface_smoke.csv    闭环验证与复用性证据
    results/csv/G4_meta.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.linalg import rotation_error_rad  # noqa: E402
from src.analysis.metrics import analyze_joint_trajectory  # noqa: E402
from src.analysis.results_io import write_csv, write_meta_json  # noqa: E402
from src.analysis.singularity import (  # noqa: E402
    analyze_trajectory_singularity,
    classify_reachability,
)
from src.analysis.reference_model import ur5e_fk  # noqa: E402
from src.planning_reference import (  # noqa: E402
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
    SIGMA_MIN_THRESHOLD,
)

CSV_DIR = ROOT / "results" / "csv"


def main() -> int:
    rows = []

    # ---- 验证 1：FK -> IK -> FK 闭环（D 组验收要求的"回代验证"） --------------------
    solver = make_ik_solver(ur5e_fk, numerical_ik_dls, DLS_PARAMS)
    T_target = ur5e_fk(Q_GOAL_RAD)
    result = solver(T_target, Q_START_RAD)
    T_check = ur5e_fk(result.q)
    pos_err = float(np.linalg.norm(T_check[:3, 3] - T_target[:3, 3]))
    rot_err = rotation_error_rad(T_check[:3, :3], T_target[:3, :3])
    rows.append({
        "check": "FK->IK->FK 闭环（q_goal 生成的目标位姿）",
        "detail": f"init=q_start, iter={result.iterations}, msg={result.message}",
        "value": pos_err,
        "unit": "m",
        "extra": f"orientation_error_rad={rot_err:.3e}",
        "passed": bool(result.success and pos_err < IK_POS_TOL_M),
    })
    print(f"[闭环] success={result.success} pos_err={pos_err:.3e} m rot_err={rot_err:.3e} rad "
          f"iter={result.iterations}")

    # ---- 验证 2：同一条轨迹换一个 FK 实现，指标随之变化（证明没有硬编码） -----------
    traj = quintic_joint_trajectory(Q_START_RAD, Q_GOAL_RAD, DURATION_S, DT_S)
    m_ur5e, T_ur5e = analyze_joint_trajectory("ur5e", traj.time_s, traj.q_rad, ur5e_fk)

    # 一个"假 FK"：TCP 定义不同（沿 world-x 平移 0.1 m，相当于工具偏移）
    # 注意：**路径长度**对这种刚体平移是不变量（只依赖相对运动），所以必须用
    # "TCP 绝对位置"来检验，不能误以为路径长度能反映 TCP 定义变化。
    def shifted_fk(q):
        T = np.asarray(ur5e_fk(q), dtype=np.float64).copy()
        T[:3, 3] = T[:3, 3] + np.array([0.1, 0.0, 0.0])
        return T

    m_shift, T_shift = analyze_joint_trajectory("shifted", traj.time_s, traj.q_rad, shifted_fk)
    dx_start = float(T_shift[0, 0, 3] - T_ur5e[0, 0, 3])
    d_len = float(m_shift.cartesian_path_length_m - m_ur5e.cartesian_path_length_m)
    rows.append({
        "check": "更换 FK/TCP 定义后绝对位置随之改变（无硬编码）",
        "detail": "TCP 沿 world-x 平移 0.1 m",
        "value": dx_start,
        "unit": "m",
        "extra": (f"expected=0.1; path_length_unchanged_by_rigid_offset={abs(d_len) < 1e-9} "
                  f"(L={m_ur5e.cartesian_path_length_m:.6f} m)"),
        "passed": bool(abs(dx_start - 0.1) < 1e-12 and abs(d_len) < 1e-9),
    })
    print(f"[复用] 起点 TCP x: {T_ur5e[0, 0, 3]:.6f} -> {T_shift[0, 0, 3]:.6f} m (Δ={dx_start:+.6f})；"
          f"路径长度不变 = {abs(d_len) < 1e-9}（刚体平移是不变量，符合预期）")

    # ---- 验证 3：轨迹奇异点分析对任意来源轨迹可用 -----------------------------------
    sing = analyze_trajectory_singularity(traj.time_s, traj.q_rad, ur5e_fk,
                                          sigma_min_threshold=SIGMA_MIN_THRESHOLD)
    rows.append({
        "check": "轨迹奇异点分析（任意来源轨迹均可调用）",
        "detail": f"N={len(traj.time_s)}, dt={DT_S:g} s",
        "value": sing["min_sigma_min"],
        "unit": "1",
        "extra": f"worst_t={sing['worst_time_s']:.3f} s; n_below_threshold={sing['n_singular_samples']}",
        "passed": bool(np.isfinite(sing["min_sigma_min"])),
    })
    print(f"[奇异点] σmin_min={sing['min_sigma_min']:.4e} @ t={sing['worst_time_s']:.3f} s")

    # ---- 验证 4：可达性判定接口（供 D 组记录失败案例） ------------------------------
    far = classify_reachability(ur5e_fk, np.array([1.6, 0.0, 0.2]), max_reach_m=0.9)
    rows.append({
        "check": "可达性判定：超伸展目标",
        "detail": far["reason"],
        "value": far["target_radius_m"],
        "unit": "m",
        "extra": f"status={far['status']}",
        "passed": far["status"] == "unreachable_geometry",
    })
    print(f"[可达性] {far['status']}: {far['reason']}")

    write_csv(CSV_DIR / "G4_interface_smoke.csv", rows)
    write_meta_json(CSV_DIR / "G4_meta.json", {
        "checks": rows,
        "note": "本脚本只用到 fk(q)->T 与 traj.(time_s, q_rad) 两个约定，用于证明 G 组分析可复用到 C/D/E/F 的真实实现",
    })
    n_pass = sum(1 for r in rows if r["passed"])
    print(f"\n结果：{n_pass}/{len(rows)} 项通过；CSV -> {CSV_DIR / 'G4_interface_smoke.csv'}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
