"""G-3：奇异点定位、σmin 分布、可达性失败案例分析。

运行：
    python scripts/G3_singularity_reachability.py
    python scripts/G3_singularity_reachability.py --grid 81

输出：
    results/figures/G3_singularity_maps.png      两个关键关节切片的 σmin / 可操作度热图
    results/figures/G3_sigma_distribution.png    工作空间采样 vs 轨迹采样的 σmin 分布
    results/csv/G3_singularity_grid_wrist.csv    (q4,q5) 切片原始数据
    results/csv/G3_singularity_grid_elbow.csv    (q2,q3) 切片原始数据
    results/csv/G3_singularity_summary.csv       每个切片的 σmin / 可操作度统计
    results/csv/G3_reachability_cases.csv        失败案例表（几何不可达 / 数值失败 / 可达）
    results/csv/G3_meta.json                     可复现元数据

**本脚本刻意展示一个容易搞错的点**：
奇异位形是关节空间中的**低维曲面**（UR5e 的腕部奇异是 q5 = 0 的 5 维超曲面，
肘部奇异是一条曲线）。因此：
- 在 (q1,q2) 平面上取切片，σmin 几乎不变（因为绕基座转动 q1 与 q2 不会改变
  "腕部是否共线"），看起来"处处不奇异"——这是**切片选择错误**，不是"没有奇异"；
- 在 (q4,q5) 平面上取切片，就能看到 q5 = 0 处 σmin 直接掉到机器精度量级；
- 在 (q2,q3) 平面上取切片，可以看到肘部接近伸直的脊线。
所以报告里必须写清楚：**奇异性结论依赖所选关节切片，单张切片不能证明"不存在奇异"**。
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
from src.analysis.plots import L  # noqa: E402  # 双语标签助手：中文字体缺失时自动回退英文
from src.analysis.linalg import geometric_jacobian  # noqa: E402
from src.analysis.results_io import write_csv, write_meta_json  # noqa: E402
from src.analysis.singularity import (  # noqa: E402
    analyze_trajectory_singularity,
    classify_reachability,
    singularity_metrics,
)
from src.analysis.workspace import sample_joint_space  # noqa: E402
from src.analysis.reference_model import UR5E_NOMINAL_MAX_REACH_M  # noqa: E402
from src.planning_reference import (  # noqa: E402
    make_ik_solver,
    numerical_ik_dls,
    quintic_joint_trajectory,
)
from src.scene import (  # noqa: E402
    DLS_PARAMS,
    DT_S,
    DURATION_S,
    Q_GOAL_RAD,
    Q_START_RAD,
    SAMPLE_LIMITS_HI,
    SAMPLE_LIMITS_LO,
    SCENE_ID,
    SIGMA_MIN_THRESHOLD,
    reference_fk,
)

CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"

#: 两个诊断切片的固定关节值（其余关节取一个远离奇异的正常工作位形）
WRIST_FIXED = {"q1": 1.20, "q2": -1.60, "q3": -0.80, "q6": 0.0}   # 变化 q4, q5
ELBOW_FIXED = {"q1": 0.60, "q4": 0.30, "q5": -1.20, "q6": 0.0}    # 变化 q2, q3


def joint_slice_grid(idx_a: int, idx_b: int, fixed: dict, n_grid: int) -> dict:
    """在 (关节 idx_a, 关节 idx_b) 平面上计算 σmin 与可操作度网格。

    关节索引从 0 开始（0..5 对应 q1..q6）。fixed 给出其余关节角的固定值。
    """
    axis_a = np.linspace(-np.pi, np.pi, n_grid)
    axis_b = np.linspace(-np.pi, np.pi, n_grid)
    sigma_min = np.zeros((n_grid, n_grid))
    manip = np.zeros((n_grid, n_grid))
    q = np.zeros(6)
    for k, v in fixed.items():
        q[int(k[1:]) - 1] = v
    for i, a in enumerate(axis_a):
        for j, b in enumerate(axis_b):
            qq = q.copy()
            qq[idx_a] = a
            qq[idx_b] = b
            m = singularity_metrics(geometric_jacobian(reference_fk, qq), sigma_min_threshold=SIGMA_MIN_THRESHOLD)
            sigma_min[j, i] = m.sigma_min
            manip[j, i] = m.manipulability_yoshikawa
    return {"xs": axis_a, "ys": axis_b, "sigma_min": sigma_min, "manip": manip, "fixed": dict(fixed),
            "idx_a": idx_a, "idx_b": idx_b}


def reachability_cases():
    """构造并判定 6 个代表性目标，覆盖四类状态，每个都给数值证据。"""
    solver = make_ik_solver(reference_fk, numerical_ik_dls, DLS_PARAMS)
    cases = []

    # 1) 明显超出最大伸展（数学不可达，与算法无关）
    cases.append(("超出最大伸展 (r=1.6 m)", np.array([1.6, 0.0, 0.2]), np.zeros(6), "radius_check"))
    # 2) 可达目标：由 FK 生成（保证数学可达）
    T_goal = reference_fk(Q_GOAL_RAD)
    cases.append(("FK 生成的可达目标", T_goal[:3, 3], Q_GOAL_RAD, "fk_generated"))
    # 3) 同样目标但用零位初值（腕部奇异 q5=0 起步，检验初值敏感性）
    cases.append(("可达目标 + 奇异初值 q5=0", T_goal[:3, 3], np.zeros(6), "bad_init"))
    # 4) 腕部奇异目标：q5 = 0
    q_sing = np.array([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
    T_sing = reference_fk(q_sing)
    cases.append(("腕部奇异 q5=0 目标", T_sing[:3, 3], q_sing, "wrist_singularity"))
    # 5) 边界可达（接近最大伸展）
    T_start = reference_fk(Q_START_RAD)
    direction = T_start[:3, 3] / max(np.linalg.norm(T_start[:3, 3]), 1e-12)
    cases.append(("接近最大伸展 r=0.88 m", direction * 0.88, Q_START_RAD, "near_max_reach"))
    # 6) 基座正上方（r≈0）—— 手臂最短伸展也到不了，属于几何不可达
    cases.append(("基座正上方 r=0.028 m", np.array([0.02, 0.0, 0.02]), Q_START_RAD, "too_close"))

    rows = []
    for name, target_position, q_init, tag in cases:
        target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
        T_target = np.asarray(reference_fk(q_init), dtype=np.float64).copy()
        T_target[:3, 3] = target_position
        last_q = {"q": np.asarray(q_init, dtype=np.float64).copy()}

        def _ik(p, q0=None, _T=T_target):
            out = solver(_T, q0 if q0 is not None else np.zeros(6))
            last_q["q"] = np.asarray(out.q, dtype=np.float64)
            return out

        info = classify_reachability(
            reference_fk, target_position, q_init=q_init, ik_solver=_ik,
            max_reach_m=UR5E_NOMINAL_MAX_REACH_M,
        )
        m_init = singularity_metrics(geometric_jacobian(reference_fk, q_init),
                                     sigma_min_threshold=SIGMA_MIN_THRESHOLD)
        m_final = singularity_metrics(geometric_jacobian(reference_fk, last_q["q"]),
                                      sigma_min_threshold=SIGMA_MIN_THRESHOLD)
        rows.append({
            "case": name, "tag": tag,
            "target_x_m": target_position[0], "target_y_m": target_position[1], "target_z_m": target_position[2],
            "target_radius_m": float(np.linalg.norm(target_position)),
            "max_reach_m": UR5E_NOMINAL_MAX_REACH_M,
            "status": info["status"], "reason": info["reason"],
            "ik_success": info["ik_success"],
            "ik_position_error_m": info["ik_position_error_m"], "ik_iterations": info["ik_iterations"],
            "sigma_min_at_q_init": m_init.sigma_min,
            "sigma_min_at_ik_solution": m_final.sigma_min,
            "condition_number_at_ik_solution": m_final.condition_number,
            "q_init_rad": ";".join(f"{v:.6f}" for v in np.asarray(q_init).ravel()),
            "q_solution_rad": ";".join(f"{v:.6f}" for v in last_q["q"]),
        })
    return rows


def _grid_rows(slice_res: dict) -> list[dict]:
    rows = []
    for j, b in enumerate(slice_res["ys"]):
        for i, a in enumerate(slice_res["xs"]):
            rows.append({
                f"q{slice_res['idx_a'] + 1}_rad": a,
                f"q{slice_res['idx_b'] + 1}_rad": b,
                "sigma_min": slice_res["sigma_min"][j, i],
                "manipulability": slice_res["manip"][j, i],
            })
    return rows


def main(no_figs: bool = False, n_grid: int = 61) -> int:
    print("[G-3 奇异点与可达性分析]")

    # ---- 1. 两个诊断切片 -------------------------------------------------------------
    wrist = joint_slice_grid(3, 4, WRIST_FIXED, n_grid)   # (q4, q5)
    elbow = joint_slice_grid(1, 2, ELBOW_FIXED, n_grid)   # (q2, q3)
    write_csv(CSV_DIR / "G3_singularity_grid_wrist.csv", _grid_rows(wrist))
    write_csv(CSV_DIR / "G3_singularity_grid_elbow.csv", _grid_rows(elbow))

    summary_rows = []
    for name, sl in (("wrist_q4q5", wrist), ("elbow_q2q3", elbow)):
        s = sl["sigma_min"]
        idx = np.unravel_index(np.argmin(s), s.shape)
        summary_rows.append({
            "slice": name,
            "joint_x": f"q{sl['idx_a'] + 1}", "joint_y": f"q{sl['idx_b'] + 1}",
            "fixed_joints": ";".join(f"q{k[1:]}={v:.4f}" for k, v in sl["fixed"].items()),
            "n_grid": n_grid,
            "sigma_min_global_min": float(s.min()),
            "argmin_qx_rad": float(sl["xs"][idx[1]]), "argmin_qy_rad": float(sl["ys"][idx[0]]),
            "sigma_min_median": float(np.median(s)), "sigma_min_max": float(s.max()),
            "frac_below_threshold": float(np.mean(s < SIGMA_MIN_THRESHOLD)),
            "manipulability_min": float(sl["manip"].min()),
            "sigma_min_threshold": SIGMA_MIN_THRESHOLD,
        })
    write_csv(CSV_DIR / "G3_singularity_summary.csv", summary_rows)

    for r in summary_rows:
        print(f"  切片 {r['slice']:<12} σmin: min={r['sigma_min_global_min']:.3e} "
              f"@ (q{str(r['joint_x'])[1:]}={r['argmin_qx_rad']:+.3f}, q{str(r['joint_y'])[1:]}={r['argmin_qy_rad']:+.3f}) rad, "
              f"median={r['sigma_min_median']:.4f}, <阈值比例={r['frac_below_threshold'] * 100:.1f}%")

    # 额外观测：为什么 (q1,q2) 切片看不到腕部奇异（用于报告里的"切片依赖"说明）
    generic_fixed = dict(WRIST_FIXED)
    generic_slice = joint_slice_grid(0, 1, generic_fixed, 21)
    generic_stats = {
        "slice": "generic_q1q2",
        "sigma_min_median": float(np.median(generic_slice["sigma_min"])),
        "sigma_min_min": float(generic_slice["sigma_min"].min()),
        "sigma_min_max": float(generic_slice["sigma_min"].max()),
    }
    print(f"  对照：q5=0 的 (q1,q2) 通用切片 σmin 中位数 = {generic_stats['sigma_min_median']:.3e}，"
          f"说明该构型下整条腕部奇异面上 σmin≈0（切片依赖）")

    figures_ok = False
    if not no_figs and plots.available():
        try:
            note = (
                f"scene={SCENE_ID}; grid={n_grid}x{n_grid} (each slice) over +/-pi rad; "
                f"Jacobian=central difference eps=1e-6 rad; threshold sigma_min={SIGMA_MIN_THRESHOLD:g}; "
                f"wrist slice fixed q1,q2,q3,q6; elbow slice fixed q1,q4,q5,q6"
            )
            plots.plot_singularity_maps(
                [
                    {**wrist, "label": f"({L('腕部', 'wrist')}) q4-q5", "xlabel": "q4 (rad)", "ylabel": "q5 (rad)"},
                    {**elbow, "label": f"({L('肘部', 'elbow')}) q2-q3", "xlabel": "q2 (rad)", "ylabel": "q3 (rad)"},
                ],
                "G-3 关键关节切片上的奇异点分布（奇异集是低维曲面 -> 切片必须选对）",
                FIG_DIR / "G3_singularity_maps.png", note=note,
            )
            figures_ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"[图] G3_singularity_maps 生成失败：{type(exc).__name__}: {exc}")
    elif not no_figs:
        print("\n[图] 跳过：当前 Python 环境没有 matplotlib。安装后重跑："
              "python -m pip install -r requirements.txt")

    # ---- 2. σmin 分布：关节空间采样 vs 场景轨迹 ---------------------------------------
    n_ws = 3000
    q_ws = sample_joint_space(n_ws, SAMPLE_LIMITS_LO, SAMPLE_LIMITS_HI, seed=20261015, method="latin")
    ws_sigma = np.array([singularity_metrics(geometric_jacobian(reference_fk, q)).sigma_min for q in q_ws])
    traj = quintic_joint_trajectory(Q_START_RAD, Q_GOAL_RAD, DURATION_S, DT_S)
    sing = analyze_trajectory_singularity(traj.time_s, traj.q_rad, reference_fk,
                                          sigma_min_threshold=SIGMA_MIN_THRESHOLD)
    ws_frac = float(np.mean(ws_sigma < SIGMA_MIN_THRESHOLD))
    print(f"  关节空间采样 {n_ws} 点：σmin 中位数 {np.median(ws_sigma):.4f}，最小 {ws_sigma.min():.3e}，"
          f"< 阈值比例 {ws_frac * 100:.1f}%")
    print(f"  场景轨迹 σmin 最小值 {sing['min_sigma_min']:.3e} @ t={sing['worst_time_s']:.3f} s "
          f"（低于阈值的采样点 {sing['n_singular_samples']}/{len(traj.time_s)}）")

    if not no_figs and plots.available():
        try:
            plots.plot_sigma_distribution(
                sing["sigma_min"], ws_sigma,
                "G-3 σmin 分布对比：统一场景轨迹 vs 关节空间采样",
                FIG_DIR / "G3_sigma_distribution.png",
                note=(f"N_traj={len(traj.time_s)} (scene {SCENE_ID}); N_ws={n_ws}, seed=20261015, latin; "
                      f"threshold={SIGMA_MIN_THRESHOLD:g}; log-scale σmin(J)"),
            )
            figures_ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"[图] G3_sigma_distribution 生成失败：{type(exc).__name__}: {exc}")

    # ---- 3. 可达性 / 失败案例 --------------------------------------------------------
    case_rows = reachability_cases()
    write_csv(CSV_DIR / "G3_reachability_cases.csv", case_rows)
    print("\n  可达性案例分类：")
    label_cn = {"unreachable_geometry": "几何不可达", "reachable": "可达",
                "numerical_failure": "数值失败", "reachable_but_imprecise": "精度不足", "unknown": "未知"}
    for r in case_rows:
        print(f"    - {r['case']:<28} -> {label_cn.get(r['status'], r['status']):<8} "
              f"r={r['target_radius_m']:.3f} m, σmin(初值)={r['sigma_min_at_q_init']:.3e}, "
              f"IK成功={r['ik_success']}")
    n_fail = sum(1 for r in case_rows if r["status"] == "numerical_failure")
    n_unreach = sum(1 for r in case_rows if r["status"] == "unreachable_geometry")
    print(f"  小结：{n_unreach} 个几何不可达、{n_fail} 个数值失败（两者必须分开表述）")

    write_meta_json(
        CSV_DIR / "G3_meta.json",
        {
            "scene_id": SCENE_ID,
            "jacobian": "central finite difference on reference FK, eps=1e-6 rad, body-frame angular rows",
            "sigma_min_threshold": SIGMA_MIN_THRESHOLD,
            "slices": {
                "wrist_q4q5": {"n_grid": n_grid, "fixed": WRIST_FIXED},
                "elbow_q2q3": {"n_grid": n_grid, "fixed": ELBOW_FIXED},
                "generic_q1q2": {"n_grid": 21, "fixed": WRIST_FIXED},
            },
            "slice_summary": summary_rows,
            "generic_slice_stats": generic_stats,
            "joint_space_sampling": {
                "n": n_ws, "seed": 20261015, "method": "latin",
                "median_sigma_min": float(np.median(ws_sigma)),
                "min_sigma_min": float(ws_sigma.min()),
                "frac_below_threshold": ws_frac,
            },
            "scene_trajectory": {
                "min_sigma_min": sing["min_sigma_min"], "worst_time_s": sing["worst_time_s"],
                "n_below_threshold": sing["n_singular_samples"], "n_samples": int(len(traj.time_s)),
            },
            "reachability_cases": case_rows,
            "interpretation_notes": [
                "奇异位形是关节空间的低维曲面：腕部奇异 = {q | q5 = 0}（5 维超曲面），"
                "肘部奇异 = 手臂完全伸直的低维曲线。",
                "因此单张二维切片不能证明'某处不奇异'：切片必须与奇异面横截相交。",
                "σmin 的判定阈值 0.02 是工程约定值（无量纲），必须与报告中的定义一致。",
            ],
            "figures": sorted(p.name for p in FIG_DIR.glob("G3_*.png")) if figures_ok else [],
            "figures_generated": figures_ok,
            "csv": sorted(p.name for p in CSV_DIR.glob("G3_*.csv")),
        },
    )
    print(f"\n[输出] {CSV_DIR} / {FIG_DIR}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--grid", type=int, default=61, help="每个切片的网格分辨率")
    args = ap.parse_args()
    raise SystemExit(main(no_figs=args.no_figs, n_grid=args.grid))
