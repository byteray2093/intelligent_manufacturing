"""G-6：真实 MuJoCo 模型上的奇异位形分析（用解析 Jacobian，不是数值微分）。

与研究问题对应关系：
- "哪些位形奇异？" -> 关节空间均匀采样 + σmin 分布 + 奇异构型的关节角直方图
- "给定轨迹会不会碰到奇异？" -> 沿 F 组真实执行日志（q_cmd / q_actual）逐点算 σmin
- "为什么二维切片看不全？" -> 奇异集是关节空间的低维曲面，切片必须选对关节

关键约定（报告必须写清）：
1. Jacobian 用 MuJoCo `mj_jacSite`（世界系，解析真值），不是 G 第一阶段的数值微分。
   G 已在 G-5 里验证两者逐元素一致到 1.25e-9。
2. 采样区间 = 模型自己声明的 `jnt_range`（1/2/4/5/6 轴 ±2π，3 轴 ±π），
   均匀/拉丁超立方采样，种子 20261015。
3. σmin 阈值 0.02（无单位，因为线速度行 m/rad 与角速度行 rad/rad 混合，
   **只用于相对比较**，绝对值没有物理意义——这一点在报告里必须说明）。

输出：
    results/csv/G6_sigma_distribution.csv       σmin 分位数/分布统计
    results/csv/G6_singular_config_stats.csv    奇异采样在各关节上的聚集程度
    results/csv/G6_singularity_grid_wrist.csv   腕部奇异切片 (q3 wrist_1, q4 wrist_2)
    results/csv/G6_singularity_grid_elbow.csv   肘部切片 (q1 shoulder_lift, q2 elbow)
    results/csv/G6_trajectory_sigma.csv         沿 F 真实轨迹的 σmin（q_cmd 与 q_actual）
    results/figures/G6_sigma_distribution.png
    results/figures/G6_singularity_maps.png
    results/figures/G6_singular_config_histograms.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import plots  # noqa: E402
from src.analysis.exec_log import load_exec_log  # noqa: E402
from src.analysis.mujoco_backend import UR5eModel  # noqa: E402
from src.analysis.results_io import write_csv, write_meta_json, write_series_csv  # noqa: E402
from src.analysis.singularity import analyze_trajectory_singularity  # noqa: E402
from src.analysis.workspace import sample_joint_space  # noqa: E402
from src.scene import (  # noqa: E402
    REAL_SLICE_N,
    REAL_WORKSPACE_N_SAMPLES,
    REAL_WORKSPACE_SEED,
    SCENE_ID_REAL,
    SIGMA_MIN_THRESHOLD,
    SLICE_SEED,
)

CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"

PERCENTILES = (0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0)
L_SLICE_WRIST = "(腕部) q3-q4 切片"
L_SLICE_ELBOW = "(肘部) q1-q2 切片"


def sigma_min_of_batch(model: UR5eModel, q_batch: np.ndarray) -> np.ndarray:
    """批量算 σmin：解析 Jacobian + 堆叠 SVD（一次调用算完，比逐点快得多）。"""
    J = model.jacobian_batch(q_batch)
    sv = np.linalg.svd(J, compute_uv=False)
    return sv[:, -1]


def slice_grid(model: UR5eModel, x_idx: int, y_idx: int, xs: np.ndarray, ys: np.ndarray,
               base_q: np.ndarray) -> dict:
    """固定其余 4 个关节，在 (x_idx, y_idx) 两个关节上扫网格，返回 σmin/可操作度。"""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    q = np.tile(np.asarray(base_q, dtype=np.float64), (X.size, 1))
    q[:, x_idx] = X.ravel()
    q[:, y_idx] = Y.ravel()
    J = model.jacobian_batch(q)
    sv = np.linalg.svd(J, compute_uv=False)
    return {
        "xs": xs,
        "ys": ys,
        "sigma_min": sv[:, -1].reshape(X.shape),
        "manip": np.prod(sv, axis=1).reshape(X.shape),
    }


def main(no_figs: bool = False) -> int:
    model = UR5eModel()
    d = model.describe()
    lo, hi = model.joint_limits()
    home = model.home_q()
    print(f"[场景] {SCENE_ID_REAL}")
    print(f"  模型 = {d['model_path']}  (sha256[:16]={d['model_sha256_head16']})")
    print(f"  关节限位 lo = {np.round(lo, 3)}")
    print(f"  关节限位 hi = {np.round(hi, 3)}")
    print(f"  home q = {np.round(home, 4)}")

    # ---- 1. 关节空间采样 -> σmin 分布 ------------------------------------------------
    t0 = time.time()
    q_all = sample_joint_space(int(REAL_WORKSPACE_N_SAMPLES), lo, hi,
                               seed=REAL_WORKSPACE_SEED, method="latin")
    sig_all = sigma_min_of_batch(model, q_all)
    pos_all = model.positions_batch(q_all)
    r_all = np.linalg.norm(pos_all, axis=1)
    dt = time.time() - t0
    print(f"\n[采样] {len(q_all)} 个关节构型（拉丁超立方，seed={REAL_WORKSPACE_SEED}）"
          f" 耗时 {dt:.2f} s")

    pcts = np.percentile(sig_all, PERCENTILES)
    n_sing = int(np.sum(sig_all < SIGMA_MIN_THRESHOLD))
    print(f"  σmin 分位数：")
    for p, v in zip(PERCENTILES, pcts):
        print(f"    {p:5.1f}%  {v:.6e}")
    print(f"  σmin < {SIGMA_MIN_THRESHOLD:g} 的比例：{100 * n_sing / len(sig_all):.3f}% "
          f"({n_sing}/{len(sig_all)})")
    print(f"  TCP 半径：min {r_all.min():.4f} m, max {r_all.max():.4f} m (实测最大伸展)")

    dist_rows = [{"stat": f"p{p:g}", "sigma_min": float(v)} for p, v in zip(PERCENTILES, pcts)]
    dist_rows += [
        {"stat": "n_samples", "sigma_min": float(len(sig_all))},
        {"stat": "n_below_threshold", "sigma_min": float(n_sing)},
        {"stat": "fraction_below_threshold", "sigma_min": float(n_sing / len(sig_all))},
        {"stat": "sigma_min_threshold", "sigma_min": SIGMA_MIN_THRESHOLD},
        {"stat": "radius_max_m", "sigma_min": float(r_all.max())},
        {"stat": "radius_min_m", "sigma_min": float(r_all.min())},
    ]
    write_csv(CSV_DIR / "G6_sigma_distribution.csv", dist_rows)

    # ---- 2. 奇异构型的关节角特征 -----------------------------------------------------
    sing_q = q_all[sig_all < SIGMA_MIN_THRESHOLD]
    joint_names = d["joint_names"]
    cfg_rows = []
    for j, name in enumerate(joint_names):
        v = np.rad2deg(sing_q[:, j]) if len(sing_q) else np.array([np.nan])
        allv = np.rad2deg(q_all[:, j])
        cfg_rows.append({
            "joint": name,
            "index": j,
            "all_abs_mean_deg": float(np.mean(np.abs(allv))),
            "sing_abs_mean_deg": float(np.mean(np.abs(v))),
            "sing_abs_p05_deg": float(np.percentile(np.abs(v), 5)),
            "sing_abs_p50_deg": float(np.percentile(np.abs(v), 50)),
            "sing_frac_near_0_or_180_deg": float(np.mean(
                np.minimum(np.abs(v), np.abs(np.abs(v) - 180.0)) < 5.0)),
        })
    # 腕部奇异判据：wrist_2（index 4）接近 0 或 ±180°
    if len(sing_q):
        w2 = np.rad2deg(sing_q[:, 4])
        near = np.minimum(np.abs(w2), np.abs(np.abs(w2) - 180.0))
        frac_wrist = float(np.mean(near < 5.0))
    else:
        frac_wrist = 0.0
    write_csv(CSV_DIR / "G6_singular_config_stats.csv", cfg_rows)
    print(f"\n[奇异构型特征] 采样中 σmin<{SIGMA_MIN_THRESHOLD:g} 的构型里，"
          f"{100 * frac_wrist:.1f}% 满足 |wrist_2|≈0 或 ≈180°（腕部奇异判据，容差 5°）")
    for row in cfg_rows:
        print(f"  {row['joint']:<20} 奇异样本 |角| 中位数 {row['sing_abs_p50_deg']:7.2f}°  "
              f"（全体 {row['all_abs_mean_deg']:6.2f}°）")

    # ---- 3. 关节空间二维切片 ---------------------------------------------------------
    xs = np.linspace(-np.pi, np.pi, int(REAL_SLICE_N))
    base = home.copy()

    # 腕部奇异：wrist_2 = 0 时 wrist_1 与 wrist_3 轴共线 -> 扫 (wrist_1, wrist_2)
    wrist = slice_grid(model, 3, 4, xs, xs, base)
    elbow = slice_grid(model, 1, 2, xs, xs, base)
    shoulder = slice_grid(model, 0, 1, xs, xs, base)

    write_csv(CSV_DIR / "G6_singularity_grid_wrist.csv",
              [{"q3_wrist1_rad": float(x), "q4_wrist2_rad": float(y),
                "sigma_min": float(wrist["sigma_min"][i, j]),
                "manipulability": float(wrist["manip"][i, j])}
               for i, x in enumerate(xs) for j, y in enumerate(xs)])
    write_csv(CSV_DIR / "G6_singularity_grid_elbow.csv",
              [{"q1_shoulder_lift_rad": float(x), "q2_elbow_rad": float(y),
                "sigma_min": float(elbow["sigma_min"][i, j]),
                "manipulability": float(elbow["manip"][i, j])}
               for i, x in enumerate(xs) for j, y in enumerate(xs)])
    for name, sl in (("腕部 q3(wrist_1)-q4(wrist_2)", wrist), ("肘部 q1(shoulder_lift)-q2(elbow)", elbow)):
        print(f"[切片 {name}] σmin min={sl['sigma_min'].min():.3e} "
              f"median={np.median(sl['sigma_min']):.4f} "
              f"低于阈值比例={100 * np.mean(sl['sigma_min'] < SIGMA_MIN_THRESHOLD):.2f}%")

    # ---- 3b. 切片上"精确奇异点"的位置（找出奇异流形到底落在哪） --------------------
    manifold_rows = []
    for label, xi, yi, xname, yname, sl in (
        ("腕部切片 q3(wrist_1)-q4(wrist_2)", 3, 4, "wrist_1_joint", "wrist_2_joint", wrist),
        ("肘部切片 q1(shoulder_lift)-q2(elbow)", 1, 2, "shoulder_lift_joint", "elbow_joint", elbow),
        ("肩部切片 q0(shoulder_pan)-q1(shoulder_lift)", 0, 1, "shoulder_pan_joint",
         "shoulder_lift_joint", shoulder),
    ):
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        exact = sl["sigma_min"] < 1e-9
        vals_y = np.unique(np.round(Y[exact] if exact.any() else np.zeros(1), 6))
        vals_x = np.unique(np.round(X[exact] if exact.any() else np.zeros(1), 6))
        manifold_rows.append({
            "slice": label,
            "x_joint": xname,
            "y_joint": yname,
            "n_grid": int(sl["sigma_min"].size),
            "n_sigma_min_below_1e-9": int(exact.sum()),
            "n_sigma_min_below_threshold": int(np.sum(sl["sigma_min"] < SIGMA_MIN_THRESHOLD)),
            "frac_below_threshold": float(np.mean(sl["sigma_min"] < SIGMA_MIN_THRESHOLD)),
            "sigma_min_grid_min": float(sl["sigma_min"].min()),
            "exact_singular_y_values_rad": ";".join(f"{v:.6f}" for v in vals_y) if exact.any() else "",
            "exact_singular_x_values_rad": ";".join(f"{v:.6f}" for v in vals_x) if exact.any() else "",
            "interpretation": (
                "σmin<1e-9 的点恰好是 y 方向整行/整列 -> "
                f"{yname} 取固定值时为奇异流形，与 {xname} 无关"
                if exact.any() and len(vals_x) > 3 else
                "该切片没有精确奇异点，只有病态区域"
            ),
        })
    write_csv(CSV_DIR / "G6_singular_manifold_evidence.csv", manifold_rows)
    print("\n[奇异流形定位] 切片上 σmin<1e-9 的精确奇异点：")
    for r in manifold_rows:
        print(f"  {r['slice']}: {r['n_sigma_min_below_1e-9']}/{r['n_grid']} 点精确奇异，"
              f"y 取值 {r['exact_singular_y_values_rad'] or '(无)'}")
        print(f"      -> {r['interpretation']}")

    # 奇异样本能否被两类流形解释（容差扫描）
    if len(sing_q):
        w = np.minimum(np.abs(sing_q[:, 4]), np.abs(np.abs(sing_q[:, 4]) - np.pi))
        e = np.minimum(np.abs(sing_q[:, 2]), np.abs(np.abs(sing_q[:, 2]) - np.pi))
        for tol_deg in (2.0, 5.0, 10.0, 20.0):
            m = np.deg2rad(tol_deg)
            print(f"  [解释度] 容差 {tol_deg:4.1f}°：腕部 {100 * np.mean(w < m):5.1f}%  "
                  f"肘部 {100 * np.mean(e < m):5.1f}%  并集 {100 * np.mean((w < m) | (e < m)):5.1f}%")

    # ---- 4. 沿 F 组真实执行日志的奇异性 ----------------------------------------------
    traj_sigma = None
    try:
        log = load_exec_log()
        t0 = time.time()
        sing_cmd = analyze_trajectory_singularity(
            log.t, log.q_cmd, fk=None, jac_fn=model.jacobian_tcp,
            sigma_min_threshold=SIGMA_MIN_THRESHOLD)
        sing_act = analyze_trajectory_singularity(
            log.t, log.q_actual, fk=None, jac_fn=model.jacobian_tcp,
            sigma_min_threshold=SIGMA_MIN_THRESHOLD)
        print(f"\n[真实轨迹] F 日志 {log.n} 点，解析 Jacobian 逐点分析耗时 {time.time() - t0:.2f} s")
        print(f"  q_cmd     : σmin min={sing_cmd['min_sigma_min']:.6f} @ t={sing_cmd['worst_time_s']:.3f} s，"
              f"低于阈值 {sing_cmd['n_singular_samples']}/{log.n} 点")
        print(f"  q_actual  : σmin min={sing_act['min_sigma_min']:.6f} @ t={sing_act['worst_time_s']:.3f} s，"
              f"低于阈值 {sing_act['n_singular_samples']}/{log.n} 点")
        print(f"  条件数最大值（q_cmd）= {np.nanmax(sing_cmd['condition_number']):.3e}")
        write_series_csv(CSV_DIR / "G6_trajectory_sigma.csv", {
            "time_s": log.t,
            "cycle": log.cycle,
            "sigma_min_q_cmd": sing_cmd["sigma_min"],
            "condition_number_q_cmd": sing_cmd["condition_number"],
            "manipulability_q_cmd": sing_cmd["manipulability"],
            "sigma_min_q_actual": sing_act["sigma_min"],
            "condition_number_q_actual": sing_act["condition_number"],
        })
        traj_sigma = (sing_cmd, sing_act, log)
    except FileNotFoundError as exc:
        print(f"\n[真实轨迹] 跳过：{exc}")

    # ---- 5. 图 -------------------------------------------------------------------------
    figures_ok = False
    if not no_figs:
        if not plots.available():
            print("\n[图] 跳过：当前 Python 环境没有 matplotlib。")
        else:
            try:
                note = (f"scene={SCENE_ID_REAL}; model={Path(d['model_path']).name} "
                        f"sha256[:16]={d['model_sha256_head16']}; "
                        f"jacobian=mj_jacSite(world, analytic); "
                        f"N={len(q_all)}, latin-hypercube seed={REAL_WORKSPACE_SEED}; "
                        f"threshold sigma_min={SIGMA_MIN_THRESHOLD:g}; unit: rad, 1/rad")
                # σmin 分布对比：关节空间采样 vs 真实轨迹
                if traj_sigma is not None:
                    plots.plot_sigma_distribution(
                        traj_sigma[0]["sigma_min"], sig_all,
                        "G-6 σmin 分布：F 组真实轨迹 (q_cmd) vs 关节空间均匀采样",
                        FIG_DIR / "G6_sigma_distribution.png",
                        note=f"log-scale sigma_min(J); threshold={SIGMA_MIN_THRESHOLD:g}; "
                             f"trajectory points={len(traj_sigma[0]['sigma_min'])}",
                    )
                slices = [
                    {"xs": xs, "ys": xs, "sigma_min": wrist["sigma_min"], "manip": wrist["manip"],
                     "xlabel": "q3 wrist_1_joint (rad)", "ylabel": "q4 wrist_2_joint (rad)",
                     "label": L_SLICE_WRIST},
                    {"xs": xs, "ys": xs, "sigma_min": elbow["sigma_min"], "manip": elbow["manip"],
                     "xlabel": "q1 shoulder_lift_joint (rad)", "ylabel": "q2 elbow_joint (rad)",
                     "label": L_SLICE_ELBOW},
                ]
                plots.plot_singularity_maps(
                    slices, "G-6 真实模型关键关节切片上的奇异位形分布",
                    FIG_DIR / "G6_singularity_maps.png", note=note)
                plots.plot_singular_config_histograms(
                    q_all, sing_q, joint_names,
                    "G-6 奇异构型（σmin<阈值）在各关节上的分布",
                    FIG_DIR / "G6_singular_config_histograms.png", note=note)
                figures_ok = True
            except Exception as exc:  # noqa: BLE001
                print(f"\n[图] 生成失败（CSV 已保留）：{type(exc).__name__}: {exc}")

    meta = write_meta_json(
        CSV_DIR / "G6_meta.json",
        {
            "scene_id": SCENE_ID_REAL,
            "model": d,
            "sampling": {
                "n": int(REAL_WORKSPACE_N_SAMPLES),
                "seed": REAL_WORKSPACE_SEED,
                "method": "latin",
                "limits_lo_rad": lo.tolist(),
                "limits_hi_rad": hi.tolist(),
            },
            "jacobian": "mj_jacSite (analytic, world frame)",
            "sigma_min_threshold": SIGMA_MIN_THRESHOLD,
            "threshold_note": (
                "σmin 的行单位混合了线速度(m/rad)与角速度(rad/rad)，"
                "绝对数值没有物理意义，只用于同一 Jacobian 口径下的相对比较"
            ),
            "slice_n": int(REAL_SLICE_N),
            "slice_seed": SLICE_SEED,
            "trajectory": None if traj_sigma is None else {
                "source": "results/logs/exec_log_latest.csv",
                "n": int(traj_sigma[2].n),
                "sigma_min_q_cmd_min": traj_sigma[0]["min_sigma_min"],
                "sigma_min_q_cmd_worst_time_s": traj_sigma[0]["worst_time_s"],
                "n_below_threshold_q_cmd": traj_sigma[0]["n_singular_samples"],
                "sigma_min_q_actual_min": traj_sigma[1]["min_sigma_min"],
                "n_below_threshold_q_actual": traj_sigma[1]["n_singular_samples"],
            },
            "figures_generated": figures_ok,
            "csv": sorted(p.name for p in CSV_DIR.glob("G6_*.csv")),
        },
    )
    print(f"\n[输出] CSV -> {CSV_DIR}\n       图  -> {FIG_DIR}\n       meta -> {meta}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true", help="不生成图（只出 CSV）")
    args = ap.parse_args()
    raise SystemExit(main(no_figs=args.no_figs))
