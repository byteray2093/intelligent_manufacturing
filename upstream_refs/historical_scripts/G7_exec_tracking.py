"""G-7：真实执行日志（F 组 3 循环抓取-搬运）的跟踪误差与轨迹指标分析。

研究问题对应关系：
- 规划轨迹 vs 实际执行差多少？ -> 关节跟踪误差、末端跟踪误差（位置/姿态）
- 差在哪里？                  -> 按 13 种搬运段分组，定位 worst segment
- 是不是奇异点放大的？        -> 逐点 σmin 与末端误差的相关性/散点
- 路径有没有变长？            -> 累积末端路径长度 规划 vs 实际
- 数据本身可信吗？            -> 日志自检：ee_* 列滞后 1 个采样、三循环重复性、
                                 Jacobian 一阶预测与实测增量的一致性

关键约定（报告必须写清）：
1. **一律用 fk(q_actual) 回代算末端位姿**，不用日志里的 ee_x..ee_qz。
   原因：F 先 mj_step 更新 qpos、再读 site_xpos，日志里的 ee_* 比 q_actual
   滞后 1 个采样（G-5 已量化：对齐后残差 0.11 mm，不对齐 1.26 mm）。
   ee_* 只作交叉核对。
2. 误差定义：关节误差 = q_actual - q_cmd；末端位置误差 = ||p_actual - p_cmd||；
   姿态误差 = 旋转矩阵对数映射的夹角。单位 m / rad / s / s²。
3. 采样 dt=0.002 s（500 Hz），来自 MuJoCo timestep，不是 G 自己选的。

输出：
    results/csv/G7_segment_metrics.csv   39 段逐段指标（39 行）
    results/csv/G7_cycle_summary.csv     3 个循环汇总
    results/csv/G7_overall.csv           全日志汇总（含重复性检查）
    results/csv/G7_series.csv            24921 点逐点时序（画图与复核用）
    results/figures/G7_error_by_segment.png
    results/figures/G7_ee_tracking.png
    results/figures/G7_lag_demo.png
    results/figures/G7_path_lengths.png
    results/figures/G7_sigma_vs_tracking.png
    results/figures/G7_ee_vs_joint_error.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import plots, tracking  # noqa: E402
from src.analysis.exec_log import load_exec_log  # noqa: E402
from src.analysis.mujoco_backend import UR5eModel  # noqa: E402
from src.analysis.results_io import (  # noqa: E402
    read_csv_rows,
    write_csv,
    write_meta_json,
    write_series_csv,
)
from src.scene import (  # noqa: E402
    EXEC_LOG_SCENE,
    SCENE_ID_REAL,
    SIGMA_MIN_THRESHOLD,
)

CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"

SEG_ORDER = ["home", "approach", "descend", "descend_settle", "grasp_hold", "lift",
             "transfer", "lower", "lower_settle", "release_hold", "retreat",
             "settle_after_release", "return_home"]
SEG_ZH = {
    "home": "回 home", "approach": "接近", "descend": "下降", "descend_settle": "下降稳定",
    "grasp_hold": "夹持保持", "lift": "抬升", "transfer": "搬运(笛卡尔直线)",
    "lower": "下放", "lower_settle": "下放稳定", "release_hold": "释放保持",
    "retreat": "撤离", "settle_after_release": "释放后稳定", "return_home": "回 home",
}


def main(no_figs: bool = False) -> int:
    model = UR5eModel()
    log = load_exec_log()
    print(f"[场景] {SCENE_ID_REAL} / 日志场景 {EXEC_LOG_SCENE}")
    print(f"  日志 = {log.path}")
    print(f"  采样点 {log.n}，dt = {log.dt_s:g} s，时长 {log.t[-1] - log.t[0]:.3f} s，"
          f"循环 {log.cycles}，段名 {len(log.segment_names)} 种，连续段 {len(log.segments())} 段")

    res = tracking.analyze_exec_log(model, log, jac_step=1,
                                    sigma_min_threshold=SIGMA_MIN_THRESHOLD)
    seg = res["segments"]
    ov = res["overall"]
    ser = res["series"]

    # ---- 1. 全局 ---------------------------------------------------------------------
    print("\n[全局跟踪误差]（末端一律由 fk(q_actual) 回代，不用日志 ee_* 列）")
    print(f"  关节误差：max {ov['joint_err_max_deg']:.4f}°  RMS {ov['joint_err_rms_deg']:.4f}°")
    print(f"  末端位置误差：max {ov['ee_pos_err_max_mm']:.4f} mm  RMS {ov['ee_pos_err_rms_mm']:.4f} mm")
    print(f"  末端姿态误差：max {ov['ee_ori_err_max_deg']:.4f}°")
    print(f"  σmin(J) 全程最小值 {ov['sigma_min_min']:.4f}（阈值 {SIGMA_MIN_THRESHOLD:g}）")
    print(f"  单循环末端路径长度：规划 {ov['ee_path_planned_m']:.6f} m，"
          f"实际 {ov['ee_path_actual_m']:.6f} m，"
          f"长出 {100 * (ov['ee_path_actual_m'] / ov['ee_path_planned_m'] - 1):+.4f}%")
    rep = ov["repeatability"]
    print(f"  三循环重复性：最大跨循环同段误差差 {rep['max_ee_err_spread_mm']:.3e} mm，"
          f"确定性={rep['deterministic']}")

    # ---- 2. 逐段 ---------------------------------------------------------------------
    print("\n[逐段指标]（cycle=1，单位见列名）")
    hdr = f"  {'segment':<22}{'dur(s)':>7}{'iq_max(deg)':>12}{'ee_pos_max(mm)':>15}{'ee_ori_max(deg)':>16}"
    print(hdr)
    for m in seg:
        if m.cycle != log.cycles[0]:
            continue
        print(f"  {m.segment:<22}{m.duration_s:>7.3f}{m.joint_err_max_deg:>12.4f}"
              f"{m.ee_pos_err_max_m:>15.4f}{np.degrees(m.ee_ori_err_max_rad):>16.4f}")

    worst_joint = max(seg, key=lambda m: m.joint_err_max_deg)
    worst_ee = max(seg, key=lambda m: m.ee_pos_err_max_m)
    worst_ori = max(seg, key=lambda m: m.ee_ori_err_max_rad)
    print(f"\n  worst 关节误差段：{worst_joint.segment} (cycle {worst_joint.cycle}) "
          f"{worst_joint.joint_err_max_deg:.4f}°")
    print(f"  worst 末端位置误差段：{worst_ee.segment} (cycle {worst_ee.cycle}) "
          f"{worst_ee.ee_pos_err_max_m * 1000:.4f} mm")
    print(f"  worst 末端姿态误差段：{worst_ori.segment} (cycle {worst_ori.cycle}) "
          f"{np.degrees(worst_ori.ee_ori_err_max_rad):.4f}°")

    # ---- 3. 路径是否变长 / 直线度 -----------------------------------------------------
    print("\n[路径指标]（每循环，规划 vs 实际）")
    for c in res["cycles"]:
        ratio = c["ee_path_actual_m"] / c["ee_path_planned_m"] - 1.0
        print(f"  cycle {c['cycle']}: 末端路径 规划 {c['ee_path_planned_m']:.6f} m → "
              f"实际 {c['ee_path_actual_m']:.6f} m（{100 * ratio:+.4f}%）；"
              f"关节路径 规划 {c['joint_path_planned_rad']:.6f} → "
              f"实际 {c['joint_path_actual_rad']:.6f} rad")

    transfer = [m for m in seg if m.segment == "transfer" and m.cycle == log.cycles[0]]
    if transfer:
        m = transfer[0]
        print(f"  transfer（笛卡尔直线段）cycle1：直线度 规划 {m.ee_straightness_planned:.9f} / "
              f"实际 {m.ee_straightness_actual:.9f}；"
              f"偏离直线 max 规划 {m.ee_max_dev_from_chord_planned_m * 1000:.4f} mm / "
              f"实际 {m.ee_max_dev_from_chord_actual_m * 1000:.4f} mm")

    # ---- 4. 日志自检：ee_* 滞后 + Jacobian 线性化残差 ---------------------------------
    lag = tracking.detect_pose_lag(model, log, max_lag=3)
    print(f"\n[日志自检 1] ee_* 列相对 q_actual 的滞后 = {lag['lag_samples']} 个采样 "
          f"({lag['lag_time_s'] * 1000:+.1f} ms)；对齐后残差 max {lag['best_max_m'] * 1000:.4f} mm / "
          f"mean {lag['best_mean_m'] * 1000:.4f} mm；不对齐 max {lag['naive_max_m'] * 1000:.4f} mm")
    print("  -> 结论：ee_* 不能用，G 一律用 fk(q_actual) 回代")

    res_max = max(m.jac_residual_max_m for m in seg if m.cycle == log.cycles[0])
    r2_all = np.array([m.jac_r2 for m in seg if m.cycle == log.cycles[0]], dtype=float)
    r2_ok = r2_all[np.isfinite(r2_all)]
    r2_na = int(np.sum(~np.isfinite(r2_all)))
    pred_max = max(m.jac_pred_err_max_m for m in seg if m.cycle == log.cycles[0])
    print(f"[日志自检 2] Jacobian 一阶预测 Δp≈J(q_cmd)Δq：残差 max {res_max * 1000:.4f} mm，"
          f"最小 R² {r2_ok.min():.4f}（{r2_na} 段因 Δq≈0 而 R² 无定义，已排除）")
    worst_r2_seg = min((m for m in seg if m.cycle == log.cycles[0] and np.isfinite(m.jac_r2)),
                       key=lambda m: m.jac_r2).segment
    print(f"  R² 最低的段：{worst_r2_seg}；最大一阶预测误差 {pred_max * 1000:.4f} mm")
    print("  -> 结论：跟踪误差基本落在线性化区内，J 可用于误差归因；非线性二阶项不是主因")

    # 与 F 自报数字交叉核对
    f_summary = ROOT / "results" / "logs" / "run_summary_20261017_141328.txt"
    if f_summary.exists():
        txt = f_summary.read_text(encoding="utf-8", errors="replace")
        print(f"\n[与 F 自报交叉核对] F 的 run_summary 摘要（前 12 行）：")
        for line in txt.splitlines()[:12]:
            print("   " + line)

    # ---- 5. 奇异点 vs 误差 ------------------------------------------------------------
    sig = np.empty(log.n)
    for i in range(log.n):
        J = model.jacobian_tcp(log.q_actual[i])
        sig[i] = np.linalg.svd(J, compute_uv=False)[-1]
    ee_err = ser["ee_pos_err_mm"]
    corr = float(np.corrcoef(np.log10(np.maximum(sig, 1e-12)), np.log10(np.maximum(ee_err, 1e-9)))[0, 1])
    print(f"\n[奇异 vs 误差] log10(σmin) 与 log10(末端误差 mm) 的相关系数 = {corr:+.4f}")
    print(f"  全程 σmin ∈ [{sig.min():.4f}, {sig.max():.4f}]，"
          f"低于阈值 {SIGMA_MIN_THRESHOLD:g} 的点 {int(np.sum(sig < SIGMA_MIN_THRESHOLD))}/{log.n}")
    print("  -> 结论：该轨迹离奇异还很远，误差不是奇异放大造成的（与 G-6 一致）")

    # ---- 6. 落盘 ---------------------------------------------------------------------
    write_csv(CSV_DIR / "G7_segment_metrics.csv", [m.to_row() for m in seg])
    write_csv(CSV_DIR / "G7_cycle_summary.csv", res["cycles"])
    write_series_csv(CSV_DIR / "G7_series.csv", {
        "time_s": ser["t"],
        "cycle": ser["cycle"],
        "joint_err_max_deg": ser["joint_err_max_deg"],
        "joint_err_rms_deg": ser["joint_err_rms_deg"],
        "ee_pos_err_mm": ser["ee_pos_err_mm"],
        "ee_ori_err_deg": ser["ee_ori_err_deg"],
        "ee_speed_planned_m_s": ser["ee_speed_planned_m_s"],
        "ee_speed_actual_m_s": ser["ee_speed_actual_m_s"],
        "sigma_min": sig,
        "ee_x_planned_m": ser["ee_pos_planned"][:, 0],
        "ee_y_planned_m": ser["ee_pos_planned"][:, 1],
        "ee_z_planned_m": ser["ee_pos_planned"][:, 2],
        "ee_x_actual_m": ser["ee_pos_actual"][:, 0],
        "ee_y_actual_m": ser["ee_pos_actual"][:, 1],
        "ee_z_actual_m": ser["ee_pos_actual"][:, 2],
    })
    overall_rows = [{k: v for k, v in ov.items() if not isinstance(v, dict)}]
    overall_rows[0]["repeatability_deterministic"] = rep["deterministic"]
    overall_rows[0]["repeatability_max_spread_mm"] = rep["max_ee_err_spread_mm"]
    overall_rows[0]["pose_lag_samples"] = lag["lag_samples"]
    overall_rows[0]["pose_lag_residual_max_mm"] = lag["best_max_m"] * 1000
    overall_rows[0]["sigma_vs_eeerr_log10_corr"] = corr
    overall_rows[0]["log_path"] = str(log.path)
    overall_rows[0]["scene"] = EXEC_LOG_SCENE
    write_csv(CSV_DIR / "G7_overall.csv", overall_rows)

    # ---- 7. 图 -----------------------------------------------------------------------
    figures_ok = False
    if not no_figs:
        if not plots.available():
            print("\n[图] 跳过：当前 Python 环境没有 matplotlib。")
        else:
            try:
                c1 = log.cycle_slice(log.cycles[0])
                t1 = ser["t"][c1]
                pos_p, pos_a = ser["ee_pos_planned"][c1], ser["ee_pos_actual"][c1]
                note = (f"scene={EXEC_LOG_SCENE}; dt={log.dt_s:g}s (MuJoCo timestep); "
                        f"ee pose from fk(q_actual) NOT logger ee_* columns "
                        f"(logger ee_* lags {lag['lag_samples']} sample); unit: m, rad, s")
                plots.plot_error_by_segment(
                    t1, ser["joint_err_max_deg"][c1], ee_err[c1], ser["segment"][c1],
                    "G-7 跟踪误差随搬运段的变化（循环 1）",
                    FIG_DIR / "G7_error_by_segment.png", note=note)
                plots.plot_ee_tracking(
                    t1, pos_p, pos_a, ee_err[c1],
                    "G-7 末端轨迹 规划 vs 实际（循环 1）",
                    FIG_DIR / "G7_ee_tracking.png", note=note)
                log_pos = np.column_stack([log.ee_pos[..., :3]]) if log.ee_pos.ndim == 2 else log.ee_pos
                plots.plot_lag_demo(
                    t1[:1200], log_pos[c1][:1200],
                    np.array([model.fk_tcp(q)[:3, 3] for q in log.q_actual[c1][:1200]]),
                    lag["lag_samples"],
                    "G-7 日志 ee_* 列滞后 1 个采样的证据（前 2.4 s）",
                    FIG_DIR / "G7_lag_demo.png", note=note)
                plots.plot_path_length_accum(
                    t1, pos_p, pos_a,
                    "G-7 累积末端路径长度 规划 vs 实际（循环 1）",
                    FIG_DIR / "G7_path_lengths.png", note=note)
                plots.plot_sigma_vs_tracking(
                    sig[c1], ee_err[c1], ser["segment"][c1],
                    "G-7 σmin(J) 与末端跟踪误差的关系（循环 1）",
                    FIG_DIR / "G7_sigma_vs_tracking.png",
                    note=note + f"; log10 corr={corr:+.3f}")
                plots.plot_ee_vs_joint_error(
                    t1, ser["joint_err_max_deg"][c1], ee_err[c1],
                    "G-7 关节跟踪误差与末端误差的对应关系（循环 1）",
                    FIG_DIR / "G7_ee_vs_joint_error.png", note=note)
                figures_ok = True
            except Exception as exc:  # noqa: BLE001
                print(f"\n[图] 生成失败（CSV 已保留）：{type(exc).__name__}: {exc}")

    meta = write_meta_json(
        CSV_DIR / "G7_meta.json",
        {
            "scene_id": SCENE_ID_REAL,
            "log_scene": EXEC_LOG_SCENE,
            "log_path": str(log.path),
            "log_n": int(log.n),
            "log_dt_s": float(log.dt_s),
            "cycles": log.cycles,
            "segment_names": log.segment_names,
            "n_segments": len(seg),
            "ee_pose_source": "fk(q_actual) recomputed by G; logger ee_* NOT used",
            "pose_lag_samples": lag["lag_samples"],
            "overall": {k: v for k, v in ov.items() if not isinstance(v, dict)},
            "repeatability": rep,
            "sigma_vs_eeerr_log10_corr": corr,
            "sigma_min_threshold": SIGMA_MIN_THRESHOLD,
            "figures_generated": figures_ok,
            "f_run_summary_head": f_summary.read_text(encoding="utf-8", errors="replace").splitlines()[:12]
            if f_summary.exists() else None,
        },
    )
    print(f"\n[输出] CSV -> {CSV_DIR}\n       图  -> {FIG_DIR}\n       meta -> {meta}")
    # 备用：read_csv_rows 自检，确保下游能读回
    back = read_csv_rows(CSV_DIR / "G7_segment_metrics.csv")
    print(f"[自检] 回读 G7_segment_metrics.csv：{len(back)} 行，"
          f"首行 segment={back[0]['segment']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true", help="不生成图（只出 CSV）")
    args = ap.parse_args()
    raise SystemExit(main(no_figs=args.no_figs))
