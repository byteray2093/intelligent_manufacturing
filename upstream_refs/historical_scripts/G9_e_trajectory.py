"""G-9 E 组轨迹验证与 E→F 集成实验。

本脚本属于 G 组"分析"职责，**不改 E、F 的任何代码**，只做：

1. 校验 E 交付的整条搬运轨迹（数据契约 / 起末位形 / 限位 / 速度加速度 / 关键高度）；
2. 用 E 自己的轨迹模块重建设置取**权威分段**，逐段给规划侧指标（路径、直线度、
   速度加速度），并把"E 报告的请求时长"与"实际轨迹时长"并列，避免混用；
3. 沿 E 的轨迹逐点算 Jacobian 奇异值（解析 Jacobian），判断这条规划是否贴近奇异；
4. 把 E 的规划轨迹与 **F 内置规划器**产出的 `q_cmd`（取自 F 的交付日志）做同口径对比；
5. 若存在"用 `--traj-file` 让 F 控制器执行 E 轨迹"的复跑日志，则分析其跟踪误差并定位
   两模块的接口不匹配（G 只测量与定位，不修改上游）。

运行：
    python scripts/G9_e_trajectory.py                 # 全量
    python scripts/G9_e_trajectory.py --no-rebuild    # 跳过重建分段（用静止点检测）
    python scripts/G9_e_trajectory.py --no-figs       # 只出数据

产物（results/csv/）：G9_e_checks.csv、G9_e_segments.csv、G9_e_sigma.csv、
G9_e_compare.csv、G9_e_exec_segments.csv、G9_e_exec_summary.csv、G9_meta.json
产物（results/figures/）：G9_e_joint_profiles.png、G9_e_tcp_path.png、
G9_e_segment_straightness.png、G9_e_sigma.png、G9_e_exec_jump.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import e_traj as et  # noqa: E402
from src.analysis import exec_log as exec_log_mod  # noqa: E402
from src.analysis import plots, singularity, tracking  # noqa: E402
from src.analysis.results_io import write_csv, write_meta_json  # noqa: E402
from src.scene import SCENE_ID_E, SIGMA_MIN_THRESHOLD, real_model  # noqa: E402

CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"
LOG_DIR = ROOT / "results" / "logs"
#: G 用 F 控制器执行 E 轨迹所得的复跑日志（由 F 的 --traj-file 生成，F 代码未改动）
E_EXEC_LOG = LOG_DIR / "exec_log_e_traj_1cycle.csv"
#: F 交付的原始执行日志（F 内置规划器 + 3 循环），只用于同口径对比
F_BASE_LOG = LOG_DIR / "exec_log_latest.csv"

#: E 的 7 段 → F 日志里的段名（用于"E 规划 vs F 内置规划"的同口径对比）
E_TO_F_SEGMENT = {
    "home->hover": "approach",
    "descend": "descend",
    "lift": "lift",
    "transfer": "transfer",
    "lower": "lower",
    "retreat": "retreat",
    "return->home": "return_home",
}


def _seg_names_array(n: int, names, windows) -> np.ndarray:
    """把分段窗口展开成逐采样的段名数组（给绘图函数按段着色用）。"""
    arr = np.empty(n, dtype=object)
    arr[:] = "unknown"
    for name, (i0, i1) in zip(names, windows):
        arr[i0:i1 + 1] = str(name)
    return arr


def _f_builtin_plan_rows(model, log) -> list[dict]:
    """F 内置规划器（第 1 循环）的逐段规划侧指标，口径与 E 的分段一致。"""
    rows: list[dict] = []
    cycle1 = log.cycle == log.cycles[0]
    q_cmd = log.q_cmd[cycle1]
    seg = np.asarray(log.segment)[cycle1]
    t = log.t[cycle1] - log.t[cycle1][0]
    pos = model.positions_batch(q_cmd)
    for e_name, f_name in E_TO_F_SEGMENT.items():
        sel = np.where(seg == f_name)[0]
        if len(sel) == 0:
            continue
        one = et.per_segment_rows(t, q_cmd, pos, [e_name], [(int(sel[0]), int(sel[-1]))])[0]
        one["source"] = "F 内置规划器（F 交付日志 q_cmd，循环 1）"
        rows.append(one)
    return rows


def _exec_summary(model, log, base_log) -> tuple[list[dict], dict]:
    """分析"F 控制器执行 E 轨迹"的复跑日志，定位接管点的不匹配。"""
    res = tracking.analyze_exec_log(
        model, log, jac_step=1, sigma_min_threshold=SIGMA_MIN_THRESHOLD
    )
    seg_rows = [m.to_row() for m in res["segments"]]
    for row in seg_rows:
        row["source"] = "F 控制器 --traj-file 执行 E 轨迹（G 触发复跑）"

    seg = np.asarray([str(v) for v in log.segment])
    q_cmd, q_act = log.q_cmd, log.q_actual
    err_all = np.degrees(np.abs(q_cmd - q_act).max(axis=1))
    idx = np.where(np.char.startswith(seg, "transfer_e"))[0]
    summary: dict = {
        "n_samples": int(log.n),
        "duration_s": float(log.t[-1] - log.t[0]),
        "joint_err_max_deg": float(err_all.max()),
        "joint_err_max_time_s": float(log.t[int(np.argmax(err_all))]),
        "transfer_e_present": bool(len(idx)),
    }
    if len(idx):
        i0 = int(idx[0])
        summary["transfer_e_start_s"] = float(log.t[i0])
        summary["transfer_e_n"] = int(len(idx))
        # 接管瞬间：q_cmd 从 F 上一段末位形跳到 E 轨迹首点（E 首点 = home）
        if i0 > 0:
            jump = np.abs(q_cmd[i0] - q_cmd[i0 - 1])
            summary["handover_jump_max_rad"] = float(jump.max())
            summary["handover_jump_max_deg"] = float(np.degrees(jump.max()))
            summary["handover_jump_max_joint"] = int(np.argmax(jump)) + 1
            summary["handover_jump_prev_q_rad"] = [float(v) for v in q_cmd[i0 - 1]]
            summary["handover_jump_next_q_rad"] = [float(v) for v in q_cmd[i0]]
        # 接管后的瞬态与稳态必须分开报：跳变引起的误差会在约 1 s 内衰减，
        # 稳态误差才代表"F 执行器执行 E 规划轨迹"的真实跟踪精度。
        tail = err_all[idx]
        good = np.where(tail < 0.5)[0]
        settle = int(good[0]) if len(good) else len(tail) - 1
        summary["handover_recover_to_0p5deg_s"] = float(settle * log.dt_s)
        summary["handover_spike_n_samples_over_1deg"] = int((tail > 1.0).sum())
        summary["transfer_e_settled_joint_err_max_deg"] = float(tail[settle:].max())
        summary["transfer_e_settled_joint_err_p95_deg"] = float(
            np.percentile(tail[settle:], 95)
        )
        summary["transfer_e_n_settled"] = int(len(tail) - settle)

    # 基线（F 内置规划器，同一执行器）：只取第 1 循环同口径
    b = tracking.analyze_exec_log(
        model, base_log, jac_step=1, sigma_min_threshold=SIGMA_MIN_THRESHOLD
    )
    cyc = b["cycles"]
    first = cyc[0] if isinstance(cyc, list) and cyc else {}
    if isinstance(first, dict):
        for key in ("joint_err_max_deg", "ee_pos_err_max_mm", "ee_ori_err_max_deg"):
            if key in first:
                summary[f"baseline_{key}"] = first[key]
    return seg_rows, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G-9 E 组轨迹验证与 E→F 集成实验")
    ap.add_argument("--e-traj", default=None, help="E 的 pick_place_traj.npz（默认自动查找）")
    ap.add_argument("--no-rebuild", action="store_true", help="不重建 E 的规划，改用静止点检测分段")
    ap.add_argument("--exec-log", default=None, help="F 执行 E 轨迹的复跑日志（可选）")
    ap.add_argument("--no-figs", action="store_true", help="不画图")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("G-9 E 组轨迹验证与 E→F 集成实验")
    print("=" * 74)

    traj = et.load_e_trajectory(args.e_traj)
    print(f"[E] 轨迹 : {traj.source}")
    print(f"[E] 后端 : {traj.backend}")
    print(f"[E] 采样 : N={traj.n}, dt={traj.dt_s:.4f} s, 时长={traj.duration_s:.4f} s")

    model = real_model()
    print(f"[模型] {model.describe()['model_path']}")

    seg = et.resolve_segments(traj, rebuild=not args.no_rebuild)
    windows, info = seg["windows"], seg["info"]
    raw_names = [str(n) for n in info["segment_names"]]
    # 重建用的是 E 模块内部的段名；对外统一改用 E 报告里的 7 段名（同顺序），
    # 这样 CSV 与 E 交付的说明文档能直接对照；内部名保留在元数据里。
    names = [
        et.SEGMENT_NAMES[i] if i < len(et.SEGMENT_NAMES) else n
        for i, n in enumerate(raw_names)
    ]
    info["rebuild_segment_names"] = raw_names
    print(f"[分段] 来源={info['source']}")
    for nm, (i0, i1) in zip(names, windows):
        print(f"       {nm:14s} i={i0:5d}-{i1:5d}  {traj.time_s[i1] - traj.time_s[i0]:6.3f} s")

    pos = model.positions_batch(traj.q_rad)

    # ---------------- 1) 轨迹自检 ----------------
    checks = et.check_e_trajectory(traj, model, windows=windows, names=names)
    write_csv(CSV_DIR / "G9_e_checks.csv", checks)
    n_pass = sum(1 for r in checks if r["passed"])
    print(f"\n[1] 轨迹自检 {n_pass}/{len(checks)} 通过 -> results/csv/G9_e_checks.csv")
    for r in checks:
        flag = "OK  " if r["passed"] else "注意"
        print(f"    [{flag}] {r['check']:11s} {r['metric']:32s} = {r['value']} {r['unit']}")

    # ---------------- 2) 逐段规划侧指标 ----------------
    seg_rows = et.per_segment_rows(traj.time_s, traj.q_rad, pos, names, windows)
    for row, req in zip(seg_rows, et.REPORTED_DURATIONS_S):
        row["reported_duration_s"] = float(req)
        row["duration_ratio_vs_report"] = row["duration_s"] / req if req else float("nan")
    for row in seg_rows:
        row["scene_id"] = SCENE_ID_E
    write_csv(CSV_DIR / "G9_e_segments.csv", seg_rows)
    print("\n[2] 逐段规划侧指标 -> results/csv/G9_e_segments.csv")
    for row in seg_rows:
        print(
            f"    {row['segment']:14s} {row['duration_s']:5.3f}s(报告 {row['reported_duration_s']:.3f}) "
            f"TCP {row['tcp_path_m']:.6f} m  直线度 {row['straightness']:.9f}  "
            f"偏离 {row['max_dev_from_chord_m'] * 1000:.4f} mm  vmax {row['joint_speed_max_rad_s']:.4f}"
        )

    # ---------------- 3) 奇异性（解析 Jacobian，逐点） ----------------
    seg_of_sample = _seg_names_array(traj.n, names, windows)
    sing = singularity.analyze_trajectory_singularity(
        traj.time_s,
        traj.q_rad,
        None,
        sigma_min_threshold=SIGMA_MIN_THRESHOLD,
        jac_fn=model.jacobian_tcp,
    )
    write_csv(
        CSV_DIR / "G9_e_sigma.csv",
        [
            {
                "t_s": float(traj.time_s[i]),
                "segment": str(seg_of_sample[i]),
                "q1_rad": float(traj.q_rad[i, 0]),
                "sigma_min": float(sing["sigma_min"][i]),
                "sigma_max": float(sing["sigma_max"][i]),
                "condition_number": float(sing["condition_number"][i]),
            }
            for i in range(traj.n)
        ],
    )
    print("\n[3] 奇异值（逐点） -> results/csv/G9_e_sigma.csv")
    print(
        f"    sigma_min: min={sing['min_sigma_min']:.6e} @ t={sing['worst_time_s']:.3f}s  "
        f"中位数={float(np.median(sing['sigma_min'])):.6e}  "
        f"最大条件数={float(np.max(sing['condition_number'])):.4g}"
    )
    print(f"    低于阈值 {SIGMA_MIN_THRESHOLD} 的点: {sing['n_singular_samples']}/{traj.n}")

    # ---------------- 4) E 规划 vs F 内置规划 ----------------
    cmp_rows: list[dict] = []
    base_log = None
    if F_BASE_LOG.is_file():
        base_log = exec_log_mod.load_exec_log(F_BASE_LOG)
        by_name = {r["segment"]: r for r in _f_builtin_plan_rows(model, base_log)}
        for row in seg_rows:
            f = by_name.get(row["segment"])
            if f is None:
                continue
            cmp_rows.append(
                {
                    "segment": row["segment"],
                    "E_duration_s": row["duration_s"],
                    "F_duration_s": f["duration_s"],
                    "E_tcp_path_m": row["tcp_path_m"],
                    "F_tcp_path_m": f["tcp_path_m"],
                    "E_straightness": row["straightness"],
                    "F_straightness": f["straightness"],
                    "E_max_dev_mm": row["max_dev_from_chord_m"] * 1000.0,
                    "F_max_dev_mm": f["max_dev_from_chord_m"] * 1000.0,
                    "E_joint_path_rad": row["joint_path_rad"],
                    "F_joint_path_rad": f["joint_path_rad"],
                    "E_joint_speed_max_rad_s": row["joint_speed_max_rad_s"],
                    "F_joint_speed_max_rad_s": f["joint_speed_max_rad_s"],
                }
            )
        write_csv(CSV_DIR / "G9_e_compare.csv", cmp_rows)
        print("\n[4] E 规划 vs F 内置规划（同口径） -> results/csv/G9_e_compare.csv")
        for r in cmp_rows:
            print(
                f"    {r['segment']:14s} 时长 E {r['E_duration_s']:.3f}/F {r['F_duration_s']:.3f}s  "
                f"TCP E {r['E_tcp_path_m']:.6f}/F {r['F_tcp_path_m']:.6f} m  "
                f"vmax E {r['E_joint_speed_max_rad_s']:.3f}/F {r['F_joint_speed_max_rad_s']:.3f} rad/s"
            )
    else:
        print(f"\n[4] 跳过对比：未找到 F 交付日志 {F_BASE_LOG}")

    # ---------------- 5) E→F 集成复跑 ----------------
    exec_rows: list[dict] = []
    exec_summary: dict = {}
    exec_log_path = Path(args.exec_log) if args.exec_log else E_EXEC_LOG
    if exec_log_path.is_file() and base_log is not None:
        e_log = exec_log_mod.load_exec_log(exec_log_path)
        exec_rows, exec_summary = _exec_summary(model, e_log, base_log)
        write_csv(CSV_DIR / "G9_e_exec_segments.csv", exec_rows)
        write_csv(
            CSV_DIR / "G9_e_exec_summary.csv",
            [{"metric": k, "value": v} for k, v in exec_summary.items()],
        )
        print("\n[5] E→F 集成复跑（F 控制器 --traj-file 执行 E 轨迹）")
        print(
            f"    样本 {exec_summary['n_samples']}，时长 {exec_summary['duration_s']:.3f} s，"
            f"整体关节误差 max {exec_summary['joint_err_max_deg']:.4f}°"
        )
        if "handover_jump_max_rad" in exec_summary:
            print(
                f"    接管跳变: {exec_summary['handover_jump_max_rad']:.4f} rad "
                f"({exec_summary['handover_jump_max_deg']:.2f}°) "
                f"@ J{exec_summary['handover_jump_max_joint']}，"
                f"t={exec_summary['transfer_e_start_s']:.3f} s"
            )
            print(
                f"    跳变后 {exec_summary['handover_recover_to_0p5deg_s']:.3f} s 内回落到 0.5° 以内；"
                f"稳态（后 {exec_summary['transfer_e_n_settled']} 点）关节误差 max "
                f"{exec_summary['transfer_e_settled_joint_err_max_deg']:.4f}° / "
                f"p95 {exec_summary['transfer_e_settled_joint_err_p95_deg']:.4f}°"
            )
    else:
        print(f"\n[5] 跳过集成复跑：未找到 {exec_log_path}")

    # ---------------- 6) 图 ----------------
    if not args.no_figs:
        note = (
            f"scene={SCENE_ID_E}; E 轨迹 N={traj.n}, dt={traj.dt_s:.3f}s, "
            f"时长={traj.duration_s:.3f}s; model={Path(model.describe()['model_path']).name}; "
            "unit: m, rad, s"
        )
        plots.plot_e_joint_profiles(
            traj.time_s,
            traj.q_rad,
            traj.dq_rad_s,
            traj.ddq_rad_s2,
            seg_of_sample,
            "G-9 E 组轨迹：关节角 / 速度 / 加速度（按 7 段着色）",
            FIG_DIR / "G9_e_joint_profiles.png",
            note,
        )
        plots.plot_tcp_path_3d(
            pos,
            seg_of_sample,
            et.KEY_HEIGHTS_M,
            "G-9 E 组轨迹：TCP 路径与关键高度（TCP=attachment_site）",
            FIG_DIR / "G9_e_tcp_path.png",
            note,
        )
        plots.plot_segment_straightness(
            [r["segment"] for r in seg_rows],
            [r["straightness"] for r in seg_rows],
            [r["max_dev_from_chord_m"] * 1000.0 for r in seg_rows],
            "G-9 E 组轨迹：逐段直线度（笛卡尔段应≈1.0，关节空间段必然弯曲）",
            FIG_DIR / "G9_e_segment_straightness.png",
            note,
        )
        plots.plot_e_sigma(
            traj.time_s,
            sing["sigma_min"],
            sing["condition_number"],
            SIGMA_MIN_THRESHOLD,
            seg_of_sample,
            "G-9 E 组轨迹上的奇异值：判断规划是否贴近奇异",
            FIG_DIR / "G9_e_sigma.png",
            note,
        )
        if exec_rows and exec_summary.get("handover_jump_max_joint"):
            e_log = exec_log_mod.load_exec_log(exec_log_path)
            seg = np.asarray([str(v) for v in e_log.segment])
            err = np.degrees(np.abs(e_log.q_cmd - e_log.q_actual).max(axis=1))
            j = int(exec_summary["handover_jump_max_joint"]) - 1
            idx = int(np.where(np.char.startswith(seg, "transfer_e"))[0][0])
            plots.plot_exec_jump(
                e_log.t,
                np.degrees(e_log.q_cmd[:, j]),
                np.degrees(e_log.q_actual[:, j]),
                err,
                seg,
                idx,
                f"J{j + 1}",
                "G-9 E→F 集成复跑：接管瞬间的指令跳变与跟踪误差",
                FIG_DIR / "G9_e_exec_jump.png",
                f"scene={SCENE_ID_E}; F 控制器 --traj-file 执行 E 轨迹（F 代码未改动）, "
                f"N={e_log.n}, dt={e_log.dt_s:.3f}s; ee pose 由 fk(q_actual) 回代; unit: deg, s",
            )
        print("\n[6] 图 -> results/figures/G9_e_*.png")

    # ---------------- 7) 元数据 ----------------
    write_meta_json(
        ROOT / "results" / "csv" / "G9_meta.json",
        {
            "scene_id": SCENE_ID_E,
            "script": "scripts/G9_e_trajectory.py",
            "e_trajectory": traj.summary(),
            "segmentation": info["source"],
            "segments": names,
            "segmentation_rebuild_names": info.get("rebuild_segment_names"),
            "segmentation_starts_s": [float(v) for v in info["starts_s"]],
            "segmentation_durations_s": [float(v) for v in info["durations_s"]],
            "ik_fail_counts": info.get("ik_fail_counts"),
            "sigma_min_threshold": SIGMA_MIN_THRESHOLD,
            "sigma_min_min": sing["min_sigma_min"],
            "n_singular_samples": sing["n_singular_samples"],
            "exec_log": str(exec_log_path) if exec_log_path.is_file() else None,
            "exec_summary": exec_summary,
            "checks_passed": n_pass,
            "checks_total": len(checks),
        },
    )
    print("\n[7] 元数据 -> results/csv/G9_meta.json")
    print("\nG-9 完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
