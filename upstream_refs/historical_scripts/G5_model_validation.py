"""G-5：上游数据一致性验证（C 的 FK / D 的 IK / F 的日志 / MuJoCo 自身）。

运行：
    python scripts/G5_model_validation.py
    python scripts/G5_model_validation.py --no-figs

为什么 G 组要做这件事：报告里的所有误差数字都建立在"上游位姿、Jacobian、日志字段
可信"这个前提上。本脚本把这个前提变成 15 行可复现的"实测值 / 阈值 / 通过与否"，
其中两条是真正的坑（写进报告）：

1. **坐标系陷阱**：C 的 `fk(q)` 输出的是法兰 `wrist_3_link`，不是 TCP
   `attachment_site`。两者相差 100.7 mm 与 90°。任何"拿 C 的 FK 去算末端误差"的
   做法都会凭空多出 10 cm 级误差。
2. **日志滞后**：F 的执行日志里 `ee_*` 列比 `q_actual` 滞后 1 个采样（2 ms，原因是
   先 `mj_step` 后读 `site_xpos`）。不校正就拿来算跟踪误差，最坏会凭空多出 1.26 mm。

输出：
    results/csv/G5_validation_checks.csv   每项检查：实测值/单位/阈值/是否通过/说明
    results/csv/G5_ik_damping_sweep.csv    ik_dls_limited 阻尼系数敏感性扫描
    results/csv/G5_meta.json              模型指纹、环境、命令、场景
    results/figures/G5_validation_summary.png
    results/figures/G5_ik_damping_sweep.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import plots, validation  # noqa: E402
from src.analysis.exec_log import load_exec_log  # noqa: E402
from src.analysis.mujoco_backend import UR5eModel  # noqa: E402
from src.analysis.results_io import write_csv, write_meta_json  # noqa: E402
from src.scene import SCENE_ID_REAL, VALIDATION_SEED  # noqa: E402

CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"


def main(no_figs: bool = False) -> int:
    model = UR5eModel()
    d = model.describe()
    print(f"[场景] {SCENE_ID_REAL}")
    print(f"  模型 = {d['model_path']}")
    print(f"  sha256[:16] = {d['model_sha256_head16']}, nq={d['nq']}, nu={d['nu']}, "
          f"timestep={d['timestep_s']:g} s, TCP 偏移={d['tcp_offset_m']:.4f} m")

    rows: list[dict] = []
    rows += validation.check_c_fk(model)
    rows += validation.check_numeric_jacobian(model)
    rows += validation.check_flange_tcp(model)
    rows += validation.check_d_ik(model, seed=VALIDATION_SEED)
    rows += validation.check_d_ik_damping(model, seed=VALIDATION_SEED)

    print("\n[F 组执行日志]")
    try:
        log = load_exec_log()
        print(f"  {log.n} 个采样, dt={log.dt_s:g} s, 时长={log.n * log.dt_s:.3f} s, "
              f"cycles={log.cycles}, 段类型={len(log.segment_names)} 种")
        rows += validation.check_f_log(model, log)
    except FileNotFoundError as exc:
        print(f"  跳过：{exc}")

    # ---- 汇总 ------------------------------------------------------------------------
    write_csv(CSV_DIR / "G5_validation_checks.csv", rows)

    n_auto = sum(1 for r in rows if r["passed"])
    manual = {"c_fk_tcp": "坐标系陷阱（不是失败，是要给报告读者的提醒）",
              "f_log_residual": "对照值"}
    n_attention = sum(1 for r in rows if not r["passed"])
    print("\n[验证结果]")
    print(f"  {'检查项':<32}{'指标':<28}{'实测':>16}{'阈值':>14}  结论")
    print("  " + "-" * 100)
    for r in rows:
        flag = "PASS" if r["passed"] else "注意"
        thr = f"{r['threshold']:g}" if r["unit"] != "bool" else "-"
        print(f"  {r['check']:<32}{r['metric']:<28}{r['value']:>16.6g}{thr:>14}  {flag}")
    print(f"\n  合计 {len(rows)} 项：{n_auto} 项通过/符合预期，{n_attention} 项需要关注"
          f"（其中坐标系陷阱与日志滞后是**已知的物理/实现差异**，不是错误）")

    sweep_rows = [r for r in rows if r["check"] == "d_ik_damping"]
    if sweep_rows:
        write_csv(CSV_DIR / "G5_ik_damping_sweep.csv", sweep_rows)
        print("\n[IK 阻尼扫描] ik_dls_limited 在 D 的 seed42 十目标上：")
        for r in sweep_rows:
            print(f"  lambda={r['lam']:<7g} 成功率={100 * r['value']:5.1f}%  "
                  f"最差残差={r['worst_mm']:8.4f} mm  {'全部收敛' if r['passed'] else '有失败'}")

    # ---- 图 --------------------------------------------------------------------------
    figures_ok = False
    if not no_figs:
        if not plots.available():
            print("\n[图] 跳过：当前 Python 环境没有 matplotlib。")
        else:
            try:
                note = (f"scene={SCENE_ID_REAL}; model={Path(d['model_path']).name} "
                        f"sha256[:16]={d['model_sha256_head16']}; seed={VALIDATION_SEED}; unit: see column names")
                plots.plot_validation_summary(
                    rows, "G-5 上游数据一致性验证：实测 / 阈值（<1 为通过）",
                    FIG_DIR / "G5_validation_summary.png", note=note,
                )
                if sweep_rows:
                    plots.plot_ik_damping_sweep(
                        [r["lam"] for r in sweep_rows],
                        [r["value"] for r in sweep_rows],
                        [r["worst_mm"] for r in sweep_rows],
                        "G-5 IK 阻尼系数敏感性（D 的 seed42 十目标，带限位增强版 DLS）",
                        FIG_DIR / "G5_ik_damping_sweep.png", note=note,
                    )
                figures_ok = True
            except Exception as exc:  # noqa: BLE001
                print(f"\n[图] 生成失败（CSV 已保留）：{type(exc).__name__}: {exc}")

    meta = write_meta_json(
        CSV_DIR / "G5_meta.json",
        {
            "scene_id": SCENE_ID_REAL,
            "model": d,
            "seed": VALIDATION_SEED,
            "checks": {"total": len(rows), "passed": n_auto, "attention": n_attention},
            "tolerances": validation.TOL,
            "known_differences": {
                "c_fk_outputs_flange_not_tcp": True,
                "f_log_ee_columns_lag_samples": -1,
            },
            "figures_generated": figures_ok,
            "csv": sorted(p.name for p in CSV_DIR.glob("G5_*.csv")),
        },
    )
    print(f"\n[输出] CSV -> {CSV_DIR}\n       图  -> {FIG_DIR}\n       meta -> {meta}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true", help="不生成图（只出 CSV）")
    args = ap.parse_args()
    raise SystemExit(main(no_figs=args.no_figs))
