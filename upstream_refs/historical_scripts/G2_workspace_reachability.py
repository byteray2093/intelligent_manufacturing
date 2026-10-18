"""G-2：工作空间采样与可达性分布。

运行：
    python scripts/G2_workspace_reachability.py
    python scripts/G2_workspace_reachability.py --n 80000 --seed 7 --method uniform

输出：
    results/csv/G2_workspace_meta.csv        采样条件与包络统计（含单位列名）
    results/csv/G2_radial_histogram.csv      半径直方图数据
    results/csv/G2_voxel_summary.csv         体素占用摘要（含占用体积上界估计）
    results/figures/G2_workspace.png         三视图：XZ 密度 / XY 密度 / 半径分布
    results/csv/G2_meta.json                 可复现元数据

要点（报告必须照抄）：
- 采样区间 ±π（6 维超立方体），不是完整的 ±2π 关节行程；
- 体素边长 0.05 m，占用体积是上界估计（不是精确体积）；
- 随机种子与样本数写进 CSV 与图注，换机器同样命令可复现。
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
from src.analysis.results_io import write_csv, write_meta_json  # noqa: E402
from src.analysis.workspace import box_envelope_m, radial_histogram, sample_workspace  # noqa: E402
from src.scene import (  # noqa: E402
    SAMPLE_LIMITS_HI,
    SAMPLE_LIMITS_LO,
    SCENE_ID,
    WORKSPACE_METHOD,
    WORKSPACE_N_SAMPLES,
    WORKSPACE_SEED,
    WORKSPACE_VOXEL_M,
    reference_fk,
)

CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"


def main(n: int, seed: int, method: str, no_figs: bool = False) -> int:
    print(f"[G-2 工作空间采样] N={n}, seed={seed}, method={method}, voxel={WORKSPACE_VOXEL_M} m")
    print(f"  采样区间：q_i ∈ [{SAMPLE_LIMITS_LO[0]:.3f}, {SAMPLE_LIMITS_HI[0]:.3f}] rad（6 维超立方体）")

    res = sample_workspace(
        reference_fk,
        SAMPLE_LIMITS_LO,
        SAMPLE_LIMITS_HI,
        n_samples=n,
        seed=seed,
        method=method,
        voxel_size_m=WORKSPACE_VOXEL_M,
    )
    s = res.radial_stats
    env = box_envelope_m(res.positions)
    print(f"  半径：min={s['radius_min_m']:.4f} m, max={s['radius_max_m']:.4f} m, "
          f"mean={s['radius_mean_m']:.4f} m, p95={s['radius_p95_m']:.4f} m")
    print(f"  包络：x∈[{s['x_min_m']:.3f},{s['x_max_m']:.3f}], y∈[{s['y_min_m']:.3f},{s['y_max_m']:.3f}], "
          f"z∈[{s['z_min_m']:.3f},{s['z_max_m']:.3f}] m")
    print(f"        边长=(dx,dy,dz)={tuple(round(v, 4) for v in env['span_m'])} m, "
          f"对角线={env['diagonal_m']:.4f} m")
    print(f"  体素占用：{res.voxel['occupied_voxels']} 个（边长 {res.voxel['voxel_size_m']} m）, "
          f"体积上界 ≈ {res.voxel['approx_volume_m3']:.4f} m³")

    write_csv(CSV_DIR / "G2_workspace_meta.csv", [res.provenance()])
    write_csv(
        CSV_DIR / "G2_envelope_summary.csv",
        [{
            "n_samples": env["n"],
            "span_x_m": env["span_m"][0], "span_y_m": env["span_m"][1], "span_z_m": env["span_m"][2],
            "diagonal_m": env["diagonal_m"],
            "min_x_m": env["min_m"][0], "min_y_m": env["min_m"][1], "min_z_m": env["min_m"][2],
            "max_x_m": env["max_m"][0], "max_y_m": env["max_m"][1], "max_z_m": env["max_m"][2],
            "definition": "world-frame axis-aligned bounding box (AABB) of sampled TCP positions",
        }],
    )
    hist = radial_histogram(res.positions, bins=60)
    write_csv(
        CSV_DIR / "G2_radial_histogram.csv",
        [
            {"bin_left_m": hist["bin_edges_m"][i], "bin_right_m": hist["bin_edges_m"][i + 1], "count": hist["counts"][i]}
            for i in range(len(hist["counts"]))
        ],
    )
    write_csv(
        CSV_DIR / "G2_voxel_summary.csv",
        [{
            "voxel_size_m": res.voxel["voxel_size_m"],
            "occupied_voxels": res.voxel["occupied_voxels"],
            "approx_volume_m3_upper_bound": res.voxel["approx_volume_m3"],
            "index_min": res.voxel["index_min"],
            "index_max": res.voxel["index_max"],
        }],
    )

    figures_ok = False
    if not no_figs:
        if not plots.available():
            print("\n[图] 跳过：当前 Python 环境没有 matplotlib。安装后重跑："
                  "python -m pip install -r requirements.txt")
        else:
            try:
                note = (
                    f"scene={SCENE_ID}; model=UR5e reference DH (placeholder); N={n}, seed={seed}, "
                    f"method={method}; q_i in [-pi, pi] rad; voxel={WORKSPACE_VOXEL_M:g} m; unit: m"
                )
                plots.plot_workspace(
                    res.positions, "G-2 UR5e 工作空间采样（关节空间均匀采样 + FK）",
                    FIG_DIR / "G2_workspace.png", voxel_size_m=WORKSPACE_VOXEL_M, note=note,
                )
                figures_ok = True
            except Exception as exc:  # noqa: BLE001
                print(f"\n[图] 生成失败（CSV 已保留）：{type(exc).__name__}: {exc}")

    write_meta_json(
        CSV_DIR / "G2_meta.json",
        {
            "scene_id": SCENE_ID,
            "parameters": {
                "n_samples": n, "seed": seed, "method": method,
                "limits_lo_rad": SAMPLE_LIMITS_LO.tolist(), "limits_hi_rad": SAMPLE_LIMITS_HI.tolist(),
                "voxel_size_m": WORKSPACE_VOXEL_M,
            },
            "radial_stats": s,
            "envelope_aabb_m": env,
            "voxel": {k: v for k, v in res.voxel.items() if k != "occupancy"},
            "model_source": "UR5e reference DH table (G placeholder) — replace with B-group MJCF after freeze",
            "figures": sorted(p.name for p in FIG_DIR.glob("G2_*.png")) if figures_ok else [],
            "figures_generated": figures_ok,
            "csv": sorted(p.name for p in CSV_DIR.glob("G2_*.csv")),
        },
    )
    print(f"\n[输出] {CSV_DIR} / {FIG_DIR}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=WORKSPACE_N_SAMPLES, help="采样点数")
    ap.add_argument("--seed", type=int, default=WORKSPACE_SEED, help="随机种子（写进元数据）")
    ap.add_argument("--method", choices=["latin", "uniform"], default=WORKSPACE_METHOD)
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(args.n, args.seed, args.method, args.no_figs))
