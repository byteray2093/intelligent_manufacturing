"""G-8：真实模型的可达性分析（可达范围 + 数值不可达/几何不可达的区分）。

研究问题对应关系：
- 机器人到底能到多远？        -> 关节空间大规模采样 + 局部优化精修最大/最小伸展
- 工作空间长什么样、多大？    -> 半径分布、AABB 包络、体素占用、三视图
- 哪些点"到不了"？为什么？    -> 逐案分类：几何不可达 vs 数值 IK 失败（验收硬要求）
- 占位 DH 模型与真实模型差多少？-> 同一指标下两模型工作空间对比

关键约定（报告必须写清）：
1. 采样区间用模型自己声明的 `jnt_range`（1/2/4/5/6 轴 ±2π、3 轴 ±π），
   拉丁超立方，种子 20261015。所以"最大伸展"是**采样+优化得到的经验值**，
   不是解析证明出来的理论上界。
2. **伸展量到"肩部球心"而不是基座原点**。UR5e 的臂长从 shoulder_lift 轴心
   （基座上方约 0.163 m）算起；若用基座原点半径判可达，"基座正下方 1.1 m"
   这种点会被误判为可达。球心取自 `UR5eModel.reach_center()`，它在第 1 轴
   轴线上，随基座旋转不变。图与统计里的"半径"仍按基座原点给（方便对照 F 的
   坐标），但可达性判据用肩部伸展。
3. 内腔边界同样是经验值：伸展小于采样到的最小值即判为几何不可达，
   依据是 20 万次采样 + 局部优化都没有到达过更小的伸展。
4. IK 用 D 组算法的可移植版（`src/kinematics/ik.py`），阻尼 λ 取 0.005
   （G-5 的阻尼扫描实测：λ=0.005 时 10/10 收敛，λ=0.08 时最差 29.6 mm）。
   单初值（home）失败时再做 5 组多初值重试——这是"区分算法失败与几何不可达"
   的核心证据：多初值能解出 → 是算法/初值问题，不是数学不可达。
5. 判"可达"用的是**位置**；姿态单独看，因为同一位置换姿态可能解不出来。

输出：
    results/csv/G8_reach_stats.csv        半径/AABB 统计
    results/csv/G8_max_min_reach.csv      最大/最小伸展（采样、精修、标称值对照）
    results/csv/G8_radial_histogram.csv   半径分布
    results/csv/G8_voxel_summary.csv      体素占用
    results/csv/G8_reach_cases.csv        逐案可达性分类（核心表）
    results/figures/G8_workspace_views.png
    results/figures/G8_radial_reach.png
    results/figures/G8_workspace_compare.png
"""
from __future__ import annotations

import argparse
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import plots, workspace  # noqa: E402
from src.analysis.mujoco_backend import UR5eModel  # noqa: E402
from src.analysis.reference_model import UR5E_REFERENCE, fk_dh  # noqa: E402
from src.analysis.results_io import write_csv, write_meta_json  # noqa: E402
from src.analysis.singularity import classify_reachability  # noqa: E402,F401  （保留给 G3 复用同一判据，本脚本用肩部伸展版）
from src.scene import (  # noqa: E402
    REAL_IK_POS_TOL_M,
    REAL_SLICE_N,
    REAL_WORKSPACE_N_SAMPLES,
    REAL_WORKSPACE_SEED,
    REAL_WORKSPACE_VOXEL_M,
    RECOMMENDED_IK_LAMBDA,
    SCENE_ID_REAL,
    SIGMA_MIN_THRESHOLD,
)

CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"

R_DOWN = np.diag([1.0, -1.0, -1.0])
R_ID = np.eye(3)
# 单次 IK 的初值：home、零位、以及 4 组固定偏置（多初值重试用）
MULTISTART_OFFSETS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.6, -0.4, 0.5, -0.6, 0.7, 0.4),
    (-0.7, 0.5, -0.6, 0.5, -0.8, -0.5),
    (1.2, -1.0, 0.9, -1.1, 1.0, 1.3),
    (-1.3, 1.1, -1.0, 1.2, -1.1, -1.2),
)


def pose_from(pos, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


class IKAdapter:
    """把 D/F 组 `ik_dls()` 的 (q, ok, info) 包装成 classify_reachability 要的对象。"""

    def __init__(self, ik_module, q_init, R=R_ID, lam=RECOMMENDED_IK_LAMBDA, max_iter=400):
        self.ik = ik_module
        self.q_init = np.asarray(q_init, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.lam = lam
        self.max_iter = max_iter

    def __call__(self, p, q_init=None):
        T = pose_from(p, self.R)
        q0 = self.q_init if q_init is None else np.asarray(q_init, dtype=np.float64)
        _q, ok, info = self.ik.ik_dls(T, q0, max_iter=self.max_iter, lam=self.lam)
        return SimpleNamespace(
            success=bool(ok),
            position_error_m=float(info.get("pos_err", np.inf)),
            orientation_error_rad=float(info.get("rot_err", np.inf)),
            iterations=int(info.get("iterations", -1)),
            message="",
        )


def refine_extremum(model: UR5eModel, q0: np.ndarray, sign: float = -1.0,
                    center0: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """局部优化找极值伸展：sign=-1 求最大，sign=+1 求最小。

    伸展按"TCP 到肩部枢轴"的距离量（枢轴随第 1 轴旋转：Rz(q[0]) @ center0），
    而不是到基座原点——UR5e 的臂长从肩部枢轴算起，用基座原点会把"基座正下方"
    误判成可达。

    用 Nelder-Mead 与 Powell 各跑一次取更好者（无需梯度）。
    优化可能把关节推出 `jnt_range`，但 UR5e 各轴是周期关节，等价构型可折回
    (-π, π]，因此最后统一折回并在报告中注明"关节角按 2π 周期性归一"。
    """
    from scipy.optimize import minimize

    c0 = np.zeros(3) if center0 is None else np.asarray(center0, dtype=np.float64).reshape(3)

    def pivot(q):
        c, s = np.cos(q[0]), np.sin(q[0])
        return np.array([c * c0[0] - s * c0[1], s * c0[0] + c * c0[1], c0[2]])

    def obj(q):
        q = np.asarray(q, dtype=np.float64)
        return sign * float(np.linalg.norm(model.fk_tcp(q)[:3, 3] - pivot(q)))

    def wrap(q):
        return (np.asarray(q, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi

    best = (obj(q0), np.asarray(q0, dtype=np.float64))
    for method, opts in (("Nelder-Mead", {"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-14}),
                         ("Powell", {"maxiter": 20000, "xtol": 1e-10, "ftol": 1e-14})):
        res = minimize(obj, best[1], method=method, options=opts)
        if res.fun < best[0]:
            best = (float(res.fun), np.asarray(res.x, dtype=np.float64))
    q_best = wrap(best[1])
    radius = float(np.linalg.norm(model.fk_tcp(q_best)[:3, 3] - pivot(q_best)))
    return radius, q_best


def shoulder_distance_bounds(p: np.ndarray, center0: np.ndarray) -> tuple[float, float]:
    """目标点到"肩部枢轴可能所在位置"的距离下界/上界（对基座转角取极值）。

    枢轴相对基座轴线的水平偏置为 c_rho、高度为 z_c；给定目标 p（水平半径 rho），
    基座转角可以任意选择，于是枢轴与 p 的水平距离可落在 [|rho-c_rho|, rho+c_rho]：
      d_min  = sqrt(max(0, rho-c_rho)^2 + (pz-z_c)^2)  —— 最有利的转角
      d_max  = sqrt((rho+c_rho)^2 + (pz-z_c)^2)        —— 最不利的转角
    若 d_min > 实测最大伸展 → 任何转角都够不到 → **几何不可达**（与 IK 实现无关）；
    若 d_max < 实测最小伸展 → 任何转角都太近 → 同样几何不可达（内腔）。
    """
    p = np.asarray(p, dtype=np.float64).reshape(3)
    c_rho = float(np.hypot(center0[0], center0[1]))
    rho = float(np.hypot(p[0], p[1]))
    dz = float(p[2] - center0[2])
    d_min = float(np.sqrt(max(0.0, rho - c_rho) ** 2 + dz ** 2))
    d_max = float(np.sqrt((rho + c_rho) ** 2 + dz ** 2))
    return d_min, d_max


def classify_case(p, center0, r_min, r_max, ik_solver, q_init, position_tol_m) -> dict:
    """先做几何必要性判定（肩部伸展），再交给 IK；两者结论分开记录。"""
    d_min, d_max = shoulder_distance_bounds(p, center0)
    out = {
        "target_position_m": [float(v) for v in p],
        "d_min_shoulder_m": d_min,
        "d_max_shoulder_m": d_max,
        "r_min_shoulder_m": float(r_min),
        "r_max_shoulder_m": float(r_max),
        "status": None, "reason": "",
        "ik_success": None, "ik_position_error_m": None, "ik_iterations": None,
    }
    if d_min > r_max:
        out["status"] = "unreachable_geometry"
        out["reason"] = (f"目标到肩部枢轴的距离即使取最有利的基座转角也有 {d_min:.4f} m，"
                         f"超过实测最大伸展 {r_max:.4f} m；任何算法都不可能到达（与 IK 实现无关）")
        return out
    if d_max < r_min:
        out["status"] = "unreachable_geometry"
        out["reason"] = (f"目标到肩部枢轴的距离即使取最不利的基座转角也只有 {d_max:.4f} m，"
                         f"小于实测最小伸展 {r_min:.4f} m；落在工作空间内腔（经验边界）")
        return out
    res = ik_solver(p)
    out["ik_success"] = bool(res.success)
    out["ik_position_error_m"] = float(res.position_error_m)
    out["ik_iterations"] = int(res.iterations)
    if res.success and res.position_error_m <= position_tol_m:
        out["status"] = "reachable"
        out["reason"] = f"IK 收敛，位置误差 {res.position_error_m * 1000:.3f} mm"
    else:
        out["status"] = "numerical_failure"
        out["reason"] = ("目标在几何可达范围之内（"
                         f"{r_min:.4f} ≤ d ≤ {r_max:.4f} 的必要条件满足），但数值 IK 未收敛；"
                         "属于算法/初值/奇异问题，不是数学不可达")
    return out


def main(no_figs: bool = False) -> int:
    model = UR5eModel()
    lo, hi = model.joint_limits()
    home = model.home_q()

    # ---- 1. 采样 ---------------------------------------------------------------------
    t0 = time.time()
    q_all = workspace.sample_joint_space(int(REAL_WORKSPACE_N_SAMPLES), lo, hi,
                                         seed=REAL_WORKSPACE_SEED, method="latin")
    pos = model.positions_batch(q_all)
    center0 = model.reach_center()
    pivots = model.reach_center_batch(q_all)
    r_arm = np.linalg.norm(pos - pivots, axis=1)
    print(f"[采样] {len(q_all)} 个关节构型，耗时 {time.time() - t0:.2f} s，seed={REAL_WORKSPACE_SEED}")
    stats = workspace.radial_reach_stats(pos)
    env = workspace.box_envelope_m(pos)
    vox = workspace.voxel_occupancy(pos, REAL_WORKSPACE_VOXEL_M)
    hist = workspace.radial_histogram(pos, bins=60)
    print(f"  到基座原点：半径 min {stats['radius_min_m']:.4f} / mean {stats['radius_mean_m']:.4f} / "
          f"p95 {stats['radius_p95_m']:.4f} / max {stats['radius_max_m']:.4f} m")
    print(f"  到肩部枢轴：伸展 r_arm ∈ [{r_arm.min():.4f}, {r_arm.max():.4f}] m"
          f"（判几何可达用这个口径；枢轴 q=0 时位于 {np.round(center0, 4)}，随 pan 旋转）")
    if r_arm.min() < 0.02:
        print("  -> 实测最小伸展接近 0：TCP 能折回到肩部枢轴附近，"
              "所以真实 UR5e 在枢轴附近**没有内腔**（与第一阶段占位模型的结论不同）")
    print(f"  AABB 边长 {tuple(round(v, 4) for v in env['span_m'])} m，"
          f"对角线 {env['diagonal_m']:.4f} m")
    print(f"  体素 {vox['occupied_voxels']} 个（{vox['voxel_size_m']} m），"
          f"体积上界 {vox['approx_volume_m3']:.4f} m^3")
    print(f"  z 范围 [{stats['z_min_m']:.3f}, {stats['z_max_m']:.3f}] m"
          f"（z<0 表示 TCP 能到基座安装面以下）")

    # ---- 2. 极值精修 -----------------------------------------------------------------
    i_max = int(np.argmax(r_arm))
    i_min = int(np.argmin(r_arm))
    r_max_ref, q_max_ref = refine_extremum(model, q_all[i_max], sign=-1.0, center0=center0)
    r_min_ref, q_min_ref = refine_extremum(model, q_all[i_min], sign=+1.0, center0=center0)
    flange_max = float(max(np.linalg.norm(model.fk_flange(q)[:3, 3] - model.reach_center_batch([q])[0])
                           for q in q_all[:: max(1, len(q_all) // 5000)]))
    print(f"\n[伸展极值]（到肩部枢轴）采样 max {r_arm.max():.6f} m -> 精修 {r_max_ref:.6f} m")
    print(f"  q@max = {np.round(q_max_ref, 4)}  TCP = "
          f"{np.round(model.fk_tcp(q_max_ref)[:3, 3], 4)}")
    print(f"  采样 min {r_arm.min():.6f} m -> 精修 {r_min_ref:.6f} m")
    print(f"  q@min = {np.round(q_min_ref, 4)}  TCP = "
          f"{np.round(model.fk_tcp(q_min_ref)[:3, 3], 4)}")
    print(f"  法兰最大伸展（抽样，到枢轴） {flange_max:.4f} m；与 TCP 差 "
          f"{r_max_ref - flange_max:+.4f} m（TCP 偏移 0.1 m 的影响）")
    print(f"  对照：第一阶段占位 DH 模型曾用名义最大伸展 0.90 m —— 真实模型到肩部枢轴 "
          f"{r_max_ref:.4f} m、到基座原点 {np.linalg.norm(model.fk_tcp(q_max_ref)[:3, 3]):.4f} m，"
          "报告中不得再引用 0.90 m 作为 UR5e 的真实伸展。")

    write_csv(CSV_DIR / "G8_reach_stats.csv", [{
        **{k: v for k, v in stats.items()},
        "aabb_span_x_m": env["span_m"][0],
        "aabb_span_y_m": env["span_m"][1],
        "aabb_span_z_m": env["span_m"][2],
        "aabb_diagonal_m": env["diagonal_m"],
        "voxel_size_m": vox["voxel_size_m"],
        "occupied_voxels": vox["occupied_voxels"],
        "approx_volume_upper_m3": vox["approx_volume_m3"],
    }])
    write_csv(CSV_DIR / "G8_max_min_reach.csv", [
        {"quantity": "radius_arm_max_sampled_m", "value": float(r_arm.max()),
         "q_rad": ";".join(f"{v:.6f}" for v in q_all[i_max])},
        {"quantity": "radius_arm_max_refined_m", "value": r_max_ref,
         "q_rad": ";".join(f"{v:.6f}" for v in q_max_ref)},
        {"quantity": "radius_arm_min_sampled_m", "value": float(r_arm.min()),
         "q_rad": ";".join(f"{v:.6f}" for v in q_all[i_min])},
        {"quantity": "radius_arm_min_refined_m", "value": r_min_ref,
         "q_rad": ";".join(f"{v:.6f}" for v in q_min_ref)},
        {"quantity": "radius_base_max_sampled_m", "value": stats["radius_max_m"], "q_rad": ""},
        {"quantity": "radius_base_min_sampled_m", "value": stats["radius_min_m"], "q_rad": ""},
        {"quantity": "flange_max_sampled_m", "value": flange_max, "q_rad": ""},
        {"quantity": "nominal_max_reach_stage1_placeholder_m", "value": 0.90, "q_rad": ""},
        {"quantity": "sample_count", "value": len(q_all), "q_rad": ""},
        {"quantity": "seed", "value": REAL_WORKSPACE_SEED, "q_rad": ""},
    ])
    write_csv(CSV_DIR / "G8_radial_histogram.csv",
              [{"radius_center_m": 0.5 * (hist["bin_edges_m"][i] + hist["bin_edges_m"][i + 1]),
                "radius_lo_m": hist["bin_edges_m"][i], "radius_hi_m": hist["bin_edges_m"][i + 1],
                "count": hist["counts"][i]} for i in range(len(hist["counts"]))])
    write_csv(CSV_DIR / "G8_voxel_summary.csv", [{
        "voxel_size_m": vox["voxel_size_m"],
        "occupied_voxels": vox["occupied_voxels"],
        "approx_volume_upper_m3": vox["approx_volume_m3"],
        "index_min": vox["index_min"],
        "index_max": vox["index_max"],
    }])

    # ---- 3. 逐案可达性分类（核心：几何不可达 vs 数值 IK 失败） ------------------------
    from src.kinematics import ik as ikmod

    max_reach = r_max_ref
    min_reach = r_min_ref
    cases = [
        ("上料点 TCP 抓取姿态", [-0.35, 0.40, 0.055], R_DOWN, "R_DOWN"),
        ("下料点 TCP 抓取姿态", [0.35, 0.40, 0.055], R_DOWN, "R_DOWN"),
        ("上料悬停", [-0.35, 0.40, 0.30], R_DOWN, "R_DOWN"),
        ("下料抬升", [0.35, 0.40, 0.38], R_DOWN, "R_DOWN"),
        ("搬运中点", [0.00, 0.40, 0.38], R_DOWN, "R_DOWN"),
        ("半径 0.5 Rmax", [0.5 * max_reach, 0.0, 0.0], R_ID, "identity"),
        ("半径 0.95 Rmax", [0.95 * max_reach, 0.0, 0.0], R_ID, "identity"),
        ("半径 1.25 Rmax", [1.25 * max_reach, 0.0, 0.0], R_ID, "identity"),
        ("基座正上方小半径", [0.02, 0.0, 0.50], R_ID, "identity"),
        ("基座下方远点", [0.0, 0.0, -1.30], R_ID, "identity"),
        ("基座顶部近点", [0.005, 0.0, 0.015], R_ID, "identity"),
        ("可达位置 + 翻转姿态", [0.30, 0.30, 0.40], np.diag([-1.0, 1.0, -1.0]), "diag(-1,1,-1)"),
    ]
    case_rows = []
    for name, p, R, rname in cases:
        adapter = IKAdapter(ikmod, home, R=R)
        res = classify_case(p, center0, min_reach, max_reach, adapter, home, REAL_IK_POS_TOL_M)
        # 多初值重试：只在单初值失败时做，用来区分"算法/初值"与"几何不可达"
        multi_ok = False
        multi_best = np.inf
        if res["status"] == "numerical_failure":
            for off in MULTISTART_OFFSETS:
                a2 = IKAdapter(ikmod, np.clip(home + np.asarray(off), lo, hi), R=R)
                r2 = a2(p)
                multi_best = min(multi_best, r2.position_error_m)
                if r2.success and r2.position_error_m <= REAL_IK_POS_TOL_M:
                    multi_ok = True
                    break
        r_base = float(np.linalg.norm(np.asarray(p, dtype=np.float64)))
        row = {
            "case": name,
            "target_x_m": float(p[0]), "target_y_m": float(p[1]), "target_z_m": float(p[2]),
            "radius_base_m": r_base,
            "d_min_shoulder_m": res["d_min_shoulder_m"],
            "d_max_shoulder_m": res["d_max_shoulder_m"],
            "r_min_shoulder_m": res["r_min_shoulder_m"],
            "r_max_shoulder_m": res["r_max_shoulder_m"],
            "orientation": rname,
            "status": res["status"],
            "reason": res["reason"],
            "ik_success": res["ik_success"],
            "ik_pos_err_mm": None if res["ik_position_error_m"] is None
            else float(res["ik_position_error_m"] * 1000),
            "ik_iterations": res["ik_iterations"],
            "multistart_success": multi_ok,
            "multistart_best_pos_err_mm": None if not np.isfinite(multi_best) else multi_best * 1000,
        }
        case_rows.append(row)
        flag = ""
        if res["status"] == "numerical_failure":
            flag = f"  [多初值重试{'成功' if multi_ok else '仍失败'}]"
        print(f"  {name:<22} d_肩∈[{res['d_min_shoulder_m']:.4f}, {res['d_max_shoulder_m']:.4f}] m  "
              f"status={res['status']:<22}{flag}")
        if res["status"] != "reachable":
            print(f"     原因：{res['reason']}")
            if multi_ok:
                print("     -> 多初值能解出 ⇒ 单初值下的失败属于算法/初值问题，不是几何不可达")
    write_csv(CSV_DIR / "G8_reach_cases.csv", case_rows)

    n_unreach = sum(1 for x in case_rows if x["status"] == "unreachable_geometry")
    n_numfail = sum(1 for x in case_rows if x["status"] == "numerical_failure")
    n_reach = sum(1 for x in case_rows if x["status"] == "reachable")
    print(f"\n[分类汇总] 可达 {n_reach} / 几何不可达 {n_unreach} / 数值 IK 失败 {n_numfail} "
          f"（共 {len(case_rows)} 例）")

    # ---- 4. 与占位 DH 模型对比 -------------------------------------------------------
    ref_lo = np.full(6, -np.pi)
    ref_hi = np.full(6, np.pi)
    n_ref = min(50000, len(q_all))
    q_ref = workspace.sample_joint_space(n_ref, ref_lo, ref_hi, seed=REAL_WORKSPACE_SEED,
                                        method="latin")
    pos_ref = np.empty((len(q_ref), 3))
    for i, q in enumerate(q_ref):
        pos_ref[i] = fk_dh(q, UR5E_REFERENCE)[:3, 3]
    env_ref = workspace.box_envelope_m(pos_ref)
    r_ref = np.linalg.norm(pos_ref, axis=1)
    z_ref = np.linalg.norm(fk_dh(np.zeros(6), UR5E_REFERENCE)[:3, 3])
    z_real = float(np.linalg.norm(model.fk_tcp(np.zeros(6))[:3, 3]))
    print(f"\n[模型对比] 占位 DH 模型（stage1，±π 采样）：到基座原点半径 max {r_ref.max():.4f} m，"
          f"AABB 对角线 {env_ref['diagonal_m']:.4f} m")
    print(f"           真实 MuJoCo 模型：到基座原点半径 max {stats['radius_max_m']:.4f} m，"
          f"AABB 对角线 {env['diagonal_m']:.4f} m")
    print(f"           （口径统一为基座原点；真实模型到肩部枢轴的最大伸展 {r_max_ref:.4f} m）")
    print(f"  -> 两模型的工作空间尺度接近，但**不能**互相替代：零位 TCP 半径 "
          f"占位 {z_ref:.4f} m vs 真实 {z_real:.4f} m，差 {abs(z_ref - z_real) * 1000:.1f} mm")

    # ---- 5. 图 -----------------------------------------------------------------------
    figures_ok = False
    if not no_figs:
        if not plots.available():
            print("\n[图] 跳过：当前 Python 环境没有 matplotlib。")
        else:
            try:
                note = (f"scene={SCENE_ID_REAL}; model={Path(model.model_path).name}; "
                        f"N={len(q_all)}, latin-hypercube seed={REAL_WORKSPACE_SEED}; "
                        f"limits=jnt_range; unit: m")
                plots.plot_workspace(pos, "G-8 真实 UR5e 工作空间点云（TCP）",
                                     FIG_DIR / "G8_workspace_views.png",
                                     voxel_size_m=REAL_WORKSPACE_VOXEL_M, note=note)
                plots.plot_radial_reach(hist, stats, "G-8 TCP 半径分布与伸展边界",
                                        FIG_DIR / "G8_radial_reach.png", note=note)
                plots.plot_workspace_compare(pos, pos_ref,
                                             "G-8 真实模型 vs 占位 DH 模型的工作空间对比",
                                             FIG_DIR / "G8_workspace_compare.png",
                                             note=note + "; reference = stage-1 placeholder DH")
                figures_ok = True
            except Exception as exc:  # noqa: BLE001
                print(f"\n[图] 生成失败（CSV 已保留）：{type(exc).__name__}: {exc}")

    meta = write_meta_json(
        CSV_DIR / "G8_meta.json",
        {
            "scene_id": SCENE_ID_REAL,
            "model": model.describe(),
            "sampling": {"n": int(len(q_all)), "seed": REAL_WORKSPACE_SEED, "method": "latin",
                         "limits_lo_rad": lo.tolist(), "limits_hi_rad": hi.tolist()},
            "reach_center_m": center0.tolist(),
            "radius_arm_max_refined_m": r_max_ref,
            "radius_arm_min_refined_m": r_min_ref,
            "radius_base_max_sampled_m": stats["radius_max_m"],
            "radius_base_min_sampled_m": stats["radius_min_m"],
            "q_at_max_rad": q_max_ref.tolist(),
            "q_at_min_rad": q_min_ref.tolist(),
            "ik_lambda": RECOMMENDED_IK_LAMBDA,
            "ik_position_tol_m": REAL_IK_POS_TOL_M,
            "orientation_singularity_threshold": SIGMA_MIN_THRESHOLD,
            "cases_classification": {"reachable": n_reach, "unreachable_geometry": n_unreach,
                                     "numerical_failure": n_numfail},
            "figures_generated": figures_ok,
            "csv": sorted(p.name for p in CSV_DIR.glob("G8_*.csv")),
        },
    )
    print(f"\n[输出] CSV -> {CSV_DIR}\n       图  -> {FIG_DIR}\n       meta -> {meta}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true", help="不生成图（只出 CSV）")
    args = ap.parse_args()
    raise SystemExit(main(no_figs=args.no_figs))
