"""上游交叉验证：C（正运动学）/ D（逆运动学）/ F（执行日志）/ MuJoCo 自身。

G 的定位是"分析"，所以除了自己出图，还要回答一个更硬的问题：
**上游交给我的东西，真的能对上吗？** 本模块把四件事做成可复现的数值检查，
每项都给出"阈值 + 实测值 + 通过/不通过"，写进 `results/csv/G5_validation_checks.csv`。

检查清单
========
``c_fk``            C 的 `src/kinematics/fk.py`（DH + 补偿矩阵）vs MuJoCo 法兰位姿
``c_fk_tcp``        若 C 的 FK 直接当 TCP 用，会差多少（坐标系陷阱量化）
``numeric_jacobian`` G 的数值微分 Jacobian vs MuJoCo `mj_jacSite` 解析 Jacobian
``flange_tcp``      法兰→TCP 固定变换的自洽性（T_tcp == T_flange @ T_flange_tcp）
``d_ik_replay``     D 的 `ik_dls` 复现 D 自己 CSV 里的 10 组结果（seed=42）
``d_ik_limited``    D 的 `ik_dls_limited`（带限位）对同一批目标的收敛性
``f_log_pose_lag``  F 日志 `ee_*` 列相对 `q_actual` 的采样滞后
``f_log_residual``  滞后对齐后 `ee_*` 与 `fk(q_actual)` 的残差

阈值来源：`docs/G_METRICS_SPEC.md`。改阈值必须同步改文档，否则报告与代码不一致。
"""

from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path

import numpy as np

from . import metrics
from .exec_log import ExecLog, quat_wxyz_to_matrix
from .linalg import geometric_jacobian, so3_log

# ---------------------------------------------------------------- 阈值（可追溯）
TOL = {
    "c_fk_pos_mm": 2.0,          # C 组自报 ~0.9 mm；给 2 mm 容差
    "c_fk_rot_deg": 0.05,        # 姿态应几乎精确
    "numeric_jac_abs": 1e-8,     # 数值 vs 解析 Jacobian 的逐元素最大偏差
    "flange_tcp_pos_m": 1e-9,
    "flange_tcp_rot": 1e-9,
    "d_ik_pos_m": 1e-3,          # D 的收敛判据是 1e-4 m，给 1e-3 容差
    "d_ik_rot_rad": 1e-3,
    "d_ik_replay_tol": 5e-4,     # 复现 D 自己 CSV 的数值一致度
    "f_log_lag_samples": 1.0,    # 滞后应当正好 1 个采样
    "f_log_residual_mm": 0.2,    # 对齐后残差（CSV 6 位小数舍入 ~0.1 mm）
}


def _row(check: str, item: str, metric: str, value: float, unit: str,
         threshold: float, passed: bool, note: str = "") -> dict:
    return {
        "check": check,
        "item": item,
        "metric": metric,
        "value": float(value),
        "unit": unit,
        "threshold": float(threshold),
        "passed": bool(passed),
        "note": note,
    }


# ======================================================================
# 载入上游模块（路径可移植，找不到就返回 None 并跳过该项）
# ======================================================================
def _load_module(name: str, candidates: list[Path]):
    for p in candidates:
        if p and Path(p).exists():
            spec = importlib.util.spec_from_file_location(name, str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod, Path(p)
    return None, None


def load_c_fk(project_root: Path | None = None):
    """载入 C 组的 `fk.py`（返回 (模块, 路径) 或 (None, None)）。"""
    from .mujoco_backend import project_root as _root

    root = project_root or _root()
    return _load_module("g_upstream_c_fk", [
        root / "src" / "kinematics" / "fk.py",
        root / "upstream_refs" / "c_fk.py",
    ])


def load_d_ik(project_root: Path | None = None):
    """载入 D 组的 `ik.py`（G 整合时只改了模型路径解析，算法未动）。"""
    from .mujoco_backend import project_root as _root

    root = project_root or _root()
    return _load_module("g_upstream_d_ik", [
        root / "src" / "kinematics" / "ik.py",
        root / "upstream_refs" / "d_ik.py",
    ])


# ======================================================================
# 1. C 的 FK
# ======================================================================
def check_c_fk(model, n: int = 10, seed: int = 42, low: float = -1.5, high: float = 1.5) -> list[dict]:
    """复现 C 组自己的测试（`np.random.seed(42)`，q∈[-1.5,1.5]），与 MuJoCo 比。

    C 的 `fk(q)` 输出**法兰 wrist_3_link**，所以对比基准是 `model.fk_flange(q)`。
    """
    mod, path = load_c_fk()
    if mod is None:
        return [_row("c_fk", "-", "module_found", 0, "bool", 1, False,
                     "未找到 src/kinematics/fk.py，跳过 C 的 FK 交叉验证")]
    rs = np.random.RandomState(seed)
    pos, rot = [], []
    for _ in range(n):
        q = rs.uniform(low, high, 6)
        T_c = np.asarray(mod.fk(q), dtype=np.float64)
        T_m = model.fk_flange(q)
        pos.append(float(np.linalg.norm(T_c[:3, 3] - T_m[:3, 3])))
        rot.append(float(np.linalg.norm(so3_log(T_m[:3, :3], T_c[:3, :3]))))
    pos_mm = np.array(pos) * 1000.0
    rot_deg = np.degrees(np.array(rot))
    rows = [
        _row("c_fk", str(path), "pos_max_mm", float(pos_mm.max()), "mm",
             TOL["c_fk_pos_mm"], float(pos_mm.max()) <= TOL["c_fk_pos_mm"],
             f"{n} 组随机关节角（seed={seed}, ±1.5 rad），对比 MuJoCo 法兰位姿"),
        _row("c_fk", str(path), "pos_mean_mm", float(pos_mm.mean()), "mm",
             TOL["c_fk_pos_mm"], float(pos_mm.mean()) <= TOL["c_fk_pos_mm"], ""),
        _row("c_fk", str(path), "rot_max_deg", float(rot_deg.max()), "deg",
             TOL["c_fk_rot_deg"], float(rot_deg.max()) <= TOL["c_fk_rot_deg"],
             "C 用补偿矩阵对齐 MuJoCo，姿态应当几乎精确"),
    ]
    # 坐标系陷阱：拿法兰位姿当 TCP 用会差多少
    q = np.zeros(6)
    T_c = np.asarray(mod.fk(q), dtype=np.float64)
    T_tcp = model.fk_tcp(q)
    gap_mm = float(np.linalg.norm(T_c[:3, 3] - T_tcp[:3, 3])) * 1000
    gap_deg = float(np.degrees(np.linalg.norm(so3_log(T_tcp[:3, :3], T_c[:3, :3]))))
    rows.append(
        _row("c_fk_tcp", str(path), "flange_vs_tcp_pos_mm", gap_mm, "mm", 1.0, False,
             "把 C 的法兰 FK 直接当 TCP 用会产生的偏差（坐标系陷阱，见 F 的接口说明）")
    )
    rows.append(
        _row("c_fk_tcp", str(path), "flange_vs_tcp_rot_deg", gap_deg, "deg", 1.0, False,
             "法兰与 TCP 的姿态差（模型里 attachment_site 相对 wrist_3_link 有 90° 旋转）")
    )
    return rows


# ======================================================================
# 2. G 的数值 Jacobian
# ======================================================================
def check_numeric_jacobian(model, n: int = 5, seed: int = 7,
                           eps_list=(1e-4, 1e-5, 1e-6, 1e-7)) -> list[dict]:
    """数值微分 Jacobian vs `mj_jacSite` 解析 Jacobian。

    第一步：`linalg.geometric_jacobian` 的角速度行是**体坐标**（右乘增量），
    而 `mj_jacSite` 是世界系，所以比较前要左乘 R 换算到世界系：
    ``J_world[3:] = R @ J_body[3:]``。

    第二步：扫差分步长，给出"中心差分误差 ~h²"的实测验证。
    """
    rs = np.random.RandomState(seed)
    qs = [rs.uniform(-2.0, 2.0, 6) for _ in range(n)]
    rows: list[dict] = []
    for eps in eps_list:
        worst = 0.0
        for q in qs:
            T = model.fk_tcp(q)
            J_num = geometric_jacobian(model.fk_tcp, q, eps=eps, method="central")
            J_num[3:, :] = T[:3, :3] @ J_num[3:, :]
            J_ref = model.jacobian_tcp(q)
            worst = max(worst, float(np.max(np.abs(J_num - J_ref))))
        rows.append(_row("numeric_jacobian", f"eps={eps:g}", "max_abs_diff", worst, "-",
                         TOL["numeric_jac_abs"], worst <= max(TOL["numeric_jac_abs"], 100 * eps),
                         f"中心差分，{n} 个随机关节角，逐元素最大偏差"))
    return rows


# ======================================================================
# 3. 法兰 → TCP
# ======================================================================
def check_flange_tcp(model) -> list[dict]:
    rs = np.random.RandomState(3)
    worst_p = worst_r = 0.0
    for _ in range(20):
        q = rs.uniform(-2.5, 2.5, 6)
        T_pred = model.fk_flange(q) @ model.T_flange_tcp
        T_true = model.fk_tcp(q)
        worst_p = max(worst_p, float(np.linalg.norm(T_pred[:3, 3] - T_true[:3, 3])))
        worst_r = max(worst_r, float(np.linalg.norm(T_pred[:3, :3] - T_true[:3, :3])))
    return [
        _row("flange_tcp", "T_flange_tcp", "pos_err_m", worst_p, "m",
             TOL["flange_tcp_pos_m"], worst_p <= TOL["flange_tcp_pos_m"],
             f"T_tcp == T_flange @ T_flange_tcp，offset={model.tcp_offset_m:.4f} m"),
        _row("flange_tcp", "T_flange_tcp", "rot_err", worst_r, "-",
             TOL["flange_tcp_rot"], worst_r <= TOL["flange_tcp_rot"], ""),
    ]


# ======================================================================
# 4. D 的 IK
# ======================================================================
def find_d_ik_csv(project_root: Path | None = None) -> Path | None:
    from .mujoco_backend import project_root as _root

    root = project_root or _root()
    for rel in ("upstream_refs/d_ik_cases.csv",
                os.path.join("results", "logs", "d_ik_cases.csv")):
        p = root / rel
        if p.exists():
            return p
    return None


def check_d_ik(model, n: int = 10, seed: int = 42) -> list[dict]:
    """复现 D 组的 10 组测试，并与 D 自己 `ik_cases.csv` 的数值逐行比对。

    D 的测试逻辑（`tests/test_ik_10cases.py`）：
    ``np.random.seed(42)`` → `q_true = uniform(-2,2,6)` → `T = get_fk(q_true)`
    → `q_sol = ik_dls(T, zeros(6))` → 用 FK 回代算误差。这里完全照抄。
    """
    mod, path = load_d_ik()
    if mod is None:
        return [_row("d_ik_replay", "-", "module_found", 0, "bool", 1, False,
                     "未找到 src/kinematics/ik.py，跳过 D 的 IK 复现")]

    rs = np.random.RandomState(seed)
    got = []
    for i in range(n):
        q_true = rs.uniform(-2.0, 2.0, 6)
        T_t = np.asarray(mod.get_fk(q_true), dtype=np.float64)
        q_sol, ok, info = mod.ik_dls(T_t, np.zeros(6))
        T_c = np.asarray(mod.get_fk(q_sol), dtype=np.float64)
        pos = float(np.linalg.norm(T_c[:3, 3] - T_t[:3, 3]))
        rot = float(np.linalg.norm(so3_log(T_t[:3, :3], T_c[:3, :3])))
        got.append({"i": i + 1, "ok": bool(ok), "pos": pos, "rot": rot,
                    "iters": int(info.get("iterations", -1))})

    pos = np.array([g["pos"] for g in got])
    rot = np.array([g["rot"] for g in got])
    n_ok = sum(g["ok"] for g in got)
    rows = [
        _row("d_ik_replay", str(path), "success_rate", n_ok / n, "-", 1.0, n_ok == n,
             f"D 的 ik_dls 在 seed={seed} 的 {n} 组目标上成功率 {n_ok}/{n}"),
        _row("d_ik_replay", str(path), "pos_max_m", float(pos.max()), "m",
             TOL["d_ik_pos_m"], float(pos.max()) <= TOL["d_ik_pos_m"],
             "FK 回代的位置误差（D 的收敛判据是 1e-4 m）"),
        _row("d_ik_replay", str(path), "rot_max_rad", float(rot.max()), "rad",
             TOL["d_ik_rot_rad"], float(rot.max()) <= TOL["d_ik_rot_rad"], ""),
    ]

    # 与 D 交付的 CSV 逐行比对（验证"D 报的数字能被别人复现"）
    d_csv = find_d_ik_csv()
    if d_csv is not None:
        reported = _read_d_ik_csv(d_csv)
        if reported is not None and len(reported["pos"]) == n:
            dpose = np.abs(np.array([g["pos"] for g in got]) - reported["pos"])
            drot = np.abs(np.array([g["rot"] for g in got]) - reported["rot"])
            rows.append(_row(
                "d_ik_replay", str(d_csv), "replay_pos_delta_max_m",
                float(dpose.max()), "m", TOL["d_ik_replay_tol"],
                float(dpose.max()) <= TOL["d_ik_replay_tol"],
                "与 D 交付 CSV 中同一行位置误差的最大绝对差（应≈0，说明可复现）"))
            rows.append(_row(
                "d_ik_replay", str(d_csv), "replay_rot_delta_max_rad",
                float(drot.max()), "rad", TOL["d_ik_rot_rad"],
                float(drot.max()) <= TOL["d_ik_rot_rad"], ""))
    else:
        rows.append(_row("d_ik_replay", "-", "csv_found", 0, "bool", 1, False,
                         "未找到 D 的 ik_cases.csv，只能验证收敛性、无法逐行复现比对"))

    # 带限位的增强版：默认参数（F 组交付值 lam=0.08）
    if hasattr(mod, "ik_dls_limited"):
        n_ok2, worst2 = 0, 0.0
        for q_true in _d_seed42_targets(n, seed):
            T_t = np.asarray(mod.get_fk(q_true), dtype=np.float64)
            _, ok2, info2 = mod.ik_dls_limited(T_t, np.zeros(6))
            n_ok2 += int(bool(ok2))
            worst2 = max(worst2, float(info2.get("pos_mm", np.inf)))
        rows.append(_row("d_ik_limited", str(path), "success_rate", n_ok2 / n, "-", 1.0,
                         n_ok2 == n,
                         f"ik_dls_limited 默认 lam=0.08：{n_ok2}/{n} 收敛，最差 {worst2:.2f} mm；"
                         "该变体在少数目标上因阻尼过大停滞（详见 G6 的 lam 扫描）"))
    return rows


def check_d_ik_damping(model, n: int = 10, seed: int = 42,
                       lambdas=(0.005, 0.01, 0.02, 0.05, 0.08)) -> list[dict]:
    """`ik_dls_limited` 的阻尼系数敏感性扫描（G 组对 D/F 算法的诊断性分析）。

    D 的原始 `ik_dls` 用 lam=0.01 且无步长上限；F 的增强版把默认 lam 提到 0.08
    并加限位裁剪。本函数量化"阻尼取多大才不影响成功率"，为 E 组选参数提供依据。
    """
    mod, path = load_d_ik()
    if mod is None or not hasattr(mod, "ik_dls_limited"):
        return [_row("d_ik_damping", "-", "module_found", 0, "bool", 1, False,
                     "未找到 ik_dls_limited，跳过阻尼扫描")]
    targets = _d_seed42_targets(n, seed)
    rows = []
    for lam in lambdas:
        ok, worst = 0, 0.0
        for q_true in targets:
            T_t = np.asarray(mod.get_fk(q_true), dtype=np.float64)
            _, good, info = mod.ik_dls_limited(T_t, np.zeros(6), lam=float(lam))
            ok += int(bool(good))
            worst = max(worst, float(info.get("pos_mm", np.inf)))
        row = _row("d_ik_damping", f"lam={lam:g}", "success_rate", ok / n, "-", 1.0,
                   ok == n, f"最差位置误差 {worst:.4f} mm")
        row["lam"] = float(lam)
        row["worst_mm"] = float(worst)
        rows.append(row)
    return rows


def _d_seed42_targets(n: int, seed: int) -> list[np.ndarray]:
    rs = np.random.RandomState(seed)
    return [rs.uniform(-2.0, 2.0, 6) for _ in range(n)]


def _read_d_ik_csv(path: Path) -> dict | None:
    """读 D 的 `ik_cases.csv`（中文表头，UTF-8 BOM）。列序：序号,成功,位置误差,姿态误差,迭代。"""
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh))
        if len(rows) < 2:
            return None
        pos, rot, ok = [], [], []
        for r in rows[1:]:
            ok.append(str(r[1]).strip().lower() in {"true", "1", "yes"})
            pos.append(float(r[2]))
            rot.append(float(r[3]))
        return {"pos": np.array(pos), "rot": np.array(rot), "ok": np.array(ok)}
    except Exception:
        return None


# ======================================================================
# 5. F 的执行日志
# ======================================================================
def check_f_log(model, log: ExecLog) -> list[dict]:
    """F 日志的两项数据质量检查：`ee_*` 采样滞后、对齐后残差。"""
    from .tracking import detect_pose_lag

    lag = detect_pose_lag(model, log)
    rows = [
        _row("f_log_pose_lag", "ee_*", "lag_samples", lag["lag_samples"], "sample",
             TOL["f_log_lag_samples"], abs(lag["lag_samples"]) <= TOL["f_log_lag_samples"],
             f"最优滞后 {lag['lag_samples']} 个采样 = {lag['lag_time_s'] * 1000:.1f} ms；"
             "F 先 mj_step 后读 site_xpos，导致末端列落后关节角一步"),
        _row("f_log_residual", "ee_*", "residual_max_mm", lag["best_max_m"] * 1000, "mm",
             TOL["f_log_residual_mm"], lag["best_max_m"] * 1000 <= TOL["f_log_residual_mm"],
             "滞后校正后 ee_* 与 fk(q_actual) 的最大残差（应≈CSV 6 位小数舍入）"),
        _row("f_log_residual", "ee_*", "naive_max_mm", lag["naive_max_m"] * 1000, "mm",
             TOL["f_log_residual_mm"], False,
             "**不校正滞后**直接用 ee_* 算误差时会凭空多出的最大偏差"),
    ]
    return rows
