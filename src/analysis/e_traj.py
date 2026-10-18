"""E 组轨迹（E 模块交付）的加载、分段与规划侧指标。

G 组不重新规划、不改 E 的代码，本模块只做三件事：

1. 读 E 的 ``pick_place_traj.npz`` 并按数据契约校验（time/q/dq/ddq）；
2. 用 E 自己的轨迹模块（``upstream_refs/e_plan``，原样副本）重建一次规划的
   **分段边界与分段时长**，作为权威分段依据；重建失败时退化为"各段端点静止"
   的数据驱动检测，并在输出里注明来源；
3. 计算每段与全轨迹的规划侧指标（TCP 路径长度/直线度、关节路径长度、
   峰值速度/加速度）——全部基于 E 交付的**数据**，不是重新生成的轨迹。

为什么分段要信 E 的模块：E 的 7 段都是 rest-to-rest，段边界处关节速度≈0，
单看 ``dq`` 最小值只能在 ±0.1 s 内定位边界；而 E 的分段时长会因 vmax/amax
自动缩放，不能直接用报告里的请求时长累加。重建规划只需十几秒，且能同时
给出每段笛卡尔 IK 的失败点数，是更强的证据。

单位约定：时间 s、关节角 rad、关节速度 rad/s、关节加速度 rad/s²、位置 m。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: E 交付的轨迹文件名（.npz 主格式）
E_TRAJ_FILENAME = "pick_place_traj.npz"
#: E 轨迹模块副本所在目录（相对工程根）
E_PLAN_RELPATH = Path("upstream_refs") / "e_plan"
#: E 轨迹在工程内的默认落位（G 把 E 的 npz 放入 upstream_refs/e_traj/）
E_TRAJ_RELPATH = Path("upstream_refs") / "e_traj" / E_TRAJ_FILENAME

#: E 报告的 7 段顺序（与重建出的 segment_names 一一对应）
SEGMENT_NAMES = (
    "home->hover",
    "descend",
    "lift",
    "transfer",
    "lower",
    "retreat",
    "return->home",
)
#: E 报告表里的**请求**时长（s）。实际轨迹因 vmax/amax 自动缩放会与之不同，
#: 两者都要写进报告，不能混用。
REPORTED_DURATIONS_S = (2.0, 1.6, 1.6, 3.0, 1.6, 1.2, 2.0)

#: E 声明的关键高度（m，TCP 到基座坐标），用于校验轨迹几何
KEY_HEIGHTS_M = {
    "hover": 0.30,
    "grasp": 0.055,
    "lift": 0.38,
    "place": 0.065,
    "clear": 0.12,
}
#: E 声明的速度/加速度上限（rad/s, rad/s²）
E_VMAX_RAD_S = 1.0
E_AMAX_RAD_S2 = 3.0


def project_root() -> Path:
    """返回工程根（含 ``src/`` 的目录），与其它分析模块保持一致。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir():
            return parent
    return Path.cwd()


def find_e_traj_path(explicit: str | Path | None = None) -> Path | None:
    """按"显式参数 → 环境变量 → 工程内约定位置"的顺序找 E 的 npz。"""
    import os

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("G_E_TRAJ")
    if env:
        candidates.append(Path(env))
    root = project_root()
    candidates.extend(
        [
            root / E_TRAJ_RELPATH,
            root / "data" / E_TRAJ_FILENAME,
            root / "results" / "e_traj" / E_TRAJ_FILENAME,
        ]
    )
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


@dataclass
class ETrajectory:
    """E 交付的整条搬运轨迹（一次完整 pick-and-place，13.386 s / 6694 点）。"""

    time_s: np.ndarray
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    ddq_rad_s2: np.ndarray
    dt_s: float
    backend: str = ""
    source: str = ""

    @property
    def n(self) -> int:
        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    def summary(self) -> dict:
        return {
            "source": self.source,
            "backend": self.backend,
            "n_samples": self.n,
            "dt_s": float(self.dt_s),
            "duration_s": self.duration_s,
            "q_start_rad": [float(v) for v in self.q_rad[0]],
            "q_final_rad": [float(v) for v in self.q_rad[-1]],
        }


def _load_npz(path: Path) -> ETrajectory:
    data = np.load(path, allow_pickle=False)
    missing = [k for k in ("time", "q_traj") if k not in data]
    if missing:
        raise ValueError(f"{path} 缺少必需键 {missing}；实际键={list(data.keys())}")
    time_s = np.asarray(data["time"], dtype=np.float64)
    q = np.asarray(data["q_traj"], dtype=np.float64)
    dq = np.asarray(data["dq_traj"], dtype=np.float64) if "dq_traj" in data else np.zeros_like(q)
    ddq = np.asarray(data["ddq_traj"], dtype=np.float64) if "ddq_traj" in data else np.zeros_like(q)
    dt = float(data["dt"]) if "dt" in data else float(np.median(np.diff(time_s)))
    backend = str(data["backend"]) if "backend" in data else ""
    return ETrajectory(time_s=time_s, q_rad=q, dq_rad_s=dq, ddq_rad_s2=ddq,
                       dt_s=dt, backend=backend, source=str(path))


def _load_csv(path: Path) -> ETrajectory:
    """备用格式：``t,q1..q6,dq1..dq6,ddq1..ddq6``（E 同时导出的 .csv）。"""
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    cols = reader.fieldnames or []
    if not rows or not cols:
        raise ValueError(f"{path} 为空或没有表头")
    time_s = np.array([float(r["t"]) for r in rows])
    q = np.array([[float(r[f"q{i + 1}"]) for i in range(6)] for r in rows])
    dq = np.array([[float(r[f"dq{i + 1}"]) for i in range(6)] for r in rows])
    ddq = np.array([[float(r[f"ddq{i + 1}"]) for i in range(6)] for r in rows])
    return ETrajectory(time_s=time_s, q_rad=q, dq_rad_s=dq, ddq_rad_s2=ddq,
                       dt_s=float(np.median(np.diff(time_s))), backend="csv", source=str(path))


def load_e_trajectory(path: str | Path | None = None) -> ETrajectory:
    """加载 E 的整条搬运轨迹；找不到文件时抛出带指引的 ``FileNotFoundError``。"""
    p = find_e_traj_path(path)
    if p is None:
        raise FileNotFoundError(
            "未找到 E 组轨迹文件（pick_place_traj.npz）。请把它放到 "
            f"{E_TRAJ_RELPATH}，或用 --e-traj 指定，或设置环境变量 G_E_TRAJ。"
        )
    traj = _load_csv(p) if p.suffix.lower() == ".csv" else _load_npz(p)
    if traj.q_rad.ndim != 2 or traj.q_rad.shape[1] != 6:
        raise ValueError(f"q_traj 形状应为 (N,6)，实际 {traj.q_rad.shape}")
    if traj.time_s.shape != (traj.q_rad.shape[0],):
        raise ValueError("time 与 q_traj 点数不一致")
    return traj


# ---------------------------------------------------------------------------
# 分段
# ---------------------------------------------------------------------------
def rebuild_plan_segments(upstream_dir: str | Path | None = None) -> dict | None:
    """用 E 自己的轨迹模块重建规划，取回权威分段边界与每段 IK 失败点数。

    重建使用与 E 交付时相同的默认参数（dt=0.002、home=keyframe、上下料盘
    ``(-0.35,0.40)``/``(0.35,0.40)``、默认关键高度）。返回 ``None`` 表示当前
    环境不具备重建条件（缺 E 模块副本或缺 MuJoCo），调用方应退化到数据驱动检测。
    """

    root = project_root()
    plan_dir = Path(upstream_dir) if upstream_dir is not None else root / E_PLAN_RELPATH
    if not (plan_dir / "trajectory.py").is_file():
        return None
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from upstream_refs.e_plan import trajectory as etraj  # noqa: WPS433 - 延迟导入
    except Exception:  # noqa: BLE001 - 重建只是"更强证据"，失败不应中断分析
        return None

    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    keypoints = {"up_pos": (-0.35, 0.40), "down_pos": (0.35, 0.40)}
    try:
        plan = etraj.plan_pick_and_place(home, keypoints, dt=0.002)
    except Exception:  # noqa: BLE001
        return None

    durations = [float(seg.t[-1]) for seg in plan.segments]
    starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    n_points = [int(len(seg.q)) - 1 for seg in plan.segments]
    ik_fail = [int(len(getattr(seg, "fail_idx", []))) for seg in plan.segments]
    return {
        "segment_names": list(plan.segment_names),
        "starts_s": [float(v) for v in starts],
        "durations_s": durations,
        "n_points": n_points,
        "ik_fail_counts": ik_fail,
        "n_samples": int(len(plan.t)),
        "total_duration_s": float(plan.T_total),
        "backend": str(getattr(plan, "backend", "")),
        "within_limits": bool(getattr(plan, "within_limits", True)),
        "peak_velocity_rad_s": [float(v) for v in plan.peak_velocity],
        "peak_acceleration_rad_s2": [float(v) for v in plan.peak_acceleration],
        "source": "rebuild(upstream_refs/e_plan)",
    }


def detect_rest_boundaries(time_s: np.ndarray, dq_rad_s: np.ndarray,
                           threshold_rad_s: float = 2e-2) -> list[int]:
    """数据驱动的段边界：所有段都是 rest-to-rest，边界处 ``max|dq| ≈ 0``。

    返回 6 个内部边界下标（不含首尾）。只用于重建失败时的退化路径，
    边界精度受平滑影响约 ±10 个采样点，报告里必须注明来源。
    """
    vmax = np.abs(dq_rad_s).max(axis=1)
    low = vmax < threshold_rad_s
    idx = np.where(low)[0]
    # 把连续的低速段压成一个代表点（取该段内最小值处）
    runs: list[list[int]] = []
    for i in idx:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    cores = [r[0] + int(np.argmin(vmax[r[0]:r[1] + 1])) for r in runs]
    inner = [c for c in cores if 0 < c < len(time_s) - 1]
    return inner[:6]


def segment_windows_from_starts(time_s: np.ndarray, starts_s) -> list[tuple[int, int]]:
    """把"各段起始时间"换算成首尾相接的下标窗口 ``[(i0,i1), ...]``。"""
    idx = [int(np.searchsorted(time_s, float(s) - 1e-9)) for s in starts_s]
    idx.append(len(time_s) - 1)
    return [(idx[k], idx[k + 1]) for k in range(len(starts_s))]


def resolve_segments(traj: ETrajectory, rebuild: bool = True,
                     upstream_dir: str | Path | None = None) -> dict:
    """确定 7 段的分段窗口。优先重建 E 的规划，失败则用静止点检测。"""
    info: dict = {"source": "rest-detect", "rebuild": None}
    if rebuild:
        info["rebuild"] = rebuild_plan_segments(upstream_dir)
    plan = info["rebuild"]
    if plan is not None and abs(plan["total_duration_s"] - traj.duration_s) < 1e-6:
        windows = segment_windows_from_starts(traj.time_s, plan["starts_s"])
        info.update(
            source="rebuild(upstream_refs/e_plan)",
            segment_names=list(plan["segment_names"]),
            starts_s=list(plan["starts_s"]),
            durations_s=list(plan["durations_s"]),
            ik_fail_counts=list(plan["ik_fail_counts"]),
        )
        return {"windows": windows, "info": info}

    inner = detect_rest_boundaries(traj.time_s, traj.dq_rad_s)
    idx = [0] + inner + [len(traj.time_s) - 1]
    windows = [(idx[k], idx[k + 1]) for k in range(len(idx) - 1)]
    info.update(
        source="rest-detect",
        segment_names=list(SEGMENT_NAMES[: len(windows)]),
        starts_s=[float(traj.time_s[i0]) for i0, _ in windows],
        durations_s=[float(traj.time_s[i1] - traj.time_s[i0]) for i0, i1 in windows],
        ik_fail_counts=[None] * len(windows),
    )
    return {"windows": windows, "info": info}


# ---------------------------------------------------------------------------
# 规划侧指标
# ---------------------------------------------------------------------------
def per_segment_rows(time_s: np.ndarray, q_rad: np.ndarray, positions: np.ndarray,
                     names, windows) -> list[dict]:
    """每段的 TCP/关节路径与直线度、峰值速度加速度（全部是规划侧指标）。"""
    rows: list[dict] = []
    for name, (i0, i1) in zip(names, windows):
        p = positions[i0:i1 + 1]
        q = q_rad[i0:i1 + 1]
        chord = float(np.linalg.norm(p[-1] - p[0]))
        path = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0
        if chord > 1e-9 and len(p) > 2:
            u = (p[-1] - p[0]) / chord
            w = p - p[0]
            proj = w @ u
            dev = float(np.linalg.norm(w - np.outer(proj, u), axis=1).max())
        else:
            dev = 0.0
        dq = np.gradient(q, time_s[i0:i1 + 1], axis=0) if len(q) > 2 else np.zeros_like(q)
        ddq = np.gradient(dq, time_s[i0:i1 + 1], axis=0) if len(q) > 2 else np.zeros_like(q)
        rows.append(
            {
                "segment": name,
                "i0": int(i0),
                "i1": int(i1),
                "n_samples": int(i1 - i0 + 1),
                "t0_s": float(time_s[i0]),
                "t1_s": float(time_s[i1]),
                "duration_s": float(time_s[i1] - time_s[i0]),
                "tcp_path_m": path,
                "tcp_chord_m": chord,
                "straightness": (chord / path) if path > 0 else float("nan"),
                "max_dev_from_chord_m": dev,
                "joint_path_rad": float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))),
                "joint_speed_max_rad_s": float(np.abs(dq).max()) if len(q) else 0.0,
                "joint_accel_max_rad_s2": float(np.abs(ddq).max()) if len(q) else 0.0,
                "tcp_z_start_m": float(p[0, 2]),
                "tcp_z_end_m": float(p[-1, 2]),
                "tcp_x_start_m": float(p[0, 0]),
                "tcp_x_end_m": float(p[-1, 0]),
            }
        )
    return rows


def derivative_consistency(time_s: np.ndarray, q_rad: np.ndarray,
                           dq_rad_s: np.ndarray, ddq_rad_s2: np.ndarray) -> dict:
    """校验 E 给的 dq/ddq 与对 q 做数值微分的结果是否一致。

    E 的 dq/ddq 是分段 Savitzky-Golay 平滑后求导，边界处会有平滑尾迹，
    因此 ``max|Δ|`` 只能作为量级检查（报告里要说明这一点）。
    """
    dq_fd = np.gradient(q_rad, time_s, axis=0)
    ddq_fd = np.gradient(dq_fd, time_s, axis=0)
    return {
        "dq_max_abs_diff_rad_s": float(np.abs(dq_rad_s - dq_fd).max()),
        "dq_rms_diff_rad_s": float(np.sqrt(np.mean((dq_rad_s - dq_fd) ** 2))),
        "ddq_max_abs_diff_rad_s2": float(np.abs(ddq_rad_s2 - ddq_fd).max()),
        "ddq_rms_diff_rad_s2": float(np.sqrt(np.mean((ddq_rad_s2 - ddq_fd) ** 2))),
    }


def _row(check: str, metric: str, value, unit: str, threshold, passed: bool, note: str = "") -> dict:
    return {
        "check": check,
        "metric": metric,
        "value": value,
        "unit": unit,
        "threshold": threshold,
        "passed": bool(passed),
        "note": note,
    }


def check_e_trajectory(traj: ETrajectory, model, windows=None, names=None) -> list[dict]:
    """对 E 交付的轨迹做数据契约与几何自检，返回逐项结论行。

    ``model`` 需提供 ``positions_batch(q)``、``fk_batch(q)``、``joint_limits()``、
    ``home_q()``（即 :class:`src.analysis.mujoco_backend.UR5eModel`）。
    """
    t, q, dq, ddq = traj.time_s, traj.q_rad, traj.dq_rad_s, traj.ddq_rad_s2
    rows: list[dict] = []
    dt = np.diff(t)
    rows.append(_row("contract", "n_samples", traj.n, "count", ">=2", traj.n >= 2))
    rows.append(_row("contract", "dt_uniform_max_dev", float(np.abs(dt - traj.dt_s).max()),
                     "s", 1e-9, bool(np.allclose(dt, traj.dt_s, atol=1e-9)),
                     "E 声明 dt=0.002 s（= F 仿真步长）"))
    rows.append(_row("contract", "t0_s", float(t[0]), "s", 0.0, abs(float(t[0])) < 1e-12))
    rows.append(_row("contract", "monotonic", bool(np.all(dt > 0)), "bool", True, bool(np.all(dt > 0))))
    rows.append(_row("contract", "finite", bool(np.isfinite(q).all() and np.isfinite(dq).all()),
                     "bool", True, bool(np.isfinite(q).all() and np.isfinite(dq).all())))
    rows.append(_row("endpoint", "duration_s", traj.duration_s, "s", None, traj.duration_s > 0,
                     f"E 报告 13.39 s；time[-1]=(N-1)*dt={traj.duration_s:.6f}"))

    home = np.asarray(model.home_q(), dtype=np.float64)
    rows.append(_row("endpoint", "start_home_max_dev", float(np.abs(q[0] - home).max()), "rad", 1e-3,
                     float(np.abs(q[0] - home).max()) < 1e-3, "起末必须等于 keyframe home"))
    rows.append(_row("endpoint", "final_home_max_dev", float(np.abs(q[-1] - home).max()), "rad", 1e-3,
                     float(np.abs(q[-1] - home).max()) < 1e-3))

    lo, hi = model.joint_limits()
    viol = float(max(np.max(np.maximum(lo - q, 0.0)), np.max(np.maximum(q - hi, 0.0))))
    rows.append(_row("limits", "joint_limit_violation_max", viol, "rad", 1e-9, viol <= 1e-9,
                     "E 声明 J3 限 ±3.1415、其余 ±6.28319"))

    peak_v = float(np.abs(dq).max())
    peak_a = float(np.abs(ddq).max())
    rows.append(_row("limits", "peak_joint_speed", peak_v, "rad/s", E_VMAX_RAD_S,
                     peak_v <= E_VMAX_RAD_S + 1e-6, f"逐关节峰值={np.round(np.abs(dq).max(axis=0), 4).tolist()}"))
    rows.append(_row("limits", "peak_joint_accel", peak_a, "rad/s2", E_AMAX_RAD_S2,
                     peak_a <= E_AMAX_RAD_S2 + 1e-6, f"逐关节峰值={np.round(np.abs(ddq).max(axis=0), 4).tolist()}"))

    cons = derivative_consistency(t, q, dq, ddq)
    rows.append(_row("derivative", "dq_max_abs_diff", cons["dq_max_abs_diff_rad_s"], "rad/s",
                     E_VMAX_RAD_S * 0.05, cons["dq_max_abs_diff_rad_s"] < E_VMAX_RAD_S * 0.05,
                     "E 的 dq 为分段 Savitzky-Golay 平滑后求导，与数值微分不逐点相同"))
    rows.append(_row("derivative", "ddq_max_abs_diff", cons["ddq_max_abs_diff_rad_s2"], "rad/s2",
                     E_AMAX_RAD_S2, cons["ddq_max_abs_diff_rad_s2"] < E_AMAX_RAD_S2,
                     "同上，平滑导致峰值差异"))

    pos = model.positions_batch(q)
    if windows is not None and names is not None:
        for name, (i0, i1) in zip(names, windows):
            z0, z1 = float(pos[i0, 2]), float(pos[i1, 2])
            if name == "descend":
                rows.append(_row("geometry", "descend_target_z", z1, "m", KEY_HEIGHTS_M["grasp"],
                                  abs(z1 - KEY_HEIGHTS_M["grasp"]) < 1e-3, "抓取高度"))
            elif name == "lift":
                rows.append(_row("geometry", "lift_target_z", z1, "m", KEY_HEIGHTS_M["lift"],
                                  abs(z1 - KEY_HEIGHTS_M["lift"]) < 1e-3, "抬升高度"))
            elif name == "transfer":
                straight = _straightness(pos[i0:i1 + 1])
                rows.append(_row("geometry", "transfer_straightness", straight, "ratio", 0.9995,
                                  straight >= 0.9995, "笛卡尔直线段：弦长/路径长"))
                zspan = float(np.ptp(pos[i0:i1 + 1, 2]))
                rows.append(_row("geometry", "transfer_z_span", zspan, "m", 1e-3,
                                  zspan < 1e-3, "搬运段 TCP 高度应恒定"))
            elif name == "lower":
                rows.append(_row("geometry", "lower_target_z", z1, "m", KEY_HEIGHTS_M["place"],
                                  abs(z1 - KEY_HEIGHTS_M["place"]) < 1e-3, "放置高度"))
            elif name == "retreat":
                rows.append(_row("geometry", "retreat_target_z", z1, "m", KEY_HEIGHTS_M["clear"],
                                  abs(z1 - KEY_HEIGHTS_M["clear"]) < 1e-3, "退让高度"))
        # 段边界速度（C1 拼接）
        inner = [i0 for i0, _ in windows[1:]]
        jump = float(np.abs(dq[inner]).max()) if inner else 0.0
        rows.append(_row("continuity", "boundary_speed_max", jump, "rad/s", 5e-3, jump < 5e-3,
                         "各段 rest-to-rest，边界速度应≈0（E 报告 ≤2.5e-3）"))
    return rows


def _straightness(points: np.ndarray) -> float:
    chord = float(np.linalg.norm(points[-1] - points[0]))
    path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    return chord / path if path > 0 else float("nan")
