"""解析 F 组（控制与搬运）的真实执行日志 —— G 第二阶段数据来源。

为什么 G 要用 F 的日志
======================
任务书对 G 的要求是"统计规划与实际执行的末端误差、路径长度、速度/加速度指标"。
"执行"这一侧的真实数据只有 F 组能产生。F 组交付的
`results/logs/exec_log_latest.csv` 是 500 Hz 的逐步记录，关键列：

=====================  ====================================================
`t`                    仿真时间（s），步长 0.002
`cycle`                第几次搬运循环（1/2/3）
`segment`              流程段名：home/approach/descend/lift/transfer/lower/
                       retreat/return（**这就是"规划侧"的天然分组**）
`grip`                 夹爪指令（0/1）
`q_cmd_0..5`           **规划（指令）关节角** rad —— 轨迹来自 F 的临时
                       `src/control/traj_adapter.py`（E 组未交付时的占位）
`q_actual_0..5`        **实际（执行后）关节角** rad
`track_err_max`        F 自己算的关节跟踪误差峰值（rad）
`ee_x/y/z`             实际末端 TCP 位置（m）
`ee_qw/qx/qy/qz`       实际末端 TCP 姿态四元数（w,x,y,z）
`wp_x/y/z`             工件位置（m）
=====================  ====================================================

**重要边界（必须写进报告）**：`q_cmd` 来自 F 的临时轨迹生成器，不是 E 的
`src/planning/trajectory.py`。E 交付后只需把 `q_cmd` 换掉，G 的下游分析
（误差、路径长度、速度加速度、与 Jacobian 的耦合）**完全不用改** —— 这正是
G 坚持"只依赖 `time_s` + `q_rad` 两个契约字段"的价值。

零新增接口：G 不需要 F 增加任何列。F 没记 `ee_pose_cmd`，G 用 `fk_tcp(q_cmd)`
自己回代（`metrics.poses_from_joint_trajectory` 就是为此存在的）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REQUIRED_COLUMNS = [
    "t", "cycle", "segment", "grip",
    "q_cmd_0", "q_cmd_1", "q_cmd_2", "q_cmd_3", "q_cmd_4", "q_cmd_5",
    "q_actual_0", "q_actual_1", "q_actual_2", "q_actual_3", "q_actual_4", "q_actual_5",
    "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
]

_RELATIVE_CANDIDATES = (
    os.path.join("results", "logs", "exec_log_latest.csv"),
    os.path.join("results", "logs", "exec_log.csv"),
    os.path.join("upstream_refs", "results", "logs", "exec_log_latest.csv"),
    os.path.join("results", "logs", "exec_log_20261017_141328.csv"),
)


def find_exec_log_path(explicit: str | Path | None = None) -> Path | None:
    """按优先级查找 F 的执行日志；找不到返回 None。"""
    if explicit is not None:
        p = Path(explicit).expanduser()
        if p.exists():
            return p.resolve()

    env = os.environ.get("G_EXEC_LOG", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()

    # 复用 mujoco_backend 的工程根定位（不 import mujoco，安全）
    try:
        from .mujoco_backend import project_root
    except ImportError:  # pragma: no cover - 直接以脚本方式运行时的兜底
        from mujoco_backend import project_root  # type: ignore[no-redef]

    root = project_root()
    for rel in _RELATIVE_CANDIDATES:
        p = root / rel
        if p.exists():
            return p.resolve()
    # 兜底：任何 exec_log*.csv
    logs = root / "results" / "logs"
    if logs.is_dir():
        cands = sorted(logs.glob("exec_log*.csv"))
        if cands:
            return cands[0].resolve()
    return None


@dataclass
class SegmentRun:
    """一次连续的同名流程段（每循环重来一次）。"""

    cycle: int
    segment: str
    i0: int
    i1: int          # 不含
    time_s: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def n(self) -> int:
        return self.i1 - self.i0

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if self.n >= 2 else float("nan")

    def __str__(self) -> str:  # pragma: no cover
        return (f"cycle{self.cycle}/{self.segment}[{self.i0}:{self.i1}] "
                f"N={self.n} {self.duration_s:.3f}s")


@dataclass
class ExecLog:
    """F 执行日志的结构化视图（数组都是 NumPy，方便直接喂给 metrics）。"""

    path: str
    t: np.ndarray
    cycle: np.ndarray
    segment: np.ndarray
    grip: np.ndarray
    q_cmd: np.ndarray
    q_actual: np.ndarray
    ee_pos: np.ndarray
    ee_quat_wxyz: np.ndarray
    wp_pos: np.ndarray
    track_err_max: np.ndarray
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ 基本属性
    @property
    def n(self) -> int:
        return int(len(self.t))

    @property
    def dt_s(self) -> float:
        return float(np.median(np.diff(self.t))) if self.n > 1 else float("nan")

    @property
    def cycles(self) -> list[int]:
        return sorted({int(c) for c in self.cycle})

    @property
    def segment_names(self) -> list[str]:
        seen: list[str] = []
        for s in self.segment:
            if s not in seen:
                seen.append(str(s))
        return seen

    # ------------------------------------------------------------------ 切片
    def segments(self) -> list[SegmentRun]:
        """按 (cycle, segment) 切出所有连续段，顺序即执行顺序。"""
        runs: list[SegmentRun] = []
        start = 0
        for i in range(1, self.n + 1):
            changed = i == self.n or (
                self.segment[i] != self.segment[start] or self.cycle[i] != self.cycle[start]
            )
            if changed:
                runs.append(SegmentRun(
                    cycle=int(self.cycle[start]),
                    segment=str(self.segment[start]),
                    i0=start,
                    i1=i,
                    time_s=self.t[start:i],
                ))
                start = i
        return runs

    def cycle_slice(self, cycle: int) -> slice:
        mask = self.cycle == cycle
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise ValueError(f"日志中没有 cycle={cycle}（现有 {self.cycles}）")
        return slice(int(idx[0]), int(idx[-1]) + 1)

    def cycle_segments(self, cycle: int) -> list[SegmentRun]:
        return [r for r in self.segments() if r.cycle == cycle]

    # ------------------------------------------------------------------ 元数据
    def describe(self) -> dict:
        d = {
            "path": self.path,
            "n_samples": self.n,
            "dt_s": self.dt_s,
            "duration_s": float(self.t[-1] - self.t[0]),
            "cycles": self.cycles,
            "segment_names": self.segment_names,
            "n_segments": len(self.segments()),
            "q_cmd_absmax_rad": float(np.max(np.abs(self.q_cmd))),
            "q_actual_absmax_rad": float(np.max(np.abs(self.q_actual))),
            "grip_fraction": float(np.mean(self.grip > 0.5)),
        }
        d.update(self.meta)
        return d


def load_exec_log(path: str | Path | None = None) -> ExecLog:
    """读取 F 的 `exec_log_latest.csv`。

    解析策略：手写 split(',') 而不是 `np.loadtxt`，因为这张表混了字符串列
    （`segment`）且行数 2.5 万——手写解析约 0.2 s，比 pandas 依赖更轻。

    校验（宁可在入口报错，也不让错误数据流进分析）：
    * 表头必须含 `_REQUIRED_COLUMNS` 里的列；
    * `t` 必须单调不减、长度为 N 且与各数组一致；
    * `ee_qw..qz` 四元数必须接近单位长度。
    """
    p = find_exec_log_path(path)
    if p is None:
        raise FileNotFoundError(
            "找不到 F 组执行日志（exec_log_latest.csv）。已尝试环境变量 G_EXEC_LOG 与 "
            f"工程根下的 {list(_RELATIVE_CANDIDATES)}。"
        )
    with open(p, "r", encoding="utf-8-sig", newline="") as fh:
        header = fh.readline().strip().split(",")
        missing = [c for c in _REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"执行日志缺少必需列 {missing}；现有列 {header}")
        col = {name: i for i, name in enumerate(header)}
        rows = [line.rstrip("\n").split(",") for line in fh if line.strip()]

    n = len(rows)
    if n < 2:
        raise ValueError(f"执行日志样本太少：{n} 行")

    def fcol(name: str) -> np.ndarray:
        i = col[name]
        return np.array([float(r[i]) for r in rows], dtype=np.float64)

    t = fcol("t")
    cycle = fcol("cycle").astype(int)
    segment = np.array([r[col["segment"]] for r in rows], dtype=object)
    grip = fcol("grip")
    q_cmd = np.column_stack([fcol(f"q_cmd_{j}") for j in range(6)])
    q_actual = np.column_stack([fcol(f"q_actual_{j}") for j in range(6)])
    ee_pos = np.column_stack([fcol("ee_x"), fcol("ee_y"), fcol("ee_z")])
    ee_quat = np.column_stack([fcol("ee_qw"), fcol("ee_qx"), fcol("ee_qy"), fcol("ee_qz")])
    track_err_max = fcol("track_err_max") if "track_err_max" in col else np.zeros(n)
    wp_pos = (np.column_stack([fcol("wp_x"), fcol("wp_y"), fcol("wp_z")])
              if "wp_x" in col else np.zeros((n, 3)))

    if np.any(np.diff(t) < 0):
        raise ValueError("执行日志的 t 不是单调不减，数据可能被拼接错")
    qn = np.linalg.norm(ee_quat, axis=1)
    if np.any(np.abs(qn - 1.0) > 1e-3):
        raise ValueError(
            f"ee 四元数模长偏离 1 超过 1e-3（min={qn.min():.6f}, max={qn.max():.6f}），"
            "无法确定列顺序或归一化方式"
        )

    meta = {
        "source": "F 组（MuJoCo 控制与搬运）",
        "planning_side": (
            "q_cmd 由 F 的临时轨迹生成器 src/control/traj_adapter.py 产生"
            "（E 组未交付时的占位）；E 交付后替换 q_cmd 即可，下游分析不变"
        ),
        "columns": header,
    }
    return ExecLog(path=str(p), t=t, cycle=cycle, segment=segment, grip=grip,
                   q_cmd=q_cmd, q_actual=q_actual, ee_pos=ee_pos,
                   ee_quat_wxyz=ee_quat, wp_pos=wp_pos,
                   track_err_max=track_err_max, meta=meta)


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    """(w,x,y,z) 四元数 → 3x3 旋转矩阵（批量或单个）。"""
    q = np.asarray(quat, dtype=np.float64)
    single = q.ndim == 1
    Q = q.reshape(-1, 4)
    w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((len(Q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R[0] if single else R
