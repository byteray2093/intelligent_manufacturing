"""可达性（工作空间）采样分析。

方法（报告必须照抄的条件说明）：
- 采样对象：关节空间。UR5e 六个关节都是 ±2π 连续旋转关节，理论上关节空间的
  工作空间**无限大**（同一空间点有无穷多组解）。因此"可达性分布"必须明确
  采样区间：默认取 ±π 的 2π 立方体（比 ±2π 更贴合"不绕整圈"的工程实际，
  也避免把同一空间区域重复计数）。
- 采样方式：`uniform`（均匀随机）或 `latin`（拉丁超立方，分层，低差异）。
  两种都要支持，便于说明"结果不依赖某一随机实现"。
- 随机性：用 `seed` 固定 NumPy Generator，报告中记录 seed 与样本数 N。
- 坐标为世界系（与 FK 的输出一致）；若 B 组定义了非平凡 T_world_base，
  需要对结果左乘 T_world_base 后再统计，并在报告中注明。
- 采样得到的"可达点云"是**离散近似**：体素占用率随 N 增长收敛，
  报告中给出 N，并说明体素边长。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_joint_space(
    n: int,
    limits_lo: np.ndarray,
    limits_hi: np.ndarray,
    seed: int = 0,
    method: str = "latin",
) -> np.ndarray:
    """在关节限位内采样，返回 q_samples (n, 6)。

    method="uniform"：均匀独立采样
    method="latin"  ：拉丁超立方（每维分层打乱），低差异、覆盖更均匀
    """
    lo = np.asarray(limits_lo, dtype=np.float64).reshape(6)
    hi = np.asarray(limits_hi, dtype=np.float64).reshape(6)
    if n <= 0:
        raise ValueError("n must be positive")
    if np.any(hi <= lo):
        raise ValueError("关节上限必须大于下限")

    rng = np.random.default_rng(seed)
    if method == "uniform":
        u = rng.random((n, 6))
    elif method == "latin":
        # 每维把 [0,1) 分成 n 层，层内均匀采样后按维打乱
        cut = (np.arange(n)[:, None] + rng.random((n, 6))) / n
        u = np.empty_like(cut)
        for j in range(6):
            u[:, j] = cut[rng.permutation(n), j]
    else:
        raise ValueError("method must be 'uniform' or 'latin'")
    return lo + u * (hi - lo)


def fk_positions(fk, q_samples: np.ndarray) -> np.ndarray:
    """对每个采样构型求 TCP 位置，返回 (n, 3)（只保留平移，省内存）。"""
    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim != 2 or q_samples.shape[1] != 6:
        raise ValueError("q_samples must be (n, 6)")
    out = np.empty((len(q_samples), 3), dtype=np.float64)
    for i, q in enumerate(q_samples):
        T = np.asarray(fk(q), dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("fk must return 4x4")
        out[i] = T[:3, 3]
    return out


def radial_reach_stats(positions: np.ndarray) -> dict:
    """从基座原点看出去的半径统计（m）：min/max/mean/分位数。"""
    p = np.asarray(positions, dtype=np.float64)
    r = np.linalg.norm(p, axis=1)
    return {
        "n": int(len(r)),
        "radius_min_m": float(np.min(r)),
        "radius_max_m": float(np.max(r)),
        "radius_mean_m": float(np.mean(r)),
        "radius_p05_m": float(np.percentile(r, 5)),
        "radius_p50_m": float(np.percentile(r, 50)),
        "radius_p95_m": float(np.percentile(r, 95)),
        "x_min_m": float(np.min(p[:, 0])),
        "x_max_m": float(np.max(p[:, 0])),
        "y_min_m": float(np.min(p[:, 1])),
        "y_max_m": float(np.max(p[:, 1])),
        "z_min_m": float(np.min(p[:, 2])),
        "z_max_m": float(np.max(p[:, 2])),
    }


def voxel_occupancy(positions: np.ndarray, voxel_size_m: float = 0.05) -> dict:
    """把点云离散成体素，统计占用数量与占用体积近似。

    返回：
      occupancy : dict[(i,j,k)] -> 点数
      occupied_voxels : int
      approx_volume_m3 : occupied * voxel_size^3（是**上界**估计，非精确体积）
      bounds / shape   : 体素网格的索引范围，便于画图
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("positions must be (n, 3)")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    idx = np.floor(p / voxel_size_m).astype(np.int64)
    keys, counts = np.unique(idx, axis=0, return_counts=True)
    occupancy = {tuple(int(v) for v in k): int(c) for k, c in zip(keys, counts)}
    return {
        "voxel_size_m": float(voxel_size_m),
        "occupancy": occupancy,
        "occupied_voxels": int(len(keys)),
        "approx_volume_m3": float(len(keys) * voxel_size_m**3),
        "index_min": idx.min(axis=0).tolist(),
        "index_max": idx.max(axis=0).tolist(),
    }


def radial_histogram(positions: np.ndarray, bins: int = 30) -> dict:
    """半径直方图（用于"可达性沿径向的分布"图）。"""
    p = np.asarray(positions, dtype=np.float64)
    r = np.linalg.norm(p, axis=1)
    counts, edges = np.histogram(r, bins=bins)
    return {"bin_edges_m": edges.tolist(), "counts": counts.tolist()}


def box_envelope_m(positions: np.ndarray) -> dict:
    """点云的世界系轴对齐包围盒（AABB），单位 m。

    用途：把"工作空间有多大"从一个模糊说法变成可核对指标（边长、对角线）。
    与 `metrics.trajectory_envelope_m` 用同一套定义（AABB + 对角线），
    保证"轨迹包络"与"工作空间包络"两组数字可以直接比。
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("positions must be (n, 3)")
    lo = np.min(p, axis=0)
    hi = np.max(p, axis=0)
    span = hi - lo
    return {
        "n": int(len(p)),
        "span_m": tuple(float(v) for v in span),
        "diagonal_m": float(np.linalg.norm(span)),
        "min_m": tuple(float(v) for v in lo),
        "max_m": tuple(float(v) for v in hi),
    }


def reachability_grid_slice(
    positions: np.ndarray,
    axis: str = "z",
    slab_half_width_m: float = 0.025,
    value_m: float | None = None,
    bins: int = 64,
) -> dict:
    """取一个薄片（例如 z ≈ 0.3 m），把点云投影到平面网格上做"可达密度图"。

    三维点云很难直观看，工程报告通常给一个水平薄片 + 一个垂直剖面。
    """
    p = np.asarray(positions, dtype=np.float64)
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis)
    if axis_index is None:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    if value_m is None:
        value_m = float(np.median(p[:, axis_index]))
    mask = np.abs(p[:, axis_index] - value_m) <= slab_half_width_m
    sub = p[mask]
    others = [i for i in range(3) if i != axis_index]
    if len(sub) == 0:
        return {
            "axis": axis, "value_m": float(value_m), "slab_half_width_m": float(slab_half_width_m),
            "n_in_slab": 0, "other_axes": others, "edges_a": [], "edges_b": [], "counts": np.zeros((0, 0)),
        }
    counts, edges_a, edges_b = np.histogram2d(sub[:, others[0]], sub[:, others[1]], bins=bins)
    return {
        "axis": axis,
        "value_m": float(value_m),
        "slab_half_width_m": float(slab_half_width_m),
        "n_in_slab": int(len(sub)),
        "other_axes": others,
        "edges_a": edges_a.tolist(),
        "edges_b": edges_b.tolist(),
        "counts": counts,
    }


@dataclass
class WorkspaceResult:
    """一次工作空间采样的完整结果（含复现所需的元数据）。"""

    n_samples: int
    seed: int
    method: str
    limits_lo: np.ndarray
    limits_hi: np.ndarray
    positions: np.ndarray
    radial_stats: dict
    voxel: dict

    def provenance(self) -> dict:
        """写进 CSV 的"可复现元数据"。"""
        return {
            "n_samples": self.n_samples,
            "seed": self.seed,
            "sampling_method": self.method,
            "joint_limits_lo_rad": ";".join(f"{v:.6f}" for v in self.limits_lo),
            "joint_limits_hi_rad": ";".join(f"{v:.6f}" for v in self.limits_hi),
            **self.radial_stats,
            "voxel_size_m": self.voxel["voxel_size_m"],
            "occupied_voxels": self.voxel["occupied_voxels"],
            "approx_volume_m3": self.voxel["approx_volume_m3"],
        }


def sample_workspace(
    fk,
    limits_lo: np.ndarray,
    limits_hi: np.ndarray,
    n_samples: int = 20000,
    seed: int = 20261015,
    method: str = "latin",
    voxel_size_m: float = 0.05,
) -> WorkspaceResult:
    """完整工作空间采样流程：采样 -> FK -> 半径统计 -> 体素占用。"""
    q = sample_joint_space(n_samples, limits_lo, limits_hi, seed=seed, method=method)
    positions = fk_positions(fk, q)
    return WorkspaceResult(
        n_samples=n_samples,
        seed=seed,
        method=method,
        limits_lo=np.asarray(limits_lo, dtype=np.float64).reshape(6).copy(),
        limits_hi=np.asarray(limits_hi, dtype=np.float64).reshape(6).copy(),
        positions=positions,
        radial_stats=radial_reach_stats(positions),
        voxel=voxel_occupancy(positions, voxel_size_m=voxel_size_m),
    )
