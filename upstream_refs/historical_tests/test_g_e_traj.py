"""G 组第三阶段（接入 E 组轨迹）的回归测试。

覆盖面：E 交付的 npz 数据契约、分段与时长（来自 G9 落盘结果）、逐段几何、
沿轨迹的奇异性、E 规划与 F 内置规划的同口径对比、E→F 集成复跑的接管跳变。

设计取舍：这里**不重建** E 的规划（那要 12 s，会让 pytest 变慢），
分段相关的断言一律读 ``results/csv/G9_e_segments.csv`` / ``G9_meta.json``，
这些文件由 ``scripts/G9_e_trajectory.py`` 生成；文件缺失时跳过而不是失败。

直接运行（不装 pytest 也能跑）：
    python tests/test_g_e_traj.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import e_traj as et  # noqa: E402
from src.scene import SCENE_ID_E  # noqa: E402

CSV_DIR = ROOT / "results" / "csv"


def _read_csv(name: str) -> list[dict]:
    path = CSV_DIR / name
    if not path.is_file():
        _skip_missing(name)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_meta(name: str = "G9_meta.json") -> dict:
    path = CSV_DIR / name
    if not path.is_file():
        _skip_missing(name)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _skip_missing(name: str) -> None:
    try:
        import pytest

        pytest.skip(f"缺少 {name}，先跑 scripts/G9_e_trajectory.py")
    except ImportError:  # 没装 pytest 时直接抛错，避免静默通过
        raise FileNotFoundError(f"缺少 {name}，先跑 scripts/G9_e_trajectory.py")


def _traj():
    path = et.find_e_traj_path()
    if path is None:
        _skip_missing("E 的 pick_place_traj.npz")
    return et.load_e_trajectory(path)


# ---------------------------------------------------------------------------
# E 交付物本身
# ---------------------------------------------------------------------------
def test_e_trajectory_contract():
    """E 的 npz 满足 data 契约：6694 点、dt=0.002、time 从 0 起单调、全有限。"""
    traj = _traj()
    assert traj.n == 6694
    assert abs(traj.dt_s - 0.002) < 1e-12
    assert abs(traj.duration_s - 13.386) < 1e-9
    assert np.all(np.isfinite(traj.q_rad))
    assert np.all(np.isfinite(traj.dq_rad_s))
    assert np.all(np.diff(traj.time_s) > 0)
    assert traj.q_rad.shape == (6694, 6)


def test_e_trajectory_starts_and_ends_at_home():
    """首尾都必须等于 keyframe home（否则整条轨迹不能循环执行）。"""
    from src.scene import real_model

    traj = _traj()
    home = np.asarray(real_model().home_q(), dtype=np.float64)
    assert np.max(np.abs(traj.q_rad[0] - home)) < 1e-3
    assert np.max(np.abs(traj.q_rad[-1] - home)) < 1e-3


def test_e_trajectory_within_joint_limits_and_limits():
    """不越关节限位、峰值速度/加速度不超过 E 自己声明的上限。"""
    from src.scene import real_model

    traj = _traj()
    lo, hi = real_model().joint_limits()
    viol = max(float(np.max(np.maximum(lo - traj.q_rad, 0.0))),
               float(np.max(np.maximum(traj.q_rad - hi, 0.0))))
    assert viol <= 1e-9
    assert float(np.abs(traj.dq_rad_s).max()) <= et.E_VMAX_RAD_S + 1e-6
    assert float(np.abs(traj.ddq_rad_s2).max()) <= et.E_AMAX_RAD_S2 + 1e-6


# ---------------------------------------------------------------------------
# G9 落盘结果：分段、几何、奇异、对比、集成
# ---------------------------------------------------------------------------
def test_segments_rebuilt_from_e_module():
    """7 段、内部段名与 E 模块一致、时长之和等于整条轨迹时长。"""
    meta = _read_meta()
    assert meta["segmentation"].startswith("rebuild")
    assert len(meta["segments"]) == 7
    assert meta["segments"] == list(et.SEGMENT_NAMES)
    assert abs(sum(meta["segmentation_durations_s"]) - 13.386) < 1e-6
    # E 声明"笛卡尔直线段 IK 不可解点为 0"
    assert meta["ik_fail_counts"] is not None
    assert sum(meta["ik_fail_counts"]) == 0


def test_segment_geometry_from_g9():
    """每段几何：transfer 是严格直线且高度恒定；起末两段是关节空间运动，允许弯曲。"""
    rows = {r["segment"]: r for r in _read_csv("G9_e_segments.csv")}
    assert set(rows) == set(et.SEGMENT_NAMES)
    transfer = rows["transfer"]
    assert float(transfer["straightness"]) > 0.9995
    assert float(transfer["max_dev_from_chord_m"]) < 1e-3
    assert abs(float(transfer["tcp_path_m"]) - 0.70) < 0.01
    # 搬运段 TCP 高度恒定在 LIFT=0.38 m
    assert abs(float(transfer["tcp_z_start_m"]) - 0.38) < 1e-2
    assert abs(float(transfer["tcp_z_end_m"]) - 0.38) < 1e-2
    # 关节空间段（home->hover / return->home）不可能直线
    assert float(rows["home->hover"]["straightness"]) < 0.999
    assert float(rows["return->home"]["straightness"]) < 0.999
    # E 报告的是"请求时长"，实际时长会被 vmax/amax 自动缩放：两者都落盘
    assert abs(float(transfer["reported_duration_s"]) - 3.0) < 1e-9
    assert float(transfer["duration_ratio_vs_report"]) > 1.0


def test_geometry_checks_match_reported_heights():
    """G9 的几何自检：抓取 0.055 / 抬升 0.38 / 放置 0.065 / 退让 0.12 m。"""
    rows = {r["metric"]: r for r in _read_csv("G9_e_checks.csv")}
    assert abs(float(rows["descend_target_z"]["value"]) - 0.055) < 2e-3
    assert abs(float(rows["lift_target_z"]["value"]) - 0.38) < 2e-3
    assert abs(float(rows["lower_target_z"]["value"]) - 0.065) < 2e-3
    assert abs(float(rows["retreat_target_z"]["value"]) - 0.12) < 2e-3
    assert float(rows["transfer_z_span"]["value"]) < 1e-3
    assert rows["descend_target_z"]["passed"].lower() == "true"


def test_e_trajectory_stays_away_from_singularity():
    """E 的规划全程远离奇异：σmin 最小值应远高于阈值，且无点低于阈值。"""
    rows = _read_csv("G9_e_sigma.csv")
    assert len(rows) == 6694
    sigma = np.array([float(r["sigma_min"]) for r in rows])
    cond = np.array([float(r["condition_number"]) for r in rows])
    assert sigma.min() > 0.2
    assert np.median(sigma) > 0.2
    assert cond.max() < 10.0
    meta = _read_meta()
    assert int(meta["n_singular_samples"]) == 0


def test_e_vs_f_builtin_plan_same_scene():
    """E 规划与 F 内置规划在同口径下对比：搬运段路径长度接近，时长不同。"""
    rows = {r["segment"]: r for r in _read_csv("G9_e_compare.csv")}
    transfer = rows["transfer"]
    assert abs(float(transfer["E_tcp_path_m"]) - float(transfer["F_tcp_path_m"])) < 5e-3
    assert abs(float(transfer["E_tcp_path_m"]) - 0.70) < 0.01
    # E 为了压峰值速度把搬运段拉长到 3.34 s（F 是 3.0 s）
    assert float(transfer["E_duration_s"]) > float(transfer["F_duration_s"])
    # 起末两段两套规划器的路径几乎一致（同一个 home 与上下料点）
    assert abs(float(rows["home->hover"]["E_tcp_path_m"])
               - float(rows["home->hover"]["F_tcp_path_m"])) < 5e-3


def test_exec_log_handover_jump_is_interface_mismatch():
    """E→F 集成：接管瞬间有约 0.464 rad 的指令跳变，稳态跟踪误差 < 0.5°。

    这条测试把"接口语义不匹配"与"轨迹质量"分开：跳变是 F 的 ``--traj-file``
    只替换 transfer 段却把整条 E 轨迹塞进去造成的，不是 E 的规划问题。
    """
    summary = {r["metric"]: r["value"] for r in _read_csv("G9_e_exec_summary.csv")}
    jump = float(summary["handover_jump_max_rad"])
    assert 0.4 < jump < 0.5
    assert int(summary["handover_jump_max_joint"]) == 1
    assert 6.0 < float(summary["transfer_e_start_s"]) < 8.0
    # 瞬态恢复时间约 0.8 s，之后稳态误差与 F 内置规划同量级
    assert 0.5 < float(summary["handover_recover_to_0p5deg_s"]) < 1.5
    assert float(summary["transfer_e_settled_joint_err_max_deg"]) < 0.6
    assert float(summary["transfer_e_settled_joint_err_p95_deg"]) < 0.2
    assert int(summary["handover_spike_n_samples_over_1deg"]) > 100
    # 基线：F 内置规划器执行同一控制器时关节误差 0.147°
    assert abs(float(summary["baseline_joint_err_max_deg"]) - 0.1468) < 1e-3


def test_exec_segments_cover_one_cycle():
    """复跑日志的分段表：13 段、含 transfer_e、每段都有指标。"""
    rows = _read_csv("G9_e_exec_segments.csv")
    names = [r["segment"] for r in rows]
    assert "transfer_e" in names
    assert len(rows) >= 13
    for row in rows:
        assert float(row["duration_s"]) >= 0.0
        assert np.isfinite(float(row["joint_err_max_deg"]))


def test_scene_id_is_stage3():
    """第三阶段场景号写进元数据，便于和 G5–G8 的第二阶段产物区分。"""
    assert SCENE_ID_E == "G-stage3-e-traj-v1"
    meta = _read_meta()
    assert meta["scene_id"] == SCENE_ID_E


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[ OK ] {name}")
            except Exception as exc:  # noqa: BLE001 - 自检脚本需要继续跑完
                fails += 1
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len([n for n in globals() if n.startswith('test_')]) - fails} passed, {fails} failed")
    raise SystemExit(1 if fails else 0)
