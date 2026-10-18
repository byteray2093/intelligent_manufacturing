"""G 组第二阶段自测：真实模型（B）+ 上游交付（C/D/F）+ F 的真实执行日志。

与 `tests/test_g_analysis.py` 的分工：
- `test_g_analysis.py`：只依赖 NumPy，检查分析算法本身（差分、指标、奇异值……）；
- 本文件：检查"接上真实数据之后"是否仍然成立，依赖 `mujoco`，并要求仓库里有
  `universal_robots_ur5e/ur5e.xml`（B 的模型）与 `results/logs/exec_log_latest.csv`
  （F 的日志）。缺任何一样就 skip，而不是让整个测试集报错。

设计原则（和第一阶段一致的"能解析对照就绝不看差不多"）：
1. 真实 MuJoCo FK 必须和 C 组的 DH FK 对上（阈值 2 mm / 0.05°）；
2. 法兰→TCP 的固定变换必须是纯刚体变换（位置/旋转残差 ~1e-9）；
3. 数值 Jacobian 必须和 `mj_jacSite` 对上（阈值 1e-8）；
4. F 日志的 `ee_*` 列滞后一个采样，这个"坑"必须有测试守着，否则以后有人
   直接拿 `ee_*` 算误差会得出错误结论；
5. 可达性判据必须能区分"基座正下方"（几何不可达）与普通可达点。

运行：
    python -m pytest tests/test_g_real_data.py -q
    python tests/test_g_real_data.py          # 不装 pytest 也能跑
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import exec_log, plots, tracking, validation  # noqa: E402
from src.analysis.mujoco_backend import (  # noqa: E402
    FLANGE_BODY,
    JOINT_NAMES,
    SITE_NAME,
    UR5eModel,
    find_model_path,
)
from src.analysis.singularity import (  # noqa: E402
    classify_reachability,
    singularity_metrics,
)
from src.analysis.linalg import pose_errors  # noqa: E402


# --------------------------------------------------------------------------- 夹具
_MODEL: UR5eModel | None = None
_LOG = None


def _model() -> UR5eModel:
    """整个模块复用一个 MuJoCo 模型（加载一次约 0.2 s，但每步 forward 很便宜）。"""
    global _MODEL
    if _MODEL is None:
        if find_model_path() is None:
            import pytest

            pytest.skip("仓库里找不到 ur5e.xml（B 组模型未就位）")
        _MODEL = UR5eModel()
    return _MODEL


def _log():
    global _LOG
    if _LOG is None:
        if exec_log.find_exec_log_path() is None:
            import pytest

            pytest.skip("仓库里找不到 F 组的执行日志 exec_log_latest.csv")
        _LOG = exec_log.load_exec_log()
    return _LOG


# --------------------------------------------------------------------------- 模型
def test_model_metadata_matches_b_delivery():
    """B 的模型：6 关节、关节名/顺序、TCP site、法兰 body、0.1 m TCP 偏移。"""
    model = _model()
    data = model.describe()
    assert data["nq"] == 6 and data["nv"] == 6 and data["nu"] == 6
    assert tuple(data["joint_names"]) == JOINT_NAMES
    assert data["site_name"] == SITE_NAME
    assert data["flange_body"] == FLANGE_BODY
    assert abs(data["tcp_offset_m"] - 0.1000) < 1e-9
    # 第 3 轴（elbow）限位只有 ±π，其余是 ±2π —— 这是 F 组反复提醒的坑
    lo, hi = model.joint_limits()
    assert abs(lo[2] + np.pi) < 1e-3 and abs(hi[2] - np.pi) < 1e-3
    assert abs(lo[0] + 2 * np.pi) < 1e-3 and abs(hi[0] - 2 * np.pi) < 1e-3


def test_fk_batch_matches_single_and_is_fast():
    """批量 FK 与逐点 FK 完全一致（同一实现的两条路径不能有差异）。"""
    model = _model()
    rng = np.random.default_rng(20261015)
    q = rng.uniform(-1.0, 1.0, size=(7, 6))
    batch = model.fk_batch(q)
    for i in range(len(q)):
        assert np.allclose(batch[i], model.fk_tcp(q[i]), atol=1e-12)
    pos = model.positions_batch(q)
    assert np.allclose(pos, batch[:, :3, 3], atol=1e-12)


def test_fk_matches_c_group_dh():
    """真实 MuJoCo FK（法兰）必须和 C 组 DH FK 对上：位置 < 2 mm、姿态 < 0.05°。

    注意 `check_c_fk` 里还有两行 `c_fk_tcp/flange_vs_tcp_*`，它们是"把法兰当
    TCP 用会差多少"的**说明性**行（故意 passed=False，约 100.7 mm / 90°），不参与
    通过判定——这正是 C 组交付里最容易被误用的地方。
    """
    model = _model()
    rows = validation.check_c_fk(model)
    picked = [r for r in rows if r["check"] == "c_fk"]
    assert picked, "check_c_fk 没有产出 c_fk 行"
    for r in picked:
        assert r["passed"] is True, f"{r['metric']}={r['value']} 超阈值 {r.get('threshold')}"
    trap = [r for r in rows if r["check"] == "c_fk_tcp"]
    assert trap and all(r["value"] > 1.0 for r in trap)


def test_flange_tcp_is_rigid_transform():
    """法兰→TCP 是固定刚体变换：用它换出来的位姿必须和直接 FK 一致（残差 ~1e-9）。"""
    model = _model()
    rows = validation.check_flange_tcp(model)
    assert any(r["passed"] is True for r in rows), rows


def test_numeric_jacobian_matches_mujoco():
    """数值（差分）Jacobian 与 MuJoCo 的 mj_jacSite 一致：逐元素差 < 1e-8。"""
    model = _model()
    rows = validation.check_numeric_jacobian(model)
    picked = [r for r in rows if r["check"] == "numeric_jacobian"]
    assert picked
    for r in picked:
        assert r["passed"] is True, f"{r['metric']}={r['value']} 超阈值 {r['tolerance']}"


def test_sigma_min_at_zero_pose_is_singular():
    """零位 q5=0 是腕部奇异：最小奇异值应接近 0、条件数极大。"""
    model = _model()
    J = model.jacobian_tcp(np.zeros(6))
    m = singularity_metrics(J)
    assert m.sigma_min < 1e-10
    assert m.condition_number > 1e10
    # home 位形则远离奇异
    m_home = singularity_metrics(model.jacobian_tcp(model.home_q()))
    assert m_home.sigma_min > 0.1


def test_jacobian_batch_matches_single():
    model = _model()
    rng = np.random.default_rng(7)
    q = rng.uniform(-1.0, 1.0, size=(5, 6))
    Jb = model.jacobian_batch(q)
    assert Jb.shape == (5, 6, 6)
    for i in range(len(q)):
        assert np.allclose(Jb[i], model.jacobian_tcp(q[i]), atol=1e-12)


# --------------------------------------------------------------------------- 日志
def test_exec_log_shape_and_metadata():
    """F 的日志：24921 个采样、3 个循环、13 种段名、39 段、t 单调。"""
    log = _log()
    assert log.n == 24921
    assert abs(log.dt_s - 0.002) < 1e-12
    assert log.cycles == [1, 2, 3]
    assert len(log.segment_names) == 13
    assert len(log.segments()) == 39
    assert np.all(np.diff(log.t) > 0)
    assert set(["home", "approach", "descend", "lift", "transfer",
                "lower", "retreat", "return_home"]).issubset(set(log.segment_names))


def test_exec_log_pose_lag_is_one_sample():
    """F 的 `ee_*` 列比 `q_actual` 滞后 1 个采样（先 mj_step 后读 site 的坑）。

    这是本阶段最重要的"数据陷阱"：若直接拿 `ee_*` 与 `fk(q_cmd)` 比，末端
    误差会被高估到 ~2.6 mm；按 lag=-1 对齐后残差只有 ~0.11 mm。
    """
    model = _model()
    lag = tracking.detect_pose_lag(model, _log())
    assert lag["lag_samples"] == -1
    assert lag["best_max_m"] < 2.0e-4
    assert lag["naive_max_m"] > 5.0 * lag["best_max_m"]


def test_segment_metrics_are_consistent():
    """单段指标自洽：跟踪误差、路径长度、Jacobian 一阶分解都应有合理量级。"""
    model = _model()
    log = _log()
    run = log.cycle_segments(1)[6]  # cycle1 的第 7 段 = transfer（笛卡尔直线 3 s）
    assert run.segment == "transfer"
    m = tracking.analyze_segment(model, log, run)
    assert 0.0 < m.joint_err_max_rad < 0.01          # < 0.57°
    assert 0.0 < m.ee_pos_err_max_m < 0.01           # < 10 mm
    assert m.ee_straightness_actual > 0.99           # transfer 是笛卡尔直线
    assert m.joint_path_actual_rad > 0.0
    assert m.ee_path_actual_m > 0.0
    assert m.jac_r2 > 0.99                           # 一阶差分能解释实际末端误差
    assert 0.0 <= m.sigma_min_min <= m.sigma_min_mean
    # 实际走的路不一定比规划短，但同量级
    assert abs(m.ee_path_actual_m - m.ee_path_planned_m) < 0.02
    row = m.to_row()
    assert "ee_pos_err_max_m" in row and "cycle" in row


def test_f_log_pose_residual_check():
    """F 日志自洽性检查：滞后必须是 -1 采样、对齐残差 < 0.2 mm。

    `naive_max_mm` 那一行是**对照**（不对齐直接比，~1.26 mm），按设计
    passed=False——它的存在本身就是要证明"不对齐会得出错误结论"。
    """
    model = _model()
    rows = validation.check_f_log(model, _log())
    assert rows
    lag = [r for r in rows if r["check"] == "f_log_pose_lag"]
    resid = [r for r in rows if r["check"] == "f_log_residual" and r["metric"] == "residual_max_mm"]
    naive = [r for r in rows if r["check"] == "f_log_residual" and r["metric"] == "naive_max_mm"]
    assert lag and lag[0]["value"] == -1 and lag[0]["passed"] is True
    assert resid and resid[0]["passed"] is True, resid
    assert naive and naive[0]["passed"] is False and naive[0]["value"] > 1.0


# --------------------------------------------------------------------------- 可达性
def test_reachability_separates_geometry_from_solver():
    """可达性判据：基座正下方远点=几何不可达；工作区内的点不判几何不可达。"""
    # 纯几何判据（不给 IK，避免依赖 D 的求解器）：
    # 20 万次采样的经验伸展上界 ~0.958 m（到肩部枢轴），下界 ~0。
    r_min, r_max = 0.0, 0.9584
    far_below = classify_reachability(None, [0.0, 0.0, -1.30],
                                      max_reach_m=r_max, min_reach_m=r_min,
                                      center_m=np.array([0.0, -0.138, 0.163]))
    assert far_below["status"] == "unreachable_geometry"
    # 工作空间内部点：不能判成几何不可达（否则就是把算法问题说成数学问题）
    inside = classify_reachability(None, [0.35, 0.40, 0.055],
                                   max_reach_m=r_max, min_reach_m=r_min,
                                   center_m=np.array([0.0, -0.138, 0.163]))
    assert inside["status"] == "unknown"  # 没给 IK：未知，而不是"不可达"


def test_workspace_radius_bound_from_real_model():
    """采样得到的伸展上界必须落在物理上界内（UR5e 臂长 0.425+0.3922+腕偏置）。"""
    model = _model()
    rng = np.random.default_rng(20261016)
    lo, hi = model.joint_limits()
    q = rng.uniform(lo, hi, size=(2000, 6))
    pos = model.positions_batch(q)
    pivots = model.reach_center_batch(q)
    r_arm = np.linalg.norm(pos - pivots, axis=1)
    assert r_arm.max() < 1.00       # a2 + a3 + sqrt(wrist offset^2) ~ 0.98 m
    assert r_arm.max() > 0.90


# --------------------------------------------------------------------------- 绘图
def test_ascii_safe_covers_problem_glyphs():
    """绘图文本 ASCII 化：组合上点/上标/Unicode 减号都必须能被替换掉。"""
    src = "q\u0307 与 q\u0308，rad/s\u00b2，m\u00b3，\u2212 0.5"
    out = plots.ascii_safe(src)
    for bad in ("\u0307", "\u0308", "\u00b2", "\u00b3", "\u2212"):
        assert bad not in out
    assert "q_dot" in out and "q_ddot" in out and "rad/s^2" in out


# --------------------------------------------------------------------------- 手跑入口
def _run_all() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"[ OK ] {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
