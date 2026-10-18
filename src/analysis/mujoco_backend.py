"""MuJoCo 真实 UR5e 模型后端 —— G 第二阶段（真实模型 + 真实执行数据）。

为什么需要这个文件
==================
G 第一阶段（`docs/G_ANALYSIS_PLAN.md`）用的是 G 自己造的 UR5e 参考 DH 表
（`src/analysis/reference_model.py`，零位 TCP 半径 0.8521 m），它只能证明
"分析方法可用"。**B 组交付的 MuJoCo 模型才是全组统一的真值**：
`universal_robots_ur5e/ur5e.xml`（mujoco_menagerie 的 UR5e），
坐标与关节约定见 B 组 `docs/坐标系说明.md`。

本模块把该模型封装成 G 的分析后端，与 C/D/F 的约定严格对齐：

===========  ============================================================
FK           `fk_tcp(q)`    工具坐标系（`attachment_site` 站点）→ 与 D/F 一致
             `fk_flange(q)` 法兰 `wrist_3_link` → 与 C 的 `fk.py` 对齐
Jacobian     `jacobian_tcp(q)`  世界系 6x6，直接调用 `mj_jacSite`
             `jacobian_flange(q)` 法兰原点处世界系 6x6，调用 `mj_jac`
限位         `ctrl_limits()`  执行器 `ctrlrange`（真正能执行的范围）
             `joint_limits()` 关节 `jnt_range`（运动学范围）
批量 FK      `positions_batch()` / `fk_batch()`，用 `mj_kinematics` 跳开
             碰撞检测，比逐点 `mj_forward` 快一个数量级
===========  ============================================================

设计约束（与第一阶段一致）
==========================
* **零新增接口**：不改 `src/contracts.py`、`docs/INTERFACES.md`、`src/main.py`。
  G 只消费上游能力，本模块是 G 的内部实现。
* 没有安装 MuJoCo 时**导入本模块不报错**，只有实例化 `UR5eModel` 才失败，
  这样纯 NumPy 的自检（`tests/test_g_analysis.py`）在没有 MuJoCo 的机器上仍能跑。
* 模型路径可移植：按候选路径查找，支持 `G_UR5E_XML` 环境变量覆盖。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

# ---- 与 B 组模型一致的名字常量（写死是有意的：模型改名应当报错而不是静默跑错）----
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
SITE_NAME = "attachment_site"      # TCP：工具坐标系原点
FLANGE_BODY = "wrist_3_link"       # 法兰：C 组 fk.py 输出的坐标系
MODEL_FILENAME = "ur5e.xml"

# 候选相对路径（相对工程根），按优先级排列
_RELATIVE_CANDIDATES = (
    os.path.join("universal_robots_ur5e", MODEL_FILENAME),   # B 组交付布局
    "ur5e.xml",                                              # F 模块布局
    os.path.join("upstream_refs", "universal_robots_ur5e", MODEL_FILENAME),
    os.path.join("universal_robots_ur5e", "scene.xml"),      # 场景（含地板）兜底
)

_HERE = Path(__file__).resolve().parent


def project_root() -> Path:
    """定位工程根目录：从本文件向上找到含 `src/` 子目录的那一层。

    支持 `G_PROJECT_ROOT` 覆盖（打包成 wheel 或被复制到别处时用）。
    """
    override = os.environ.get("G_PROJECT_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    for cand in (_HERE, *_HERE.parents):
        if (cand / "src").is_dir() and cand.name != "src":
            return cand
    return _HERE


def find_model_path(explicit: str | Path | None = None) -> Path | None:
    """按优先级查找 UR5e 模型文件；找不到返回 None（不抛异常，便于测试跳过）。"""
    tried: list[str] = []
    if explicit is not None:
        p = Path(explicit).expanduser()
        if p.exists():
            return p.resolve()
        tried.append(str(p))

    env = os.environ.get("G_UR5E_XML", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()
        tried.append(env)

    root = project_root()
    for rel in _RELATIVE_CANDIDATES:
        p = root / rel
        if p.exists():
            return p.resolve()

    # 最后再试：本文件所在包内嵌一份（用户把包单独拷走时的兜底）
    for rel in _RELATIVE_CANDIDATES:
        p = _HERE / rel
        if p.exists():
            return p.resolve()
    return None


def _file_sha256(path: str | Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()[:16]


def mujoco_available() -> bool:
    """MuJoCo 是否可用（用于测试跳过与脚本前置检查）。"""
    try:
        import mujoco  # noqa: F401
    except Exception:  # pragma: no cover - 仅环境缺失时触发
        return False
    return True


class UR5eModel:
    """UR5e 真实模型后端：FK / Jacobian / 限位 / TCP 常量。

    用法::

        m = UR5eModel()
        T = m.fk_tcp([0, -1.2, 1.2, -1.57, -1.57, 0])
        J = m.jacobian_tcp(q)          # 世界系，0:3 线速度，3:6 角速度

    所有方法都接受长度 6 的关节角（rad），顺序 = `JOINT_NAMES`。
    """

    def __init__(self, xml_path: str | Path | None = None) -> None:
        try:
            import mujoco
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "需要 MuJoCo 才能使用真实模型后端：pip install 'mujoco>=3.1,<4'"
            ) from exc

        path = find_model_path(xml_path)
        if path is None:
            raise FileNotFoundError(
                "找不到 UR5e 模型文件。已尝试环境变量 G_UR5E_XML、"
                f"工程根 {project_root()} 下的 {list(_RELATIVE_CANDIDATES)}。"
                "请确认 B 组的 universal_robots_ur5e/ 目录（含 assets/）已就位。"
            )

        self._mj = mujoco
        self.model_path = str(path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
        self.flange_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, FLANGE_BODY)
        if self.site_id < 0 or self.flange_id < 0:
            raise RuntimeError(
                f"模型缺少 {SITE_NAME}(site) 或 {FLANGE_BODY}(body)，"
                "请核对是否为 B 组交付的 ur5e.xml"
            )

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES
        ]
        if any(j < 0 for j in self.joint_ids):
            raise RuntimeError(f"模型关节名与约定不符，需要 {JOINT_NAMES}")

        self._q = np.zeros(6, dtype=np.float64)
        self._T_flange_tcp = self._compute_flange_tcp()

    # ------------------------------------------------------------------ 内部
    def _compute_flange_tcp(self) -> np.ndarray:
        """法兰 → TCP 的固定变换（T_flange^-1 @ T_tcp），与 q 无关。"""
        T_f = self.fk_flange(np.zeros(6))
        T_t = self.fk_tcp(np.zeros(6))
        T = np.linalg.inv(T_f) @ T_t
        T[:3, :3] = _orthonormalize(T[:3, :3])
        return T

    def _set_q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(6)
        self.data.qpos[:6] = q
        return q

    def _read_site(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return p, R

    def _read_body(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.data.xpos[self.flange_id].copy()
        R = self.data.xmat[self.flange_id].reshape(3, 3).copy()
        return p, R

    # ------------------------------------------------------------------ FK
    def reach_center(self) -> np.ndarray:
        """伸展量的参考点：第 2 轴（shoulder_lift）的轴心，**在 q=0 时的世界坐标**。

        为什么用它而不用世界原点：UR5e 的臂长是从肩部枢轴算起的，肩部枢轴在
        基座上方约 0.163 m、并相对基座轴线偏 ~0.14 m，所以"到基座原点的半径"
        与"到肩部的伸展"不是同一个概念。判几何可达必须用肩部伸展，否则会把
        "基座正下方"这类点误判为可达。

        ⚠️ 该点相对基座轴线有偏置，因此**随第 1 轴（pan）旋转**：给定关节角
        q 时它位于 Rz(q[0]) @ reach_center()。调用方要么按 q[0] 旋转它，要么
        直接用下面的 `reach_center_batch()`。
        """
        self._set_q(np.zeros(6))
        self._mj.mj_forward(self.model, self.data)
        return self.data.xanchor[self.joint_ids[1]].copy()

    def reach_center_batch(self, q_batch) -> np.ndarray:
        """一组关节角对应的肩部枢轴世界坐标 (N,3)：Rz(q[0]) @ reach_center()。"""
        q = np.atleast_2d(np.asarray(q_batch, dtype=np.float64))
        c0 = self.reach_center()
        c, s = np.cos(q[:, 0]), np.sin(q[:, 0])
        out = np.empty((q.shape[0], 3))
        out[:, 0] = c * c0[0] - s * c0[1]
        out[:, 1] = s * c0[0] + c * c0[1]
        out[:, 2] = c0[2]
        return out

    def fk_tcp(self, q) -> np.ndarray:
        """TCP（`attachment_site`）位姿 4x4，世界系。与 D/F 的 `get_fk` 等价。"""
        self._set_q(q)
        self._mj.mj_kinematics(self.model, self.data)
        p, R = self._read_site()
        return _homogeneous(R, p)

    def fk_flange(self, q) -> np.ndarray:
        """法兰（`wrist_3_link`）位姿 4x4，世界系。与 C 的 `fk(q)` 对齐。"""
        self._set_q(q)
        self._mj.mj_kinematics(self.model, self.data)
        p, R = self._read_body()
        return _homogeneous(R, p)

    def fk_batch(self, q_batch) -> np.ndarray:
        """批量 TCP 位姿，(N,6) → (N,4,4)。"""
        Q = np.asarray(q_batch, dtype=np.float64).reshape(-1, 6)
        out = np.empty((len(Q), 4, 4), dtype=np.float64)
        for i, q in enumerate(Q):
            self._set_q(q)
            self._mj.mj_kinematics(self.model, self.data)
            p, R = self._read_site()
            out[i, :3, :3] = R
            out[i, :3, 3] = p
            out[i, 3, :3] = 0.0
            out[i, 3, 3] = 1.0
        return out

    def positions_batch(self, q_batch) -> np.ndarray:
        """批量 TCP 位置，(N,6) → (N,3)。工作空间采样用这个（比 fk_batch 省内存）。"""
        Q = np.asarray(q_batch, dtype=np.float64).reshape(-1, 6)
        out = np.empty((len(Q), 3), dtype=np.float64)
        for i, q in enumerate(Q):
            self._set_q(q)
            self._mj.mj_kinematics(self.model, self.data)
            out[i] = self.data.site_xpos[self.site_id]
        return out

    def poses_batch(self, q_batch) -> tuple[np.ndarray, np.ndarray]:
        """批量 TCP 位置与旋转矩阵，(N,3) 与 (N,3,3)。"""
        Q = np.asarray(q_batch, dtype=np.float64).reshape(-1, 6)
        pos = np.empty((len(Q), 3), dtype=np.float64)
        rot = np.empty((len(Q), 3, 3), dtype=np.float64)
        for i, q in enumerate(Q):
            self._set_q(q)
            self._mj.mj_kinematics(self.model, self.data)
            pos[i] = self.data.site_xpos[self.site_id]
            rot[i] = self.data.site_xmat[self.site_id].reshape(3, 3)
        return pos, rot

    # ------------------------------------------------------------------ Jacobian
    def jacobian_tcp(self, q) -> np.ndarray:
        """TCP 处的几何 Jacobian（世界系，6x6）。

        行 0:3 = 线速度（m/s per rad/s），行 3:6 = 角速度（rad/s per rad/s）。
        直接调用 `mj_jacSite`，因此这是**解析真值**，可作为 G 数值微分实现的
        交叉验证基准（见 `scripts/G5_model_validation.py`）。
        """
        self._set_q(q)
        self._mj.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self._mj.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, :6], jacr[:, :6]])

    def jacobian_flange(self, q) -> np.ndarray:
        """法兰原点处的几何 Jacobian（世界系，6x6）。"""
        self._set_q(q)
        self._mj.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        point = self.data.xpos[self.flange_id]
        self._mj.mj_jac(self.model, self.data, jacp, jacr, point, self.flange_id)
        return np.vstack([jacp[:, :6], jacr[:, :6]])

    def jacobian_batch(self, q_batch) -> np.ndarray:
        """批量 TCP Jacobian，返回 (N, 6, 6) 数组（逐点解析，不做数值微分）。"""
        qs = np.asarray(q_batch, dtype=np.float64).reshape(-1, 6)
        out = np.empty((len(qs), 6, 6), dtype=np.float64)
        for i, q in enumerate(qs):
            out[i] = self.jacobian_tcp(q)
        return out

    # ------------------------------------------------------------------ 限位
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """关节 `jnt_range` 下/上界（rad）。运动学上关节能达到的范围。"""
        rng = np.asarray(self.model.jnt_range[:6], dtype=np.float64)
        return rng[:, 0].copy(), rng[:, 1].copy()

    def ctrl_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """执行器 `ctrlrange` 下/上界（rad）。

        **这是"真正能执行"的范围**：位置执行器会把指令静默截断到 ctrlrange，
        所以 G 的可达性采样、IK 限位裁剪都必须用它（见 F 组接口说明第三节）。
        """
        rng = np.asarray(self.model.actuator_ctrlrange[:6], dtype=np.float64)
        return rng[:, 0].copy(), rng[:, 1].copy()

    # ------------------------------------------------------------------ 常量与元数据
    @property
    def T_flange_tcp(self) -> np.ndarray:
        """法兰 → TCP 固定变换 4x4（位置约 0.1 m，姿态 90°）。"""
        return self._T_flange_tcp.copy()

    @property
    def tcp_offset_m(self) -> float:
        """TCP 相对法兰原点的距离（m）。"""
        return float(np.linalg.norm(self._T_flange_tcp[:3, 3]))

    def home_q(self) -> np.ndarray:
        """模型 keyframe `home`（与 F 组归位姿态一致）。"""
        try:
            kid = self._mj.mj_name2id(self.model, self.mjtObj_key, "home")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            kid = -1
        if kid is None or kid < 0:
            return np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        return np.asarray(self.model.key_qpos[kid][:6], dtype=np.float64).copy()

    def describe(self) -> dict:
        """模型事实清单（写进 meta.json，报告里引用）。"""
        lo, hi = self.ctrl_limits()
        jlo, jhi = self.joint_limits()
        return {
            "model_path": self.model_path,
            "model_sha256_head16": _file_sha256(self.model_path),
            "model_name": getattr(self.model, "name", Path(self.model_path).parent.name),
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "timestep_s": float(self.model.opt.timestep),
            "joint_names": list(JOINT_NAMES),
            "site_name": SITE_NAME,
            "flange_body": FLANGE_BODY,
            "ctrl_limits_lo_rad": [float(v) for v in lo],
            "ctrl_limits_hi_rad": [float(v) for v in hi],
            "joint_limits_lo_rad": [float(v) for v in jlo],
            "joint_limits_hi_rad": [float(v) for v in jhi],
            "tcp_offset_m": self.tcp_offset_m,
            "T_flange_tcp": self._T_flange_tcp.tolist(),
            "home_q_rad": self.home_q().tolist(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"UR5eModel(path={self.model_path!r}, tcp_offset_m={self.tcp_offset_m:.4f})"


# ---------------------------------------------------------------------- 工具函数
def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """把数值上略有偏差的旋转矩阵投影回 SO(3)（SVD 正交化）。"""
    u, _, vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def _homogeneous(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def try_model(xml_path: str | Path | None = None) -> "UR5eModel | None":
    """尽力构造模型；不可用（缺 MuJoCo / 缺模型）时返回 None 供测试跳过。"""
    if not mujoco_available():
        return None
    try:
        return UR5eModel(xml_path)
    except Exception:
        return None
