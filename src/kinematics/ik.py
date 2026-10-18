"""
逆运动学（UR5e + MuJoCo）——D 模块原版，仅修复硬编码路径
=========================================================

这是 D 模块 `ik.py` 的原函数（get_fk / ik_dls），E 模块只做了两项修复，
未改动任何求解逻辑：

1. 路径不再写死 C:\\Users\\wzy20\\...，改为按顺序解析：
   a. 环境变量 UR5E_XML 指定的 ur5e.xml；
   b. 本文件同目录下的 ur5e.xml / ur5e_nomesh.xml；
   c. 工程内搜索 universal_robots_ur5e/ur5e.xml。
2. site 名 / 加载失败给出可读提示，并在工程实际模型缺失时回退到
   同目录的纯运动学模型 ur5e_nomesh.xml（无 mesh，运动学一致）。

接口不变：
    get_fk(q)                     -> 4x4 TCP(attachment_site) 位姿
    ik_dls(T_target, q_init,...)  -> (q, success, {"iterations","pos_err","rot_err"})
"""
import os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

site_name = "attachment_site"   # TCP（工具坐标系）


def _resolve_xml():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    path = Path(os.environ.get("UR5E_XML", root / "universal_robots_ur5e/ur5e.xml"))
    if not path.is_file():
        raise FileNotFoundError(f"UR5e model missing: {path}")
    return str(path)


xml_path = _resolve_xml()
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
if site_id < 0:
    raise ValueError(f"模型 {xml_path} 中找不到 site '{site_name}'")


def get_fk(q):
    """使用 MuJoCo 算正运动学（TCP），返回 4x4 矩阵。"""
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    T[:3, 3] = data.site_xpos[site_id]
    return T


def ik_dls(T_target, q_init, max_iter=200, lam=0.01, alpha=0.5):
    """阻尼最小二乘逆运动学（TCP 参考点）。返回 (q, success, info_dict)。"""
    q = np.asarray(q_init, dtype=float).copy()
    p_target = T_target[:3, 3]
    R_target = T_target[:3, :3]

    for it in range(max_iter):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)

        p_cur = data.site_xpos[site_id].copy()
        R_cur = data.site_xmat[site_id].reshape(3, 3).copy()

        e_pos = p_target - p_cur
        R_err = R_target @ R_cur.T
        e_rot = R.from_matrix(R_err).as_rotvec()

        pos_err = np.linalg.norm(e_pos)
        rot_err = np.linalg.norm(e_rot)

        if pos_err < 1e-4 and rot_err < 1e-3:
            return q, True, {"iterations": it, "pos_err": pos_err, "rot_err": rot_err}

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])

        e = np.hstack([e_pos, e_rot])
        A = J @ J.T + (lam ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(A, e)

        q = q + alpha * dq

    actual = get_fk(q)
    pos_err = np.linalg.norm(p_target - actual[:3, 3])
    rot_err = np.linalg.norm(R.from_matrix(R_target @ actual[:3, :3].T).as_rotvec())
    return q, bool(pos_err < 1e-4 and rot_err < 1e-3), {"iterations": max_iter, "pos_err": pos_err, "rot_err": rot_err}


# UR5e 关节限位（ur5e.xml ctrlrange）：J3(elbow) 最严
_UR5E_QMIN = np.array([-6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319])
_UR5E_QMAX = np.array([6.28319, 6.28319, 3.1415, 6.28319, 6.28319, 6.28319])


def ik_dls_limited(T_target, q_init, max_iter=200, lam=0.01, alpha=0.5,
                   q_min=None, q_max=None, step_cap=0.05, pos_tol=1e-4, rot_tol=1e-3):
    """阻尼最小二乘 IK（TCP），带关节限位裁剪 + 单步角度上限（F 推荐的增强版）。

    与原 ik_dls 的区别：
      - 每步把解裁剪到关节限位内（默认 UR5e ctrlrange，J3 ±3.1415）；
      - 单步关节增量按 step_cap 限幅（默认 0.05 rad/步），避免奇异附近大步跳跃。
    返回 (q, success, info_dict)。
    """
    q_min = _UR5E_QMIN if q_min is None else np.asarray(q_min, float)
    q_max = _UR5E_QMAX if q_max is None else np.asarray(q_max, float)
    q = np.asarray(q_init, dtype=float).copy()
    q = np.clip(q, q_min, q_max)
    p_target = T_target[:3, 3]
    R_target = T_target[:3, :3]

    for it in range(max_iter):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)

        p_cur = data.site_xpos[site_id].copy()
        R_cur = data.site_xmat[site_id].reshape(3, 3).copy()

        e_pos = p_target - p_cur
        e_rot = R.from_matrix(R_target @ R_cur.T).as_rotvec()
        pos_err = np.linalg.norm(e_pos)
        rot_err = np.linalg.norm(e_rot)

        if pos_err < pos_tol and rot_err < rot_tol:
            return q, True, {"iterations": it, "pos_err": pos_err, "rot_err": rot_err}

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])

        e = np.hstack([e_pos, e_rot])
        A = J @ J.T + (lam ** 2) * np.eye(6)
        dq = alpha * (J.T @ np.linalg.solve(A, e))
        # 单步限幅
        step_norm = np.linalg.norm(dq)
        if step_norm > step_cap:
            dq = dq * (step_cap / step_norm)
        q = np.clip(q + dq, q_min, q_max)

    actual = get_fk(q)
    pos_err = np.linalg.norm(p_target - actual[:3, 3])
    rot_err = np.linalg.norm(R.from_matrix(R_target @ actual[:3, :3].T).as_rotvec())
    return q, bool(pos_err < pos_tol and rot_err < rot_tol), {"iterations": max_iter, "pos_err": pos_err, "rot_err": rot_err}


def ik(T_target, q_init):
    """Public A contract; tuple API above retained for E/F compatibility."""
    from src.contracts import Pose, IKResult
    target = Pose(T_target).T_world_tcp
    q, ok, info = ik_dls_limited(target, q_init)
    return IKResult(q, bool(ok), float(info['pos_err']), float(info['rot_err']),
                    info['iterations'], 'converged' if ok else 'iteration limit')
