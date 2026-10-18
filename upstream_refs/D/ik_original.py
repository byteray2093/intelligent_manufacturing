import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

# ⚠️ 换成你自己的 ur5e.xml 路径
xml_path = r"C:\Users\wzy20\Desktop\mujoco_menagerie-main\mujoco_menagerie-main\universal_robots_ur5e\ur5e.xml"
site_name = "attachment_site" # 如果之前你改过，用你实际打印出来的名字

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

def get_fk(q):
    """使用 MuJoCo 算正运动学，返回 4x4 矩阵"""
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    T[:3, 3] = data.site_xpos[site_id]
    return T

def ik_dls(T_target, q_init, max_iter=200, lam=0.01, alpha=0.5):
    """阻尼最小二乘逆运动学"""
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
        
    return q, False, {"iterations": max_iter, "pos_err": pos_err, "rot_err": rot_err}

print("ik.py 编写完成！")