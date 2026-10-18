"""
F 模块｜MuJoCo 控制与完整搬运演示
====================================

任务书要求：
  把 E 的目标关节序列交给 MuJoCo 执行；设计抓取、移动、放置状态流程；
  先用稳定、可解释的抓取约束跑通基础演示。
  交付：src/control/controller.py、可重复执行的搬运脚本、
        results/videos/demo.mp4、实际执行轨迹日志。
  验收：至少连续完成 3 次同一设置下的搬运流程；记录失败情况，不剪辑掩盖问题。

────────────────────────────────────────────────────────────
运行方式
────────────────────────────────────────────────────────────
    cd F_mujoco_pick_place_demo
    python src/control/controller.py                  # 3 次循环 + 录制 demo.mp4
    python src/control/controller.py --cycles 5       # 改循环次数
    python src/control/controller.py --mode pingpong  # 往返搬运，中途不复位
    python src/control/controller.py --no-video       # 只跑物理与日志，不出视频
    python src/control/controller.py --viewer         # 打开 MuJoCo 交互窗口实时看
    python src/control/controller.py --traj-file results/logs/e_traj.npz
                                                      # 接入 E 模块真实轨迹

依赖：pip install mujoco numpy scipy matplotlib imageio imageio-ffmpeg

────────────────────────────────────────────────────────────
关键实现说明（答辩要点）
────────────────────────────────────────────────────────────
1) 抓取约束的正确激活方式 —— 这是本模块踩到的最大的坑
   B 模块在 pick_place.xml 中定义了 weld 焊接约束 grip_constraint，
   绑定 workpiece ↔ wrist_3_link，默认 active="false"。
   若直接把 eq_active 置 1，MuJoCo 会按【编译 XML 时记录的相对位姿】
   （m.eq_data 中的 relpose）去强制对齐两个刚体。而编译时机械臂处于
   qpos=0 的展开姿态，与抓取姿态相差极大，实测工件瞬间被拽走
   1157 mm —— 演示直接失败。

   解决方案：激活前把 m.eq_data 的相对位姿【改写为当前真实值】
   （见 _sync_weld_relpose），这样焊接等于"就地锁定"，不发生任何位移。
   实测抓取瞬间工件位移 0.023 mm。

   物理可解释性：本场景没有夹爪指部几何，attachment_site 相当于
   吸盘/真空吸嘴的工具中心点。TCP 下降到工件顶面中心后锁定，
   等价于吸盘吸附，符合任务书"先用稳定、可解释的抓取约束"的要求。

2) 执行方式
   ur5e.xml 使用 <general> 位置型执行器（gaintype=fixed,
   biastype=affine, gainprm=2000, biasprm="0 -2000 -400"），
   即 PD 位置伺服：ctrl 直接填目标关节角（rad），无需自己写 PID。
   腕部三个关节 forcerange=±28 N·m，肩部 ±150 N·m；工件仅 0.1 kg
   （0.981 N），抬升力矩裕度充足。

3) 轨迹来源
   按接口约定，F 读取 E 模块的 time[N] / q_traj[N,6]。
   E 尚未交付时，由 src/planning/trajectory.py 内置生成器按同样的
   两种方案（关节空间五次多项式 / 笛卡尔直线 + 逐点 IK）产生轨迹，
   E 交付后用 --traj-file 直接接管，controller 无需改动。

4) 坐标系
   工具坐标系 TCP = wrist_3_link 上的 attachment_site（B 模块第 1 周冻结，
   见 docs/坐标系说明.md）。日志中的 ee_pose_actual 即该 site 的位姿。
   注意：C 模块 fk.py 返回的是法兰 wrist_3_link 位姿，与本文件的 TCP
   相差固定变换（位置约 99.5 mm、姿态 90°），不可混用。

单位：关节角 rad，位置 m，时间 s。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

# ------------------------------------------------------------------
# 路径：让本文件既能以 python src/control/controller.py 运行，
#       也能被 main.py 作为模块导入
# ------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_THIS, ".."))
_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
import mujoco
from src.kinematics.ik import ik_dls_limited
from src.planning import io as traj_mod

# ------------------------------------------------------------------
# 常量
# ------------------------------------------------------------------
SCENE_XML = os.path.join(_ROOT, "scenes", "pick_place.xml")
RESULTS = os.path.join(_ROOT, "results")
LOG_DIR = os.path.join(RESULTS, "logs")
VIDEO_DIR = os.path.join(RESULTS, "videos")
FIG_DIR = os.path.join(RESULTS, "figures")

SITE_NAME = "attachment_site"      # TCP，工具坐标系
WORKPIECE = "workpiece"            # 被搬运工件
WRIST3 = "wrist_3_link"            # 焊接约束的机械臂侧刚体
EQ_GRIP = "grip_constraint"        # weld 约束名
KEY_HOME = "home"                  # B 场景中的归位 keyframe

# 工作空间关键点（基坐标系，单位 m）—— 依据 B 场景几何
#   上料盘中心 (-0.35, 0.4)，下料盘中心 (0.35, 0.4)，盘面 z=0.01
#   工件为边长 0.05 的立方体，中心 z=0.03 → 顶面 z=0.055
FEED_XY = np.array([-0.35, 0.4])
PLACE_XY = np.array([0.35, 0.4])
BLOCK_TOP_Z = 0.055                # 工件顶面：吸盘吸附高度
# 工件静止在托盘上时的中心高度：盘面 z=0.01（tray pos 0.005 + half 0.005）
#                                + 工件半边长 0.025 = 0.035
# 注意：B 场景 keyframe 中工件初始 z=0.03，比静止高度低 5 mm（轻微嵌入盘面），
#       仿真启动后会上浮到约 0.0348。这是初始条件的设定，不影响搬运功能，
#       但判定放置精度时必须以静止高度 0.035 为基准，否则会误报 5 mm 偏差。
BLOCK_REST_Z = 0.035
HOVER_Z = 0.30                     # 安全过渡高度
LIFT_Z = 0.38                      # 搬运抬升高度
CLEAR_Z = 0.12                     # 释放后抬离高度

# TCP 朝正下方的姿态：Z=[0,0,-1], X=[1,0,0], Y=[0,-1,0]
# 实测 home keyframe 下 attachment_site 的世界姿态即为此，det=+1 合法
R_DOWN = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])

# 各段时长（s）
T_APPROACH = 2.0
T_DESCEND = 1.6
T_GRASP_HOLD = 0.6
T_LIFT = 1.6
T_TRANSFER = 3.0
T_LOWER = 1.6
T_RELEASE_HOLD = 0.6
T_RETREAT = 1.2
T_RETURN = 2.0

# 判定阈值
PLACE_TOL_M = 0.015        # 放置精度：工件中心落在下料盘 ±15 mm 内视为成功
# 关节跟踪误差阈值。ur5e.xml 的位置执行器为 PD 伺服
#   gainprm="2000"(kp) / biasprm="0 -2000 -400"(kv)
# 跟踪速度为 v 的斜坡指令时，稳态滞后 e = (kv/kp)·v = 0.2·v。
# 本演示峰值关节速度约 0.35 rad/s → 理论滞后约 0.07 rad，
# 叠加重力负载与腕部 forcerange=±28 N·m 的限幅后取 0.12 rad（≈6.9°）为工程阈值。
# 该滞后是位置伺服的固有特性，不是控制逻辑缺陷；实测值见
# results/figures/tracking_error_grip.png 与日志 track_err_max 列。
TRACK_TOL_RAD = 0.12


# ==================================================================
# 工具函数
# ==================================================================
def _T_from_pose(pos, rot=R_DOWN):
    """由位置和姿态构造 4x4 齐次变换。"""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rot, dtype=float)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def rotmat2quat(R):
    """3x3 旋转矩阵 → MuJoCo 四元数 (w, x, y, z)。"""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s,
                          (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array([(R[0, 2] - R[2, 0]) / s,
                          (R[0, 1] + R[1, 0]) / s,
                          0.25 * s,
                          (R[1, 2] + R[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array([(R[1, 0] - R[0, 1]) / s,
                          (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


# ==================================================================
# 搬运控制器
# ==================================================================
class PickPlaceController:
    """UR5e 抓取-搬运-放置状态机。

    状态流程（每段均为五次多项式或笛卡尔直线插补，位置/速度/加速度连续）：
        HOME → APPROACH  移到源上方 HOVER_Z
             → DESCEND   下降到工件顶面 BLOCK_TOP_Z
             → GRASP     同步 weld 相对位姿后激活，锁定工件
             → LIFT      抬升到 LIFT_Z
             → TRANSFER  笛卡尔直线平移到目标上方（逐点 IK）
             → LOWER     下降到放置高度
             → RELEASE   解除 weld，工件自由落下到位
             → RETREAT   抬离
             → HOME      归位
    """

    def __init__(self, dt=None, verbose=True):
        if not os.path.exists(SCENE_XML):
            raise FileNotFoundError(f"未找到场景文件：{SCENE_XML}")

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.verbose = verbose

        # 仿真步长固定为模型 timestep，一个轨迹点对应一次 mj_step，
        # 保证 time[N] 与仿真步严格对齐，便于 G 模块做误差分析
        self.dt = float(dt) if dt is not None else float(self.model.opt.timestep)
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        self.model.opt.timestep = self.dt

        self.site_id = self._body_id(SITE_NAME, mujoco.mjtObj.mjOBJ_SITE)
        self.wp_id = self._body_id(WORKPIECE, mujoco.mjtObj.mjOBJ_BODY)
        self.w3_id = self._body_id(WRIST3, mujoco.mjtObj.mjOBJ_BODY)
        self.eq_id = self._body_id(EQ_GRIP, mujoco.mjtObj.mjOBJ_EQUALITY)

        if self.model.nu != 6:
            raise RuntimeError(f"期望 6 个执行器，实际 {self.model.nu}")

        # 执行器指令范围，用于裁剪前馈修正后的控制量
        cr = np.asarray(self.model.actuator_ctrlrange, dtype=float)
        self.ctrl_lo = cr[:, 0].copy()
        self.ctrl_hi = cr[:, 1].copy()

        # PD 增益：<general gaintype="fixed" biastype="affine"
        #            gainprm="kp" biasprm="0 -kp -kv">
        #   执行器输出力矩 F = kp*(ctrl - q) - kv*qdot
        # gainprm[:,0] = kp，biasprm[:,2] = -kv
        self.kp = np.asarray(self.model.actuator_gainprm, dtype=float)[:, 0].copy()
        self.kv = -np.asarray(self.model.actuator_biasprm, dtype=float)[:, 2].copy()
        self.ff_enabled = True          # 前馈补偿开关

        self.home_q = self._keyframe_q(KEY_HOME)
        self.log_rows = []          # 执行轨迹日志
        self.events = []            # 状态切换与异常事件
        self._t_offset = 0.0        # 当前循环的累计时间
        self._cycle_idx = 0         # 当前循环编号
        self._video_writer = None   # 视频写入器（未录制时为 None）
        self._video_renderer = None
        self._video_path = None
        self._video_kind = "none"   # mujoco_opengl / skeleton_fallback / none
        self._video_frame_i = 0     # 实际写入视频的帧数
        self._video_step_i = 0      # 录制期间累计的仿真步数
        self._video_decimation = 1
        self._e_traj = None         # E 模块交付的轨迹（未提供时为 None）

    # ---------------- 基础查询 ----------------
    def _body_id(self, name, obj_type):
        i = mujoco.mj_name2id(self.model, obj_type, name)
        if i < 0:
            raise RuntimeError(f"场景中未找到 {name}")
        return i

    def _keyframe_q(self, name):
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            raise RuntimeError(f"场景中未找到 keyframe {name}")
        return np.array(self.model.key_qpos[kid][:6], dtype=float).copy()

    @property
    def q_actual(self):
        """实际关节角（长度 6，rad）。"""
        return np.array(self.data.qpos[:6], dtype=float)

    @property
    def ee_pose_actual(self):
        """实际末端 TCP 位姿，4x4 齐次矩阵（attachment_site）。"""
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site_id]
        return T

    @property
    def ee_pos(self):
        return np.array(self.data.site_xpos[self.site_id], dtype=float)

    @property
    def wp_pos(self):
        """工件中心位置（世界系，m）。"""
        return np.array(self.data.xpos[self.wp_id], dtype=float)

    @property
    def grip_closed(self):
        return bool(self.data.eq_active[self.eq_id])

    # ---------------- 抓取约束 ----------------
    def _sync_weld_relpose(self):
        """把 weld 的相对位姿改写为当前真实值，使激活时工件零位移。

        weld 的 eq_data 布局（MuJoCo 3.x，eq_type=1）：
            [0:3]  anchor（锚点，保持 0）
            [3:6]  相对位置：body2(wrist_3) 在 body1(workpiece) 局部系下的坐标
            [6:10] 相对姿态四元数 (w,x,y,z)
            [10]   torque scale

        不改写而直接激活 → 按 XML 编译期相对位姿对齐 → 实测工件瞬移 1157 mm。
        """
        mujoco.mj_forward(self.model, self.data)
        R_wp = self.data.xmat[self.wp_id].reshape(3, 3)
        R_w3 = self.data.xmat[self.w3_id].reshape(3, 3)
        p_rel = R_wp.T @ (self.data.xpos[self.w3_id] - self.data.xpos[self.wp_id])
        q_rel = rotmat2quat(R_wp.T @ R_w3)

        self.model.eq_data[self.eq_id][0:3] = 0.0
        self.model.eq_data[self.eq_id][3:6] = p_rel
        self.model.eq_data[self.eq_id][6:10] = q_rel
        self.model.eq_data[self.eq_id][10] = 1.0
        return p_rel, q_rel

    def grasp(self):
        """闭合抓取：同步相对位姿 → 激活 weld → 短程稳态。"""
        wp_before = self.wp_pos.copy()
        p_rel, q_rel = self._sync_weld_relpose()
        self.data.eq_active[self.eq_id] = 1
        self._event("GRASP", f"激活 weld；relpose p={np.round(p_rel,4).tolist()} "
                            f"q={np.round(q_rel,4).tolist()}")
        self._hold(T_GRASP_HOLD, f"grasp_hold")
        shift = float(np.linalg.norm(self.wp_pos - wp_before) * 1000.0)
        self._event("GRASP", f"工件抓取瞬间位移 {shift:.4f} mm（阈值 2 mm）")
        if shift > 2.0:
            self._event("WARN", f"抓取位移过大 {shift:.2f} mm，relpose 同步可能失效")
        return shift

    def release(self):
        """松开抓取：解除 weld，工件自由落下到位。"""
        self.data.eq_active[self.eq_id] = 0
        self._event("RELEASE", "解除 weld 约束")
        self._hold(T_RELEASE_HOLD, "release_hold")

    # ---------------- 运动原语 ----------------
    def _event(self, tag, msg):
        cyc = getattr(self, "_cycle_idx", 0)
        self.events.append({"t": self._t_offset + self.data.time,
                            "cycle": cyc, "tag": tag, "msg": msg})
        if self.verbose:
            print(f"      [{tag}] {msg}")

    def _log_row(self, seg, q_cmd):
        T = self.ee_pose_actual
        q = self.q_actual
        wp = self.wp_pos
        quat = rotmat2quat(T[:3, :3])
        err = q - q_cmd
        self.log_rows.append({
            "t": round(self._t_offset + self.data.time, 6),
            "cycle": self._cycle_idx,
            "segment": seg,
            "grip": int(self.grip_closed),
            "q_cmd": [round(float(v), 6) for v in q_cmd],
            "q_actual": [round(float(v), 6) for v in q],
            "track_err_max": round(float(np.max(np.abs(err))), 6),
            "ee_pos": [round(float(v), 6) for v in T[:3, 3]],
            "ee_quat": [round(float(v), 6) for v in quat],
            "wp_pos": [round(float(v), 6) for v in wp],
        })

    def _step(self, q_cmd, seg, n_sub=1, dq_cmd=None, ddq_cmd=None):
        """推进 n_sub 个仿真步，把目标关节角送给位置执行器。

        控制律（计算力矩前馈 + PD 反馈）
        ------------------------------
        ur5e.xml 的执行器为
            <general gaintype="fixed" biastype="affine"
                     gainprm="kp" biasprm="0 -kp -kv">
        其输出力矩 F = kp*(ctrl - q) - kv*qdot。
        仅令 ctrl = q_cmd 时，纯反馈控制存在两项稳态误差：

          (1) 重力下垂：静态时 F 需平衡 qfrc_bias，故 kp*(ctrl-q) = qfrc_bias，
              产生恒定的位置偏差；
          (2) 速度滞后：跟踪速度为 v 的斜坡指令时，kv*qdot 项需要
              e = (kv/kp)*v 的位置误差来补偿，实测 kv/kp = 0.2。

        前馈把这两项直接抵消：
              ctrl = q_cmd + [M*ddq_cmd + qfrc_bias + kv*dq_cmd] / kp
        反馈项 kp*(ctrl-q) 仍然保留，因此闭环稳定性不受影响；
        qfrc_bias 与质量矩阵 M 取上一步仿真结束时的值（dt=2 ms，动态变化
        远慢于步长，滞后一拍不影响补偿效果）。
        最终 ctrl 裁剪到执行器 ctrlrange，超出 forcerange 的力矩由 MuJoCo 限幅。
        """
        for _ in range(n_sub):
            if self.ff_enabled:
                self.data.ctrl[:] = self._feedforward_ctrl(
                    q_cmd, dq_cmd, ddq_cmd)
            else:
                self.data.ctrl[:] = q_cmd
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._log_row(seg, q_cmd)
        self._maybe_grab_frame(seg)
        handle = getattr(self, "_viewer_handle", None)
        if handle is not None and handle.is_running():
            handle.sync()

    def _feedforward_ctrl(self, q_cmd, dq_cmd=None, ddq_cmd=None):
        """计算带前馈的执行器指令，返回裁剪到 ctrlrange 的长度 6 数组。"""
        d = self.data
        # 重力 + 科氏力补偿（上一步仿真已填充，无需额外 mj_forward）
        f_ff = np.array(d.qfrc_bias[:6], dtype=float)

        # 速度前馈：抵消 PD 中的 kv*qdot 阻尼项造成的斜坡滞后
        if dq_cmd is not None:
            f_ff += self.kv * np.asarray(dq_cmd, dtype=float)

        # 加速度前馈：M(q) * ddq_cmd
        # 用 mj_mulM 直接做「质量矩阵 × 向量」，比 mj_fullM 展开 nv×nv 全矩阵
        # 更省；且 M 取当前真实状态，焊接激活后工件质量已并入机械臂动力学，
        # 补偿量会自动包含负载惯性，无需手工加重物模型。
        if ddq_cmd is not None and np.any(ddq_cmd):
            vec = np.zeros(self.model.nv)
            vec[:6] = np.asarray(ddq_cmd, dtype=float)
            res = np.zeros(self.model.nv)
            mujoco.mj_mulM(self.model, d, res, vec)
            f_ff += res[:6]

        ctrl = np.asarray(q_cmd, dtype=float) + f_ff / self.kp
        return np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

    def _hold(self, duration, seg):
        """保持当前关节角不动 duration 秒（用于抓取/释放稳态）。

        静止段不加速度/速度前馈，仅做重力补偿 —— 此时 dq_cmd = ddq_cmd = 0，
        前馈退化为纯重力项，正好消除静态下垂。
        """
        q = self.q_actual.copy()
        zero = np.zeros(6)
        n = max(1, int(round(duration / self.dt)))
        for _ in range(n):
            self._step(q, seg, dq_cmd=zero, ddq_cmd=zero)

    def _settle(self, duration, seg, q_target=None):
        """伺服到目标关节角并稳定，用于消除残余跟踪误差。"""
        q = self.q_actual.copy() if q_target is None else np.asarray(q_target, float)
        zero = np.zeros(6)
        n = max(1, int(round(duration / self.dt)))
        for _ in range(n):
            self._step(q, seg, dq_cmd=zero, ddq_cmd=zero)
        return self.q_actual.copy()

    def _run_traj(self, traj, seg):
        """执行一段轨迹：逐点把 q_traj 喂给位置执行器。

        若轨迹字典带有 dq_traj / ddq_traj（quintic_joint_traj 与
        cartesian_line_traj 均提供），则同时用于前馈补偿。

        时间尺度对齐（重要）
        --------------------
        本函数的执行方式是「相邻轨迹点之间的时间间隔 = 一次 mj_step(dt=self.dt)」，
        因此一段轨迹的实际仿真时长 = (N−1) × self.dt。

        内置生成器始终用 self.dt 采样，两者天然一致。
        但【外部轨迹】（E 模块交付物）的采样周期可能是任意值，
        例如 E 用 dt=0.01 采样、N=301 点、声明 duration=3.00 s；
        若直接逐点执行，实际时长只有 301 × 0.002 = 0.602 s，
        等于把轨迹按 5 倍速播放 —— 速度、加速度全部放大 5 倍，
        前馈补偿也随之失真（实测该场景确实发生）。

        处理方式：检测到轨迹 dt 与仿真步长不一致时，
        在【真实时间轴】上把 q_traj 重采样到 self.dt 网格，
        使实际执行时长与轨迹声明的 duration 一致。
        dq/ddq 同样重采样（它们是轨迹自身的解析/数值导数，
        单位是 rad/s、rad/s²，不随重采样缩放）。
        """
        q_traj = np.asarray(traj["q_traj"], dtype=float)
        dq = traj.get("dq_traj")
        ddq = traj.get("ddq_traj")

        from src.contracts import JointTrajectory
        tt = np.asarray(traj["time"], dtype=float)
        JointTrajectory(tt, q_traj)
        duration = float(tt[-1])
        n_steps = int(round(duration / self.dt))
        if not np.isclose(n_steps * self.dt, duration, atol=1e-9):
            raise ValueError("Trajectory duration must align with simulation timestep")
        t_out = np.arange(n_steps + 1) * self.dt
        if len(tt) == len(t_out) and np.allclose(tt, t_out, atol=1e-10):
            q_out, dq_out, ddq_out = q_traj, dq, ddq
        else:
            from scipy.interpolate import CubicSpline
            spline = CubicSpline(tt, q_traj, axis=0)
            q_out = spline(t_out)
            dq_out = self._resample_deriv(dq, tt, t_out) if dq is not None else spline(t_out, 1)
            ddq_out = self._resample_deriv(ddq, tt, t_out) if ddq is not None else spline(t_out, 2)
        # t=0 is the current state. Advance N-1 intervals, never N steps.
        for i in range(1, len(q_out)):
            self._step(q_out[i], seg,
                       dq_cmd=(dq_out[i] if dq_out is not None
                               and i < len(dq_out) else None),
                       ddq_cmd=(ddq_out[i] if ddq_out is not None
                                and i < len(ddq_out) else None))
        return self.q_actual.copy()

    @staticmethod
    def _resample_deriv(arr, t_in, t_out):
        """把速度/加速度数组重采样到新的时间网格。

        这些量单位是 rad/s、rad/s²，是轨迹本身的属性，
        重采样只改变采样点位置，不做任何缩放。
        """
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or len(arr) != len(t_in):
            return None
        from scipy.interpolate import interp1d
        out = interp1d(t_in, arr, axis=0, bounds_error=False,
                       fill_value=(arr[0], arr[-1]))(t_out)
        return np.asarray(out, dtype=float)

    # ---------------- IK ----------------
    def _ik(self, T_target, q_seed, tag):
        q, ok, info = ik_dls_limited(T_target, q_seed)
        if not ok:
            self._event("IK_FAIL",
                        f"{tag}: 逆解未收敛 pos_err={(info['pos_err'] * 1000):.4f} mm "
                        f"rot_err={np.degrees(info['rot_err']):.4f}° iters={info['iterations']}")
        elif self.verbose and info.get("iterations", 0) > 30:
            self._event("IK_SLOW", f"{tag}: 迭代 {info['iterations']} 次")
        return q, ok, info

    # ---------------- 单段生成 ----------------
    def _seg_joint(self, q0, q1, dur, seg, method="quintic"):
        """关节空间五次多项式段。"""
        tr = traj_mod.quintic_joint_traj(q0, q1, dur, self.dt)
        self._event("SEG", f"{seg} [{method}] {dur:.1f}s N={len(tr['time'])} "
                           f"q0={np.round(q0,4).tolist()} → q1={np.round(q1,4).tolist()}")
        return self._run_traj(tr, seg), tr

    def _seg_cartesian(self, p0, p1, dur, seg, q_seed, rot=R_DOWN):
        """笛卡尔直线段：末端沿直线运动，逐点 IK。"""
        T0 = _T_from_pose(p0, rot)
        T1 = _T_from_pose(p1, rot)
        tr = traj_mod.cartesian_line_traj(
            T0, T1, dur, self.dt, ik_dls_limited, q_seed, fail_fast=False)
        if tr["ik_failures"]:
            for f in tr["ik_failures"][:5]:
                self._event("IK_UNSOLVABLE",
                            f"{seg} 第{f['index']}点 t={f['t']:.3f}s "
                            f"tcp={np.round(f['tcp_target'],4).tolist()} "
                            f"pos_err={f['pos_err']:.3e}")
            self._event("WARN", f"{seg} 共 {len(tr['ik_failures'])} 个不可解点，"
                                f"已完整记录不掩盖")
        q_end = self._run_traj(tr, seg)
        self._event("SEG", f"{seg} [cartesian_line] {dur:.1f}s N={len(tr['time'])} "
                           f"p0={np.round(p0,4).tolist()} → p1={np.round(p1,4).tolist()} "
                           f"success={tr['success']}")
        return q_end, tr

    # ==================================================================
    # 一次完整搬运
    # ==================================================================
    def run_e_cycle(self, cycle_idx, trajectory):
        """Execute E's seven phases once, inserting grasp/release holds."""
        phases = traj_mod.split_phases(trajectory)
        if np.max(np.abs(phases[0]["q_traj"][0] - self.home_q)) > 0.01:
            raise ValueError("External trajectory must start at home")
        self._settle(0.5, "home", self.home_q)
        shift = tcp_err = 0.0
        for i, phase in enumerate(phases):
            self._run_traj(phase, phase["name"])
            if i == 1:
                self._settle(0.3, "descend_settle", phase["q_traj"][-1])
                tcp_err = float(np.linalg.norm(self.ee_pos - _T_from_pose(
                    [*FEED_XY, BLOCK_TOP_Z])[:3, 3]) * 1000)
                if tcp_err > 3.0:
                    return self._cycle_result(cycle_idx, False, [], abort="grasp_pose_error")
                shift = self.grasp()
                if shift > 2.0:
                    return self._cycle_result(cycle_idx, False, [], abort="grasp_shift")
            elif i == 4:
                self._settle(0.3, "lower_settle", phase["q_traj"][-1])
                self.release()
            elif i == 5:
                self._settle(0.3, "settle_after_release", phase["q_traj"][-1])
        wp = self.wp_pos.copy()
        dxy = float(np.linalg.norm(wp[:2] - PLACE_XY))
        dz = float(abs(wp[2] - BLOCK_REST_Z))
        placed = dxy <= PLACE_TOL_M and dz <= 0.02 and not self.grip_closed
        self._event("RESULT", f"E full trajectory: success={placed}, place error={dxy*1000:.3f} mm")
        return self._cycle_result(cycle_idx, placed, [], place_dxy_mm=dxy*1000,
                                  place_dz_mm=dz*1000, wp_final=wp,
                                  grasp_shift_mm=shift, tcp_err_mm=tcp_err)

    def run_cycle(self, cycle_idx, src_xy, dst_xy, use_cartesian=True):
        """执行一次 抓取→搬运→放置 完整流程。

        参数：
            cycle_idx : 循环编号（从 1 开始）
            src_xy    : 源料盘中心 (x, y)
            dst_xy    : 目标料盘中心 (x, y)
            use_cartesian : 平移段是否用笛卡尔直线（True）或关节空间（False）
        返回：
            dict，含成功标志、放置误差、跟踪误差、各段轨迹等
        """
        self._cycle_idx = cycle_idx
        segs = []
        e_traj = self._e_traj
        if self.verbose:
            print(f"\n{'='*66}\n  第 {cycle_idx} 次搬运：{np.round(src_xy,3).tolist()} "
                  f"→ {np.round(dst_xy,3).tolist()}\n{'='*66}")

        # ---- 0. 归位 ----
        q = self._settle(T_APPROACH * 0.5, "home", q_target=self.home_q)
        self._event("STATE", f"归位完成 q={np.round(q,4).tolist()}")

        # ---- 1. APPROACH：移到源上方 ----
        T_hover = _T_from_pose([src_xy[0], src_xy[1], HOVER_Z])
        q_hover, ok, info = self._ik(T_hover, q, "hover_src")
        if not ok:
            self._event("ABORT", "预抓取点逆解失败，本次循环中止")
            return self._cycle_result(cycle_idx, False, segs,
                                      abort="approach_ik_fail")
        q, _ = self._seg_joint(q, q_hover, T_APPROACH, "approach")
        segs.append(("approach", q_hover, ok, info))

        # ---- 2. DESCEND：下降到工件顶面 ----
        T_grasp = _T_from_pose([src_xy[0], src_xy[1], BLOCK_TOP_Z])
        q_grasp, ok, info = self._ik(T_grasp, q, "grasp_point")
        if not ok:
            self._event("ABORT", f"抓取点逆解失败 pos_err={(info['pos_err'] * 1000):.4f} mm")
            return self._cycle_result(cycle_idx, False, segs,
                                      abort="descend_ik_fail")
        q, _ = self._seg_joint(q, q_grasp, T_DESCEND, "descend")
        segs.append(("descend", q_grasp, ok, info))

        # 到位确认：伺服稳态，保证抓取瞬间无残余速度
        q = self._settle(0.3, "descend_settle", q_target=q_grasp)
        tcp_err = float(np.linalg.norm(self.ee_pos - T_grasp[:3, 3]) * 1000)
        self._event("STATE", f"抓取点到位 TCP 误差 {tcp_err:.4f} mm，"
                             f"工件在 {np.round(self.wp_pos,4).tolist()}")

        # ---- 3. GRASP ----
        shift = self.grasp()
        if shift > 2.0:
            return self._cycle_result(cycle_idx, False, segs, abort="grasp_shift")

        # ---- 4. LIFT ----
        T_lift = _T_from_pose([src_xy[0], src_xy[1], LIFT_Z])
        q_lift, ok, info = self._ik(T_lift, q, "lift")
        if not ok:
            self._event("ABORT", "抬升点逆解失败")
            return self._cycle_result(cycle_idx, False, segs, abort="lift_ik_fail")
        q, _ = self._seg_joint(q, q_lift, T_LIFT, "lift")
        segs.append(("lift", q_lift, ok, info))

        # ---- 5. TRANSFER ----
        if e_traj is not None:
            # 接口约定：F 直接读取并执行 E 模块输出的 q_traj[N,6]
            self._event("SEG", f"transfer [E模块轨迹] N={len(e_traj['time'])} "
                               f"时长={e_traj['duration']:.2f}s "
                               f"来源={e_traj.get('source', 'in-memory')}")
            q = self._run_traj(e_traj, "transfer_e")
        elif use_cartesian:
            p_from = [src_xy[0], src_xy[1], LIFT_Z]
            p_to = [dst_xy[0], dst_xy[1], LIFT_Z]
            q, tr_cart = self._seg_cartesian(p_from, p_to, T_TRANSFER,
                                             "transfer", q)
            if not tr_cart["success"]:
                self._event("WARN", "平移段存在不可解点，轨迹可能不完整")
        else:
            T_over = _T_from_pose([dst_xy[0], dst_xy[1], LIFT_Z])
            q_over, ok, info = self._ik(T_over, q, "over_dst")
            if not ok:
                self._event("ABORT", "目标上方逆解失败")
                return self._cycle_result(cycle_idx, False, segs,
                                          abort="transfer_ik_fail")
            q, _ = self._seg_joint(q, q_over, T_TRANSFER, "transfer",
                                   method="quintic_joint")
            segs.append(("transfer", q_over, ok, info))

        # ---- 6. LOWER ----
        T_place = _T_from_pose([dst_xy[0], dst_xy[1], BLOCK_TOP_Z + 0.01])
        q_place, ok, info = self._ik(T_place, q, "place_point")
        if not ok:
            self._event("ABORT", "放置点逆解失败")
            return self._cycle_result(cycle_idx, False, segs, abort="place_ik_fail")
        q, _ = self._seg_joint(q, q_place, T_LOWER, "lower")
        segs.append(("lower", q_place, ok, info))
        q = self._settle(0.3, "lower_settle", q_target=q_place)

        # ---- 7. RELEASE ----
        self.release()

        # ---- 8. RETREAT ----
        T_clear = _T_from_pose([dst_xy[0], dst_xy[1], CLEAR_Z])
        q_clear, ok, info = self._ik(T_clear, q, "retreat")
        if not ok:
            self._event("ABORT", "退让点逆解失败")
            return self._cycle_result(cycle_idx, False, segs, abort="retreat_ik_fail")
        q, _ = self._seg_joint(q, q_clear, T_RETREAT, "retreat")
        segs.append(("retreat", q_clear, ok, info))

        # ---- 9. 判定 ----
        # 释放后给工件一点时间稳定落位（weld 已解除，工件自由下落到位）
        self._hold(0.8, "settle_after_release")
        wp = self.wp_pos.copy()
        dxy = float(np.linalg.norm(wp[:2] - np.asarray(dst_xy, dtype=float)))
        dz = float(abs(wp[2] - BLOCK_REST_Z))
        placed = dxy <= PLACE_TOL_M and dz <= 0.02 and not self.grip_closed

        self._event("RESULT",
                    f"{'✓ 放置成功' if placed else '✗ 放置失败'} "
                    f"工件={np.round(wp,4).tolist()} "
                    f"目标={list(np.round(dst_xy,3))+[BLOCK_REST_Z]} "
                    f"水平偏差={dxy*1000:.3f} mm 高度偏差={dz*1000:.3f} mm")

        # ---- 10. 归位，为下一次循环做准备 ----
        q, _ = self._seg_joint(self.q_actual, self.home_q, T_RETURN, "return_home")

        return self._cycle_result(cycle_idx, placed, segs,
                                  place_dxy_mm=dxy * 1000, place_dz_mm=dz * 1000,
                                  wp_final=wp, grasp_shift_mm=shift,
                                  tcp_err_mm=tcp_err)

    def _cycle_result(self, idx, placed, segs, **kw):
        rows = [r for r in self.log_rows if r["cycle"] == idx]
        errs = np.array([r["track_err_max"] for r in rows]) if rows else np.zeros(1)
        res = {
            "cycle": idx,
            "success": bool(placed),
            "steps": len(rows),
            "duration_s": float(len(rows) * self.dt),
            "track_err_max_rad": float(np.max(errs)) if len(errs) else 0.0,
            "track_err_mean_rad": float(np.mean(errs)) if len(errs) else 0.0,
            "ik_segments": len(segs),
            "ik_all_converged": all(s[2] for s in segs) if segs else True,
        }
        res.update(kw)
        return res

    # ==================================================================
    # 复位
    # ==================================================================
    def reset_to_home(self, wp_xy=FEED_XY):
        """用 keyframe 复位机械臂，并把工件放回指定料盘中心。

        对应验收要求"同一设置"：每次循环起始状态完全一致，
        这是标准的可重复性实验协议（初始条件受控）。
        """
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, KEY_HOME)
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        self.data.ctrl[:] = self.home_q
        self.data.eq_active[self.eq_id] = 0
        # 工件自由关节 qpos[6:13] = [x,y,z, qw,qx,qy,qz]
        self.data.qpos[6:9] = [float(wp_xy[0]), float(wp_xy[1]), BLOCK_REST_Z]
        self.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ==================================================================
    # 主流程
    # ==================================================================
    def run(self, cycles=3, mode="reset", record_video=True,
            video_path=None, viewer=False, traj_file=None,
            width=960, height=720, fps=50):
        """执行完整演示。

        参数：
            cycles : 循环次数（验收要求 ≥3）
            mode   : 'reset'    每次循环复位到同一初始状态（严格可重复性）
                     'pingpong' 往返搬运，中途不复位，上料盘↔下料盘交替
            record_video : 是否录制 mp4
            traj_file    : E 模块交付的轨迹文件；提供时按显式阶段边界执行完整搬运
        返回：
            dict(summary=..., cycles=[...], events=[...])
        """
        if cycles < 1:
            raise ValueError("cycles must be positive")
        os.makedirs(LOG_DIR, exist_ok=True)
        if record_video:
            os.makedirs(VIDEO_DIR, exist_ok=True)

        renderer = None
        writer = None
        if record_video:
            renderer, writer = self._open_video(video_path, width, height, fps)
            if renderer is None:
                record_video = False

        viewer_handle = None
        if viewer:
            import mujoco.viewer
            viewer_handle = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer_handle = viewer_handle

        # E 模块轨迹接管（可选）
        e_traj = None
        if traj_file:
            e_traj = traj_mod.load_e_trajectory(traj_file)
            if e_traj is None:
                raise ValueError(f"Cannot load trajectory: {traj_file}")
            if e_traj.get("scope") != "full_pick_place":
                raise ValueError("Full trajectory with explicit phase metadata required; regenerate with python -m src.main plan")
            if mode != "reset":
                raise ValueError("External full trajectory requires --mode reset")
            if np.any(e_traj["q_traj"] < self.model.jnt_range[:6, 0] - 1e-8) or np.any(
                e_traj["q_traj"] > self.model.jnt_range[:6, 1] + 1e-8):
                raise ValueError("External trajectory violates joint limits")
            self._e_traj = e_traj
            if e_traj is not None:
                self._event("E_TRAJ", f"已加载 E 模块轨迹：{traj_file} "
                                      f"N={len(e_traj['time'])} "
                                      f"时长={e_traj['duration']:.2f}s")
            else:
                self._event("E_TRAJ", f"未能加载 {traj_file}，回退到内置轨迹生成器")

        decimation = max(1, int(round(1.0 / (fps * self.dt))))
        results = []
        t_wall = time.time()

        for c in range(1, cycles + 1):
            self._cycle_idx = c
            if mode == "reset":
                src, dst = FEED_XY.copy(), PLACE_XY.copy()
                self.reset_to_home(FEED_XY)
            else:                                  # pingpong：交替方向
                if c % 2 == 1:
                    src, dst = FEED_XY.copy(), PLACE_XY.copy()
                else:
                    src, dst = PLACE_XY.copy(), FEED_XY.copy()
                if c == 1:
                    self.reset_to_home(FEED_XY)
                else:
                    self.data.eq_active[self.eq_id] = 0
            self._t_offset = (sum(r["duration_s"] for r in results)
                              if mode == "reset" else 0.0)

            # 若加载了 E 模块轨迹，用它作为本次 transfer 段的目标序列
            res = (self.run_e_cycle(c, e_traj) if e_traj is not None
                   else self.run_cycle(c, src, dst, use_cartesian=True))
            if e_traj is not None:
                res["used_e_traj"] = True
            results.append(res)

            if viewer_handle is not None and viewer_handle.is_running():
                viewer_handle.sync()

        if record_video and writer is not None:
            writer.close()
            if renderer is not None:
                renderer.close()
            self._event("VIDEO", f"演示视频已保存：{self._video_path}"
                                 f"（后端 {self._video_kind}，"
                                 f"{self._video_frame_i} 帧）")

        if viewer_handle is not None:
            viewer_handle.close()

        elapsed = time.time() - t_wall
        summary = {
            "cycles_requested": cycles,
            "cycles_completed": len(results),
            "cycles_succeeded": sum(1 for r in results if r["success"]),
            "mode": mode,
            "dt": self.dt,
            "control_freq_hz": round(1.0 / self.dt, 2),
            "total_steps": sum(r["steps"] for r in results),
            "total_sim_time_s": round(sum(r["duration_s"] for r in results), 3),
            "wall_time_s": round(elapsed, 2),
            "track_err_max_rad": round(max((r["track_err_max_rad"] for r in results),
                                           default=0.0), 6),
            "track_err_max_deg": round(np.degrees(
                max((r["track_err_max_rad"] for r in results), default=0.0)), 4),
            "all_consecutive_success": all(r["success"] for r in results)
                                       and len(results) >= 3,
            "video": getattr(self, "_video_path", None) if record_video else None,
            "render_backend": getattr(self, "_video_kind", "none")
                              if record_video else "disabled",
            "frames": getattr(self, "_video_frame_i", 0),
        }
        return {"summary": summary, "cycles": results, "events": self.events,
                "log_rows": self.log_rows}

    # ---------------- 视频 ----------------
    def _open_video(self, path, width, height, fps):
        """初始化渲染器与视频写入器。

        优先使用 mujoco.Renderer（OpenGL 真实渲染，带网格与光照）；
        若当前环境无可用显卡或无桌面会话导致 gladLoadGL 失败，
        自动降级到 fallback_render.SkeletonRenderer（纯 CPU 三维骨架示意），
        保证 results/videos/demo.mp4 这一交付项在任何环境下都能产出。
        """
        path = path or os.path.join(VIDEO_DIR, "demo.mp4")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        renderer = None
        self._video_kind = "none"
        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._video_kind = "mujoco_opengl"
            print(f"  渲染后端   : mujoco.Renderer（OpenGL 真实渲染）")
        except Exception as e:                                 # noqa: BLE001
            print(f"\n[提示] OpenGL 离屏渲染不可用（{type(e).__name__}: "
                  f"{str(e)[:90]}）。")
            print(f"       常见于无显卡驱动或无桌面会话的环境。")
            print(f"       自动降级到纯 CPU 骨架渲染，视频照常产出；"
                  f"物理数据仍取自真实仿真状态。")
            print(f"       在有显卡的机器上重跑本脚本即可得到真实渲染画面。\n")
            try:
                from src.control.fallback_render import SkeletonRenderer
            except ImportError:
                from fallback_render import SkeletonRenderer
            try:
                renderer = SkeletonRenderer(self.model, height=height,
                                            width=width)
                self._video_kind = "skeleton_fallback"
            except Exception as e2:                            # noqa: BLE001
                print(f"[提示] 后备渲染也不可用（{type(e2).__name__}: {e2}）。\n"
                      f"       本次跳过视频录制，物理仿真、日志与图表照常产出。")
                return None, None

        try:
            import imageio.v2 as imageio
            writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                        quality=8, macro_block_size=1)
        except Exception as e:                                 # noqa: BLE001
            print(f"[提示] 视频编码器不可用（{type(e).__name__}: {e}）。"
                  f"请执行 pip install imageio-ffmpeg")
            renderer.close()
            return None, None
        self._video_renderer = renderer
        self._video_writer = writer
        self._video_path = path
        self._video_decimation = max(1, int(round(1.0 / (fps * self.dt))))
        self._video_frame_i = 0
        self._video_cam = mujoco.MjvCamera()
        self._video_cam.azimuth = 135
        self._video_cam.elevation = -22
        self._video_cam.distance = 2.4
        self._video_cam.lookat[:] = [0.0, 0.35, 0.25]
        return renderer, writer

    def _maybe_grab_frame(self, seg=None):
        w = getattr(self, "_video_writer", None)
        if w is None:
            return
        self._video_step_i += 1
        if self._video_step_i % self._video_decimation != 0:
            return
        r = self._video_renderer
        if self._video_kind == "skeleton_fallback":
            # 骨架渲染器需要显式设置画面标注，且相机参数用 set_caption/elev/azim
            if seg:
                r.set_caption(seg)
            cam = self._video_cam
            r.cam = cam
            r.update_scene(self.data)
        else:
            r.update_scene(self.data, camera=self._video_cam)
        w.append_data(r.render())
        self._video_frame_i += 1


# ==================================================================
# 结果输出
# ==================================================================
def save_log_csv(log_rows, path):
    """把执行轨迹日志写成 CSV（接口约定：q_actual 与 ee_pose_actual）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "cycle", "segment", "grip"]
                   + [f"q_cmd_{i}" for i in range(6)]
                   + [f"q_actual_{i}" for i in range(6)]
                   + ["track_err_max"]
                   + ["ee_x", "ee_y", "ee_z",
                      "ee_qw", "ee_qx", "ee_qy", "ee_qz"]
                   + ["wp_x", "wp_y", "wp_z"])
        for r in log_rows:
            w.writerow([r["t"], r["cycle"], r["segment"], r["grip"]]
                       + r["q_cmd"] + r["q_actual"] + [r["track_err_max"]]
                       + r["ee_pos"] + r["ee_quat"] + r["wp_pos"])
    return path


def save_events_csv(events, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "cycle", "tag", "message"])
        for e in events:
            w.writerow([round(e["t"], 4), e["cycle"], e["tag"], e["msg"]])
    return path


def save_summary_txt(summary, cycles, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("=" * 70)
    lines.append("F 模块｜MuJoCo 控制与完整搬运演示 —— 运行摘要")
    lines.append("=" * 70)
    for k, v in summary.items():
        lines.append(f"  {k:26s} : {v}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("逐次循环明细")
    lines.append("-" * 70)
    for r in cycles:
        lines.append(f"\n  循环 {r['cycle']}: {'✓ 成功' if r['success'] else '✗ 失败'}")
        for k in ("steps", "duration_s", "track_err_max_rad", "track_err_mean_rad",
                  "ik_segments", "ik_all_converged", "grasp_shift_mm",
                  "tcp_err_mm", "place_dxy_mm", "place_dz_mm"):
            if k in r:
                lines.append(f"      {k:24s} = {r[k]}")
        if "abort" in r:
            lines.append(f"      !! 中止原因: {r['abort']}")
        if "wp_final" in r:
            lines.append(f"      工件最终位置        = "
                         f"{np.round(r['wp_final'], 5).tolist()}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("验收对照（任务书 F 模块）")
    lines.append("=" * 70)
    n_ok = summary["cycles_succeeded"]
    n_req = summary["cycles_requested"]
    lines.append(f"  [{'✓' if n_ok >= 3 else '✗'}] 至少连续完成 3 次同一设置下的搬运流程："
                 f" 请求 {n_req} 次，成功 {n_ok} 次")
    lines.append(f"  [{'✓' if summary.get('video') else '✗'}] results/videos/demo.mp4："
                 f" {summary.get('video') or '未生成（渲染环境不可用）'}")
    lines.append(f"  [{'✓' if n_ok == n_req else '✗'}] 记录失败情况不剪辑掩盖：全部 "
                 f"{n_req} 次结果均写入日志，含失败与告警事件")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return path


def plot_figures(log_rows, cycles, fig_dir):
    """生成 F 模块图表：关节跟踪、TCP 轨迹、工件位置、跟踪误差。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                       # noqa: BLE001
        print(f"[提示] matplotlib 不可用，跳过绘图：{e}")
        return []

    os.makedirs(fig_dir, exist_ok=True)
    made = []
    t = np.array([r["t"] for r in log_rows])
    q_cmd = np.array([r["q_cmd"] for r in log_rows])
    q_act = np.array([r["q_actual"] for r in log_rows])
    ee = np.array([r["ee_pos"] for r in log_rows])
    wp = np.array([r["wp_pos"] for r in log_rows])
    grip = np.array([r["grip"] for r in log_rows])
    cyc = np.array([r["cycle"] for r in log_rows])

    # 图 1：关节指令 vs 实际
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    names = ["J1 shoulder_pan", "J2 shoulder_lift", "J3 elbow",
             "J4 wrist_1", "J5 wrist_2", "J6 wrist_3"]
    for i, ax in enumerate(axes.flat):
        ax.plot(t, q_cmd[:, i], lw=1.0, color="#1f77b4", label="q_cmd (E/traj)")
        ax.plot(t, q_act[:, i], lw=1.0, ls="--", color="#d62728",
                label="q_actual (sim)")
        ax.set_ylabel("rad")
        ax.set_title(names[i], fontsize=10)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("t (s)")
    axes[-1, 1].set_xlabel("t (s)")
    fig.suptitle("Joint Command vs Actual — F module execution tracking",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(fig_dir, "joint_tracking.png")
    fig.savefig(p, dpi=160); plt.close(fig); made.append(p)

    # 图 2：跟踪误差 + 抓取状态
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    err = np.max(np.abs(q_act - q_cmd), axis=1)
    for c in np.unique(cyc):
        msk = cyc == c
        a1.plot(t[msk], np.degrees(err[msk]), lw=0.9, label=f"cycle {int(c)}")
    a1.axhline(np.degrees(TRACK_TOL_RAD), color="k", ls=":", lw=1,
               label=f"tolerance {np.degrees(TRACK_TOL_RAD):.1f}°")
    a1.set_ylabel("max joint tracking error (deg)")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a1.set_title("Tracking error per cycle", fontsize=11)
    a2.plot(t, grip, lw=1.0, color="#2ca02c", drawstyle="steps-post")
    a2.set_ylabel("grip (1=holding)")
    a2.set_xlabel("t (s)")
    a2.set_ylim(-0.15, 1.15)
    a2.set_yticks([0, 1])
    a2.grid(alpha=0.3)
    a2.set_title("Gripper weld constraint state", fontsize=11)
    fig.tight_layout()
    p = os.path.join(fig_dir, "tracking_error_grip.png")
    fig.savefig(p, dpi=160); plt.close(fig); made.append(p)

    # 图 3：TCP 与工件轨迹（XY + Z）
    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(ee[:, 0], ee[:, 1], lw=0.9, color="#1f77b4", label="TCP path")
    ax1.plot(wp[:, 0], wp[:, 1], lw=1.2, color="#d62728", label="workpiece path")
    for xy, nm in [(FEED_XY, "feed tray"), (PLACE_XY, "place tray")]:
        ax1.plot(xy[0], xy[1], "ks", ms=9, mfc="none")
        ax1.annotate(nm, xy, textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_title("XY trajectory"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax1.set_aspect("equal", adjustable="datalim")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, ee[:, 2], lw=0.9, color="#1f77b4", label="TCP z")
    ax2.plot(t, wp[:, 2], lw=1.2, color="#d62728", label="workpiece z")
    ax2.axhline(BLOCK_TOP_Z, color="gray", ls=":", lw=1, label="block top z")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("z (m)")
    ax2.set_title("Height profile"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(fig_dir, "tcp_and_workpiece_path.png")
    fig.savefig(p, dpi=160); plt.close(fig); made.append(p)

    return made


# ==================================================================
# 入口
# ==================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="F 模块｜UR5e MuJoCo 抓取-搬运-放置完整演示")
    ap.add_argument("--cycles", type=int, default=3,
                    help="搬运循环次数，验收要求 ≥3（默认 3）")
    ap.add_argument("--mode", choices=["reset", "pingpong"], default="reset",
                    help="reset=每次复位到同一初始状态；pingpong=往返搬运不复位")
    ap.add_argument("--no-video", action="store_true", help="不录制 mp4")
    ap.add_argument("--video-path", default=None, help="视频输出路径")
    ap.add_argument("--viewer", action="store_true",
                    help="打开 MuJoCo 交互窗口实时观察")
    ap.add_argument("--traj-file", default=None,
                    help="E 模块轨迹文件（.npz/.npy/.csv），提供后按阶段执行完整搬运")
    ap.add_argument("--width", type=int, default=640, help="视频宽度")
    ap.add_argument("--height", type=int, default=480, help="视频高度")
    ap.add_argument("--fps", type=int, default=30, help="视频帧率")
    ap.add_argument("--no-figures", action="store_true", help="不生成图表")
    ap.add_argument("--no-ff", action="store_true",
                    help="关闭前馈补偿，仅用纯位置反馈（用于对比跟踪误差，"
                         "可写进报告说明控制策略的作用）")
    args = ap.parse_args(argv)

    print("=" * 70)
    print("F 模块｜MuJoCo 控制与完整搬运演示")
    print("=" * 70)
    print(f"  场景文件   : {SCENE_XML}")
    print(f"  循环次数   : {args.cycles}")
    print(f"  运行模式   : {args.mode}")
    print(f"  视频录制   : {'否' if args.no_video else '是'}")

    ctrl = PickPlaceController()
    if args.no_ff:
        ctrl.ff_enabled = False
    print(f"  仿真步长   : {ctrl.dt} s  ({1.0/ctrl.dt:.0f} Hz)")
    print(f"  执行器     : {ctrl.model.nu} 个位置型 (general/position)")
    print(f"  前馈补偿   : {'开启（计算力矩 + 速度前馈）' if ctrl.ff_enabled else '关闭（纯位置反馈）'}")
    print(f"  归位关节角 : {np.round(ctrl.home_q, 4).tolist()}")
    print(f"  抓取约束   : {EQ_GRIP} (weld, {WORKPIECE} ↔ {WRIST3})")

    out = ctrl.run(cycles=args.cycles, mode=args.mode,
                   record_video=not args.no_video,
                   video_path=args.video_path, viewer=args.viewer,
                   traj_file=args.traj_file,
                   width=args.width, height=args.height, fps=args.fps)

    summary, cycles, events, rows = (out["summary"], out["cycles"],
                                     out["events"], out["log_rows"])

    # ---- 保存日志与图表 ----
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_csv = save_log_csv(rows, os.path.join(LOG_DIR, f"exec_log_{stamp}.csv"))
    log_latest = save_log_csv(rows, os.path.join(LOG_DIR, "exec_log_latest.csv"))
    ev_csv = save_events_csv(events, os.path.join(LOG_DIR, f"events_{stamp}.csv"))
    summary_txt = save_summary_txt(summary, cycles,
                                   os.path.join(LOG_DIR, f"run_summary_{stamp}.txt"))

    import json
    with open(os.path.join(LOG_DIR, "run_summary_latest.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "cycles": cycles}, fh, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    figs = [] if args.no_figures else plot_figures(rows, cycles, FIG_DIR)

    # ---- 控制台报告 ----
    print("\n" + "=" * 70)
    print("运行摘要")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:26s} : {v}")
    print("\n逐次结果：")
    for r in cycles:
        flag = "✓ 成功" if r["success"] else "✗ 失败"
        extra = ""
        if "place_dxy_mm" in r:
            extra = (f"  放置偏差 {r['place_dxy_mm']:.2f} mm"
                     f"  抓取位移 {r.get('grasp_shift_mm', float('nan')):.4f} mm"
                     f"  跟踪误差峰值 {np.degrees(r['track_err_max_rad']):.3f}°")
        if "abort" in r:
            extra += f"  中止于 {r['abort']}"
        print(f"  循环 {r['cycle']}: {flag}{extra}")

    print("\n输出文件：")
    print(f"  执行轨迹日志 : {log_csv}")
    print(f"                {log_latest}")
    print(f"  事件与告警   : {ev_csv}")
    print(f"  运行摘要     : {summary_txt}")
    for p in figs:
        print(f"  图表         : {p}")
    if summary.get("video"):
        print(f"  演示视频     : {summary['video']}")

    ok = summary["cycles_succeeded"] == args.cycles
    print("\n" + "=" * 70)
    print(f"验收结论：{'✓ 通过' if ok else '✗ 未通过'} —— "
          f"连续完成 {summary['cycles_succeeded']}/{summary['cycles_requested']} 次搬运")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
