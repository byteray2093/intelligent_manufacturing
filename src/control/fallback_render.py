"""无 GPU 环境下的后备渲染器（matplotlib 三维骨架示意）。

为什么需要它
------------
MuJoCo 官方渲染器 mujoco.Renderer 依赖 OpenGL 上下文，Windows 上走
wgl / glfw。当机器没有可用的显卡驱动，或进程运行在无桌面会话中
（CI、远程沙箱、服务模式）时，构造渲染器会抛：

    FatalError: gladLoadGL error

此时 results/videos/demo.mp4 就无法产出，而它是任务书 F 模块的明确交付项。
本模块提供一条纯 CPU 的后备路径，使视频在任何环境下都能生成。

画面性质（重要，答辩时如实说明）
------------------------------
这是【示意渲染】，不含网格贴图与光照，绘制内容为：
    工作台台面、上/下料盘、机械臂连杆骨架与关节、TCP 工具中心点、
    工件立方体、抓取状态连线。

但所有位姿数据均取自 MuJoCo 的真实仿真状态
（data.xpos / data.site_xpos / data.eq_active），逐帧读取，
不是另行编造的假动画 —— 物理过程与主仿真完全一致。

若运行环境有可用显卡，controller.py 会优先使用 mujoco.Renderer
输出带网格与光照的真实渲染画面，本模块不会被调用。
"""
from __future__ import annotations

import numpy as np

# 机械臂运动链（基座 → 法兰），用于绘制连杆骨架
ARM_CHAIN = ["base", "shoulder_link", "upper_arm_link", "forearm_link",
             "wrist_1_link", "wrist_2_link", "wrist_3_link"]

# 立方体 12 条棱的顶点索引（顶点按 dx,dy,dz ∈ (-1,1) 的三重循环顺序编号）
_CUBE_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
               (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def _cube_edges(cx, cy, cz, half):
    """返回立方体 12 条棱的折线坐标（用 NaN 分隔，单次 plot 即可绘制）。"""
    corners = np.array([[cx + dx * half, cy + dy * half, cz + dz * half]
                        for dx in (-1.0, 1.0)
                        for dy in (-1.0, 1.0)
                        for dz in (-1.0, 1.0)])
    xs, ys, zs = [], [], []
    for a, b in _CUBE_EDGES:
        xs += [corners[a, 0], corners[b, 0], np.nan]
        ys += [corners[a, 1], corners[b, 1], np.nan]
        zs += [corners[a, 2], corners[b, 2], np.nan]
    return xs, ys, zs


class SkeletonRenderer:
    """接口对齐 mujoco.Renderer：update_scene(data, camera) → render() → HxWx3 uint8。"""

    def __init__(self, model, height=480, width=640, dpi=100,
                 title_prefix="MuJoCo UR5e pick & place"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import mujoco

        self._plt = plt
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self.title_prefix = title_prefix

        self.fig = plt.figure(figsize=(self.width / dpi, self.height / dpi),
                              dpi=dpi, tight_layout=False)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # 预解析 id，避免每帧查询
        self._ids = {}
        for n in ARM_CHAIN:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if i < 0:
                raise RuntimeError(f"后备渲染器：场景中未找到 body {n}")
            self._ids[n] = i
        self._wp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
        self._site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                      "attachment_site")
        self._eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                    "grip_constraint")

        self.data = None
        self.cam = None
        self.caption = ""
        # 固定视野范围，避免镜头随内容跳动
        self.xlim = (-0.95, 0.95)
        self.ylim = (-0.85, 0.95)
        self.zlim = (-0.45, 1.05)

    # ---------------- 接口 ----------------
    def update_scene(self, data, camera=None):
        self.data = data
        if camera is not None:
            self.cam = camera

    def set_caption(self, text):
        self.caption = text or ""

    def render(self):
        """绘制当前帧，返回 HxWx3 的 uint8 RGB 数组。"""
        if self.data is None:
            raise RuntimeError("render() 前必须先调用 update_scene()")

        d = self.data
        ax = self.ax
        ax.cla()

        # ---- 工作台台面（z=0，x∈[-0.5,0.5], y∈[-0.3,0.6]）----
        X, Y = np.meshgrid(np.linspace(-0.5, 0.5, 2), np.linspace(-0.3, 0.6, 2))
        ax.plot_surface(X, Y, np.zeros_like(X), color="#bdc3c7", alpha=0.9,
                        linewidth=0, rstride=1, cstride=1, zorder=1)

        # ---- 上料盘 / 下料盘 ----
        for (cx, cy), col in (((-0.35, 0.4), "#2980b9"), ((0.35, 0.4), "#2980b9")):
            Xs, Ys = np.meshgrid(np.linspace(cx - 0.1, cx + 0.1, 2),
                                 np.linspace(cy - 0.1, cy + 0.1, 2))
            ax.plot_surface(Xs, Ys, np.full_like(Xs, 0.011), color=col,
                            alpha=0.8, linewidth=0, rstride=1, cstride=1,
                            zorder=2)

        # ---- 机械臂连杆骨架 ----
        pts = np.array([d.xpos[self._ids[n]] for n in ARM_CHAIN])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#2c3e50", lw=7,
                solid_capstyle="round", zorder=5)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="#3498db", s=42,
                   zorder=6, edgecolors="white", linewidths=0.8)

        # ---- TCP（attachment_site，工具坐标系原点）----
        tcp = d.site_xpos[self._site]
        ax.scatter(tcp[0], tcp[1], tcp[2], color="#e67e22", s=85, marker="^",
                   zorder=8, edgecolors="black", linewidths=0.6)

        # ---- 工件 ----
        wp = d.xpos[self._wp]
        xs, ys, zs = _cube_edges(wp[0], wp[1], wp[2], 0.025)
        ax.plot(xs, ys, zs, color="#c0392b", lw=2.2, zorder=7)

        # ---- 抓取状态：weld 激活时画 TCP→工件 连线 ----
        gripping = self._eq >= 0 and d.eq_active[self._eq] > 0.5
        if gripping:
            ax.plot([tcp[0], wp[0]], [tcp[1], wp[1]], [tcp[2], wp[2]],
                    color="#27ae60", lw=1.8, ls="--", zorder=4)

        # ---- 视野与标题 ----
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_zlim(*self.zlim)
        ax.set_box_aspect((self.xlim[1] - self.xlim[0],
                           self.ylim[1] - self.ylim[0],
                           self.zlim[1] - self.zlim[0]))
        if self.cam is not None:
            # MuJoCo elevation 向下为负，matplotlib elev 向上为正
            ax.view_init(elev=-float(self.cam.elevation),
                         azim=float(self.cam.azimuth))
        else:
            ax.view_init(elev=22, azim=135)

        ax.set_xlabel("x (m)", fontsize=7, labelpad=-2)
        ax.set_ylabel("y (m)", fontsize=7, labelpad=-2)
        ax.set_zlabel("z (m)", fontsize=7, labelpad=-2)
        ax.tick_params(labelsize=6, pad=1)
        grip_txt = "GRIP" if gripping else "OPEN"
        ax.set_title(f"{self.title_prefix}   t={d.time:.2f}s   [{grip_txt}]"
                     + (f"   {self.caption}" if self.caption else ""),
                     fontsize=9, pad=2)

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return np.ascontiguousarray(buf[:, :, :3])

    def close(self):
        try:
            self._plt.close(self.fig)
        except Exception:                                      # noqa: BLE001
            pass
