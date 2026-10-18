"""绘图工具：所有 G 组图表统一风格、统一输出路径、统一中文字体处理。

设计要点：
- matplotlib 采用**延迟导入**：核心分析（metrics / singularity / workspace）只依赖 NumPy，
  没有 matplotlib 的机器上也能出 CSV 与结论，不会因为画图失败而让整套分析跑不起来。
- 强制无界面后端（Agg），避免在没有显示的环境崩溃。
- 中文字体：优先尝试常见 Windows/Linux 中文字体；找不到时回退英文标签并在
  图注中标注，绝不出现方框乱码。也可用环境变量 `G_FONT=English` 强制英文标签。
- 每张图都会把"采样条件 / 单位 / 模型来源"写进图注——对应验收要求"注明采样条件与单位"。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_PLT = None
_CJK_OK: bool | None = None
_CJK_ERROR: str | None = None

_CJK_CANDIDATES = [
    # 顺序有讲究：中文字体必须同时具备拉丁字形与上下标字形，否则中文标签里的
    # "q̇ (rad/s)"、"deg/s²" 会出现缺字框并触发 UserWarning。
    # 实测（Windows）：SimHei 缺 U+00B2/U+0307，SimSun 缺 U+2212，
    # Microsoft YaHei 覆盖最全，因此把它放在首位；其余作为跨平台回退。
    "Microsoft YaHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "SimHei",
]
# 单字符 -> ASCII 等价写法。matplotlib 的中文字体常缺这些码位，而 matplotlib 的
# 字体回退对**带组合符号的字符串**并不可靠（实测仍会告警并画成缺字框），
# 因此在保存前统一做一次 ASCII 化，保证任何环境下都画得出来。
_ASCII_FALLBACKS = {
    "\u0307": "_dot",   # 组合上点：q̇ -> q_dot
    "\u0308": "_ddot",  # 组合分音符：q̈ -> q_ddot
    "\u00b2": "^2",     # 上标 2
    "\u00b3": "^3",     # 上标 3
    "\u00b9": "^1",
    "\u2070": "^0",
    "\u2074": "^4",
    "\u2075": "^5",
    "\u2076": "^6",
    "\u2077": "^7",
    "\u2078": "^8",
    "\u2079": "^9",
    "\u2212": "-",      # 数学减号（axes.unicode_minus=False 时也不会出现在刻度）
}
_ASCII_TABLE = str.maketrans(_ASCII_FALLBACKS)
_EN_FORCED = os.environ.get("G_FONT", "").lower() in {"english", "en", "ascii"}


def _ensure_mpl_config_dir() -> None:
    """在**导入 matplotlib 之前**把字体缓存目录指到仓库内，避免权限问题。

    背景：受限环境 / 沙箱下 `%LOCALAPPDATA%\\matplotlib` 可能不可写，实测报
    `mkdir -p failed for path ...: [WinError 5] 拒绝访问`，matplotlib 会退回到
    工作目录里创建 `matplotlib-xxxx` 临时目录（污染仓库根目录）。
    在模块导入时就设置 MPLCONFIGDIR，可彻底消除该告警与临时目录。
    """
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache = Path(__file__).resolve().parents[2] / "results" / ".mplconfig"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
    except OSError:
        pass  # 建不了就交给 matplotlib 自己回退（不影响绘图正确性）


_ensure_mpl_config_dir()


def available() -> bool:
    """matplotlib 是否可用（不可用时脚本会跳过画图并给出安装提示）。"""
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _plt():
    """延迟导入 matplotlib 并完成中文字体配置。"""
    global _PLT, _CJK_OK, _CJK_ERROR
    if _PLT is not None:
        return _PLT
    _ensure_mpl_config_dir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if _EN_FORCED:
        _CJK_OK = False
    else:
        available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        for name in _CJK_CANDIDATES:
            if name in available_fonts:
                plt.rcParams["font.family"] = "sans-serif"
                # 中文字体优先，拉丁字体兜底：个别字符缺失时给 matplotlib 一个回退目标。
                # （组合符号如 U+0307 仍可能在部分后端下告警，故保存前另有 ASCII 化处理。）
                chain = [name]
                for fallback in ("DejaVu Sans", "Segoe UI", "Arial"):
                    if fallback in available_fonts and fallback not in chain:
                        chain.append(fallback)
                plt.rcParams["font.sans-serif"] = chain
                plt.rcParams["axes.unicode_minus"] = False
                _CJK_OK = True
                break
        else:
            _CJK_OK = False
            _CJK_ERROR = "未找到中文字体（尝试过: " + ", ".join(_CJK_CANDIDATES) + "）"
    _PLT = plt
    return _PLT


def cjk_ok() -> bool:
    _plt()
    return bool(_CJK_OK)


def L(zh: str, en: str) -> str:
    """双语标签：中文字体可用时用中文，否则自动回退英文（避免乱码）。"""
    return zh if cjk_ok() else en


def font_note() -> str:
    _plt()
    return "" if _CJK_OK else f" [labels: EN fallback; {_CJK_ERROR or 'G_FONT=English'}]"


def ascii_safe(text: str) -> str:
    """把中文字体普遍缺失的码位换成 ASCII 等价写法（组合上点 -> ``_dot``、上标 -> ``^2``）。

    为什么必须做：matplotlib 的字体回退对**带组合符号的字符串**不可靠，实测会打印
    ``UserWarning: Glyph 113 (q) missing from font(s) ...`` 并把字符画成缺字框；
    而 G 组多张图的纵轴标签正是 ``deg/s²``、标题里是 ``q̇ / q̈``。
    统一 ASCII 化后，任何字体环境下图像都不缺字，标签含义不变。
    """
    return text.translate(_ASCII_TABLE) if isinstance(text, str) else text


def _figtexts(fig):
    """遍历 figure 上所有可改写的文本对象（标题/轴标签/图例/注释/tick 标签）。"""
    texts = list(getattr(fig, "texts", []))
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        texts.append(suptitle)
    for ax in fig.get_axes():
        texts.append(ax.title)
        texts.append(ax.xaxis.label)
        texts.append(ax.yaxis.label)
        texts.extend(ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            texts.extend(legend.get_texts())
            texts.append(legend.get_title())
        texts.extend(ax.get_xticklabels())
        texts.extend(ax.get_yticklabels())
    return texts


def save(fig, path: str | Path, dpi: int = 160, tight: bool = True) -> Path:
    """保存图并关闭，返回绝对路径。输出目录不存在时自动创建。

    保存前对全图文本做一次 ASCII 化（见 :func:`ascii_safe`），确保中文字体
    不完整时也不会出现缺字框或 ``Glyph ... missing`` 告警。

    ``tight=False``：调用方已用 ``subplots_adjust`` 手工布局（例如需要在底部
    留出脚注空间）时使用，避免 ``tight_layout`` 覆盖造成脚注与坐标轴重叠。
    """
    plt = _plt()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for txt in _figtexts(fig):
        try:
            cur = txt.get_text()
        except Exception:  # noqa: BLE001 - 非文本对象直接跳过
            continue
        fixed = ascii_safe(cur)
        if fixed != cur:
            txt.set_text(fixed)
    if tight:
        fig.tight_layout()
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p.resolve()


def footnote(ax, text: str, y: float = -0.16) -> None:
    ax.annotate(text + font_note(), xy=(0.0, y), xycoords="axes fraction",
                fontsize=7.5, va="top", ha="left", color="#444444")


def plot_joint_profiles(time_s, q_rad, qdot_rad_s, qddot_rad_s2, title: str, out_path, note: str = ""):
    """三联图：关节角 q(t) / 速度 q̇(t) / 加速度 q̈(t)。"""
    plt = _plt()
    t = np.asarray(time_s, dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.0), sharex=True)
    for j in range(6):
        axes[0].plot(t, np.rad2deg(np.asarray(q_rad)[:, j]), lw=1.4, label=f"J{j + 1}")
        axes[1].plot(t, np.rad2deg(np.asarray(qdot_rad_s)[:, j]), lw=1.2)
        axes[2].plot(t, np.rad2deg(np.asarray(qddot_rad_s2)[:, j]), lw=1.2)
    axes[0].set_ylabel(L("关节角 (deg)", "joint angle (deg)"))
    axes[1].set_ylabel(L("关节速度 (deg/s)", "joint velocity (deg/s)"))
    axes[2].set_ylabel(L("关节加速度 (deg/s²)", "joint accel (deg/s²)"))
    axes[2].set_xlabel(L("时间 t (s)", "time t (s)"))
    axes[1].axhline(0, color="#999999", lw=0.6)
    axes[2].axhline(0, color="#999999", lw=0.6)
    axes[0].legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    axes[0].set_title(title, fontsize=11)
    footnote(axes[2], note)
    return save(fig, out_path)


def plot_trajectory_comparison(time_j, T_j, time_c, T_c, title: str, out_path, note: str = ""):
    """两条轨迹对比：3D 末端路径 + XZ 剖面 + 到起终点直线的偏差。"""
    plt = _plt()
    Tj = np.asarray(T_j, dtype=np.float64)
    Tc = np.asarray(T_c, dtype=np.float64)
    fig = plt.figure(figsize=(13.5, 4.6))

    ax3d = fig.add_subplot(131, projection="3d")
    ax3d.plot(Tj[:, 0, 3], Tj[:, 1, 3], Tj[:, 2, 3], lw=2, label=L("关节空间五次多项式", "joint quintic"))
    ax3d.plot(Tc[:, 0, 3], Tc[:, 1, 3], Tc[:, 2, 3], lw=2, ls="--", label=L("笛卡尔直线", "cartesian line"))
    ax3d.scatter(*Tj[0, :3, 3], color="k", s=18)
    ax3d.scatter(*Tj[-1, :3, 3], color="k", s=18, marker="^")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title(L("末端 3D 路径", "TCP path in 3D"), fontsize=10)
    ax3d.legend(fontsize=8)

    ax2 = fig.add_subplot(132)
    ax2.plot(Tj[:, 0, 3], Tj[:, 2, 3], lw=2, label=L("关节空间", "joint quintic"))
    ax2.plot(Tc[:, 0, 3], Tc[:, 2, 3], lw=2, ls="--", label=L("笛卡尔直线", "cartesian line"))
    ax2.plot([Tj[0, 0, 3], Tj[-1, 0, 3]], [Tj[0, 2, 3], Tj[-1, 2, 3]], color="#bbbbbb", lw=1, ls=":",
             label=L("起终点直线", "chord"))
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(alpha=0.3)
    ax2.set_title(L("XZ 剖面", "XZ section"), fontsize=10)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(133)
    for T, lab, ls in ((Tj, L("关节空间", "joint quintic"), "-"), (Tc, L("笛卡尔直线", "cartesian line"), "--")):
        p = T[:, :3, 3]
        a, b = p[0], p[-1]
        ab = b - a
        n = max(np.linalg.norm(ab), 1e-12)
        tt = ((p - a) @ ab) / (n * n)
        dev = np.linalg.norm(p - (a + np.outer(tt, ab)), axis=1)
        # 以"沿起终点直线的进度 (m)"为横轴，两条轨迹可直接比较
        ax3.plot(tt * n, dev, lw=2, ls=ls, label=lab)
    ax3.set_xlabel(L("沿起终点直线的进度 (m)", "progress along chord (m)"))
    ax3.set_ylabel(L("偏离直线的距离 (m)", "deviation from chord (m)"))
    ax3.grid(alpha=0.3)
    ax3.set_title(L("末端偏离直线程度", "deviation from straight line"), fontsize=10)
    ax3.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_tracking_error(time_s, err_rad, title: str, out_path, note: str = ""):
    """跟踪误差图：整体范数 + 逐关节。"""
    plt = _plt()
    t = np.asarray(time_s, dtype=np.float64)
    e = np.asarray(err_rad, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(t, np.rad2deg(np.linalg.norm(e, axis=1)), lw=1.8, color="#c0392b")
    axes[0].set_xlabel(L("时间 t (s)", "t (s)"))
    axes[0].set_ylabel(L("误差范数 (deg)", "error norm (deg)"))
    axes[0].set_title(L("跟踪误差范数", "tracking error norm"), fontsize=10)
    for j in range(6):
        axes[1].plot(t, np.rad2deg(e[:, j]), lw=1.2, label=f"J{j + 1}")
    axes[1].set_xlabel(L("时间 t (s)", "t (s)"))
    axes[1].set_ylabel(L("关节误差 (deg)", "joint error (deg)"))
    axes[1].legend(ncol=6, fontsize=8)
    axes[1].set_title(L("逐关节跟踪误差", "per-joint tracking error"), fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_singularity_profile(time_s, analyze_result: dict, title: str, out_path, note: str = ""):
    """奇异点随时间的演化：σmin、条件数（对数轴）、可操作度。"""
    plt = _plt()
    t = np.asarray(time_s, dtype=np.float64)
    res = analyze_result
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(t, res["sigma_min"], lw=1.8, color="#8e44ad")
    axes[0].axhline(res["sigma_min_threshold"], color="#e67e22", ls="--", lw=1,
                    label=f"threshold={res['sigma_min_threshold']:g}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(L("时间 t (s)", "t (s)"))
    axes[0].set_ylabel(r"$\sigma_{\min}$")
    axes[0].set_title(L("最小奇异值", "smallest singular value"), fontsize=10)
    axes[0].legend(fontsize=8)

    cond = np.asarray(res["condition_number"], dtype=np.float64)
    cond = np.where(np.isfinite(cond), cond, np.nan)
    axes[1].semilogy(t, cond, lw=1.8, color="#2980b9")
    axes[1].set_xlabel(L("时间 t (s)", "t (s)"))
    axes[1].set_ylabel(L("条件数 $\\sigma_1/\\sigma_6$", "condition number"))
    axes[1].set_title(L("条件数（越大越病态）", "condition number"), fontsize=10)

    axes[2].plot(t, res["manipulability"], lw=1.8, color="#27ae60")
    axes[2].set_xlabel(L("时间 t (s)", "t (s)"))
    axes[2].set_ylabel(L("可操作度 $\\prod\\sigma_i$", "manipulability"))
    axes[2].set_title(L("可操作度（0 = 奇异）", "manipulability"), fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_sigma_distribution(sigma_min_traj: np.ndarray, sigma_min_ws: np.ndarray, title: str, out_path, note: str = ""):
    """两类构型的 σmin 分布对比（对数横轴）。"""
    plt = _plt()
    a = np.asarray(sigma_min_traj, dtype=np.float64)
    b = np.asarray(sigma_min_ws, dtype=np.float64)
    positive = [v[v > 0].min() for v in (a, b) if np.any(v > 0)]
    lo = max(min(positive) if positive else 1e-8, 1e-12)
    bins = np.logspace(np.log10(lo), 0, 40)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(a, bins=bins, alpha=0.6, label=L("轨迹采样点", "trajectory samples"))
    ax.hist(b, bins=bins, alpha=0.6, label=L("关节空间均匀采样", "uniform joint samples"))
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{\min}(J)$")
    ax.set_ylabel(L("采样点数", "sample count"))
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    footnote(ax, note)
    return save(fig, out_path)


def plot_workspace(positions: np.ndarray, title: str, out_path, voxel_size_m: float = 0.05, note: str = ""):
    """工作空间三视图：XZ 投影密度、XY 投影、半径直方图。"""
    plt = _plt()
    p = np.asarray(positions, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    axes[0].hexbin(p[:, 0], p[:, 2], gridsize=60, cmap="viridis", mincnt=1)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title(L("XZ 投影（可达密度）", "XZ projection density"), fontsize=10)
    axes[1].hexbin(p[:, 0], p[:, 1], gridsize=60, cmap="magma", mincnt=1)
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].set_title(L("XY 投影（水平可达范围）", "XY projection"), fontsize=10)
    r = np.linalg.norm(p, axis=1)
    axes[2].hist(r, bins=60, color="#2c7fb8", alpha=0.85)
    axes[2].axvline(r.max(), color="#c0392b", ls="--", lw=1.2, label=f"max={r.max():.3f} m")
    axes[2].set_xlabel(L("半径 r (m)", "radius r (m)"))
    axes[2].set_ylabel(L("采样点数", "sample count"))
    axes[2].legend(fontsize=8)
    axes[2].set_title(L("半径分布", "radial distribution"), fontsize=10)
    fig.suptitle(f"{title}  (voxel={voxel_size_m:g} m)", fontsize=12)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_singularity_map(xs, ys, sigma_min, manip, title: str, out_path, xlabel: str, ylabel: str, note: str = ""):
    """在关节空间二维切片上画 σmin / 可操作度 热图（找奇异位形的直观图）。"""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    m1 = axes[0].pcolormesh(xs, ys, sigma_min.T, shading="auto", cmap="viridis")
    axes[0].set_title(r"$\sigma_{\min}(J)$", fontsize=11)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.colorbar(m1, ax=axes[0])
    m2 = axes[1].pcolormesh(xs, ys, manip.T, shading="auto", cmap="magma")
    axes[1].set_title(L("可操作度 $\\prod\\sigma_i$", "manipulability"), fontsize=11)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    fig.colorbar(m2, ax=axes[1])
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_singularity_maps(slices, title: str, out_path, note: str = ""):
    """多关节切片对照：每行一个切片的 σmin（左）与可操作度（右）。

    slices: list[dict]，每项含 keys: xs, ys, sigma_min, manip, xlabel, ylabel, label
    用途：奇异位形是关节空间的**低维曲面**，单一 (q1,q2) 切片通常看不到奇异性；
    需要针对不同类型的奇异（腕部 q5=0、肘部伸展）选择对应的关节切片。
    """
    plt = _plt()
    n = len(slices)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.2 * n), squeeze=False)
    for row, s in enumerate(slices):
        m1 = axes[row][0].pcolormesh(s["xs"], s["ys"], s["sigma_min"].T, shading="auto", cmap="viridis")
        axes[row][0].set_title(f"{s['label']} — " + r"$\sigma_{\min}(J)$", fontsize=10)
        axes[row][0].set_xlabel(s["xlabel"])
        axes[row][0].set_ylabel(s["ylabel"])
        fig.colorbar(m1, ax=axes[row][0])

        m2 = axes[row][1].pcolormesh(s["xs"], s["ys"], s["manip"].T, shading="auto", cmap="magma")
        axes[row][1].set_title(f"{s['label']} — " + L("可操作度 $\\prod\\sigma_i$", "manipulability"), fontsize=10)
        axes[row][1].set_xlabel(s["xlabel"])
        axes[row][1].set_ylabel(s["ylabel"])
        fig.colorbar(m2, ax=axes[row][1])
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.12, hspace=0.45)
    fig.text(0.01, 0.01, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


# ======================================================================
# 第二阶段（真实模型 + F 组真实日志）新增图
# ======================================================================
_SEG_COLORS = {
    "home": "#7f8c8d", "approach": "#2980b9", "descend": "#8e44ad", "descend_settle": "#af7ac5",
    "grasp_hold": "#16a085", "lift": "#27ae60", "transfer": "#e67e22", "lower": "#c0392b",
    "lower_settle": "#ec7063", "release_hold": "#f1c40f", "retreat": "#d35400",
    "settle_after_release": "#f39c12", "return_home": "#34495e",
}


def _seg_color(name: str) -> str:
    for key, col in _SEG_COLORS.items():
        if name.startswith(key):
            return col
    return "#555555"


def shade_segments(ax, t, seg_ids, alpha: float = 0.18) -> None:
    """按段名给时间轴打不同底色的竖条（同一段名同色）。"""
    t = np.asarray(t, dtype=np.float64)
    ids = np.asarray(seg_ids)
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            ax.axvspan(t[start], t[min(i, len(t) - 1)], color=_seg_color(str(ids[start])),
                       alpha=alpha, lw=0, label=str(ids[start]))
            start = i


def plot_ee_tracking(time_s, pos_planned, pos_actual, err_mm, title: str, out_path, note: str = ""):
    """末端执行跟踪：3D 规划/实际路径 + 位置误差时序（按搬运段着色）。"""
    plt = _plt()
    t = np.asarray(time_s, dtype=np.float64)
    pp = np.asarray(pos_planned, dtype=np.float64)
    pa = np.asarray(pos_actual, dtype=np.float64)
    fig = plt.figure(figsize=(14, 4.8))

    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.plot(pp[:, 0], pp[:, 1], pp[:, 2], lw=2.0, label=L("规划 (q_cmd)", "planned"))
    ax3d.plot(pa[:, 0], pa[:, 1], pa[:, 2], lw=1.2, ls="--", label=L("实际 (q_actual)", "actual"))
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title(L("末端 3D 路径：规划 vs 实际", "TCP path: planned vs actual"), fontsize=10)
    ax3d.legend(fontsize=8)

    ax = fig.add_subplot(122)
    ax.plot(t, np.asarray(err_mm, dtype=np.float64), lw=1.6, color="#c0392b")
    ax.set_xlabel(L("时间 t (s)", "t (s)"))
    ax.set_ylabel(L("末端位置误差 (mm)", "TCP position error (mm)"))
    ax.grid(alpha=0.3)
    ax.set_title(L("末端位置跟踪误差", "TCP position tracking error"), fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_error_by_segment(t, joint_err_deg, ee_err_mm, seg_ids, title: str, out_path, note: str = ""):
    """关节与末端误差时序，背景按 13 种搬运段着色（把误差归因到具体动作）。"""
    plt = _plt()
    t = np.asarray(t, dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True)
    axes[0].plot(t, np.asarray(joint_err_deg, dtype=np.float64), lw=1.2, color="#2c3e50")
    axes[0].set_ylabel(L("关节误差范数 (deg)", "joint error norm (deg)"))
    axes[0].set_title(L("关节跟踪误差", "joint tracking error"), fontsize=10)
    axes[1].plot(t, np.asarray(ee_err_mm, dtype=np.float64), lw=1.2, color="#c0392b")
    axes[1].set_ylabel(L("末端位置误差 (mm)", "TCP position error (mm)"))
    axes[1].set_xlabel(L("时间 t (s)", "t (s)"))
    axes[1].set_title(L("末端位置跟踪误差", "TCP position error"), fontsize=10)
    for ax in axes:
        shade_segments(ax, t, seg_ids)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            h2.append(h)
            l2.append(lab)
    axes[0].legend(h2, l2, ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.30),
                   frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.2, top=0.86)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_lag_demo(t, ee_pos_log, fk_pos, lag_samples: int, title: str, out_path,
                  window: tuple[float, float] | None = None, note: str = ""):
    """证明 F 日志 `ee_*` 列比 `q_actual` 滞后一个采样：两条曲线错位 2 ms。"""
    plt = _plt()
    t = np.asarray(t, dtype=np.float64)
    a = np.asarray(ee_pos_log, dtype=np.float64)
    b = np.asarray(fk_pos, dtype=np.float64)
    if window is None:
        i = int(np.argmax(np.linalg.norm(a - b, axis=1)))
        lo, hi = max(t[i] - 0.6, t[0]), min(t[i] + 0.6, t[-1])
    else:
        lo, hi = window
    m = (t >= lo) & (t <= hi)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for j, lab in ((0, "x"), (2, "z")):
        axes[0].plot(t[m], a[m, j] * 1000, lw=2.2, label=f"{lab}: ee_* (log)")
        axes[0].plot(t[m], b[m, j] * 1000, lw=1.4, ls="--", label=f"{lab}: fk(q_actual)")
    axes[0].set_xlabel(L("时间 t (s)", "t (s)"))
    axes[0].set_ylabel(L("位置 (mm)", "position (mm)"))
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(L(f"原样对比：曲线错位 {lag_samples} 个采样",
                        f"raw: misaligned by {lag_samples} sample(s)"), fontsize=10)

    n = len(t)
    shifted = np.roll(b, int(lag_samples), axis=0)
    d_raw = np.linalg.norm(a - b, axis=1) * 1000
    d_al = np.linalg.norm(a - shifted, axis=1) * 1000
    axes[1].plot(t, d_raw, lw=1.2, label=L("未校正", "raw"))
    axes[1].plot(t, d_al, lw=1.2, ls="--", label=L("滞后校正后", "lag-corrected"))
    axes[1].set_xlabel(L("时间 t (s)", "t (s)"))
    axes[1].set_ylabel(L("残差 (mm)", "residual (mm)"))
    axes[1].set_title(L("与 fk(q_actual) 的残差", "residual vs fk(q_actual)"), fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, max(d_raw.max() * 1.1, 0.05))

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_validation_summary(rows, title: str, out_path, note: str = ""):
    """验证表可视化：每项"实测/阈值"比值的对数条形图，>1 即不通过。"""
    plt = _plt()
    items, ratios, passed = [], [], []
    for r in rows:
        if r["check"] in {"c_fk_tcp", "f_log_residual"} and r["metric"].startswith("naive"):
            continue
        if r["unit"] == "bool":
            continue
        thr = r["threshold"]
        val = r["value"]
        if not np.isfinite(val) or not np.isfinite(thr) or thr <= 0:
            continue
        ratios.append(max(val / thr, 1e-6))
        items.append(f"{r['check']}:{r['metric']}")
        passed.append(r["passed"])
    order = np.argsort(ratios)[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.32 * len(items) + 2.2))
    y = np.arange(len(items))
    vals = [ratios[i] for i in order]
    cols = ["#27ae60" if passed[i] else "#c0392b" for i in order]
    ax.barh(y, vals, color=cols, alpha=0.85)
    ax.axvline(1.0, color="#2c3e50", lw=1.2, ls="--", label=L("阈值 (1.0)", "threshold"))
    ax.set_yticks(y)
    ax.set_yticklabels([items[i] for i in order], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel(L("实测值 / 阈值（<1 通过）", "measured / threshold (<1 passes)"))
    ax.set_xlim(1e-6, max(3.0, max(vals) * 2))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    ax.set_title(L("上游数据一致性验证", "upstream data consistency"), fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(0.01, 0.01, note + font_note(), fontsize=7, color="#444444")
    return save(fig, out_path, tight=False)


def plot_singular_config_histograms(all_q, sing_q, joint_names, title: str, out_path, note: str = ""):
    """奇异构型在不同关节上的分布：哪个关节的角度把机器人推到奇异。"""
    plt = _plt()
    a = np.rad2deg(np.asarray(all_q, dtype=np.float64))
    s = np.rad2deg(np.asarray(sing_q, dtype=np.float64))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2))
    for j, ax in enumerate(axes.ravel()):
        lo = min(a[:, j].min(), s[:, j].min()) if len(s) else a[:, j].min()
        hi = max(a[:, j].max(), s[:, j].max()) if len(s) else a[:, j].max()
        bins = np.linspace(lo, hi, 60)
        ax.hist(a[:, j], bins=bins, color="#95a5a6", alpha=0.55,
                density=True, label=L("全部采样", "all samples"))
        if len(s):
            ax.hist(s[:, j], bins=bins, color="#c0392b", alpha=0.65,
                    density=True, label=L("奇异采样 (σmin<阈值)", "singular samples"))
        ax.set_title(joint_names[j], fontsize=9)
        ax.set_xlabel(L("关节角 (deg)", "angle (deg)"), fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7.5)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.14, top=0.9, hspace=0.5, wspace=0.28)
    fig.text(0.01, 0.01, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_sigma_vs_tracking(sigma_min, ee_err_mm, seg_ids, title: str, out_path, note: str = ""):
    """σmin 与末端跟踪误差的散点：检验"越接近奇异，跟踪越差"是否成立。"""
    plt = _plt()
    s = np.asarray(sigma_min, dtype=np.float64)
    e = np.asarray(ee_err_mm, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for name in sorted(set(str(v) for v in seg_ids)):
        m = np.array([str(v) == name for v in seg_ids])
        axes[0].scatter(s[m], e[m], s=4, alpha=0.7, label=name, color=_seg_color(name))
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\sigma_{\min}(J(q_{cmd}))$")
    axes[0].set_ylabel(L("末端位置跟踪误差 (mm)", "TCP error (mm)"))
    axes[0].grid(alpha=0.3)
    axes[0].set_title(L("奇异性 vs 跟踪误差", "singularity vs tracking error"), fontsize=10)

    # 分箱统计：把 σmin 分位数作为横轴，看误差中位数随奇异性变化的趋势
    qs = np.quantile(s, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    qs[0] = max(qs[0], 1e-12)
    idx = np.clip(np.searchsorted(qs, s, side="right") - 1, 0, len(qs) - 2)
    centers, med, p90 = [], [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        centers.append(np.exp(np.mean(np.log(s[m]))))
        med.append(float(np.median(e[m])))
        p90.append(float(np.percentile(e[m], 90)))
    axes[1].plot(centers, med, "o-", lw=1.6, label=L("中位数", "median"))
    axes[1].plot(centers, p90, "s--", lw=1.4, label=L("90% 分位", "p90"))
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\sigma_{\min}$ 分箱中心")
    axes[1].set_ylabel(L("末端误差 (mm)", "TCP error (mm)"))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_title(L("按奇异性分箱后的误差统计", "error binned by singularity"), fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_workspace_compare(pos_real, pos_ref, title: str, out_path, note: str = ""):
    """真实 MuJoCo 模型 vs G 组占位 DH 模型的工作空间对照（XY/XZ 投影 + 半径分布）。"""
    plt = _plt()
    a = np.asarray(pos_real, dtype=np.float64)
    b = np.asarray(pos_ref, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].hexbin(a[:, 0], a[:, 2], gridsize=70, cmap="viridis", mincnt=1)
    axes[0].set_title(L("真实模型 XZ 投影", "real model XZ"), fontsize=10)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].hexbin(b[:, 0], b[:, 2], gridsize=70, cmap="magma", mincnt=1)
    axes[1].set_title(L("占位 DH 模型 XZ 投影", "reference DH model XZ"), fontsize=10)
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    axes[1].set_aspect("equal", adjustable="datalim")
    ra = np.linalg.norm(a, axis=1)
    rb = np.linalg.norm(b, axis=1)
    bins = np.linspace(0, max(ra.max(), rb.max()) * 1.02, 70)
    axes[2].hist(ra, bins=bins, alpha=0.6, label=L(f"真实模型 (max={ra.max():.3f} m)", "real"))
    axes[2].hist(rb, bins=bins, alpha=0.6, label=L(f"占位模型 (max={rb.max():.3f} m)", "reference"))
    axes[2].set_xlabel(L("半径 r (m)", "radius r (m)"))
    axes[2].set_ylabel(L("采样点数", "count"))
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_title(L("半径分布对比", "radial distribution"), fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_ik_damping_sweep(lambdas, success_rate, worst_mm, title: str, out_path, note: str = ""):
    """D/F 的 DLS 阻尼系数扫描：成功率（左轴）与最差残差（右轴，对数）。"""
    plt = _plt()
    x = np.asarray(lambdas, dtype=np.float64)
    fig, ax1 = plt.subplots(figsize=(8.6, 4.6))
    ax1.plot(x, 100.0 * np.asarray(success_rate, dtype=np.float64), "o-", lw=1.8,
             color="#2980b9", label=L("成功率 (%)", "success rate (%)"))
    ax1.set_xlabel(r"阻尼系数 $\lambda$")
    ax1.set_ylabel(L("成功率 (%)", "success rate (%)"), color="#2980b9")
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.semilogy(x, np.asarray(worst_mm, dtype=np.float64), "s--", lw=1.6, color="#c0392b",
                 label=L("最差位置残差 (mm)", "worst residual (mm)"))
    ax2.set_ylabel(L("最差位置残差 (mm)", "worst residual (mm)"), color="#c0392b")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="center right")
    ax1.set_title(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.01, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_path_length_accum(t, pos_planned, pos_actual, title: str, out_path, note: str = ""):
    """累积路径长度 vs 时间：规划 vs 实际（m）。同时给出每步增量差。"""
    plt = _plt()
    p = np.asarray(pos_planned, dtype=np.float64)
    a = np.asarray(pos_actual, dtype=np.float64)
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    da = np.linalg.norm(np.diff(a, axis=0), axis=1)
    cp = np.concatenate([[0.0], np.cumsum(dp)])
    ca = np.concatenate([[0.0], np.cumsum(da)])
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(t, cp, color="#2980b9", lw=1.8, label=L("规划 (q_cmd)", "planned (q_cmd)"))
    ax.plot(t, ca, color="#c0392b", lw=1.2, ls="--", label=L("实际 (q_actual)", "actual (q_actual)"))
    ax.set_ylabel(L("累积末端路径长度 (m)", "cumulative EE path (m)"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title(title, fontsize=12)
    ax2 = axes[1]
    inc = np.concatenate([[0.0], np.diff(ca - cp)])
    ax2.plot(t[1:], (da - dp) * 1000.0, color="#2c3e50", lw=0.9)
    ax2.axhline(0.0, color="#7f8c8d", lw=0.8)
    ax2.set_ylabel(L("每步长度差 (mm)", "step length diff (mm)"))
    ax2.set_xlabel(L("时间 t (s)", "time (s)"))
    ax2.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.01, 0.005, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_radial_reach(hist: dict, stats: dict, title: str, out_path, note: str = ""):
    """TCP 半径分布直方图 + 最小/最大伸展竖线（把"可达范围"变成可核对的数）。"""
    plt = _plt()
    edges = np.asarray(hist["bin_edges_m"], dtype=np.float64)
    counts = np.asarray(hist["counts"], dtype=np.float64)
    centers = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.bar(centers, counts, width=width, color="#2980b9", alpha=0.8,
           label=L("关节空间均匀采样", "uniform joint-space samples"))
    ax.axvline(stats["radius_min_m"], color="#c0392b", lw=1.4, ls="--",
               label=L(f"实测最小伸展 {stats['radius_min_m']:.4f} m",
                       f"min reach {stats['radius_min_m']:.4f} m"))
    ax.axvline(stats["radius_max_m"], color="#27ae60", lw=1.4, ls="--",
               label=L(f"实测最大伸展 {stats['radius_max_m']:.4f} m",
                       f"max reach {stats['radius_max_m']:.4f} m"))
    ax.set_xlabel(L("TCP 到基座原点距离 (m)", "TCP radius from base (m)"))
    ax.set_ylabel(L("采样点数", "sample count"))
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.01, 0.02, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_ee_vs_joint_error(time_s, joint_err_deg, ee_err_mm, title: str, out_path, note: str = ""):
    """关节误差与末端误差的双轴时序：说明"关节差一点，末端差多少"的放大关系。"""
    plt = _plt()
    t = np.asarray(time_s, dtype=np.float64)
    fig, ax1 = plt.subplots(figsize=(12, 4.2))
    ax1.plot(t, np.asarray(joint_err_deg, dtype=np.float64), lw=1.2, color="#2c3e50",
             label=L("关节误差范数 (deg)", "joint error (deg)"))
    ax1.set_xlabel(L("时间 t (s)", "t (s)"))
    ax1.set_ylabel(L("关节误差 (deg)", "joint error (deg)"), color="#2c3e50")
    ax2 = ax1.twinx()
    ax2.plot(t, np.asarray(ee_err_mm, dtype=np.float64), lw=1.2, color="#c0392b",
             label=L("末端位置误差 (mm)", "TCP error (mm)"))
    ax2.set_ylabel(L("末端位置误差 (mm)", "TCP error (mm)"), color="#c0392b")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.set_title(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.01, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


# ======================================================================
# 第三阶段（E 组轨迹 + E→F 集成复跑）新增图
# ======================================================================
def _span_segments(ax, t, seg_names, alpha: float = 0.15) -> None:
    """按段名打底色竖条（不加图例，避免同名段重复条目）。"""
    t = np.asarray(t, dtype=np.float64)
    names = [str(x) for x in seg_names]
    start = 0
    for i in range(1, len(names) + 1):
        if i == len(names) or names[i] != names[start]:
            ax.axvspan(t[start], t[min(i, len(t) - 1)], color=_seg_color(names[start]),
                       alpha=alpha, lw=0)
            start = i


def _seg_legend(ax, seg_names, fontsize: float = 7.5, ncol: int = 7) -> None:
    """按出现顺序去重后画段名图例（用色块代理）。"""
    from matplotlib.patches import Patch

    seen: list[str] = []
    for nm in seg_names:
        nm = str(nm)
        if nm not in seen:
            seen.append(nm)
    handles = [Patch(facecolor=_seg_color(nm), alpha=0.55, label=nm) for nm in seen]
    ax.legend(handles=handles, fontsize=fontsize, ncol=ncol, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), frameon=False)


def plot_e_joint_profiles(time_s, q_rad, dq_rad_s, ddq_rad_s2, seg_names, title: str,
                          out_path, note: str = ""):
    """E 轨迹的关节角/速度/加速度时序（按 E 的 7 段着色）。"""
    plt = _plt()
    t = np.asarray(time_s, dtype=np.float64)
    q = np.rad2deg(np.asarray(q_rad, dtype=np.float64))
    dq = np.rad2deg(np.asarray(dq_rad_s, dtype=np.float64))
    ddq = np.rad2deg(np.asarray(ddq_rad_s2, dtype=np.float64))
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.2), sharex=True)
    for j in range(6):
        axes[0].plot(t, q[:, j], lw=1.3, label=f"J{j + 1}")
        axes[1].plot(t, dq[:, j], lw=1.1)
        axes[2].plot(t, ddq[:, j], lw=1.0)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color="#95a5a6", lw=0.8)
        _span_segments(ax, t, seg_names)
    axes[0].set_ylabel(L("关节角 (deg)", "joint angle (deg)"))
    axes[0].set_title(title, fontsize=12)
    axes[0].legend(fontsize=8, ncol=6, loc="lower right")
    axes[1].set_ylabel(L("关节速度 (deg/s)", "joint speed (deg/s)"))
    axes[2].set_ylabel(L("关节加速度 (deg/s^2)", "joint accel (deg/s^2)"))
    axes[2].set_xlabel(L("时间 t (s)", "time (s)"))
    _seg_legend(axes[0], seg_names)
    fig.subplots_adjust(bottom=0.10, top=0.90)
    fig.text(0.01, 0.012, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_tcp_path_3d(pos, seg_names, key_heights: dict, title: str, out_path, note: str = ""):
    """TCP 路径：3D 按段着色 + XZ 投影并标关键高度（HOVER/GRASP/LIFT/LOWER/CLEAR）。"""
    plt = _plt()
    p = np.asarray(pos, dtype=np.float64)
    names = [str(x) for x in seg_names]
    fig = plt.figure(figsize=(14.5, 6.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    start = 0
    run = 0
    for i in range(1, len(names) + 1):
        if i == len(names) or names[i] != names[start]:
            sl = slice(start, min(i + 1, len(p)))
            ax.plot(p[sl, 0], p[sl, 1], p[sl, 2], lw=1.6, color=_seg_color(names[start]),
                    label=names[start] if run < 12 else None)
            run += 1
            start = i
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(L("TCP 3D 路径（按段着色）", "TCP path in 3D (colored by phase)"), fontsize=10.5)
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(p[:, 0], p[:, 2], color="#2c3e50", lw=1.4)
    ax2.scatter(p[0, 0], p[0, 2], s=45, color="#27ae60", zorder=5,
                label=L("起点（home）", "start (home)"))
    ax2.scatter(p[-1, 0], p[-1, 2], s=45, color="#c0392b", marker="X", zorder=5,
                label=L("终点（home）", "end (home)"))
    for name, z in key_heights.items():
        ax2.axhline(z, color="#7f8c8d", lw=0.9, ls="--")
        ax2.annotate(f"{name} z={z:.3f}", xy=(0.505, z), fontsize=7.5, color="#555555",
                     va="bottom")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_title(L("XZ 投影与关键高度", "XZ projection and key heights"), fontsize=10.5)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.10)
    fig.text(0.01, 0.012, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_segment_straightness(names, straight, dev_mm, title: str, out_path, note: str = ""):
    """各段"弦长/路径长"直线度 + 最大偏离弦线距离（mm）。"""
    plt = _plt()
    y = np.arange(len(names))
    s = np.asarray(straight, dtype=np.float64)
    d = np.asarray(dev_mm, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    colors = ["#27ae60" if v > 0.999 else "#e67e22" for v in s]
    axes[0].barh(y, s, color=colors, alpha=0.85)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(min(float(s.min()), 0.99) - 0.005, 1.0005)
    for k, v in enumerate(s):
        axes[0].annotate(f"{v:.6f}", xy=(v, k), xytext=(3, 0), textcoords="offset points",
                         va="center", fontsize=8)
    axes[0].set_xlabel(L("直线度（弦长 / 路径长，1.0 = 严格直线）", "straightness (chord/path)"))
    axes[0].set_title(L("各段直线度", "straightness per phase"), fontsize=10.5)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(y, d, color="#8e44ad", alpha=0.85)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(L("最大偏离弦线距离 (mm)", "max deviation from chord (mm)"))
    axes[1].set_title(L("偏离弦线峰值（越接近 0 越直）", "peak deviation from chord"), fontsize=10.5)
    axes[1].grid(alpha=0.3, axis="x")
    for k, v in enumerate(d):
        axes[1].annotate(f"{v:.4f}", xy=(v, k), xytext=(3, 0), textcoords="offset points",
                         va="center", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.01, 0.012, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_e_sigma(time_s, sigma_min, condition_number, threshold: float, seg_names,
                 title: str, out_path, note: str = ""):
    """E 轨迹上的最小奇异值/条件数：判断"这条规划轨迹是否贴近奇异"。"""
    plt = _plt()
    t = np.asarray(time_s, dtype=np.float64)
    smin = np.asarray(sigma_min, dtype=np.float64)
    cond = np.asarray(condition_number, dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(t, smin, color="#2980b9", lw=1.2, label=L(r"$\sigma_{\min}(J)$", "sigma_min(J)"))
    ax.axhline(threshold, color="#c0392b", lw=1.2, ls="--",
               label=L(f"阈值 {threshold}", f"threshold {threshold}"))
    ax.set_ylabel(L(r"$\sigma_{\min}(J)$", "sigma_min(J)"))
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")
    ax2 = axes[1]
    ax2.semilogy(t, cond, color="#8e44ad", lw=1.1)
    ax2.set_ylabel(L("条件数 (对数轴)", "condition number (log)"))
    ax2.set_xlabel(L("时间 t (s)", "time (s)"))
    ax2.grid(alpha=0.3, which="both")
    for a in axes:
        _span_segments(a, t, seg_names)
    _seg_legend(axes[0], seg_names)
    fig.subplots_adjust(bottom=0.12, top=0.90)
    fig.text(0.01, 0.012, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)


def plot_exec_jump(time_s, q_cmd_joint, q_act_joint, err_deg, seg_names, jump_index,
                   joint_label: str, title: str, out_path, note: str = ""):
    """E→F 集成复跑：指令/实际关节角与误差，标注轨迹接管处的跳变。"""
    plt = _plt()
    t = np.asarray(time_s, dtype=np.float64)
    qc = np.asarray(q_cmd_joint, dtype=np.float64)
    qa = np.asarray(q_act_joint, dtype=np.float64)
    e = np.asarray(err_deg, dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    ax.plot(t, qc, color="#2980b9", lw=1.2, label=L("指令 q_cmd", "command q_cmd"))
    ax.plot(t, qa, color="#e67e22", lw=1.2, ls="--", label=L("实际 q_actual", "actual q_actual"))
    ax.set_ylabel(f"{joint_label} " + L("关节角 (deg)", "(deg)"))
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")
    ax2 = axes[1]
    ax2.plot(t, e, color="#c0392b", lw=1.0)
    ax2.set_ylabel(L("关节误差范数 (deg)", "joint error norm (deg)"))
    ax2.set_xlabel(L("时间 t (s)", "time (s)"))
    ax2.grid(alpha=0.3)
    for a in axes:
        _span_segments(a, t, seg_names)
        if jump_index is not None and 0 <= int(jump_index) < len(t):
            a.axvline(t[int(jump_index)], color="#000000", lw=1.4, ls=":")
    _seg_legend(axes[0], seg_names)
    if jump_index is not None and 0 <= int(jump_index) < len(t):
        axes[1].annotate(L("轨迹接管点", "handover point"), xy=(t[int(jump_index)], e[int(jump_index)]),
                         xytext=(12, -18), textcoords="offset points", fontsize=8,
                         arrowprops=dict(arrowstyle="->", lw=0.9))
    fig.subplots_adjust(bottom=0.12, top=0.90)
    fig.text(0.01, 0.012, note + font_note(), fontsize=7.5, color="#444444")
    return save(fig, out_path, tight=False)
