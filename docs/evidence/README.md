# H 模块实验验证证据

本文件夹用于保存 H 模块在 Windows 环境下复现 UR5e 机械臂搬运仿真的实验图表与截图。

## 主要证据

- `tracking.png`：三次搬运的关节跟踪误差、TCP 位置误差和近奇异性指标。
- `planning_workspace.png`：关节空间与笛卡尔空间轨迹对比、正运动学工作空间采样。

## 验证结论

2026-10-18，在 Windows、Python 3.11.9、MuJoCo 3.16.0 环境下，连续三次搬运成功，并通过 `python scripts/verify_all.py` 集成检查。

最大关节跟踪误差约 0.22649°，TCP 最大位置误差约 3.33204 mm。

详细运行过程见 `docs/H_environment_reproduction.md`。
