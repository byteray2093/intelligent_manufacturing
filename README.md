# 六轴机器人运动学与搬运轨迹仿真

《智能制造导论》题目11，UR5e + Python + MuJoCo。B–G 组交付已集成：独立 DH 正运动学、阻尼最小二乘逆运动学、七阶段搬运规划、吸附约束搬运、轨迹/误差/奇异性/工作空间分析。

## 安装与运行

推荐 Python 3.11 或 3.12，在仓库根目录运行：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.main plan
python -m src.main demo --cycles 3 --no-video
python -m src.main analyze
```

现场窗口：`python -m src.main demo --cycles 3 --viewer --no-video`。macOS 的 MuJoCo viewer 需要使用 `mjpython -m src.main demo --viewer --no-video`。

录制视频：`python -m src.main demo --cycles 3`。Linux 无桌面环境可先设置 `export MUJOCO_GL=egl`；OpenGL 不可用时会明确标记骨架渲染后备模式。视频存于 `results/videos/demo.mp4`。

一键完整验收：

```bash
python scripts/verify_all.py
# 同时录视频：python scripts/verify_all.py --video
```

## 已验证结果

完整 E 轨迹按阶段执行，连续三次搬运成功。具体测量及环境见 [本次集成验收报告](docs/INTEGRATION_REPORT.md)，最新可复核输出见 [results/integrated](results/integrated)。

- [演示视频](results/videos/demo.mp4)
- [关节与TCP跟踪误差](results/integrated/tracking.png)
- [规划对比及工作空间采样图](results/integrated/planning_workspace.png)
- [同起终点、同时长的规划对比数据](results/integrated/planning_comparison.csv)
- [执行摘要](results/logs/run_summary_latest.json)

这些是理想吸附约束的仿真验证。当前没有真实夹爪接触抓取或自动避障；工作空间采样不等同于无碰撞位姿可达性。数值 IK 失败只能说明该初值和迭代预算下未收敛。

## 模块与协作

| 模块 | 集成位置 | 说明 |
|---|---|---|
| A 集成 | `src/main.py`、`scripts/verify_all.py` | 统一命令、验收与CI |
| B 模型场景 | `universal_robots_ur5e/`、`scenes/pick_place.xml` | 单份网格资源、场景及许可 |
| C 正运动学 | `src/kinematics/fk.py` | `fk_flange` 明确法兰；`fk_tcp` 显式接工具变换 |
| D 逆运动学 | `src/kinematics/ik.py` | DLS、限位/步长约束、公共 `IKResult` 适配 |
| E 轨迹 | `src/planning/` | 五次关节轨迹、笛卡尔直线、七阶段元数据 |
| F 控制 | `src/control/controller.py` | 跟踪、吸附/释放、日志及视频 |
| G 分析 | `src/analysis/` | G原有分析函数及当前集成分析入口 |
| H 展示 | `docs/` | 可引用验收报告；最终成员姓名、技术说明书与5–10页PPT仍需整理 |

[模型与坐标](docs/ROBOT_MODEL.md) · [冻结接口](docs/INTERFACES.md) · [AI使用记录](docs/AI_USAGE.md) · [原始交付与历史数据说明](upstream_refs/README.md)

旧版 G 脚本和会断言旧故障的测试已保留为历史参考，不混入当前验收；原始结果与当前复跑数据分开。组员后续从最新 `main` 建功能分支，通过 PR 提交，公共数据契约保持兼容。
