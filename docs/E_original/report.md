# E 模块交付说明（对齐 F 接口 · 第二版）

任务 E：轨迹生成与算法对比。本版按 F 模块**详细接口说明**逐条对齐并复测，
端到端可直接 `--traj-file` 接入 F。

## 0. 与上一版的差异（按 F 最新说明修正）

| 项 | 上一版 | 本版（F 口径） |
|---|---|---|
| 采样 dt | 0.01 s | **0.002 s**（= F 仿真步长 500 Hz，不触发重采样，导数最精确） |
| 接近高度 | z=0.38 | **HOVER z=0.30**（安全过渡） |
| 放置点 | z=0.055 | **LOWER z=0.065** |
| 抓取 / 抬升 / 退让 | 0.055 / 0.38 / 0.12 | 0.055 / 0.38 / 0.12（不变） |
| 段时长 | 各 2–3 s 自动缩放 | 按 F 参照：approach 2.0 / descend 1.6 / lift 1.6 / transfer 3.0 / lower 1.6 / retreat 1.2 / return 2.0 |
| 边界 IK | ik_dls | **ik_dls_limited**（限位裁剪 + 步长上限，F 推荐） |
| 笛卡尔求导 | 直接 np.gradient | **Savitzky-Golay 平滑后求导**（抑制 IK 噪声在 dt 小时的放大） |

## 1. 坐标系（F 强调的坑，务必注意）

- C 的 `fk(q)` 返回**法兰 `wrist_3_link`**；D 的 `get_fk(q)` 返回 **TCP `attachment_site`**，
  相差位置 ~100 mm + 姿态 90°。
- 本模块**全程以 TCP 为参考点**：笛卡尔直线逐点调用 `get_fk/ik_dls_limited`，与 D 自洽。
- 若用 C 的 fk 做规划/误差对比会凭空多出约 10 cm 偏差。

## 2. 运动学后端

| 后端 | 说明 |
|---|---|
| `src/kinematics/ik.py`（D 原版 + 路径修复 + `ik_dls_limited`） | MuJoCo + ur5e.xml，权威 TCP |
| `kinematics.RobotModel`（MuJoCo-free，含 `ik_dls_limited`） | 与 MuJoCo 逐点交叉验证 ~1e-15，无 MuJoCo 环境回退 |

启动自动优先 import D 的 `get_fk/ik_dls/ik_dls_limited`；笛卡尔逐点 IK 优先用
`ik_dls_limited`（限位裁剪 + 步长上限 0.05 rad/步）。`traj.backend` 可查。

## 3. 轨迹数据文件格式（F 的 `load_e_trajectory()` 直接加载）

`data/pick_place_traj.npz`：

```
time[N]      shape (N,)，单位 s，从 0 开始单调递增，dt=0.002 均匀
q_traj[N,6]  shape (N,6)，单位 rad，UR5e 关节顺序 J1..J6
dq_traj[N,6] 关节速度 rad/s（前馈）
ddq_traj[N,6]关节加速度 rad/s²（前馈）
```

- **dt=0.002 = F 仿真步长** → F 逐点原样执行，不触发重采样，时间尺度与导数最精确。
- 已自检：`dt 均匀`、`time[-1]=(N-1)*0.002`、`无 NaN`、`起末=home`。
- 同时导出 `pick_place_traj.csv`（同列）。

## 4. 搬运任务与关键点（F 接口口径，基坐标，单位 m）

归位 home（keyframe）= `[-1.5708,-1.5708,1.5708,-1.5708,-1.5708,0]`；
抓取姿态 `R_DOWN`：TCP 朝正下（z 轴指 −Z）。

| 阶段 | 路径 | TCP (x, y, z) | 时长 s |
|---|---|---|---|
| home → 接近(HOVER) | 关节空间五次 | 到 (−0.35, 0.40, **0.30**) | 2.0 |
| descend | 笛卡尔直线(下) | (−0.35, 0.40, 0.055) | 1.6 |
| lift | 笛卡尔直线(上) | (−0.35, 0.40, 0.38) | 1.6 |
| **transfer** | 笛卡尔直线(水平) | (0.35, 0.40, 0.38) | 3.0 |
| lower | 笛卡尔直线(下) | (0.35, 0.40, **0.065**) | 1.6 |
| retreat | 笛卡尔直线(上) | (0.35, 0.40, 0.12) | 1.2 |
| return → home | 关节空间五次 | 回到 home | 2.0 |

关键高度已实测命中：HOVER 0.300 / grasp 0.055 / LIFT 0.380 / transfer 0.380 / LOWER 0.065 / CLEAR 0.120。

## 5. 结果与约束（自检全部通过）

| 项 | 值 |
|---|---|
| 总时长 / 点数 | 13.39 s / 6694 点（dt=0.002） |
| 最大关节速度 | ≤ 1.0 rad/s（OK） |
| 最大关节加速度 | ≤ 3.0 rad/s²（max 2.38，OK） |
| 关节限位 | 全程在限位内（J3 范围 [1.571, 2.081]，限 ±3.1415）✓ |
| 段间速度跳变 | ≤ 2.5e-3 rad/s（≈0 → C1 拼接） |
| transfer 段 | 0 个 IK 不可解点；TCP 恒 z=0.38，x −0.35→0.35 |
| 后端 | D.ik.py (MuJoCo+ur5e.xml) |

## 6. 连续性 / 优缺点 / 不可解点（任务 E 验收）

- **关节空间五次多项式**：端点速度/加速度为 0，解析连续（C²），平滑、天然规避奇异；
  末端路径形状未知（会弯）。
- **笛卡尔直线**：TCP 严格走直线，利于直线焊缝/切割/插孔；路径参数用五次曲线（段端静止），
  与相邻段 C1 拼接；每点一次 IK，需逐点可达性与奇异检查。
- **不可解点**：① 目标超出工作空间（UR5e 最大半径 ≈0.85 m）→ IK 无解；
  ② 可达但奇异（肘伸直、腕部 q5≈0）→ 雅可比亏秩，IK 不收敛。
  用 `ik_dls_limited`（限位裁剪 + 步长上限）可在边界点/奇异附近稳定求解。

## 7. 复现

```bash
python run_pick_place.py      # 重新生成 npz / csv / 曲线 / 汇总
```

## 8. 交付文件

- `src/planning/trajectory.py`  轨迹生成（QuinticJointSegment / CartesianLineSegment / PickAndPlaceTrajectory / plan_pick_and_place）
- `src/planning/kinematics.py`  独立 UR5e+TCP 运动学（含 `ik_dls_limited`）
- `src/kinematics/ik.py`        D 原版 + 路径修复 + `ik_dls_limited`
- `src/kinematics/ur5e_nomesh.xml`  纯运动学 UR5e 模型
- `run_pick_place.py`           一键复现
- `data/pick_place_traj.npz` / `.csv`   给 F 的轨迹（dt=0.002）
- `data/pick_place_summary.json`
- `plots/`                      关节曲线、TCP 路径、TCP 高度时序
