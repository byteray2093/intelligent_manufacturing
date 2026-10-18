# G 组指标定义与单位规范（v1，第一周冻结）

> 本文件是 G 组所有图表、CSV、结论的**唯一口径**。任何人改动指标定义，必须同时
> 修改 `src/analysis/metrics.py`、本文件与 `docs/INTERFACES.md`，并在 PR 说明中指出。
> 生成脚本：`scripts/G1_trajectory_metrics.py`；核心代码：`src/analysis/metrics.py`。

## 0. 全局约定

| 项目 | 约定 |
|---|---|
| 关节向量 q | length-6 `np.float64`，顺序以 B 组 `docs/ROBOT_MODEL.md` 冻结值为准 |
| 位姿 | 4×4 齐次矩阵 `T_world_tcp`，`T[:3,:3] ∈ SO(3)`，末行 `[0,0,0,1]` |
| 位置 / 长度 | m |
| 时间 | s |
| 角度 | rad（图表中若用 deg 必须在轴标签注明） |
| 速度 / 加速度 | rad/s、rad/s²（关节）；m/s（末端线速度） |
| 姿态误差 | rad（测地线夹角），**与位置误差分开报告** |
| 误差符号 | `e = actual - command`（跟踪误差）；`e = target - achieved`（目标误差） |
| 采样频率 | 200 Hz（dt = 0.005 s），统一测试场景 `G-stage1-scene-v1` |
| 数值微分步长 | 中心差分 eps = 1e-6 rad |

## 1. 规划层指标（只需 JointTrajectory）

| 指标 | 定义 | 单位 | 注意事项 |
|---|---|---|---|
| `joint_path_length_rad` | Σ ‖q_{i+1} − q_i‖₂ | rad | 与采样密度有关，报告必须给 dt |
| `joint_peak_velocity_rad_s` | max‖dq/dt‖∞（np.gradient） | rad/s | 不做滤波，抖动会体现在峰值 |
| `joint_peak_acceleration_rad_s2` | max‖d²q/dt²‖∞ | rad/s² | 同上 |
| `cartesian_path_length_m` | Σ ‖p_{i+1} − p_i‖₂（p 为 TCP 位置） | m | 对刚体平移是**不变量** |
| `straightness_ratio` | 弦长 / 弧长 ∈ [0,1] | 无量纲 | 只对"笛卡尔直线"轨迹有评价意义 |
| `max_endpoint_deviation_m` | 各点到起终点直线的最大垂距 | m | 笛卡尔直线路径质量的核心指标 |
| `cartesian_peak_velocity_m_s` | max‖dp/dt‖₂ | m/s | 只算线速度，不含角速度 |
| `start_rest` / `end_rest` | 端点 ‖q̇‖∞ ≤ 1e-3 rad/s | bool | 该 1e-3 是**单侧差分分辨率**，不是物理非零 |
| `workspace_span_m` / `envelope_diagonal_m` | TCP 轨迹的**世界系轴对齐包围盒**（AABB）边长与对角线 | m | 只取决于极值点，与采样密度无关，可跨机器复现 |
| `max_axis_travel_m` | 包围盒最大边长 | m | 用于回答"这次搬运占多大空间" |
| `t_at_max_s` / `t_at_min_s` | 各轴达到极值的时刻 | s | 用于定位轨迹的极值点 |

> `trajectory_envelope_m()` 与 `workspace.box_envelope_m()` 使用**同一套 AABB 定义**，
> 因此"场景轨迹包络"与"工作空间包络"两组数字可以直接对比（单位都是 m）。

## 2. 执行层指标（需要 ExecutionLog）

| 指标 | 定义 | 单位 |
|---|---|---|
| `joint_tracking_rmse_rad` | sqrt(mean((q_meas − q_cmd)²)) | rad |
| `joint_tracking_max_abs_rad` | max‖q_meas − q_cmd‖∞ | rad |
| `joint_tracking_rmse_per_joint_rad` | 逐关节 RMSE | rad |
| TCP 逐点跟踪误差 | 逐采样点 ‖p_meas − p_cmd‖₂、姿态测地线夹角 | m / rad |

**三层误差必须分开写**（INTERFACES.md 明确要求）：
1. **规划误差**：理想轨迹自身与目标的偏差（只与轨迹算法有关）；
2. **执行跟踪误差**：MuJoCo 实测关节角与指令关节角的偏差（只与控制器有关）；
3. **末端误差**：用 FK 回代后的 TCP 位置/姿态与目标的偏差（综合结果）。
不能只报一个"误差 3 mm"而不说清是哪一层。

## 3. 奇异点指标（需要 Jacobian）

对 J(q) ∈ R^{6×6} 做 SVD，σ1 ≥ … ≥ σ6 ≥ 0：

| 指标 | 定义 | 单位 | 说明 |
|---|---|---|---|
| `sigma_min` | σ6 | 同 J 的混合单位 | 越小越接近奇异；**主判据** |
| `condition_number` | σ1/σ6 | 无量纲 | σ6→0 时发散（记为 inf） |
| `numerical_condition` | 1/σ6 | 1/单位 | 避免除零的工程写法 |
| `manipulability_yoshikawa` | ∏σi = |det J| | 混合 | 只用于**同一机器人**的相对比较 |
| `isotropy` | σ6/σ1 ∈ (0,1] | 无量纲 | 1 表示各向同性 |
| `rank` | 数值秩 | 整数 | 阈值 max(m,n)·eps·σ1 |
| `is_singular` | σ6 < 0.02 | bool | **0.02 是工程约定阈值**，无量纲，必须全局一致 |

物理意义：`‖q̇‖ = ‖J⁻¹ v‖ ~ ‖v‖/σ6`。σ6 很小时，末端方向的微小运动需要极大的关节
速度，数值 IK 的迭代矩阵病态、收敛变慢或发散。

## 4. 可达性判定（必须区分三类，不能含糊）

| 状态 | 判据 | 报告口径 |
|---|---|---|
| `unreachable_geometry` | 目标半径 > 最大伸展（或低于最小伸展） | "数学上不可达，与算法无关" |
| `reachable` | IK 收敛且位置误差 ≤ 1e-4 m | "可达，且已求得解" |
| `reachable_but_imprecise` | IK 报成功但误差 > 容差 | "精度不足，需检查收敛判据" |
| `numerical_failure` | 在伸展范围内但 IK 未收敛 | "**数值失败**，必须附 σmin、初值、迭代数证据" |
| `unknown` | 未提供 IK 求解器 | 不允许写成"不可达" |

> 验收明确要求"区分数值 IK 失败和数学上不可达"。**最常见的错误**是把一次 IK 不收敛
> 写成"该点不可达"——这正是 G 组要拦住的问题。

## 5. 工作空间采样口径

| 项目 | 值 | 说明 |
|---|---|---|
| 采样区间 | 每关节 q_i ∈ [−π, π] | 关节是 ±2π 连续旋转关节，全行程会让同一空间区域被重复计数 |
| 采样方法 | Latin hypercube（默认）/ uniform | 两种都要能跑，用于说明结论不依赖具体随机实现 |
| 样本数 | 40 000 | 报告中必须写明 N |
| 随机种子 | 20261015 | 写进 CSV 与图注，换机器可复现 |
| 体素边长 | 0.05 m | 占用体积是**上界估计**，不是精确体积 |
| 半径统计 | min / max / mean / p05 / p50 / p95 | 单位 m |

## 6. 复现要求（每次出图/出数据必带）

1. 命令行（含参数）；
2. 场景 ID `G-stage1-scene-v1` 与全部场景参数；
3. 采样条件：N、seed、方法、区间、体素边长；
4. 模型来源（当前为 UR5e 参考 DH 表，B 组冻结后替换）；
5. `results/csv/G*_meta.json`：UTC 时间戳、git commit、Python/NumPy 版本、脚本名。

## 7. 已知的"坑"清单（写进报告，避免别人重踩）

| 坑 | 现象 | 正确处理 |
|---|---|---|
| 有限差分角速度的坐标系 | 用 `so3_log(R(q−h), R(q+h))` 会得到 **−ω**，数值 IK 朝反方向迭代、永不收敛 | 用体坐标增量 `so3_log(I, R(q−h)ᵀR(q+h))`；见 `src/analysis/linalg.py` |
| 小角度近似 | `0.5(R−Rᵀ)` 在接近 180° 时严重失真，会把"差 180°"判为"已收敛" | 用精确 `so3_log`（θ=π 走四元数分支） |
| IK 误差向量量纲 | 位置(m)与姿态(rad)直接拼接，姿态项会主宰步长，迭代停在厘米级误差 | 姿态项乘权重（默认 0.25 m/rad），并用自适应阻尼 + 线搜索 |
| 关节轨迹的速度约束单位 | `max|Δq|/dt` 与 `q̇_max` 直接比较会差一个 dt 因子 | 统一换算成 rad/s 再比较 |
| 奇异点切片选择 | (q1,q2) 切片 σmin 几乎不变，容易误判"没有奇异" | 奇异集是低维曲面，必须选与奇异面横截的切片（腕部看 q5，肘部看 q3） |
| 路径长度对 TCP 平移不变 | 换 TCP 定义后 `cartesian_path_length_m` 不变 | 检验 TCP 定义要用**绝对位置**，不是路径长度 |
| 体素"体积" | 把体素数 × 体素体积当成工作空间体积 | 只能称"占用体积上界"，且说明体素边长 |

---

# 第二阶段（真实模型 + 真实执行日志）指标口径

> 本节的数字由 `scripts/G5..G8` 生成。与第一节的差别：**输入换成 B 的真实 MJCF 与 F 的真实日志**，
> 指标定义不变，但新增"按搬运段"维度和"规划 vs 实际"的分组。

## 1. 新增指标与定义

| 指标 | 定义 | 单位 | 备注 |
|---|---|---|---|
| 关节跟踪误差 | `q_actual − q_cmd`，逐关节取绝对值后取最大/RMS | deg | 指令来自 F 的 `q_cmd`（E 交付后只换这一列） |
| 末端位置跟踪误差 | `‖fk(q_actual) − fk(q_cmd)‖` | mm | **不用日志的 `ee_*` 列**，见 §2 |
| 末端姿态跟踪误差 | `rotation_error_rad(fk(q_actual), fk(q_cmd))` | deg | 旋转矩阵对数映射，精确 |
| 规划/实际末端路径长度 | `Σ‖p[k+1] − p[k]‖`，`p = fk(q)[:3,3]` | m | 与实际路径同一函数、同一 dt |
| 路径超出量 | 实际 − 规划 | m / % | 可能为负（见结论五） |
| 直线度 | `‖p_end − p_start‖ / Σ‖Δp‖` | 无量纲 | 1 = 完美直线 |
| 偏离起终点弦最大距离 | 逐点到线段距离取最大 | mm | 比"直线度"更直观 |
| Jacobian 一阶预测误差 | `J(q_cmd)·(q_actual − q_cmd)` 与 `fk(q_actual) − fk(q_cmd)` 的残差 | mm | 衡量"误差是否纯运动学" |
| 一阶解释度 R² | 逐段对上述两组位移回归 | 无量纲 | 接近 1 ⇒ 误差由 Δq 经 J 放大 |
| 位姿滞后 | 使残差最小的 `ee_*` 与 `fk(q_actual)` 采样偏移 | 采样 / ms | F 日志实测 **-1 采样（-2 ms）** |
| 伸展（到肩部枢轴） | `‖p_TCP − p_shoulder_pivot‖`，枢轴随基座旋转 | m | 判"几何可达"必须用这个 |
| 伸展（到基座原点） | `‖p_TCP‖` | m | 只是习惯叫法，**不是**判可达用的量 |

## 2. 第二阶段新增踩坑清单

| 现象 | 根因 | 处理 |
|---|---|---|
| 末端误差比预期大一个数量级 | F 日志的 `ee_*` 比 `q_actual` **慢一个采样**（先 `mj_step` 后读 `site_xpos`） | 一律用 `fk(q_actual)` 回代；`ee_*` 仅交叉核对，出 `G7_lag_demo.png` 留证 |
| 把 C 的 `fk` 当 TCP 用，笛卡尔误差差 100 mm / 90° | C 的 `fk` 输出**法兰** `wrist_3_link`，TCP 是 `attachment_site` | 笛卡尔分析用 D 的 `get_fk` 或 `fk_flange @ T_flange_tcp` |
| 可达性把"基座正下方"误判可达 | 用"到基座原点距离"当伸展，而手臂从**肩部枢轴**出发（q=0 时 z=0.163 m） | 伸展量到枢轴（`UR5eModel.reach_center()`），逐例给 `d_min/d_max_shoulder_m` |
| "20% 构型接近奇异"被误读 | 采样区间用模型 `jnt_range`（±2π），比电机实际工作角宽 | 报告写"这是**采样区间内**的比例"，并给出区间定义 |
| `OptimizeWarning: Unknown solver options: xatol` | 新版 SciPy 对 `Powell` 不再接受 `xatol/fatol` | 按 method 分别给选项（Nelder-Mead 用 `xatol/fatol`，Powell 用 `xtol/ftol`） |
| 局部优化把关节推出限位后 FK 跳变 | 优化器在无约束空间里跑 | 前后都做 `((q+π) mod 2π) − π` 归一，并用 `jnt_range` 截断 |

---

# 第三阶段（E 轨迹接入）指标口径

> 对应 `src/analysis/e_traj.py` 与 `scripts/G9_e_trajectory.py`。第三阶段的指标全部围绕
> "E 交付的规划轨迹"与"F 执行这条轨迹"，不引入新的公共接口。

## 1. E 轨迹数据契约

| 指标 | 定义 | 单位 | 阈值 |
|---|---|---|---|
| `n_samples` | `time.size` | count | E 声明 6694 |
| `dt_uniform_max_dev` | `max abs(diff(time) - dt)` | s | ≤1e-9 |
| `t0_s` | `time[0]` | s | 0 |
| `start_home_max_dev` / `final_home_max_dev` | `max abs(q[0] - home)`、`max abs(q[-1] - home)` | rad | ≤1e-3 |
| `joint_limit_violation_max` | `max(max(lo-q,0), max(q-hi,0))` | rad | ≤1e-9 |
| `peak_joint_speed` | `max abs(dq_traj)` | rad/s | ≤1.0（E 自限） |
| `peak_joint_accel` | `max abs(ddq_traj)` | rad/s² | ≤3.0（E 自限） |
| `dq_max_abs_diff` | `max abs(dq_traj - gradient(q))` | rad/s | <0.05（平滑尾迹） |
| `ddq_max_abs_diff` | `max abs(ddq_traj - gradient(dq_fd))` | rad/s² | <3.0（同上） |
| `boundary_speed_max` | 各段边界处 `max abs(dq_traj)` | rad/s | <5e-3（实测 8.58e-3，见结论说明） |

## 2. 分段与规划侧指标

**分段依据**：用 E 自己的轨迹模块（`upstream_refs/e_plan/`，原样副本）重建一次规划，
取它的 `segment_names` / 各段时长 / IK 失败点数；只有当**重建总时长与 E 交付的轨迹时长逐位相等**
（13.386 s）时才采用，否则退化为"rest-to-rest 静止点检测"并在 `G9_meta.json` 的 `segmentation`
字段注明来源（`rebuild(...)` 或 `rest-detect`）。

**"报告时长"与"实际时长"必须并列**：E 的 `report.md` 给的是**请求时长**（2.0/1.6/1.6/3.0/1.6/1.2/2.0=13.0 s），
实际轨迹因 vmax/amax 自动缩放为 13.386 s；`G9_e_segments.csv` 同时给 `reported_duration_s` 与
`duration_ratio_vs_report`，**任何引用都必须写明是哪一种**。

逐段指标（与第一/第二阶段同定义）：`tcp_path_m`、`tcp_chord_m`、`straightness`（弦长/路径长）、
`max_dev_from_chord_m`（距弦线最大垂距）、`joint_path_rad`、`joint_speed_max_rad_s`、
`joint_accel_max_rad_s2`、`tcp_z_start_m`/`tcp_z_end_m`、`tcp_x_start_m`/`tcp_x_end_m`。

## 3. E→F 集成实验的指标（瞬态 vs 稳态必须分开）

| 指标 | 定义 | 单位 |
|---|---|---|
| `handover_jump_max_rad` / `_deg` | 接管点处 `max abs(q_cmd[i0] - q_cmd[i0-1])` | rad / deg |
| `handover_recover_to_0p5deg_s` | 从接管点起，关节误差首次 <0.5° 的时间 | s |
| `handover_spike_n_samples_over_1deg` | 接管后在 `transfer_e` 段内误差 >1° 的采样点数 | count |
| `transfer_e_settled_joint_err_max_deg` | 跳过瞬态后 `transfer_e` 段内关节误差最大值 | deg |
| `transfer_e_settled_joint_err_p95_deg` | 同上，95 分位 | deg |
| `baseline_*` | F 内置规划器（`exec_log_latest.csv` 第 1 循环）同口径对照 | deg / mm |

**为什么要分瞬态/稳态**：直接报"E→F 集成最大误差 26.55°"会误导——它 100% 来自接管瞬间的指令跳变；
稳态 0.497° 才代表执行器跟踪 E 规划的真实能力。两个数都要报，并解释差异来源。

## 4. 第三阶段新增踩坑

1. **不要用 `segment_names` 的原始内部名直接对外**：E 模块内部叫 `approach_up->grasp_up` 等，
   报告里用 E 文档的 7 段名（`home->hover`/`descend`/...）；G9 已做映射，原文保留在
   `G9_meta.json:segmentation_rebuild_names`。
2. **numpy 2.x 删除了 `ndarray.ptp()`**：求极差必须用 `np.ptp(arr)`，否则 `AttributeError`。
3. **`--traj-file` 的语义是"替换 transfer 段"，不是"执行整条轨迹"**：这是 E→F 接口不匹配的根因。
4. **E 的 csv 副本与 npz 等价**：`upstream_refs/e_traj/pick_place_traj.csv` 只用于人工核对，
   代码只读 npz（csv 1.2 MB，读起来慢且精度按十进制截断）。
