# G 组｜接口方案：**不需要新增任何公共接口**（推荐，已按此实现）

> 结论先说：G 组第一阶段**不改 `src/contracts.py`、不改 `docs/INTERFACES.md`、不要求 A 组新增
> 任何接口**。G 只**消费**仓库里已经发布的四类契约，其余全部在 `src/analysis/` 里自闭环。
> 本文件记录这个决定、它需要上游满足的最低条件、以及（万一将来真的卡住时的）可选备选方案。

## 1. G 组只消费已有契约，不新增

| G 用到的上游能力 | 仓库里已有的定义 | G 怎么用 | 是否需要新接口 |
|---|---|---|---|
| 末端位姿 | `src/contracts.py::Pose`（4×4 齐次矩阵） | G 内部全用 `np.ndarray(4,4)`，字段与 `Pose` 同一校验标准 | ❌ 不需要 |
| 关节轨迹 | `JointTrajectory(time_s, q_rad)` | G 只读 `traj.time_s` / `traj.q_rad` 两个字段 | ❌ 不需要 |
| IK 结果 | `IKResult(q, success, position_error_m, orientation_error_rad, iterations, message)` | G 的失败案例表直接读这些字段 | ❌ 不需要 |
| 实测执行记录 | `ExecutionLog(time_s, q_command_rad, q_measured_rad)` + `joint_rmse_rad` | G 用前三个字段算跟踪误差；末端误差用 `fk(q_measured)` 回代 | ❌ 不需要 |
| 正运动学 | C 组 `fk(q) -> T (4,4)`（INTERFACES.md 已登记） | G 唯一的几何依赖，用于末端路径/包络/误差 | ❌ 已存在 |
| 逆运动学 | D 组 `ik(T_target, q_init) -> IKResult`（已登记） | G 的可达性判定直接调用 | ❌ 已存在 |
| 关节限位/命名 | B 组 `docs/ROBOT_MODEL.md`（文档约定，非代码接口） | 读文档填 `src/scene.py` 里的限位常量 | ❌ 属于文档，不是新接口 |
| Jacobian | **仓库里没有** | **G 自己算**：中心差分数值微分（`src/analysis/linalg.py`） | ❌ 不需要（G 组内部函数，不进公共层） |
| 误差/包络指标 | **仓库里没有** | **G 自己算**：`src/analysis/metrics.py` | ❌ 不需要（G 组交付物就是它） |

一句话：**G 的输出分析能力就是"契约的消费者"，不是"契约的扩写者"。**

## 2. G 为什么可以零新增接口

1. **Jacobian 不必是公共接口**。`INTERFACES.md` 已经把"数值微分"这条路留给了使用者——
   只要 C 组给 `fk(q) -> T`，任何人都能在自己的模块里做差分。G 的实现已用手算解析
   Jacobian 验证到 1e-10 量级（见 `tests/run_checks.py`），不依赖 MuJoCo，因此
   在没装 MuJoCo 的机器上也能跑，也不需要向 A 组申请新接口。
2. **末端误差不必新增日志字段**。`ExecutionLog` 已有 `q_measured_rad`，G 用 C 组的
   `fk(q_measured)` 回代即可得到实测末端位姿。代价是引入一次 FK 调用（O(N)，本项目
   N = 601，纯 NumPy 约毫秒级），完全可接受。
3. **工作空间/包络指标是 G 的交付物**，本来就该由 G 定义与实现（TASKS.md 明确
   "G 的误差指标须明确定义"），定义在 G 的代码与文档里，而不是塞进公共契约。
4. **统一测试场景是文档级约定**，可以写成 `src/scene.py` 里的常量 + `docs/` 说明，
   不需要成为 `contracts.py` 的数据类。

## 3. 因此 G 对上游只有"两条最低要求"

| 要求 | 上游 | 若暂时没有 | G 的兜底 |
|---|---|---|---|
| `fk(q) -> T (4,4)`，q 顺序与 B 的模型一致 | C 组 | 未交付 | `src/models/ur5e_reference.py`（参考 DH，占位） |
| 轨迹 `time_s (N,)` + `q_rad (N,6)` | E 组 | 未交付 | `src/planning_reference/`（五次多项式 + 笛卡尔直线） |

其余（B 的限位、D 的 IK、F 的日志）G 都有占位或兜底，不影响第一阶段的任何交付。
这就是 `docs/PLAN.md` 说的"上游未交付时可用 mock 数据"的具体落地。

## 4. 唯一的"软约定"（文档同步即可，不改代码）

G 组建议在 `docs/INTERFACES.md` 里**补一句约定**（不改任何数据类）：

> 凡涉及 Jacobian 的讨论，统一采用：**线速度=世界系、角速度=体坐标、参考点=TCP**；
> 数值微分公式为 `J_ω[:,i] = so3_log(I, R(q−h e_i)ᵀ R(q+h e_i)) / 2h`。

理由：角速度的坐标系约定是这类项目最容易出错的地方（G 组实际踩过：写成
`so3_log(R(q−h), R(q+h))` 会得到 −ω，导致数值 IK 朝反方向迭代、永不收敛）。
写在文档里只需要一句话，但能省掉 C/D/G 三方各自解释的成本。

这只是**文字约定**，不是接口，A 组若认为不必写也无妨——G 组的实现已按此固定，
并在 `docs/G_METRICS_SPEC.md` 第 7 节记录了原因。

## 5. 若将来真的卡住，才考虑的备选（当前**不需要**，仅备案）

| 场景 | 备选做法 | 影响面 |
|---|---|---|
| F 的 `ExecutionLog` 将来丢了 `q_measured_rad` | 请 F 用 `fk(q_measured)` 自己写一份 `T_measured` CSV 给 G | 只影响 F 的导出脚本，不改公共契约 |
| C 的 FK 在本机装不起来（纯 MuJoCo 依赖） | G 继续用参考 DH 做离线分析，并标注"离线参考模型"，实测部分等 C | 影响"交叉验证"这一条，不影响主交付 |
| 想用 MuJoCo 的解析 Jacobian 与数值法交叉验证 | G 侧写一个适配函数读 `mj_jac`，放在 G 的代码里（`jacobian_at_tcp` 已实现换算） | 纯 G 内部，A 组无需介入 |
| 需要把 G 的指标接进 `src/main.py` 主流程 | 那时才由 A 组在 `INTERFACES.md` 登记一行入口函数 | 只有这一步才涉及公共层 |

## 6. 给 A 组的一句话

**G 组申请的是"零新增接口"**：不修改 `contracts.py` / `INTERFACES.md` / `main.py`，
只新增 `src/analysis/` 目录与 `scripts/G*.py`，消费 C 的 `fk`、D 的 `ik`、E 的轨迹、
F 的日志。若您认可，G 组按 `docs/G_ANALYSIS_PLAN.md` 第 5 节的清单直接开 PR；
第 4 节的角速度约定那**一句话**是否补进 `INTERFACES.md`，由您决定。

---

# 第二阶段补充：与 C/D/F 的接口事实（**重要，建议 A 组登记进 `INTERFACES.md`**）

实际接入 C/D/F 的交付后，发现三件**不是"新接口"、但会让下游算错**的事实，需要写进接口文档：

| # | 事实 | 影响 | 建议写法（一句话） |
|---|---|---|---|
| 1 | **C 的 `fk(q)` 返回法兰 `wrist_3_link`，不是 TCP**；TCP 是 MJCF 里的 `attachment_site`，相对法兰偏移 `(0, 0.1, 0)` 且姿态旋转 90° | 直接把 C 的 FK 当末端位姿，笛卡尔误差会差 **100.7 mm / 90°** | "`fk(q)` 返回法兰位姿；末端位姿 = `fk(q) @ T_flange_tcp`，`T_flange_tcp` 见 `docs/ROBOT_MODEL.md`" |
| 2 | **F 的 `exec_log` 里 `ee_*` 列比 `q_actual` 滞后 1 个采样（-2 ms）** | 不用 `q_actual` 回代、直接拿 `ee_*` 算跟踪误差，会高估约 11 倍（1.261 mm vs 0.110 mm） | "末端误差请用 `fk(q_actual)` 回代；`ee_*` 仅供可视化" |
| 3 | **Jacobian 的角速度行存在两套约定**：MuJoCo `mj_jacSite` 给的是世界系角速度；对体坐标增量做数值差分得到的是体坐标角速度（两者差左乘 `R`） | 混用会让数值 IK 朝反方向迭代（G 一阶段踩过，`-ω` 导致永不收敛） | "角速度统一用**世界系**；数值实现见 `src/analysis/linalg.geometric_jacobian` 的 `frame` 参数" |

以上三条**都不需要新增函数或字段**，只是把已有交付的真实语义写清楚。
G 侧已自行处理（`mujoco_backend` 提供 `fk_flange` / `fk_tcp` 两个入口，`validation.check_c_fk` 会顺带报"法兰当 TCP 用会差多少"）。

## 第二阶段结论：**仍然零新增接口**

G 只消费：`fk(q) -> (4,4)`（C/B）、`ik(T, q0) -> (q, ok, info)`（D，形态与 `IKResult` 兼容）、
`exec_log` 的 `t / cycle / segment / q_cmd / q_actual` 列（F）、MJCF 模型文件（B）。
G 产出的仍是 CSV/JSON/PNG 文件，**没有向任何共享模块要求新字段**。
