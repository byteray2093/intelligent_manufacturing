# G 组（奇异点 / 可达性 / 误差分析）实现方案与第一周交付

> 角色来源：课程文档《智能制造导论》题目 11 的 **G｜奇异点、可达性与误差分析**；
> 流程图中的两个阶段：第一阶段 **G 分析**（本次范围）、第二阶段 **G 实验分析**（接入 F 实测数据后）。
> 框架来源：<https://github.com/Marco6-3/intelligent_manufacturing>（A 组长 Day-0 骨架）。
> 本包内所有数字都由本包脚本实际跑出，命令见第 8 节。

---

## 1. 任务边界：G 到底要交什么

课程原文（G 行）：

- **任务**：计算/调用 Jacobian，分析奇异值或条件数；采样生成可达性分布；统计规划与实际执行的
  末端误差、路径长度、速度/加速度等指标；比较实验结果。
- **交付**：`src/analysis/` 下分析脚本、`results/figures/` 图表、CSV 数据和结论说明。
- **验收**：至少 3 张清晰图表；注明采样条件与单位；**区分数值 IK 失败和数学上不可达**。

仓库 `docs/TASKS.md` 对 G 的最小交付写得更紧：`src/analysis/metrics.py` + 实验图，
"明确方法、单位和采样种子，分析 IK 失败原因"。`docs/PLAN.md` 给 G 的第一周任务是
**"G 设分析定义"**（Gate 0：每人有最小运行示例），第二周才是 **"G 接入日志"**。

因此本次交付对应 **第一阶段（分析：定义+离线分析）**：

| 本阶段做 | 本阶段不做（留给第二阶段） |
|---|---|
| 指标定义、单位、阈值、CSV/图管线 | 把 F 组的 MuJoCo 实测日志接进来出最终对比图 |
| Jacobian（数值微分）与奇异点分析 | 与 C 组解析 Jacobian 的交叉验证（等 C 交代码） |
| 关节空间采样 + 可达性分布（占位 FK） | 用 B 组真实 MJCF 的关节限位/命名重跑 |
| 两种轨迹的方案对比（参考实现） | 用 E 组真实 `plan_joint/plan_cartesian` 重跑 |
| 数值 IK 失败案例与"几何不可达"分类 | D 组真实 IK 的失败率统计 |
| 3 张以上图表 + 可复现元数据 | 报告/PPT 中 G 章节的定稿 |

---

## 2. 依赖关系（谁能立刻开工，谁必须等）

> **接口原则：G 组零新增公共接口。** 不修改 `src/contracts.py`、`docs/INTERFACES.md`、
> `src/main.py`；只新增 `src/analysis/` 与 `scripts/G*.py`，消费上游已有的四类契约。
> 需要的一切都可以由 `fk(q) -> T` + `JointTrajectory` + `IKResult` + `ExecutionLog` 推出
> （Jacobian 用数值微分自算、指标与包络由 G 自己定义）。详见
> `docs/G_INTERFACE_PROPOSAL.md`。

| 依赖项 | 来源 | 现在能否拿到 | G 组的处理 | 是否需要新接口 |
|---|---|---|---|---|
| 公共数据契约 `Pose/IKResult/JointTrajectory/ExecutionLog` | A 组，已有 | ✅ 已发布 | 直接按字段对齐；本包 `src/contracts_ref.py` 是字段一致的副本 | ❌ 只消费 |
| `fk(q) -> T` | C 组 | ⏳ 未交付 | 用 `src/models/ur5e_reference.py`（参考 DH）占位，接口一致，C 交付后一行替换 | ❌ 已登记 |
| `ik(T,q0) -> IKResult` | D 组 | ⏳ 未交付 | 用 `numerical_ik_dls`（DLS 参考实现）占位，只用于产生对照/失败样本 | ❌ 已登记 |
| 关节轨迹 `time_s / q_rad` | E 组 | ⏳ 未交付 | 用 `src/planning_reference/`（五次多项式 + 笛卡尔直线）占位 | ❌ 已登记 |
| 真实关节顺序/限位/TCP | B 组 MJCF + `docs/ROBOT_MODEL.md` | ⏳ 未冻结 | 参考 DH 参数 + UR5e 手册限位；B 冻结后重跑（见第 6 节） | ❌ 文档约定 |
| 实测 `q_actual` 日志 | F 组 | ⏳ 第二周 | `ExecutionLog` 三列即可算跟踪误差；末端误差用 `fk(q_measured)` 回代兜底 | ❌ 只消费 |
| 图表/报告章节 | H 组 | 并行 | G 组提供图 + 指标表 + 结论说明，H 直接引用 | ❌ 不涉及 |

关键判断：**G 组的第一阶段完全不依赖 B/C/D/E/F 的交付**（这正是 PLAN 里"上游未交付时可用
mock 数据"的意思）。本包就是把这句写成了能跑的代码。

---

## 3. 实现方式（技术路线）

### 3.1 指标层：三层误差分开算

```
规划层（不依赖仿真）     : q(t)/q̇(t)/q̈(t)、末端弧长、直线度、端点零速
        ↓  用 C 的 FK
末端层（几何）          : TCP 位置/姿态误差、末端偏离直线、工作空间包络
        ↓  接 F 的实测
执行层（MuJoCo）        : 关节跟踪 RMSE/峰值、末端跟踪误差、成功率
```

三层口径的唯一定义见 `docs/G_METRICS_SPEC.md`。**不允许只报一个"误差 3 mm"**。

### 3.2 Jacobian 与奇异点

- 中心差分数值微分：`J[i] = (T(q+h e_i) − T(q−h e_i)) / 2h`，**角速度行用体坐标增量**
  `so3_log(I, R(q−h)ᵀR(q+h))`。
- 分解：`σ1 ≥ … ≥ σ6`；输出 `σ_min`、条件数 `σ1/σ6`、Yoshikawa 可操作度 `∏σi`、各向同性度 `σ6/σ1`、数值秩。
- 主判据取 `σ_min`，工程阈值 0.02（无量纲，全局唯一）。
- 支持接入 MuJoCo：`jacobian_at_tcp` 解决 `mj_jac` 的 body 原点 → TCP 换算。

### 3.3 可达性

- 采样：关节空间 Latin hypercube，`q_i ∈ [−π, π]`，N = 40 000，seed = 20261015，体素 0.05 m。
- 判定分四类（`reachable / reachable_but_imprecise / numerical_failure / unreachable_geometry`），
  外加"未提供 IK"的 `unknown`，禁止把 `numerical_failure` 写成"不可达"。

### 3.4 两种轨迹方案对比

| 方案 | 优点 | 代价（本包实测值，参考模型、3 s、200 Hz） |
|---|---|---|
| 关节空间五次多项式 | 端点零速零加速度；关节运动平稳连续 | 末端走曲线：弦长 0.5700 m，实际弧长 0.5860 m，直线度 0.973，最大偏离直线 58.6 mm |
| 笛卡尔直线 + 逐点 IK | 末端直线：直线度 1.000，最大偏离 6.2e-7 m | 需要 601 次 IK 全部收敛；**端点速度不连续**（直线插值端点非零速）；IK 失败即路径断裂 |

这张表就是验收要的"两种轨迹方案能比较"，而且每一条都有 CSV 支撑。

---

## 4. 时间安排（对齐仓库 `docs/PLAN.md` 的 Gate）

| 时间 | G 组任务 | 产出 | 对应 Gate |
|---|---|---|---|
| 10/15–10/17 | **分析定义**（本包）：指标/单位/阈值/CSV/图管线；占位 FK 跑通；给 A 提接口建议 | 本包全部内容 | Gate 0：最小运行示例 ✅ |
| 10/18–10/21 | 换 C 的 FK、B 的限位重跑；与 C 的解析 Jacobian 交叉验证；把 3 张图定稿 | 图 + 交叉验证表 | Gate 1 |
| 10/22–10/25 | **接入 F 日志**：跟踪误差、末端误差、成功率；确认统一测试场景 | 执行层指标 CSV/图 | Gate 2（目标位姿→轨迹→执行链路） |
| 10/26–10/28 | 搬运 Demo 的误差分析；失败案例归因（初值/奇异/限位/接触） | 失败案例表 + 归因结论 | Gate 3（可复现数据） |
| 10/29–11/01 | 奇异点与可达性成图；两种轨迹对比定稿；性能指标汇总 | 4–6 张终稿图 | Gate 4（结果可解释） |
| 11/02–11/04 | 交给 H：图、CSV、结论说明；跨机器复现检查 | G 章节材料 | Gate 5 |

---

## 5. 交付物清单（本包文件 → 仓库位置）

| 本包路径 | 合并到仓库的位置 | 说明 |
|---|---|---|
| `src/analysis/linalg.py` | 同名 | 位姿/旋转误差、数值 Jacobian、TCP Jacobian 换算、DH 变换 |
| `src/analysis/metrics.py` | 同名 | 全部指标定义（唯一口径） |
| `src/analysis/singularity.py` | 同名（TASKS 要求 `metrics.py`，奇异点单独成文件更清晰） | 奇异值、条件数、可操作度、可达性分类 |
| `src/analysis/workspace.py` | 同名 | 关节空间采样、半径统计、体素占用、切片 |
| `src/analysis/results_io.py` | 同名 | CSV/元数据输出（含 git commit、时间戳、种子） |
| `src/analysis/plots.py` | 同名 | 统一图表风格；matplotlib 延迟导入 |
| `src/models/ur5e_reference.py` | 建议放 `src/analysis/reference_model.py`（避免与 B 组模型目录冲突） | 参考 DH 模型（占位） |
| `src/planning_reference/reference_trajectory.py` | 不入库（E 组交付后删除） | 占位轨迹 + DLS IK 参考实现 |
| `src/contracts_ref.py` | 不入库 | 契约字段副本，合并后改用 `src.contracts` |
| `scripts/G1_trajectory_metrics.py` | `scripts/` 或 `src/analysis/run_g1.py` | 两种轨迹指标对比 + 奇异点沿轨迹 |
| `scripts/G2_workspace_reachability.py` | 同上 | 工作空间采样与可达性分布 |
| `scripts/G3_singularity_reachability.py` | 同上 | 奇异点切片定位 + 失败案例分类 |
| `scripts/G4_interface_smoke.py` | 同上 | 接口复用性冒烟检查（换 FK 指标随之变化） |
| `tests/test_g_analysis.py` | `tests/test_g_analysis.py` | 17 项数值检查（见第 7 节） |
| `docs/G_METRICS_SPEC.md` | `docs/` | 指标与单位规范（唯一口径） |
| `docs/G_INTERFACE_PROPOSAL.md` | `docs/`（或并入 `INTERFACES.md`） | 给 A 组的接口补充建议 |
| `docs/G_结论说明.md` | `docs/` | 每张图的结论、异常与限制（供 H 组直接用） |
| `results/figures/G*.png`、`results/csv/G*.csv` | `results/` | 复现产物 |

分支名按 `CONTRIBUTING.md`：`feat/G-analysis`。

---

## 6. 必须提醒 A/B 组的三件事

1. **参考模型与真实模型的差异会导致 G 的结论整体平移**。本包用的参考 DH 表（ur_description 风格）
   零位 TCP 半径 0.8521 m；MuJoCo Menagerie 的 UR5e 约定下第 2 轴方向相反、零位半径约
   0.8394 m。两者都会让"最大伸展、工作空间包络、奇异点位置"的**具体数值**变化。
   所以 `docs/ROBOT_MODEL.md` 一冻结，G 组必须用真实模型重跑全部脚本，并在图上改模型来源标注。
2. **G 组不修改任何公共文件**：`src/contracts.py`、`docs/INTERFACES.md`、`README.md`、
   `src/main.py` 全部保持原样；G 只新增 `src/analysis/` 与 `scripts/`，**零新增接口**。
   唯一希望 A 组考虑的是一句**文字约定**（角速度用体坐标，见 `docs/G_INTERFACE_PROPOSAL.md`
   第 4 节），是否收录完全由 A 组决定，G 的代码不依赖该决定。
3. **统一测试场景必须与 E/F 一致**（本包 `src/scene.py`，scene_id = `G-stage1-scene-v1`），
   否则 E/F/G 的数字无法横向比较。这是常量约定，不是接口改动。

---

## 7. 自证：数值与接口检查（`python -m pytest tests/test_g_analysis.py -q`，16/16 通过）

| 检查 | 结果 |
|---|---|
| 数值 Jacobian vs 平面 3R 解析 Jacobian | 最大偏差 1.44e-10（阈值 1e-7） |
| UR5e Jacobian 预测实际位姿增量 | 线速度 9.9e-11 m、角速度 4.4e-11 rad |
| `so3_log` 在 30°/90°/179.9°/180° | 角度与轴均精确 |
| 五次多项式端点速度 | 3.11e-5 rad/s（单侧差分分辨率量级） |
| 路径长度/直线度/最大垂距 vs 手算 | 完全一致（1e-12） |
| 包络 AABB（单轴 1 m / 三轴 1 m） | 对角线 1.000000 m / √3 = 1.732051 m |
| `poses_from_joint_trajectory` vs 逐点 `fk` | 完全一致（唯一的上游几何依赖） |
| 奇异点指标（单位阵 / 秩亏 / 病态） | σmin、条件数、可操作度、秩均符合预期 |
| 可达性五分类 | 全部正确 |
| 采样可复现 + 落在限位内 | 通过 |
| UR5e 腕部奇异复现 | q5 = 0 处 det(J) = 1.3e-44，σmin ≈ 1e-15 |

脚本 `G4` 另做接口复用性检查：把 TCP 定义平移 0.1 m，起点绝对位置随之变化 +0.100000 m，
而路径长度保持不变（刚体平移不变量）——证明指标没有硬编码。

---

## 8. 运行方式

```bash
# 环境（与仓库 requirements.txt 一致）
python -m venv .venv
.venv\Scripts\activate            # Windows
python -m pip install -r requirements.txt

# 把本包内容按第 5 节表格合并进仓库后：
python -m pytest tests -q                        # 47 项检查（G 的 42 项 + 契约 5 项）
python -m pytest tests/test_g_analysis.py -q     # 只跑 G 的 17 项数值检查
python scripts/G1_trajectory_metrics.py        # 两种轨迹指标对比 + 奇异点沿轨迹
python scripts/G2_workspace_reachability.py    # 工作空间采样（--n --seed --method）
python scripts/G3_singularity_reachability.py  # 奇异点切片 + 失败案例（--grid 81）
python scripts/G4_interface_smoke.py           # 接口复用性冒烟检查

# 只出 CSV、不画图（无 matplotlib 的环境）
python scripts/G1_trajectory_metrics.py --no-figs
```

产物：`results/csv/G*.csv`、`results/csv/G*_meta.json`、`results/figures/G*.png`。

---

## 9. 验收对照表

| 课程验收要求 | 本包证据 |
|---|---|
| 至少 3 张清晰图表 | 计划 6 张：`G1_trajectory_compare`、`G1_joint_profiles`、`G1_singularity_profile`、`G2_workspace`、`G3_singularity_maps`、`G3_sigma_distribution`（脚本已生成数据，图在本机装好 matplotlib 后一键出图） |
| 注明采样条件与单位 | 每张图的图注 + `docs/G_METRICS_SPEC.md` 第 5 节 + `G*_meta.json` 的 seed/N/method/voxel |
| 区分数值 IK 失败与数学上不可达 | `classify_reachability` 的 4+1 类判定 + `G3_reachability_cases.csv` 逐案例的判据与 σmin 证据 |
| 明确的误差指标定义 | `docs/G_METRICS_SPEC.md` 三层误差定义 + `metrics.py` 单一实现 |
| 数据可复现 | 固定 seed、命令行参数、`G*_meta.json`（UTC 时间、git commit、版本、脚本名） |
| 有失败案例及原因 | `G3_reachability_cases.csv`：超伸展（几何）、基座正上方（几何+数值）、腕部奇异 q5=0（可达但病态） |

---

## 10. 风险与降级方案（对齐 `docs/PLAN.md` 的风险章节）

| 风险 | 触发信号 | 降级方案 |
|---|---|---|
| B 组模型迟迟不冻结 | 第 1 周结束仍无 `ROBOT_MODEL.md` 数值 | 用参考 DH 模型出全部图，图注明确标注"参考模型"，结论只作趋势性结论 |
| C 组 FK 与本包参考 FK 不一致 | 同一 q 的 TCP 位置差 > 1 cm | 以 B 的 MJCF 为仲裁，先解决零位/关节方向约定，再重跑 |
| D 组 IK 失败率高 | 可达目标回代误差 > 1e-4 m 的比例高 | G 组用 DLS 参考实现做对照基线，报告写清"自研 IK 与基线的差异"，不掩盖 |
| F 组日志缺末端位姿 | 只有 q_actual | 用 `fk(q_measured)` 兜底算末端误差，并在图中注明"末端误差由 FK 回代得到" |
| 奇异点结论被质疑"只在某个切片成立" | 有人拿单张 (q1,q2) 图说"没有奇异" | 用 `G3_singularity_maps.png` 的两张横截切片 + 采样统计（20.8% 的采样构型 σmin < 0.02）说明奇异集是低维曲面 |
| matplotlib 缺失 | 脚本打印"跳过画图" | CSV/结论不受影响；装依赖后重跑即出图（脚本 --no-figs 也可交差） |
| 提前下"抓取力学已验证"结论 | F 组用附着约束 | G 组只在报告中写"跟踪/末端误差"，不对接触力下任何结论 |

---

# 第二阶段完成情况（G5~G8，真实模型 + 真实日志）

> 触发原因：B/C/D/F 的交付到位、E 仍未交付。G 的结论**不依赖 E**（F 日志的 `q_cmd` 就是规划侧轨迹，
> 与 E 的 `JointTrajectory` 格式兼容），因此先把不依赖 E 的部分全部做完。

## 1. 已完成清单

| 产出 | 文件 | 状态 |
|---|---|---|
| 真实模型封装 | `src/analysis/mujoco_backend.py` | ✅ FK/批量 FK/Jacobian/批量 Jacobian/限位/TCP 偏移/肩部枢轴 |
| 执行日志读取 | `src/analysis/exec_log.py` | ✅ 27 列 / 24921 行 / 13 种段名 / 39 段，含四元数与单调性校验 |
| 按段跟踪分析 | `src/analysis/tracking.py` | ✅ 误差、路径、速度、加速度、Jacobian 一阶分解、滞后检测 |
| 上游一致性验证 | `src/analysis/validation.py` | ✅ 8 类检查（C 的 FK、法兰↔TCP、数值 Jacobian、D 的 IK、D 的 CSV 复现、IK 阻尼扫描、F 日志滞后） |
| 实验脚本 | `scripts/G5..G8_*.py` | ✅ 全部跑通 |
| 绘图 | `src/analysis/plots.py` | ✅ 新增 12 个图函数（分段底色、滞后对比、验证汇总、σmin-误差、工作空间对照…） |
| 测试 | `tests/test_g_real_data.py` | ✅ 14 项（缺模型/日志时自动 skip） |
| 一键验收 | `scripts/G0_verify_all.py` | ✅ 覆盖 G1~G8 + pytest + 产物清点（37 CSV/JSON、21 图） |
| 文档 | `docs/G_使用与检测指南.md`、`G_结论说明.md`、`G_METRICS_SPEC.md` | ✅ 已按第二阶段数字重写 |

## 2. 与第一阶段的差异（必须知道）

* 模型换成 B 的真实 MJCF，伸展/工作空间/奇异统计全部重算：
  最大伸展 **1.1296 m**（到基座原点）/ **0.9584 m**（到肩部枢轴），不是一阶段的 0.90 m 名义值；
* **"内腔"结论推翻**：真实 TCP 能折回肩部枢轴（最小伸展≈0），不存在一阶段假设的不可达内腔；
* 末端误差不再用 `fk(q_measured)` 兜底，而是与日志的 `q_cmd` 逐点对比，得到"规划 vs 实际"的真误差；
* 增加"数值 IK 失败 vs 几何不可达"的三分类实证（12 例）。

## 3. 仍需 E 的部分（明确挂起）

| 项 | 为什么等 E | E 交付后怎么做 |
|---|---|---|
| 用 E 的真实轨迹替换 F 的 `q_cmd` | F 目前用临时 `traj_adapter` 产生规划轨迹 | 让 F 重跑日志 → 重跑 `G7`；G 代码零改动 |
| "E 的轨迹 vs 参考轨迹"对比 | 需要 E 的轨迹 | 新增一个对比脚本即可复用 `metrics.analyze_joint_trajectory` |
| 平滑性/加加速度（jerk）对比 | 需 E 与参考两条轨迹 | `metrics` 已有速度/加速度，jerk 用同一差分逻辑补 3 行 |

**G 不会为了"看起来完整"而伪造 E 的轨迹**：报告中凡涉及 E 的都写"待 E 交付后重跑"。

---

# 第三阶段完成情况（G9，E 的规划轨迹已交付）

## 1. 已完成清单

| 项 | 位置 | 状态 |
|---|---|---|
| E 轨迹加载与数据契约校验 | `src/analysis/e_traj.py` | 完成 |
| 用 E 的模块重建分段（权威边界） | `e_traj.rebuild_plan_segments` | 完成（12.4 s，N=6694 与交付逐位一致） |
| 退化路径：rest-to-rest 静止点检测分段 | `e_traj.detect_rest_boundaries` | 完成（重建失败时自动启用） |
| 逐段规划侧指标（路径/直线度/速度/加速度/关键高度） | `e_traj.per_segment_rows` + `G9_e_segments.csv` | 完成 |
| 沿轨迹逐点奇异值（解析 Jacobian） | `G9_e_sigma.csv` | 完成（6694 点，1.2 s） |
| E 规划 vs F 内置规划同口径对比 | `G9_e_compare.csv` | 完成（7 段 × 13 列） |
| E→F 集成复跑与接口不匹配定位 | `G9_e_exec_summary.csv`、`G9_e_exec_jump.png` | 完成 |
| 第三阶段回归测试 | `tests/test_g_e_traj.py` | 完成（11 项） |
| 派生产物 | 6 CSV + 1 JSON + 5 图 | 完成 |

全量验收：`python -m pytest tests -q` → **47 passed**（契约 5 + G1~G4 17 + 第二阶段 14 + 第三阶段 11）；
`python scripts/G0_verify_all.py` → G1~G9 全绿、44 个 CSV/JSON、26 张图。

## 2. 与第二阶段的差异

| 维度 | 第二阶段（G5~G8） | 第三阶段（G9） |
|---|---|---|
| 轨迹来源 | 自写参考轨迹（G1）+ **F 内置规划器**的真实日志 | **E 交付的规划轨迹**（6694 点整周期） |
| 执行数据 | F 正式交付日志（3 循环、24921 点） | G 触发 F 控制器执行 E 轨迹（1 循环、13500 点） |
| 能回答的问题 | "F 自己的规划执行得怎么样" | "E 规划的质量如何""E 能不能直接塞进 F" |
| 关键结论 | 关节跟踪 0.147°、末端 2.17 mm、全程远离奇异 | E 规划合格（σmin≥0.222）；**接口语义不匹配，直连会跳 26.58°** |

## 3. 仍需上游配合的部分

1. **E→F 接口**：见 `docs/G_结论说明.md` 结论十的三种改法；改完 G 重跑 `G9` 即可得到"无跳变"的对比数据。
2. **E 的段边界速度声明**：E 报 ≤2.5e-3 rad/s，实测 8.58e-3 rad/s；建议 E 复核（不影响可用性）。
3. **F 的 `--traj-file` 文档**：建议在 `docs/给E模块的接口说明.md` 里明确"该参数只替换 transfer 段"。
4. **整周期演示**：若 H 报告需要"一次完整搬运"的视频/图，用 G9 的 `G9_e_tcp_path.png` 或直接让 F/E 各自跑整周期。
