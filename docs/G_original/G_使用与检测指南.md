# G 组使用与检测指南（傻瓜式）

> 一句话：**不装环境也能看图和表；想自己验，一条命令。**
> 本文覆盖 G 组的两个阶段：
> * 第一阶段（G1~G4）：用**占位 DH 模型**验证分析算法本身；
> * 第二阶段（G5~G8）：接上 **B 的真实 MuJoCo 模型 + C/D/F 的真实交付 + F 的真实执行日志**。
>
> **写进报告的是第二阶段的结论。** 第一阶段保留，因为它是"算法对不对"的自证。

---

## 0. 先看这一节就够了

| 问题 | 答案 | 证据位置 |
|---|---|---|
| G 的代码写完了吗？ | 除"必须等 E 的轨迹"以外的部分**全部写完并跑通** | 第 2、3 节 |
| 产物多少？ | **44 个 CSV/JSON + 26 张图 + 47 项测试** | `results/`、`tests/` |
| 需要 E 吗？ | **不需要**。F 日志里的 `q_cmd` 就是规划侧轨迹，与 E 的轨迹格式兼容 | 3.6 节 |
| 数值可信吗？ | 8 类上游一致性检查全过（C 的 FK、D 的 IK、F 的日志、MuJoCo 的 Jacobian） | `G5_validation_checks.csv` |
| 最值得讲的三条结论？ | ① 奇异集是低维曲面不是散点；② 数值 IK 失败 ≠ 几何不可达；③ F 日志 `ee_*` 比 `q_actual` 滞后一个采样 | 第 4 节、`G_结论说明.md` |
| 验完要多久？ | `--quick` 约 10 秒；全量约 3~5 分钟 | 第 2 节 |

---

## 1. G 的文件都在哪（各自干什么）

```
<仓库根>/
├─ src/analysis/                  ← G 的全部实现（不碰别人的文件）
│  ├─ linalg.py                   旋转误差、SO(3) 对数映射、数值几何 Jacobian、法兰→TCP 换算
│  ├─ metrics.py                  路径长度/直线度/速度/加速度/包络/跟踪误差（一阶段）
│  ├─ singularity.py              奇异值、条件数、可操作度、轨迹 σmin、可达性分类
│  ├─ workspace.py                关节空间采样（拉丁超立方）、半径分布、AABB、体素
│  ├─ results_io.py               写 CSV/JSON（UTF-8 BOM 约定 + 读回）、git commit 记录
│  ├─ plots.py                    全部绘图（中文字体 + ASCII 兜底 + 脚注 + 分段底色）
│  ├─ reference_model.py          一阶段的占位 UR5e DH 模型（现在只作对照）
│  ├─ mujoco_backend.py           【二阶段】真实模型封装：FK/批量 FK/Jacobian/限位/TCP 偏移
│  ├─ exec_log.py                 【二阶段】读 F 的 exec_log（27 列 / 24921 行 / 13 种段名 / 39 段）
│  ├─ tracking.py                 【二阶段】按段统计误差、路径长度、Jacobian 一阶分解、滞后检测
│  └─ validation.py               【二阶段】与 C/D/F 交付逐项对照（8 类检查）
├─ src/scene.py                   场景常量（统一测试场景、采样种子、阈值）——G 维护，下游只读
├─ src/kinematics/fk.py           C 组交付（原样，未改）
├─ src/kinematics/ik.py           D/F 组交付（只改模型路径解析，见 3.5）
├─ scripts/
│  ├─ G0_verify_all.py            一键验收（第 2 节）
│  ├─ G1_trajectory_metrics.py    两种轨迹的指标对比 + 奇异点
│  ├─ G2_workspace_reachability.py 占位模型工作空间采样
│  ├─ G3_singularity_reachability.py 占位模型奇异切片 + 可达性案例
│  ├─ G4_interface_smoke.py       接口复用性冒烟（换 FK、换 TCP 都不改分析代码）
│  ├─ G5_model_validation.py     【二阶段】上游一致性验证（最该先看的表）
│  ├─ G6_real_singularity.py     【二阶段】真实模型 σmin 分布 + 奇异集结构
│  ├─ G7_exec_tracking.py        【二阶段】真实执行日志的跟踪误差与路径指标
│  └─ G8_real_reachability.py    【二阶段】真实模型可达范围 + 逐案分类
├─ tests/
│  ├─ test_contracts.py          A 的契约测试（5 项，G 不动）
│  ├─ test_g_analysis.py         一阶段 17 项（纯 NumPy，可直接 python 跑）
│  └─ test_g_real_data.py        二阶段 14 项（需 mujoco + 真实模型/日志，缺文件自动 skip）
├─ docs/G_*.md                   G 的文档（本文件、结论、指标口径、计划、接口说明）
├─ results/csv/                  44 个 CSV/JSON
├─ results/figures/              26 张 PNG
├─ results/logs/                 F 的原始日志（G 只读不改）
└─ universal_robots_ur5e/        B 的模型（G 只读）
```

---

## 2. 三条命令（照抄即可）

### 2.1 只想确认"能不能交"（约 10 秒）

```powershell
cd <仓库根>
python scripts/G0_verify_all.py --quick
```

看最后一行：`结论：全部通过，可以交给人检。` 就说明文件齐、语法对、测试过。

### 2.2 完整重跑（3~5 分钟）

```powershell
python scripts/G0_verify_all.py
```

依次跑 G1~G9、pytest 全量、清点 44 个 CSV/JSON 与 26 张图；任一步失败会打印最后 12 行日志。

### 2.3 只跑测试

```powershell
python -m pytest tests -q                       # 36 项（契约 5 + 一阶段 17 + 二阶段 14）
python -m pytest tests/test_g_real_data.py -q   # 只看第二阶段 14 项
```

> 没装 pytest 也能跑：`python tests/test_g_analysis.py`（17 项）、`python tests/test_g_real_data.py`（14 项）。

### 2.4 环境（只在你自己电脑上需要）

```powershell
python -m pip install -r requirements.txt   # numpy scipy mujoco matplotlib pytest
```

* **Python 3.11 / 3.12** 都验证过（本次 3.12.14）。
* MuJoCo 仿真**不需要显示器**，纯无头可跑。
* Windows 上若 matplotlib 报 `mkdir ... 拒绝访问`：它想往用户目录写缓存，G 的代码已自动重定向到 `results/.mplconfig`；手动跑别人脚本时先设 `$env:MPLCONFIGDIR="<仓库根>\results\.mplconfig"`。
* 一定用 `python -m src.main` 这种**模块方式**，不要 `python src/main.py`（后者报 `No module named 'src'`）。

---

## 3. 每个脚本在算什么、怎么读

### 3.1 G5：上游一致性验证（**最该先看**）

`results/csv/G5_validation_checks.csv`，25 行，每行 `check / item / metric / value / unit / threshold / passed`。

| 检查 | 结论 | 数字 |
|---|---|---|
| C 的 DH FK vs MuJoCo 法兰 | ✅ 一致 | 位置最大 **1.1367 mm**、均值 0.8962 mm；姿态最大 **2.66e-14°** |
| 法兰 ↔ TCP 换算 | ✅ 刚体 | 位置残差 0.0 m、姿态 6.85e-16 rad |
| 数值 Jacobian vs `mj_jacSite` | ✅ 一致 | eps=1e-6 时逐元素最大差 **2.37e-10** |
| D 的 IK 复现（seed 42，10 组） | ✅ 一致 | 10/10 成功；位置 9.51e-5 m、姿态 7.59e-4 rad；与 D 交付 CSV 逐行差 **0** |
| D 的 IK 阻尼敏感性 | ⚠️ 选型依据 | λ=0.005 → 10/10、最差 0.0074 mm；λ=0.08 → 9/10、最差 **29.6 mm** |
| F 日志位姿滞后 | ✅ 已定量 | **-1 个采样**（-2 ms）；对齐后残差 0.110 mm，不对齐 1.261 mm |

> **两条必须知道的坑**（不知道就会得出错误结论）：
> 1. **C 的 `fk(q)` 返回法兰 `wrist_3_link`，不是 TCP**。两者差 100.7 mm / 90°。做笛卡尔分析必须用 D 的 `get_fk`（`attachment_site`）。
> 2. **F 日志的 `ee_*` 列比 `q_actual` 慢一个采样**（F 先 `mj_step` 再读 site）。所以 G 的末端误差一律用 `fk(q_actual)` 回代，`ee_*` 只作交叉核对。

### 3.2 G6：真实模型的奇异点

`G6_sigma_distribution.csv`、`G6_singular_config_stats.csv`、`G6_singular_manifold_evidence.csv`、`G6_trajectory_sigma.csv`

* 20 万构型上 `σmin(J)`：中位 0.0727、1% 分位 4.97e-4、低于阈值 0.02 的占 **20.18%**。
* 奇异集不是"散点"而是**低维曲面**：腕部奇异 = `wrist_2 = 0`（q5=0）；肘部奇异 = `elbow_joint ∈ {0, ±π}`。
  * 证据：对应二维切片里取值为 0 的格点**整行/整列**出现（363/14641 = 3×121），与另一个关节无关。
* 与搬运轨迹（F 的 24921 个指令点）对照：`σmin` 最小 **0.2222**、条件数最大 **8.31**、**0/24921** 低于阈值 ⇒ 这条搬运轨迹不经过奇异区。

### 3.3 G7：真实执行日志的跟踪误差

`G7_segment_metrics.csv`（39 段 × 60 列）、`G7_overall.csv`、`G7_cycle_summary.csv`、`G7_series.csv`

| 指标 | 数值 |
|---|---|
| 关节跟踪误差（实际−指令） | 最大 **0.1468°**、RMS **0.0349°** |
| 末端位置跟踪误差（`fk(q_actual)` vs `fk(q_cmd)`） | 最大 **2.1656 mm**、均值 0.4350 mm、RMS 0.5950 mm |
| 末端姿态跟踪误差 | 最大 **0.3381°** |
| 单循环末端路径长度 | 规划 2.623825 m → 实际 2.623528 m（**−0.0113%**） |
| 单循环关节路径长度 | 规划 8.090606 rad → 实际 8.095754 rad（+0.064%） |
| `transfer` 段直线度 | 规划 0.999997785 / 实际 0.999947178 |
| 三个循环是否一致 | 完全一致（确定性仿真，差值 0） |
| 误差最大的段 | `descend_settle`（关节 0.1468°、末端 2.17 mm）——下探末端急停的 0.3 s 误差最尖 |
| Jacobian 一阶解释能力 | 残差最大 0.0050 mm、R² ≥ 0.9987 ⇒ 误差就是"指令与实际关节差经 Jacobian 放大" |

### 3.4 G8：真实模型的可达范围

`G8_reach_stats.csv`、`G8_max_min_reach.csv`、`G8_reach_cases.csv`、`G8_radial_histogram.csv`、`G8_voxel_summary.csv`

* 伸展要量到**肩部枢轴**（q=0 时在 `(0, −0.138, 0.163)` m，随基座旋转），不是量到基座原点——否则"基座正下方"会被误判可达。
* 到肩部枢轴：最大 **0.9584 m**（采样 0.9583 → 局部优化精修），最小 **≈0**（TCP 能折回枢轴附近）。
* 到基座原点：最大 **1.1296 m**；AABB 边长 (1.9194, 1.9197, 1.9106) m、对角线 **3.3196 m**；0.05 m 体素占用 28742 个（体积上界 3.5928 m³）。
* **12 个案例**：可达 7 / 几何不可达 2 / 数值 IK 失败 3。
  * 几何不可达：1.25 倍最大伸展（到枢轴最近也要 1.0725 m）、基座下方 1.30 m（最近 1.4630 m）；
  * 数值失败：0.95 倍边界、基座正上方 `[0.02,0,0.50]`、基座顶部近点——在几何可达范围内，但 D 的 IK 从 6 个初值都不收敛，**不能说"不可达"**。
* 与第一阶段不同的发现：真实 UR5e 的 TCP 能折回肩部枢轴附近（最小伸展≈0），所以**没有"内腔"**；占位模型用"最小半径"当内腔边界，对真实模型不成立。

### 3.5 G1~G4（第一阶段，保留但不再作为报告主证据）

| 脚本 | 一句话 | 关键数字 |
|---|---|---|
| G1 | 关节五次多项式 vs 笛卡尔直线+IK | 末端偏离起终点直线 58.6 mm vs **6.2e-7 m**；直线度 0.9728 vs 0.9999999996 |
| G2 | 占位模型工作空间 | 半径 max 1.1257 m、AABB 对角线 3.2919 m |
| G3 | 占位模型奇异切片 | 腕部/肘部切片 σmin 最小 ~1e-15 / 1e-14；关节空间 20.8% 低于阈值 |
| G4 | 接口复用性 | 换 FK、平移 TCP 后分析代码零改动；闭环 IK 6 次、残差 9.6e-7 m |

### 3.6 为什么"不需要 E"

E 组的交付物是**规划侧的关节轨迹**（`time_s` + `q_traj[N,6]`）。G 需要的全部输入只有三个：

1. `fk(q) -> 4×4`（B/C 已提供）；
2. 一条"规划轨迹"（`time_s` + `q_rad`）；
3. 一条"实际轨迹"（`q_actual`，F 已提供）。

F 日志里的 `q_cmd` 就是 F 用临时 `traj_adapter` 生成的规划侧轨迹，**格式与 E 的 `JointTrajectory` 完全兼容**（`time_s` 一维、`q_rad` 形状 (N,6)、t 从 0 起严格递增）。因此：

* G 的全部结论已经在"真实规划轨迹 + 真实执行日志"上算完；
* E 交付后，只需把 F 日志的 `q_cmd` 换成 E 的轨迹（或让 F 重跑日志），**G 的分析代码一行都不用改**；
* 唯一需要重跑的是 G7（日志相关），其余不受影响。

---

## 4. 26 张图怎么看

| 图 | 看什么 | 结论 |
|---|---|---|
| `G1_joint_profiles.png` | 五次多项式的 q/q̇/q̈ | 端点速度加速度为 0，曲线光滑 |
| `G1_trajectory_compare.png` | 两种轨迹的末端三维路径 | 笛卡尔是直线，关节插值是弧线 |
| `G1_singularity_profile.png` | 沿轨迹 σmin 与条件数 | 全程远离奇异（min 0.1482） |
| `G1_sigma_distribution.png` | 关节空间 σmin 分布 | 长尾，20.8% 低于阈值 |
| `G2_workspace.png` | 占位模型工作空间三视图 | 近似球壳，中间不均匀 |
| `G3_singularity_maps.png` | 腕部/肘部切片热图 | 暗线=奇异曲面，说明是低维流形 |
| `G3_sigma_distribution.png` | 采样 σmin 直方图 | 接近 0 的样本很多 |
| `G5_validation_summary.png` | 上游一致性检查 | 8 类全过 |
| `G5_ik_damping_sweep.png` | IK 阻尼 λ 与收敛/误差 | λ 太小不稳、太大不收敛，0.005 最好 |
| `G6_sigma_distribution.png` | 真实模型 σmin 分布 | 20.18% 低于阈值 |
| `G6_singularity_maps.png` | 真实模型腕部/肘部/肩部切片 | 暗线位置与解析结论一致 |
| `G6_singular_config_histograms.png` | 奇异构型的关节角分布 | 集中在 q5=0 / q3∈{0,±π} |
| `G7_error_by_segment.png` | 39 段分段底色 + 关节/末端误差 | 峰值在 `descend_settle` |
| `G7_ee_tracking.png` | 规划 vs 实际末端轨迹 + 误差 | 实际几乎贴在规划上 |
| `G7_lag_demo.png` | 滞后 0 与 −1 采样的残差对比 | 差约 11 倍，证明必须对齐 |
| `G7_path_lengths.png` | 路径长度累积 | 实际略长，差 ~0.1% |
| `G7_sigma_vs_tracking.png` | σmin 与跟踪误差相关性 | 弱相关（轨迹不经过奇异区） |
| `G7_ee_vs_joint_error.png` | 关节误差→末端误差放大 | 与 Jacobian 预测吻合（R²≥0.9987） |
| `G8_workspace_views.png` | 真实模型工作空间三视图 | 近似球壳、前上方更密 |
| `G8_radial_reach.png` | 半径直方图 + 伸展边界 | 边界 1.130 m 与采样一致 |
| `G8_workspace_compare.png` | 真实 vs 占位模型 | 尺度接近（1.1296 vs 1.1257 m） |

---

## 5. 检测时核对这 16 个数（对不上就是有问题）

| # | 文件 | 看哪一格 | 应该是 |
|---|---|---|---|
| 1 | `G5_validation_checks.csv` | `passed` 列（共 25 行） | 17 行 `True`；8 行 `False` **全部是"已知并已解释"的**：`c_fk_tcp` 2 行（坐标系陷阱，不是失败）、`d_ik_limited` 1 行 + `d_ik_damping` 4 行（λ 选型依据）、`f_log_residual` 的 naive 对照 1 行 |
| 2 | `G5_validation_checks.csv` | `c_fk` 的 `pos_max_mm` | `1.1367` |
| 3 | `G5_validation_checks.csv` | `f_log_pose_lag` 的 `lag_samples` | `-1` |
| 4 | `G6_trajectory_sigma.csv` | `sigma_min` 最小值 | `0.222230` 左右 |
| 5 | `G7_overall.csv` | `joint_err_max_deg` | `0.1468` |
| 6 | `G7_overall.csv` | `ee_pos_err_max_m` | `0.0021656` m |
| 7 | `G8_max_min_reach.csv` | `radius_arm_max_refined_m` | `0.958431` |
| 8 | `G8_reach_cases.csv` | `status` 统计 | 可达 7 / 几何不可达 2 / 数值失败 3 |
| 9 | `G7_series.csv` | 行数 | `24921` |
| 10 | `results/figures/` | PNG 数量 | `21` |

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | 直接跑了 `python src/main.py` 或不在仓库根 | 用 `python -m src.main`；在仓库根执行 |
| `mkdir ... matplotlib ... 拒绝访问` | matplotlib 想写 `%LOCALAPPDATA%\matplotlib` | G 代码已自动重定向；手动跑时设 `MPLCONFIGDIR` |
| `找不到 ur5e.xml` | B 的模型没放到位 | 确认 `<仓库根>/universal_robots_ur5e/ur5e.xml` 存在，或设 `$env:G_UR5E_XML` |
| `找不到 exec_log_latest.csv` | F 的日志没放到位 | 放到 `results/logs/exec_log_latest.csv`，或设 `$env:G_EXEC_LOG` |
| 第二阶段测试全部 skip | 上面两个文件缺一个 | 同上；skip 是设计行为，不让整个测试集挂掉 |
| 中文变方块 / `Glyph ... missing` | 字体缺字 | 代码已做 ASCII 兜底（`q̇→q_dot` 等）；仍异常就给 `plots._CJK_CANDIDATES` 加一个本机有的中文字体 |
| CSV 第一列读出来是 `\ufefflabel` | 文件是 UTF-8 BOM（让 Excel 不乱码） | 用 `results_io.read_csv_rows()` 读，或 `encoding="utf-8-sig"` |
| G8 报 `OptimizeWarning: Unknown solver options` | SciPy 新版弃用 Nelder-Mead 的 `xatol` | 已按 method 分别给选项；若仍出现检查 SciPy 版本 |

---

## 7. 换模型 / 换日志怎么跑

```powershell
# 换模型（B 组冻结 ROBOT_MODEL.md 后）
$env:G_UR5E_XML = "D:\models\ur5e\ur5e.xml"
python scripts/G5_model_validation.py
python scripts/G6_real_singularity.py
python scripts/G8_real_reachability.py

# 换执行日志（E/F 重新交付后）
$env:G_EXEC_LOG = "D:\logs\exec_log_20261101.csv"
python scripts/G7_exec_tracking.py
```

G 只依赖两个约定（`fk(q)->4×4` 和 `time_s/q_rad`），所以换模型、换日志**都不需要改分析代码**。

---

## 8. 交给组长合并

1. **先本地人工检查**：跑 `python scripts/G0_verify_all.py`，确认全绿；
2. 在 `feat/G-analysis` 分支上按清单复制：
   * `src/analysis/*.py`（11 个文件）
   * `src/scene.py`
   * `src/kinematics/fk.py`、`src/kinematics/ik.py`（**C/D/F 的交付，合并前确认与上游一致**）
   * `scripts/G0_verify_all.py`、`scripts/G1~G8_*.py`
   * `tests/test_g_analysis.py`、`tests/test_g_real_data.py`
   * `conftest.py`
   * `docs/G_*.md`
   * `results/csv/`、`results/figures/`（**`results/.mplconfig/` 不要提交**）
   * `universal_robots_ur5e/`（B 的模型，若已合过就别覆盖）
   * `upstream_refs/`（C/D 原始交付副本，G5 对照用）
3. **不要覆盖** `src/contracts.py`、`src/main.py`、`README.md`、`docs/INTERFACES.md`（A 的）；
4. 合并后在新目录再跑 `python -m pytest tests -q`，应为 **36 passed**；
5. 由你本人 push 并开 PR。**G 从未向 GitHub 推送任何内容**，也没在任何目录执行过 `git init`。

---

## 9. 课程要求 → G 的交付对照（自证表）

| 课程/组长要求 | G 的交付 | 在哪里 |
|---|---|---|
| 计算/调用 Jacobian | 数值几何 Jacobian（体/世界系可选）+ `mj_jacSite` 对照，误差 2.4e-10 | `linalg.py`、`mujoco_backend.jacobian_tcp` |
| 分析奇异值或条件数 | σmin/σmax/条件数/可操作度/各向同性；20 万构型分布；切片热图；奇异集结构证据 | `singularity.py`、G3/G6 |
| 采样生成可达性分布 | 拉丁超立方 20 万采样、半径直方图、AABB、体素、三视图 | `workspace.py`、G2/G8 |
| 统计规划与实际执行的末端误差 | 按段统计末端位置/姿态误差、关节误差（真实日志 39 段） | `tracking.py`、G7 |
| 路径长度 | 关节/笛卡尔路径长度、超出量、直线度、偏离弦最大距离（规划 vs 实际） | `metrics.py`、`tracking.py` |
| 速度/加速度指标 | 关节速度/加速度、末端线速度（规划 vs 实际，峰值与 RMS） | 同上 |
| 比较实验结果 | 占位 vs 真实模型；两种轨迹；不同 IK 阻尼；三循环一致性 | G1/G5/G8 |
| ≥3 张清晰图表 | **26 张**，全部标注场景、种子、单位 | `results/figures/` |
| 注明采样条件与单位 | 每个 CSV 的 `meta.json` 记录 seed/N/模型 sha256/时间戳；图表脚注写单位与场景 | `results/csv/G*_meta.json` |
| **区分数值 IK 失败与数学不可达** | 12 个案例分三类；几何判据用肩部伸展而非基座半径；每个失败案例给 `d_min/d_max` 与多初值结论 | `G8_reach_cases.csv`、`singularity.classify_reachability` |
| 不得虚构结果 | 所有数字由脚本生成；无法测量的明确写"等 E/F 交付后重跑" | `G_结论说明.md` 第 5 节 |

---

# 第三阶段（E 组已交付）：G9

> E 组交付了整条搬运轨迹 `pick_place_traj.npz`（6694 点、dt=0.002 s、13.386 s）。
> G 组**没有改 E、也没改 F 的一行代码**，只新增一个脚本 `scripts/G9_e_trajectory.py`
> 做两件事：**验收 E 的规划**、**把它放进 F 的控制器里真跑一遍看会发生什么**。

## 3.7 G9 在算什么

分四块，每块都有对应的 CSV：

| 块 | 问题 | 产物 |
|---|---|---|
| ① 数据契约与几何 | E 的轨迹格式对不对？起末是不是 home？是否越限位？关键高度对不对？ | `G9_e_checks.csv` |
| ② 逐段规划指标 | 7 段各自多长、多直、峰值速度多大？"报告时长"和"实际时长"差多少？ | `G9_e_segments.csv` |
| ③ 沿轨迹的奇异值 | 这条规划会不会擦到奇异点？ | `G9_e_sigma.csv` |
| ④ E 规划 vs F 内置规划 | 同一个任务，两套规划器差在哪？ | `G9_e_compare.csv` |
| ⑤ E→F 集成复跑 | 用 F 的控制器执行 E 的轨迹，跟踪误差多少？接口对得上吗？ | `G9_e_exec_segments.csv`、`G9_e_exec_summary.csv` |

**分段为什么要重建**：E 的 7 段都是 rest-to-rest（段边界关节速度为 0），所以只用数据找边界只能准到 ±0.1 s；
而且 E 的分段时长会因 vmax/amax 自动缩放（比如 transfer 声明 3.0 s，实际 3.34 s）。
G 的做法是：用 E 自己的轨迹模块副本（`upstream_refs/e_plan/`）**重建一次规划**，拿到权威的段边界与
IK 失败点数，同时核验重建出来的总时长与 E 交付的轨迹**逐位相等**（13.386 s），确保没重错。

## 3.8 G9 怎么跑

```bash
python scripts/G9_e_trajectory.py            # 全量（约 40 秒，其中重建 E 的规划 12 秒）
python scripts/G9_e_trajectory.py --no-rebuild   # 跳过重建，用静止点检测分段
python scripts/G9_e_trajectory.py --no-figs      # 只出 CSV
python -m pytest tests/test_g_e_traj.py -q        # 11 项回归测试
```

## 3.9 第三阶段要记住的 3 个结论

**结论 A：E 的规划本身是合格的。**
数据契约全过（6694 点、dt 均匀到 1e-15、起末严格等于 home、不越限位）；
峰值关节速度 0.9996 rad/s（自限 1.0）、峰值加速度 2.384 rad/s²（自限 3.0）；
关键高度全部对上（抓取 0.0551 / 抬升 0.3799 / 放置 0.0651 / 退让 0.1199 m）；
transfer 是严格直线（直线度 0.99994，偏离弦线 0.0796 mm，高度起伏 8.3e-05 m）；
**IK 不可解点 0 个**；沿全程 σmin 最小值 0.2223（t=0 的 home 位），最大条件数 8.31，
**0/6694 点低于阈值** ⇒ 这条轨迹离奇异很远。

**结论 B：E 报告的"请求时长"不等于轨迹实际时长，报告里不能混用。**

| 段 | 报告时长 (s) | 实际时长 (s) | 比值 |
|---|---|---|---|
| home→hover | 2.000 | 2.000 | 1.000 |
| descend | 1.600 | 1.600 | 1.000 |
| lift | 1.600 | 1.646 | 1.029 |
| transfer | 3.000 | 3.340 | 1.113 |
| lower | 1.600 | 1.600 | 1.000 |
| retreat | 1.200 | 1.200 | 1.000 |
| return→home | 2.000 | 2.000 | 1.000 |
| **合计** | **13.000** | **13.386** | **1.030** |

原因是 E 为了让峰值关节速度不超过 1.0 rad/s，对超速的段做了时间缩放。**引用时长时必须写"实际 13.386 s"**。

**结论 C（最重要）：E 的轨迹与 F 的 `--traj-file` 接口语义不匹配，直接接会跳 26.58°。**
E 交付的是**一整条 pick-and-place**；而 F 的 `--traj-file` 只把它当作 **transfer 段**的替代
（`controller.py` 里 `q = self._run_traj(e_traj, "transfer_e")`），于是机器人还在 lift 位形时，
下一条指令突然变成 E 轨迹的**第 0 点（home）**，J1 一次性跳 0.464 rad（26.58°）。
之后 F 又接着从 home 做 lower/retreat/return，整条链路语义都错位了。

G 的实测数字（`G9_e_exec_summary.csv`）：

| 指标 | 数值 | 说明 |
|---|---|---|
| 接管跳变 | 0.4640 rad = **26.58°**（J1，t=7.108 s） | 接口语义不匹配的直接证据 |
| 跳变后恢复 0.5° 以内 | **0.806 s** | 执行器只是被"闪"了一下 |
| 稳态关节跟踪误差 | max **0.4968°** / p95 **0.1141°** | 剔除瞬态后，与 F 内置规划同量级 |
| F 内置规划器基线 | 0.1468°（末端 2.1656 mm / 0.3381°） | 对照 |
| 全程最大误差 | 26.55° | 全部来自接管瞬间，不是轨迹质量 |

**所以：这不是 E 的规划问题，也不是 F 的控制器问题，是两模块之间的接口约定问题。**
G 只负责测量与定位，修法建议见 `docs/G_结论说明.md` 结论十（三种改法，任选其一，都不需要 G 改代码）。

## 3.10 第三阶段新增的核对数字（接第 5 节）

| # | 文件 | 看哪一格 | 应该是 |
|---|---|---|---|
| 11 | `G9_e_checks.csv` | `descend_target_z` 的 value | `0.05507340408` m |
| 12 | `G9_e_segments.csv` | `transfer` 行 `straightness` | `0.99993825` |
| 13 | 同上 | `transfer` 行 `duration_s` | `3.340`（`reported_duration_s` 是 `3.0`） |
| 14 | `G9_e_compare.csv` | `transfer` 行 `E_tcp_path_m` / `F_tcp_path_m` | `0.699944` / `0.699999` |
| 15 | `G9_e_exec_summary.csv` | `handover_jump_max_deg` | `26.583` |
| 16 | 同上 | `transfer_e_settled_joint_err_max_deg` | `0.4967544` |

## 3.11 第三阶段的 5 张图

| 图 | 看什么 |
|---|---|
| `G9_e_joint_profiles.png` | 6 个关节角/速度/加速度，按 7 段着色：每段首末速度归零、transfer 段是梯形速度；第三行密集条纹是 E 的 ddq 里由逐点 IK 残差放大出的局部抖动（峰值 2.384 rad/s² 仍可信，做前馈前要先低通） |
| `G9_e_tcp_path.png` | TCP 3D 路径 + XZ 投影上的 5 条关键高度线（hover/grasp/lift/place/clear） |
| `G9_e_segment_straightness.png` | 7 段直线度：5 个笛卡尔段≈1.0，home→hover 与 return→home 明显弯曲（关节空间运动） |
| `G9_e_sigma.png` | σmin 与条件数沿时间：全程 >0.22，远离阈值 |
| `G9_e_exec_jump.png` | J1 指令/实际曲线 + 接管点竖线 + 下图的误差尖峰与 0.8 s 衰减 |
