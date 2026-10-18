# G 组交付说明｜奇异点 / 可达性 / 误差分析（三阶段全量）

这是 G 组（角色 G：分析）的完整可运行交付，按依赖推进分三个阶段：

- **第一阶段**（G1~G4）：分析与指标算法，只需要 NumPy/SciPy；
- **第二阶段**（G5~G8）：接入 B 的真实 MuJoCo 模型、C 的 FK、D 的 IK、F 的真实执行日志；
- **第三阶段**（G9）：接入 E 的规划轨迹，做 E 轨迹验证与 E→F 集成实验。

装了 `requirements.txt` 里的 matplotlib 后，所有脚本会自动补齐图表（共 26 张）。

## 30 秒看懂

| 想知道什么 | 看哪个文件 |
|---|---|
| **怎么跑、怎么验、怎么提交**（傻瓜式一步步） | [docs/G_使用与检测指南.md](docs/G_使用与检测指南.md) |
| 我负责什么、交付什么、时间怎么排、风险在哪 | [docs/G_ANALYSIS_PLAN.md](docs/G_ANALYSIS_PLAN.md) |
| 每个指标的定义、单位、阈值、复现要求 | [docs/G_METRICS_SPEC.md](docs/G_METRICS_SPEC.md) |
| 每张图的结论、异常与限制 | [docs/G_结论说明.md](docs/G_结论说明.md) |
| 为什么**不需要**新增任何公共接口 | [docs/G_INTERFACE_PROPOSAL.md](docs/G_INTERFACE_PROPOSAL.md) |

## 接口原则：零新增接口

G 组**不修改** `src/contracts.py`、`docs/INTERFACES.md`、`README.md`、`src/main.py`，
只新增 `src/analysis/` 与 `scripts/G*.py`。需要的一切都能从上游已发布的能力推出：

- `fk(q) -> T (4,4)`（C 组，已登记）→ 末端路径、包络、误差；
- `JointTrajectory(time_s, q_rad)`（E 组）→ 关节指标、奇异点沿轨迹；
- `IKResult`（D 组）→ 可达性判定与失败案例；
- `ExecutionLog(time_s, q_command_rad, q_measured_rad)`（F 组）→ 跟踪误差；
  末端误差用 `fk(q_measured)` 回代得到，无需新增日志字段；
- **Jacobian 由 G 自己用数值微分算**（`src/analysis/linalg.py`），不进公共层。

因此 PR 对 A 组来说是**纯新增目录**，不触碰任何既有接口。

## 怎么跑

> 下面第 1、2 步只在你**手动检测通过之后**、由你本人执行；本包不会被自动上传。

```bash
# 1) 拉仓库、装依赖
git clone https://github.com/Marco6-3/intelligent_manufacturing.git
cd intelligent_manufacturing
python -m venv .venv && .venv\Scripts\activate      # Windows
python -m pip install -r requirements.txt

# 2) 从最新 main 开分支（CONTRIBUTING.md 规定的命名）—— 由本人操作
git checkout -b feat/G-analysis

# 3) 把本包内容按下面"合并清单"复制进仓库，然后运行
python -m pytest tests -q                      # 47 项检查（G 的 42 项 + 契约 5 项）
python -m pytest tests/test_g_analysis.py -q   # 只跑 G 的 17 项数值检查
python scripts/G0_verify_all.py                # ★ 一键验收：上面全部 + 产物清单，最后给"能交/不能交"
python scripts/G1_trajectory_metrics.py        # 两种轨迹指标对比 + 奇异点沿轨迹 + 3 张图
python scripts/G2_workspace_reachability.py    # 工作空间采样与可达性分布 + 1 张图
python scripts/G3_singularity_reachability.py  # 奇异点切片 + 可达性失败案例 + 2 张图
python scripts/G4_interface_smoke.py           # 接口复用性冒烟检查

# 没有 matplotlib 也能交差（只出 CSV 与结论）
python scripts/G1_trajectory_metrics.py --no-figs
```

产物：`results/csv/G*.csv`、`results/csv/G*_meta.json`（含 UTC 时间、git commit、seed、场景参数）、`results/figures/G*.png`。

> **CSV 编码约定**：G 组 CSV 一律写成 **UTF-8 with BOM**（Excel 双击打开中文列名不乱码）。
> 读取时必须用 `encoding="utf-8-sig"`：用普通 `utf-8` 会把 BOM 读进**第一列的列名**
> （`label` 变成 `'\ufefflabel'`）。Python 侧请直接用配套入口
> `src/analysis/results_io.py` 的 `read_csv_rows(path)`，它已处理该问题（有回归测试覆盖）。
>
> 另外：首次生成图表时 matplotlib 会在 `results/.mplconfig/` 下写一个字体缓存（约 200 KB；本机 `%LOCALAPPDATA%` 不可写时的兜底位置），**属于本地缓存，不要提交**——仓库既有 `.gitignore` 没覆盖它，人工 `git add` 时请避开，或自行加一行忽略规则。

## 合并清单（本包 → 仓库）

```
src/analysis/linalg.py          -> src/analysis/linalg.py
src/analysis/metrics.py         -> src/analysis/metrics.py
src/analysis/singularity.py     -> src/analysis/singularity.py
src/analysis/workspace.py       -> src/analysis/workspace.py
src/analysis/results_io.py      -> src/analysis/results_io.py
src/analysis/plots.py           -> src/analysis/plots.py
src/analysis/reference_model.py -> src/analysis/reference_model.py   # 占位 FK（B 组冻结模型后替换）
src/scene.py                    -> src/scene.py                      # 统一测试场景（需与 E/F 约定）
src/planning_reference.py       -> src/planning_reference.py         # 占位轨迹（E 组交付后替换）
src/contracts.py                -> src/contracts.py                  # 与仓库现有文件一致，不修改
scripts/G0~G4_*.py              -> scripts/                          # G0 是一键验收
tests/test_g_analysis.py        -> tests/test_g_analysis.py
conftest.py                     -> conftest.py                       # 让测试在任何目录可收集（仓库若已有同名文件请合并）
docs/G_*.md                     -> docs/                             # 含 G_使用与检测指南.md
```

> **`src/contracts.py` 是只读参照**：仓库里已经有同名文件，合并时**不要覆盖**。
> 本包放一份只为让 17 项检查在没有上游文件时也能跑；G 的零新增接口约定见
> `docs/G_INTERFACE_PROPOSAL.md`。

## ⚠️ 提交约束：先人工检测，确认后才由本人上传

本包**只存在于本地工作区，没有、也不得自动推送到任何远程仓库**：

- 工作区、本包目录、仓库副本三处**都没有 `.git`**：本轮虽然装好了 git（用于满足环境要求），
  但**没有在任何位置执行 `git init` / `git add` / `git commit` / `git push`**，没有任何代码被提交或上传；
- **不要**在本工作区执行 `git init` / `git add` / `git commit` / `git push`；
- 正确流程是：**先用本人账号克隆仓库 → 本地人工检测本包内容与数字 → 通过后由本人手动
  复制到 `feat/G-analysis` 分支 → 由本人执行 push 并开 PR**；
- 检测入口见上一节"怎么跑"，以及 `docs/G_结论说明.md`（数字可在 `results/csv/` 逐项核对）；
- 如果后续要自动化提交，必须由本人明确指示后才可进行。

## 重要提醒

1. **模型已换成真实的**：第二阶段起使用 B 组交付的 MuJoCo Menagerie `ur5e.xml`（`universal_robots_ur5e/`），
   第一阶段的占位 DH 模型只保留在 G1~G4 里作为"算法自检"，不再作为报告主证据。
2. **零新增接口**：不改 `src/contracts.py`、`docs/INTERFACES.md`、`README.md`、`src/main.py`；
   G 只消费上游能力，Jacobian 与指标全部自算（见 `docs/G_INTERFACE_PROPOSAL.md`）。
3. **三个阶段各有自己的场景号**：`G-stage1-scene-v1`（算法自检）、`G-stage2-real-model-v1`（真实模型）、
   `G-stage3-e-traj-v1`（E 轨迹）；每个 CSV/JSON 都带 `scene_id`，跨阶段数字不能直接混比。
4. **执行层结论现在有真实数据支撑**：第二阶段的关节/末端跟踪误差来自 F 的正式交付日志，
   第三阶段的跟踪误差来自 G 触发 F 控制器执行 E 轨迹的复跑日志（两者不可混用，见结论说明）。

## 目录结构

```
G_delivery/
├── 开始看这里.md      三步验收 + 文档索引
├── conftest.py        让 pytest 在任何目录都能收集测试
├── requirements.txt
├── docs/              G_交付说明.md, G_使用与检测指南.md, G_结论说明.md,
│                      G_METRICS_SPEC.md, G_ANALYSIS_PLAN.md, G_INTERFACE_PROPOSAL.md
├── src/
│   ├── analysis/      linalg metrics singularity workspace results_io plots  ← 第一阶段
│   │                  mujoco_backend exec_log tracking validation            ← 第二阶段
│   │                  e_traj                                                 ← 第三阶段
│   │                  reference_model.py（占位 DH，仅 G1~G4 作算法自检）
│   ├── kinematics/    fk.py（C 组原样副本）, ik.py（F 组可移植版，仅改模型路径解析）
│   ├── planning_reference.py  占位轨迹 + DLS IK（E 交付后仅 G1 使用）
│   ├── contracts.py           与仓库 src/contracts.py 一致（G 不修改它）
│   └── scene.py               三个阶段统一的场景常量
├── scripts/           G0_verify_all.py（一键验收）
│                      G1~G4（第一阶段算法自检）
│                      G5_model_validation / G6_real_singularity /
│                      G7_exec_tracking / G8_real_reachability（第二阶段）
│                      G9_e_trajectory（第三阶段）
├── tests/             test_g_analysis.py(17), test_g_real_data.py(14), test_g_e_traj.py(11)
├── upstream_refs/     d_ik_cases.csv, e_traj/, e_plan/（上游只读副本）
├── universal_robots_ur5e/  B 组交付的 MuJoCo 模型 + assets（G5~G9 依赖）
└── results/
    ├── csv/           44 个 CSV/JSON
    ├── figures/       26 张 PNG
    └── logs/          F 交付日志 + G 触发的 E 轨迹复跑日志
```

> 说明：本包的目录布局与「合并进仓库后的布局」**完全一致**，因此合并时是纯复制，
> 不需要改任何 import。`src/planning_reference.py`、`src/analysis/reference_model.py`
> 是第一阶段的自检实现，不属于上游模块。
>
> 仓库根 `README.md` 不在本包内（它由 A 组维护），G 的交付说明就是本文件。

## 第三阶段增量：接入 E 的规划轨迹（G9）

E 组交付后新增的文件（都是**纯新增**，不改上游任何一行代码）：

```
src/analysis/e_traj.py          -> src/analysis/e_traj.py     # E 轨迹加载/分段/规划侧指标
scripts/G9_e_trajectory.py      -> scripts/G9_e_trajectory.py # E 轨迹验证 + E→F 集成实验
tests/test_g_e_traj.py          -> tests/test_g_e_traj.py     # 11 项回归测试
upstream_refs/e_traj/           -> upstream_refs/e_traj/      # E 的 pick_place_traj.npz（+csv 副本）
upstream_refs/e_plan/           -> upstream_refs/e_plan/      # E 的 trajectory.py/kinematics.py 原样副本（只读）
results/logs/exec_log_e_traj_1cycle.csv  G 触发的 E→F 集成复跑日志（F 代码未改动）
results/csv/G9_*.csv|json, results/figures/G9_*.png
```

`upstream_refs/e_plan/` 只用于"用 E 自己的模块重建一次规划、拿到权威分段边界"，
G 不修改也不替代 E 的规划；重建失败时脚本会自动退化为数据驱动的静止点检测，并在 CSV 里注明来源。

## 怎么跑第三阶段

```bash
python scripts/G9_e_trajectory.py                  # 全量：自检 + 分段 + 奇异 + E/F 对比 + 集成复跑 + 5 张图
python scripts/G9_e_trajectory.py --no-rebuild     # 不重建 E 的规划（12 s），改用静止点检测分段
python scripts/G9_e_trajectory.py --no-figs        # 只出数据
python -m pytest tests/test_g_e_traj.py -q         # 只跑第三阶段 11 项测试
```

产物：`G9_e_checks.csv`、`G9_e_segments.csv`、`G9_e_sigma.csv`、`G9_e_compare.csv`、
`G9_e_exec_segments.csv`、`G9_e_exec_summary.csv`、`G9_meta.json`；图 `G9_e_joint_profiles.png`、
`G9_e_tcp_path.png`、`G9_e_segment_straightness.png`、`G9_e_sigma.png`、`G9_e_exec_jump.png`。

> **本文件不要覆盖仓库根的 `README.md`**：那个文件由 A 组维护。合并时把本文件改名为
> `docs/G_交付说明.md` 放在 docs 下即可（G 的其余文档也都在 `docs/G_*.md`）。
