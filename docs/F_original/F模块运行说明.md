# F 模块技术说明｜MuJoCo 控制与完整搬运演示

《智能制造导论》题目 11 · F 模块（MuJoCo 控制与完整搬运演示）

本文记录 F 模块的实现原理、关键技术决策与踩坑过程，供答辩讲解与 A 集成时参考。
运行方式见工程根目录 `README.md`。

---

## 一、控制对象与接口

### 1.1 场景结构（B 模块交付，F 未作任何修改）

`scenes/pick_place.xml` 通过 `<include file="../ur5e.xml"/>` 引入 UR5e 本体，
并自行定义了工作台、上料盘、下料盘与工件：

| 对象 | body 名 | 位置 | 尺寸 |
|---|---|---|---|
| 工作台 | `workbench` | (0, 0.15, −0.35) | 1.0 × 0.9 × 0.7 m，台面 z=0 |
| 上料盘 | `feed_tray` | (−0.35, 0.4, 0.005) | 0.2 × 0.2 × 0.01 m |
| 下料盘 | `place_tray` | (0.35, 0.4, 0.005) | 0.2 × 0.2 × 0.01 m |
| 工件 | `workpiece` | (−0.35, 0.4, 0.03) | 0.05 m 立方体，0.1 kg，`freejoint` |
| 抓取约束 | `grip_constraint` | — | weld，`workpiece` ↔ `wrist_3_link`，默认 `active="false"` |

实测模型规模：`nq=13`（6 关节角 + 工件自由关节 7）、`nv=12`、`nu=6`、
`neq=1`、`nsite=1`、`timestep=0.002 s`。

`qpos` 布局（控制代码依赖此约定）：

```
qpos[0:6]   6 个关节角（rad）
qpos[6:9]   工件位置（m）
qpos[9:13]  工件姿态四元数 (w,x,y,z)
```

### 1.2 执行器特性

`ur5e.xml` 使用 `<general>` **位置型执行器**：

```xml
<general gaintype="fixed" biastype="affine"
         gainprm="2000" biasprm="0 -2000 -400" forcerange="-150 150"/>
<!-- 腕部三关节 class="size1"：gainprm="500" biasprm="0 -500 -100" forcerange="-28 28" -->
```

即 MuJoCo 内置 PD 位置伺服，输出力矩

```
F = kp·(ctrl − q) − kv·q̇
```

其中 `kp = gainprm[:,0]`（肩部 2000、腕部 500），`kv = −biasprm[:,2]`（400 / 100）。

**因此控制代码无需自己写 PID，`data.ctrl` 直接填目标关节角（rad）即可。**
力矩限幅：肩部 ±150 N·m，腕部 ±28 N·m。工件仅 0.1 kg（0.981 N），
抬升所需力矩远小于限幅，裕度充足。

### 1.3 坐标系与上游的差异（易错点）

按 B 模块 `docs/坐标系说明.md`（第 1 周冻结），工具坐标系 TCP =
`wrist_3_link` 上的 `attachment_site` 站点，`pos="0 0.1 0" quat="-1 1 0 0"`。

但两个上游模块的 `fk` **返回的不是同一个坐标系**：

| 函数 | 返回坐标系 | 与 MuJoCo 实测对比 |
|---|---|---|
| C 模块 `fk.py` 的 `fk(q)` | **法兰 `wrist_3_link`** | 位置误差 0.866 mm，姿态误差 0.00000° |
| D 模块 `ik.py` 的 `get_fk(q)` | **工具坐标系 `attachment_site`** | 与 IK 自洽 |

两者相差固定变换：**位置约 99.5 mm、姿态 90°**（实测数据）。

**F 模块的取舍**：日志中 `ee_pose_actual` 统一采用 `attachment_site`，
与 B 冻结的 TCP 定义、D 的 IK 求解目标保持一致。
若 G 模块要用 C 的 `fk()` 做误差分析，必须先补上这个固定变换，
否则会凭空多出约 10 cm 的位置偏差。**建议 F/G 与 C 当面确认一次。**

---

## 二、关键技术问题：weld 约束的瞬移陷阱

这是整个 F 模块踩到的最大的坑，也是最值得在答辩中讲的一点。

### 2.1 现象

B 模块提供的抓取接口非常清晰：`grip_constraint` 是一个 weld 焊接约束，
绑定 `workpiece` 与 `wrist_3_link`，默认 `active="false"`，
把它置 1 就是"抓取"、置 0 就是"释放"。

但直接这么做，工件会飞走。实测（机械臂在 keyframe `home` 位姿）：

```
激活前 工件 xpos = [-0.35, 0.4, 0.03]        ← 好好待在上料盘上
激活后 工件 xpos = [-0.20136, 0.06132, 1.12678]
位移量 = 1157.46 mm                            ← 瞬移 1.16 米，弹到空中
关节角变化 = 0.0158 rad
```

演示直接失败。

### 2.2 根因

weld 约束不是"把两个刚体在**当前**位置粘住"，而是"强制两个刚体恢复到
**XML 编译时记录的相对位姿**"。

该相对位姿存在 `model.eq_data` 中，MuJoCo 3.x 的 weld 布局为：

```
eq_data[0:3]   anchor（锚点）
eq_data[3:6]   相对位置：body2(wrist_3_link) 在 body1(workpiece) 局部系下的坐标
eq_data[6:10]  相对姿态四元数 (w,x,y,z)
eq_data[10]    torque scale
```

B 的场景文件里 `grip_constraint` 没有显式写 `relpose`，MuJoCo 便按
**编译时的初始状态**（`qpos = 0`，机械臂完全展开）自动计算并固化了这个相对位姿。
实测编译值：

```
eq_data[0] = [0, 0, 0,  -0.467, -0.534, 0.033, 0,  1, 0, 0, 1]
                        ↑ 展开姿态下 wrist_3 相对 workpiece 的位置
```

而抓取时机械臂已完全换了构型，`wrist_3_link` 的实际相对位置与编译值相差极大。
weld 一旦激活，就会把工件强行拽回那个"展开姿态下的相对位置"——于是瞬移 1157 mm。

**关键点：这不是 B 模块的 bug。** 场景文件按 MuJoCo 的标准写法是正确的，
weld 本来就是这样设计的。问题在于"运行时激活"这个用法需要控制方额外处理。

### 2.3 解决方案

激活前把 `eq_data` 的相对位姿**改写为当前真实值**，这样焊接等于"就地锁定"：

```python
def _sync_weld_relpose(self):
    mujoco.mj_forward(self.model, self.data)
    R_wp = self.data.xmat[self.wp_id].reshape(3, 3)     # 工件姿态
    R_w3 = self.data.xmat[self.w3_id].reshape(3, 3)     # 法兰姿态
    p_rel = R_wp.T @ (self.data.xpos[self.w3_id] - self.data.xpos[self.wp_id])
    q_rel = rotmat2quat(R_wp.T @ R_w3)

    self.model.eq_data[self.eq_id][0:3]  = 0.0        # anchor 保持 0
    self.model.eq_data[self.eq_id][3:6]  = p_rel      # 相对位置 ← 当前真实值
    self.model.eq_data[self.eq_id][6:10] = q_rel      # 相对姿态 ← 当前真实值
    self.model.eq_data[self.eq_id][10]   = 1.0        # torque scale
    return p_rel, q_rel
```

调用顺序必须是 **同步 → 激活 → 短程稳态**：

```python
def grasp(self):
    wp_before = self.wp_pos.copy()
    p_rel, q_rel = self._sync_weld_relpose()   # 1. 先同步
    self.data.eq_active[self.eq_id] = 1        # 2. 再激活
    self._hold(T_GRASP_HOLD, "grasp_hold")     # 3. 稳态
    shift = np.linalg.norm(self.wp_pos - wp_before) * 1000   # 4. 实测验证
```

### 2.4 实测效果

| | 直接激活 | 同步后激活 |
|---|---|---|
| 工件瞬移 | **1157.46 mm** | **0.783 mm** |
| 结果 | 演示失败 | 抓取成功 |

改善约 **1478 倍**。残余的 0.783 mm 来自 weld 的软约束特性
（`solref="0.02 1"`）与工件自重下的微小沉降，属正常范围。

代码中对该值设了 2 mm 阈值，超阈值记为 `WARN` 并中止本次循环，
不会静默放过异常。

### 2.5 物理可解释性

场景中没有夹爪指部几何，`attachment_site` 相当于**吸盘/真空吸嘴的工具中心点**。
TCP 下降到工件顶面中心（z = 0.055）后锁定，等价于吸盘吸附。

这符合任务书 F 模块的要求：

> 先用**稳定、可解释**的抓取约束跑通基础演示，若有余力再研究接触夹取。

若要做真实接触夹取，需 B 模块先在场景中增加夹爪指部几何与接触参数，
F 侧再把 weld 锁定替换为指部位置控制 + 接触力检测。

---

## 三、控制律：为什么加前馈补偿

### 3.1 纯位置反馈的问题

最初实现只是 `data.ctrl[:] = q_cmd`（纯位置反馈）。单次搬运能成功，
但跟踪误差峰值达 **11.851°**，放置水平偏差 4.85 mm。

根因在于 PD 位置伺服跟踪运动指令时存在两项**固有稳态误差**：

**(1) 重力下垂** —— 静态平衡时需要 `F = qfrc_bias`（重力 + 科氏力），
即 `kp·(ctrl − q) = qfrc_bias`，必然产生恒定位置偏差：

```
e_gravity = qfrc_bias / kp
```

**(2) 速度滞后** —— 跟踪速度为 `v` 的斜坡指令时，阻尼项 `kv·q̇` 需要
位置误差来"顶住"：

```
e_lag = (kv/kp)·v = 0.2·v
```

实测本演示峰值关节速度约 0.59 rad/s → 理论滞后 0.118 rad ≈ 6.8°，
与实测 11.851° 量级吻合（叠加重力下垂项）。

**这是位置伺服的物理特性，不是代码 bug。** 但 11.851° 的误差对
G 模块的误差分析和报告的可信度都不利。

### 3.2 计算力矩前馈

把这两项直接抵消掉：

```
ctrl = q_cmd + [ M(q)·q̈_cmd + qfrc_bias + kv·q̇_cmd ] / kp
                └─加速度前馈─┘  └重力补偿┘  └速度前馈┘
```

```python
def _feedforward_ctrl(self, q_cmd, dq_cmd=None, ddq_cmd=None):
    d = self.data
    f_ff = np.array(d.qfrc_bias[:6], dtype=float)      # 重力 + 科氏力

    if dq_cmd is not None:                             # 速度前馈
        f_ff += self.kv * np.asarray(dq_cmd, dtype=float)

    if ddq_cmd is not None and np.any(ddq_cmd):        # 加速度前馈
        vec = np.zeros(self.model.nv)
        vec[:6] = np.asarray(ddq_cmd, dtype=float)
        res = np.zeros(self.model.nv)
        mujoco.mj_mulM(self.model, d, res, vec)         # M(q)·q̈
        f_ff += res[:6]

    ctrl = np.asarray(q_cmd, dtype=float) + f_ff / self.kp
    return np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)    # 裁剪到 ctrlrange
```

实现要点：

- **用 `mj_mulM` 而非 `mj_fullM`**：直接算"质量矩阵 × 向量"，
  不必展开 12×12 全矩阵，更快。
- **`qfrc_bias` 与 `M` 取上一步仿真结束时的值**：`mj_step` 已填充，
  无需额外 `mj_forward`。dt = 2 ms，动态变化远慢于步长，滞后一拍不影响补偿效果。
- **质量矩阵取当前真实状态**：weld 激活后工件质量已并入机械臂动力学，
  补偿量自动包含负载惯性，**无需手工建"抓了东西"的模型**。
- **反馈项 `kp·(ctrl − q)` 完全保留**：闭环稳定性不受影响，前馈只负责抵消稳态误差。
- **`ctrl` 裁剪到 `ctrlrange`**：超出 `forcerange` 的力矩由 MuJoCo 限幅。
- 静止段（`_hold` / `_settle`）传 `dq_cmd = ddq_cmd = 0`，
  前馈退化为纯重力补偿，正好消除静态下垂。

### 3.3 实测对比（同一设置，仅切换 `--no-ff`）

| 指标 | 纯位置反馈 | 前馈补偿 | 改善 |
|---|---|---|---|
| 跟踪误差峰值 | 11.851° | **0.147°** | **↓ 80 倍** |
| 跟踪误差峰值 | 0.2068 rad | **0.00256 rad** | ↓ 81 倍 |
| 放置水平偏差 | 4.85 mm | **0.86 mm** | ↓ 5.6 倍 |
| 搬运成功率 | 1/1 | 1/1 | — |

跟踪误差从"十度级"降到"零点几度"，同时放置精度提升 5.6 倍。

**边界说明**：前馈补偿属于超出 F 基础要求的增强，
可用 `--no-ff` 一键关闭做对比，两种模式的实测数据都已保留，
可写进报告说明控制策略的作用。是否合并到最终集成版请与组长 A 确认。

---

## 四、轨迹执行与 E 模块接口

### 4.1 接口约定

任务书第三节明确：

> E 提供轨迹输出：`time[N]`、`q_traj[N,6]`；F 读取并执行，
> 输出 `q_actual` 和 `ee_pose_actual` 日志。

F 严格按此实现。执行时**一个轨迹点对应一次 `mj_step`**，
保证 `time[N]` 与仿真步严格对齐，便于 G 模块做误差分析：

```python
def _run_traj(self, traj, seg):
    q_traj = traj["q_traj"]
    dq  = traj.get("dq_traj")      # 有则用于速度前馈
    ddq = traj.get("ddq_traj")     # 有则用于加速度前馈
    for i in range(len(q_traj)):
        self._step(q_traj[i], seg,
                   dq_cmd=(dq[i]  if dq  is not None and i < len(dq)  else None),
                   ddq_cmd=(ddq[i] if ddq is not None and i < len(ddq) else None))
    return self.q_actual.copy()
```

### 4.2 当前轨迹来源（E 未交付的临时方案）

**F 手上没有 E 的交付物**，而 F 的输入恰恰是 E 的输出。
为使 F 能独立完成"连续 3 次搬运"的验收，在 `src/control/traj_adapter.py`
中按 E 的接口约定实现了最小可用的两种方案：

| 方案 | 函数 | 用途 |
|---|---|---|
| 关节空间五次多项式 | `quintic_joint_traj()` | approach / descend / lift / lower / retreat / return |
| 笛卡尔空间直线 + 逐点 IK | `cartesian_line_traj()` | transfer（平移段） |

**五次多项式的边界条件**（保证位置/速度/加速度全连续，避免激励机械臂振动）：

```
q(0)=q0,  q(T)=q1,  q'(0)=q'(T)=0,  q''(0)=q''(T)=0
→ q(τ) = a0 + a3·τ³ + a4·τ⁴ + a5·τ⁵,  τ = t/T
   a0 = q0,  a3 = 10(q1−q0),  a4 = −15(q1−q0),  a5 = 6(q1−q0)
```

**笛卡尔段**沿直线插补位置、姿态相同时直接沿用（否则 SVD 正交化），
逐点调用 IK 并**以上一点解作为下一点初值**，保证关节轨迹连续、不跳构型。
不收敛的点会被完整记录到 `ik_failures`，**不静默跳过**
（对应 E 的验收要求"能解释不可解点"）。

### 4.3 E 交付后的替换方式（零改动）

文件已从 `src/planning/` **移至 `src/control/` 并改名为 `traj_adapter.py`**，
`src/planning/` 目录完整留给 E，避免同名交付物互相覆盖。

E 交付后：

```bash
# 1. 把 E 的文件放到 src/planning/trajectory.py
# 2. 删除 src/control/traj_adapter.py
# 3. 用 E 的轨迹文件运行，controller.py 无需任何改动
python src/control/controller.py --cycles 3 --traj-file results/logs/e_traj.npz
```

加载器 `load_e_trajectory()` 支持的格式：

| 格式 | 要求 |
|---|---|
| `.npz` | 含 `time[N]` 与 `q_traj[N,6]` 两个数组 |
| `.npy` | 直接是 `(N,6)` 数组，时间轴按 dt=0.01 重建 |
| `.csv` | 首行表头，含 `t`（或 `time`）列与 `q0..q5`（或 `q_traj_0..q_traj_5`）列 |

加载失败时打印原因并**回退到内置生成器**，不会让演示中断。

### 4.4 时间尺度对齐（接入外部轨迹必须了解）

`_run_traj` 的执行方式是 **一个轨迹点 = 一次 `mj_step`（dt = 0.002 s）**，
所以一段轨迹的实际仿真时长 = `N × 0.002`。

内置生成器始终用 `self.dt` 采样，两者天然一致。
但 **E 交付的轨迹采样周期是任意的**，直接逐点执行会变速播放。
实测（E 若用 dt=0.01、N=301、声明 duration=3.00 s）：

| | 修复前 | 修复后 |
|---|---|---|
| 实际执行时长 | **0.602 s** | **3.002 s** |
| 等效播放速度 | 5 倍速 | 1 倍速 |
| 放置水平偏差 | 0.568 mm | 0.855 mm |
| 速度/加速度 | 全部放大 5 倍 | 与轨迹声明一致 |

5 倍速不只是"跑快了"——关节速度、加速度同步放大 5 倍，
前馈补偿的 `kv·q̇` 与 `M·q̈` 项也随之失真，跟踪误差与 G 模块的
误差分析结论都会错。

修复方式：检测到轨迹 dt 与仿真步长不一致（相对偏差 ≥1%）时，
在**真实时间轴**上把 `q_traj` 用三次样条重采样到 `self.dt` 网格，
使实际执行时长等于轨迹声明的 `duration`。
`dq_traj` / `ddq_traj` 一并重采样，**不做任何缩放**
（它们单位是 rad/s、rad/s²，是轨迹自身属性，与采样密度无关）。

重采样会写入一条 `SEG` 事件，日志中可查：

```
transfer_e 时间尺度对齐：轨迹 dt=0.0100s (N=301, 声明 3.000s)
           → 仿真 dt=0.0020s (N=1501, 实际 3.000s)，三次插值重采样
```

内置轨迹路径完全不受影响（dt 一致时跳过插值，回归实测无该事件、
3/3 成功、跟踪误差仍为 0.1468°）。

**给 E 的实用建议**：若 E 直接用 dt=0.002 s（500 Hz）输出轨迹，
则不会触发重采样，F 逐点原样执行，时间尺度与导数都最精确。
用 dt=0.01 也完全可以，F 会自动对齐。

任务书 A 模块明确要求"不允许私下各写一套"，
**集成前请与组长 A 确认本文件的处置方式**（删除 / 保留做对比）。

---

## 五、状态机与关键点定义

### 5.1 工作空间关键点（基坐标系，单位 m）

依据 B 场景几何推导：

```python
FEED_XY     = (-0.35, 0.4)     # 上料盘中心
PLACE_XY    = ( 0.35, 0.4)     # 下料盘中心
BLOCK_TOP_Z = 0.055            # 工件顶面 = 中心 0.03 + 半边长 0.025 → 吸盘吸附高度
BLOCK_REST_Z = 0.035           # 工件静止在盘上时的中心高度（判定基准）
HOVER_Z     = 0.30             # 安全过渡高度
LIFT_Z      = 0.38             # 搬运抬升高度
CLEAR_Z     = 0.12             # 释放后抬离高度
```

**`BLOCK_REST_Z` 的推导**（曾在此踩坑，见 §5.3）：

```
盘面 z = tray pos 0.005 + half height 0.005 = 0.01
工件静止中心 z = 0.01 + 半边长 0.025 = 0.035
```

**TCP 朝正下方的姿态**（实测 keyframe `home` 下 `attachment_site` 的世界姿态即为此）：

```python
R_DOWN = [[ 1,  0,  0],      # X = [1,0,0]
          [ 0, -1,  0],      # Y = [0,-1,0]
          [ 0,  0, -1]]      # Z = [0,0,-1] ← 工具朝向正下，适合吸取
# det(R_DOWN) = 1·(1−0) = +1，合法旋转矩阵
```

### 5.2 十段流程与各段时长

| # | 段名 | 时长 | 轨迹类型 | 说明 |
|---|---|---|---|---|
| 0 | `home` | 1.0 s | 伺服稳态 | 归位到 keyframe `home` |
| 1 | `approach` | 2.0 s | 五次多项式 | 移到源上方 HOVER_Z |
| 2 | `descend` | 1.6 s | 五次多项式 | 下降到工件顶面 |
| — | `descend_settle` | 0.3 s | 伺服稳态 | **到位确认**，消除残余速度 |
| 3 | `grasp_hold` | 0.6 s | 静止 | 同步 relpose → 激活 weld |
| 4 | `lift` | 1.6 s | 五次多项式 | 抬升到 LIFT_Z |
| 5 | `transfer` | 3.0 s | **笛卡尔直线** | 平移到目标上方，逐点 IK |
| 6 | `lower` | 1.6 s | 五次多项式 | 下降到放置高度 |
| — | `lower_settle` | 0.3 s | 伺服稳态 | 放置前稳定 |
| 7 | `release_hold` | 0.6 s | 静止 | 解除 weld，工件自由落下 |
| 8 | `retreat` | 1.2 s | 五次多项式 | 抬离到 CLEAR_Z |
| — | `settle_after_release` | 0.8 s | 静止 | 等工件落位稳定后判定 |
| 9 | `return_home` | 2.0 s | 五次多项式 | 归位，为下次循环准备 |

单循环仿真时长约 16.6 s，共约 8300 步。

**抓取前的到位确认**很关键：`descend_settle` 保证抓取瞬间机械臂无残余速度，
否则 weld 会在动态中激活，相对位姿同步的值与实际不符。

### 5.3 判定基准的一个错误与修正

初版判定写成 `dz = |wp_z − 0.03|`，结果工件正确落位却报"高度偏差 4.83 mm"。

原因：`0.03` 是 B 场景 **keyframe 中的初始值**，不是工件的静止高度。
keyframe 把工件放在 z=0.03，比静止高度 0.035 低 5 mm（轻微嵌入盘面），
仿真启动后工件会上浮到约 0.0348（实测）。

修正为以 `BLOCK_REST_Z = 0.035` 为基准后，高度偏差从 4.826 mm 降到 **0.174 mm**，
这才是真实的放置精度。

**工件静置稳定性实测**（未抓取，500 步）：初始 z=0.03 → 0.03483，
漂移 4.826 mm 后稳定，工件底面 z=0.0098 ≈ 盘面 0.01，**无穿模、无持续下沉**。

### 5.4 成功判定

```python
placed = (水平偏差 ≤ 15 mm) and (高度偏差 ≤ 20 mm) and (weld 已解除)
```

三项同时满足才算成功。第三项防止"工件被粘在末端上、根本没放下"被误判为成功。

---

## 六、两种运行模式

| | `--mode reset`（默认） | `--mode pingpong` |
|---|---|---|
| 每次循环起始状态 | 复位到 keyframe + 工件放回上料盘 | 不复位，沿用上次结束状态 |
| 搬运方向 | 恒定 上料盘 → 下料盘 | 交替：奇数次 →下料盘，偶数次 →上料盘 |
| 数值结果 | 三次**完全相同**（确定性） | 逐次**不同**（状态累积） |
| 验收对应 | "同一设置下连续 3 次" | 额外的鲁棒性证据 |

`reset` 模式是标准的可重复性实验协议：初始条件受控，
所以"同一设置下连续 3 次"这个验收要求得到严格满足。

**但要注意**：MuJoCo 在相同初始条件下是**确定性**的，
因此 reset 模式三次结果逐位相同（0.861 mm / 0.7827 mm / 0.147°），
这是预期行为，**不是日志复制粘贴**。答辩时若被问到，用 pingpong 数据佐证。

pingpong 实测（4 次，不复位）：

| 循环 | 方向 | 放置偏差 | 抓取位移 |
|---|---|---|---|
| 1 | feed → place | 0.86 mm | 0.7827 mm |
| 2 | place → feed | 1.05 mm | 0.7830 mm |
| 3 | feed → place | 1.59 mm | 0.7879 mm |
| 4 | place → feed | 1.84 mm | 0.7880 mm |

偏差逐次递增但仍远低于 15 mm 阈值，反映不复位时工件落位与接触状态的累积效应。
**4/4 全部成功**，比 reset 模式更能说明控制流程的鲁棒性。

---

## 七、日志与输出

### 7.1 执行轨迹日志（任务书要求的交付项）

`results/logs/exec_log_<时间戳>.csv`，另存一份 `exec_log_latest.csv` 便于取用。
UTF-8-BOM 编码，Excel 可直接打开不乱码。

| 列 | 说明 |
|---|---|
| `t` | 仿真时间（s），跨循环累加 |
| `cycle` | 循环编号 |
| `segment` | 当前所处段名 |
| `grip` | 抓取状态（1=持有工件） |
| `q_cmd_0..5` | 指令关节角（E/轨迹生成器输出） |
| `q_actual_0..5` | **实际关节角**（接口约定的 `q_actual`） |
| `track_err_max` | 该步 6 关节跟踪误差的最大值（rad） |
| `ee_x/y/z` | **实际 TCP 位置**（接口约定的 `ee_pose_actual`） |
| `ee_qw/qx/qy/qz` | **实际 TCP 姿态四元数** |
| `wp_x/y/z` | 工件中心位置 |

一次 3 循环运行约 24921 行。**这份日志可直接供 G 模块做误差分析。**

### 7.2 事件与告警日志（对应"记录失败情况，不剪辑掩盖问题"）

`results/logs/events_<时间戳>.csv`，独立记录：

| tag | 含义 |
|---|---|
| `STATE` | 状态切换、归位完成、抓取点到位 |
| `SEG` | 每段轨迹的类型、时长、点数、起止值 |
| `GRASP` / `RELEASE` | weld 激活/解除，含 relpose 数值与工件位移实测 |
| `IK_FAIL` | 逆解未收敛（含误差与迭代次数） |
| `IK_UNSOLVABLE` | 笛卡尔段的不可解点（含点序号、时间、目标 TCP、误差） |
| `IK_SLOW` | 迭代次数偏多（>30） |
| `WARN` | 抓取位移过大、平移段存在不可解点等 |
| `ABORT` | 本次循环中止及原因 |
| `RESULT` | 该次搬运的最终判定与偏差数值 |
| `VIDEO` | 视频保存路径、渲染后端、帧数 |

**失败与告警不做过滤，全部写入。** 本次运行 IK 不可解点为 0、无 ABORT。

### 7.3 运行摘要

`results/logs/run_summary_<时间戳>.txt`，含全局指标、逐次明细与验收对照。

### 7.4 图表（额外产出，供 G 模块与报告使用）

| 文件 | 内容 |
|---|---|
| `joint_tracking.png` | 6 关节 指令 vs 实际 跟踪曲线（2×3 子图） |
| `tracking_error_grip.png` | 跟踪误差（按循环分色，含阈值线）+ 抓取状态时序 |
| `tcp_and_workpiece_path.png` | TCP 与工件的 XY 轨迹 + 高度剖面（含料盘标记） |

### 7.5 演示视频

`results/videos/demo.mp4`，默认 640×480、30 fps、h264。

相机固定：`azimuth=135, elevation=-22, distance=2.4, lookat=(0, 0.35, 0.25)`。
抽帧策略：每 `1/(fps·dt)` = 每 16.7 个仿真步取一帧。

---

## 八、实测结果汇总

环境：Windows 11、Python 3.14.6、MuJoCo 3.16.0、NumPy 2.5.4、SciPy 1.18.1
运行日期：2026-10-17

### 8.1 主验收运行 `--cycles 3 --mode reset`

| 指标 | 结果 |
|---|---|
| 搬运成功 | **3 / 3 次** ✓ |
| 放置水平偏差 | 0.861 mm（阈值 15 mm） |
| 放置高度偏差 | 0.174 mm |
| 抓取瞬间工件位移 | 0.783 mm（阈值 2 mm） |
| 抓取点 TCP 到位误差 | 2.164 mm |
| 关节跟踪误差峰值 | **0.147°**（0.00256 rad） |
| 笛卡尔平移段不可解点 | **0** |
| IK 段全部收敛 | ✓ |
| 仿真总时长 | 49.842 s（24921 步 @ 500 Hz） |
| 实际耗时 | 5.94 s（约 8.4 倍实时） |
| 视频 | 1465 帧，48.83 s，640×480，h264，7.10 MB |
| 渲染后端 | `mujoco_opengl`（OpenGL 真实渲染） |

### 8.2 视频有效性校验

不只看文件大小，做了两层校验：

**逐帧抽样（帧 0 / 200 / 500 / 900 / 1300 / 1464）**

| 帧 | shape | 亮度均值 | 非黑像素占比 |
|---|---|---|---|
| 0 | (480, 640, 3) | 82.6 | 99.8% |
| 200 | (480, 640, 3) | 82.6 | 99.8% |
| 500 | (480, 640, 3) | 82.8 | 99.8% |
| 900 | (480, 640, 3) | 85.8 | 99.8% |
| 1300 | (480, 640, 3) | 85.7 | 99.8% |
| 1464 | (480, 640, 3) | 82.8 | 99.8% |

→ 全部非黑、画面有效。

**帧间差异（每 40 帧采样，36 个间隔）**

```
相邻采样帧平均像素差：均值 4.40，最小 0.91，最大 9.94
静止帧占比（差值 < 0.5）：0.0%
```

→ 画面随时间显著变化，**动画有效**，不是静止图重复。

### 8.3 IK 收敛性（前置验证，6 个目标点）

| 目标点 | TCP 位置 | 迭代 | 位置误差 | 姿态误差 |
|---|---|---|---|---|
| 预抓取（工件上方 25 cm） | (−0.35, 0.4, 0.28) | 7 | 0.00098 mm | 0.00001° |
| 抓取点（工件上方 3 cm） | (−0.35, 0.4, 0.06) | 6 | 0.00126 mm | 0.00000° |
| 抬升（工件上方 35 cm） | (−0.35, 0.4, 0.38) | 7 | 0.00596 mm | 0.00009° |
| 放置上方 | (0.35, 0.4, 0.38) | 12 | 0.00132 mm | 0.00002° |
| 放置点 | (0.35, 0.4, 0.065) | 7 | 0.00158 mm | 0.00001° |
| HOME 复现 | (−0.134, 0.492, 0.488) | 11 | 0.00704 mm | 0.00010° |

全部收敛，位置误差均在 **0.007 mm** 量级（亚微米级），
远小于放置阈值 15 mm。求解时用上一点解作初值，保证构型连续。

---

## 九、渲染后备方案

### 9.1 问题

`mujoco.Renderer` 依赖 OpenGL 上下文。Windows 上 MuJoCo 只支持
`MUJOCO_GL = wgl / glfw`（`egl`、`osmesa` 是 Linux 专用，
在 Windows 上会直接抛 `invalid value for environment variable MUJOCO_GL`）。

在无显卡驱动或无桌面会话的环境（CI、远程沙箱、服务模式）中，
构造渲染器会抛：

```
FatalError: gladLoadGL error
GLFWError: (65544) WGL: Failed to make dummy context current: 句柄无效
```

此时 `results/videos/demo.mp4` 无法产出，而它是任务书的**明确交付项**。

### 9.2 方案

`src/control/fallback_render.py` 提供纯 CPU 后备路径 `SkeletonRenderer`，
接口对齐 `mujoco.Renderer`（`update_scene()` → `render()` → `close()`）。

绘制内容：工作台台面、上/下料盘、机械臂连杆骨架与关节、
TCP 工具中心点、工件立方体线框、抓取状态连线（weld 激活时显示）。

**所有位姿数据均取自 MuJoCo 真实仿真状态**
（`data.xpos` / `data.site_xpos` / `data.eq_active`），逐帧读取，
**不是另行编造的假动画**，物理过程与主仿真完全一致。

### 9.3 自动降级逻辑

```
mujoco.Renderer 可用 → 用它（OpenGL 真实渲染，带网格与光照）
        ↓ 失败
SkeletonRenderer 可用 → 用它（CPU 骨架示意），控制台打印降级原因
        ↓ 失败
跳过视频录制，物理仿真、日志与图表照常产出
```

渲染后端会记入运行摘要的 `render_backend` 字段
（`mujoco_opengl` / `skeleton_fallback` / `none`）与 `VIDEO` 事件，
**便于说明视频来源，不掩盖降级事实**。

### 9.4 性能实测

骨架渲染约 **80 ms/帧**（640×480 与 960×720 差异很小，
matplotlib 为矢量渲染，瓶颈在绘制调用而非像素填充）。
3 次循环约 2500 帧 → **约 3.3 分钟**。
OpenGL 渲染时同样内容为 1465 帧、总耗时 5.94 s。

因此默认分辨率设为 640×480，兼顾清晰度与录制时间。

**本次主验收运行用的是 OpenGL 真实渲染**（`render_backend: mujoco_opengl`），
后备路径作为保险已就位并通过单帧测试。

---

## 十、给其他模块的接口说明

### 10.1 给 G 模块（误差分析）

- **执行日志**：`results/logs/exec_log_latest.csv`
  含 `q_cmd` / `q_actual` / `track_err_max` / `ee_pos` / `ee_quat` / `wp_pos`，
  可直接统计规划与实际的末端误差、路径长度、速度/加速度。
- **速度加速度**：如需 `dq_traj` / `ddq_traj`，
  调用 `traj_adapter.quintic_joint_traj()` 即可拿到解析导数（已换算为对真实时间 t）。
- **⚠️ 坐标系**：日志中 `ee_*` 是 **`attachment_site`（TCP）**。
  若用 C 模块 `fk(q)`（返回 `wrist_3_link` 法兰位姿）做对比，
  **必须补上固定变换（约 99.5 mm、90°）**，否则误差会虚高。
- **采样条件**：dt = 0.002 s（500 Hz），单位 rad / m / s，已在日志表头与本文注明。

### 10.2 给 E 模块（轨迹规划）

- 接口约定为 `time[N]` + `q_traj[N,6]`，
  F 用 `--traj-file` 加载，支持 `.npz` / `.npy` / `.csv`（格式见 §4.3）。
- **`src/planning/` 目录已完整留给 E**，F 的临时实现放在 `src/control/traj_adapter.py`，
  不会与 E 的同名交付物冲突。
- 若 E 的轨迹能提供 `dq_traj` / `ddq_traj`，F 会自动用于前馈补偿，跟踪精度更高；
  不提供也能正常运行（退化为纯位置反馈 + 重力补偿）。
- **TCP 定义**：F 的所有笛卡尔目标点均基于 `attachment_site`，
  姿态用 `R_DOWN`（Z 轴朝正下）。E 做笛卡尔规划时请对齐同一定义。

### 10.3 给 B 模块（场景建模）

- **`grip_constraint` 场景文件未作任何修改**，relpose 同步在 F 运行时完成。
- 但若其他模块也要用这个 weld 做抓取，**必须同样处理**，否则会瞬移。
  建议在 B 的 `坐标系说明.md` 中补一句提示。
- 若要做接触夹取（任务书"若有余力"部分），需 B 增加夹爪指部几何与接触参数。

### 10.4 给 D 模块（逆运动学）

- **`ik.py` 第 6 行的硬编码路径建议同步修掉**：
  `C:\Users\wzy20\Desktop\mujoco_menagerie-main\...\ur5e.xml`
  在他人机器上 `import` 即抛 `FileNotFoundError`，A 集成时会失败。
- D 的原 `ik_dls()` **未做关节限位处理**，目标点靠近工作空间边界时
  可能解出超出 `ctrlrange` 的关节角，位置执行器会静默截断，
  导致实际末端偏离目标。F 侧用增强版 `ik_dls_limited()`（补限位裁剪 + 步长上限），
  原函数保留未动供对比。**是否合并回 D 请与 D 同学和组长 A 确认。**

### 10.5 给 A 模块（总集成）

- 运行入口：`src/control/controller.py`，可用 `python -m` 或直接路径调用，
  内部用 `__file__` 解析路径，**不依赖 cwd**。
- 旧入口 `src/controller.py` 已改为转发（见下节），不会与包导入冲突。
- 依赖清单：`requirements.txt`（实测版本）。
- 需确认的事项：① `traj_adapter.py` 在 E 交付后如何处置；
  ② `ik_dls_limited` 是否合并回 D；③ 前馈补偿是否保留在集成版。

---

## 十一、工程内文件改动清单

| 文件 | 来源 | 改动 |
|---|---|---|
| `ur5e.xml` | B | **未改动** |
| `assets/`（20 个 .obj） | B | **未改动** |
| `scenes/pick_place.xml` | B | **未改动** |
| `scenes/assets/`（20 个 .obj） | B | **未改动** |
| `docs/坐标系说明.md` | B | **未改动** |
| `docs/模型资源来源说明.md` | B | **未改动** |
| `docs/MODEL_LICENSE.txt` | B（原 LICENSE） | 未改动，仅改名归档 |
| `src/kinematics/fk.py` | C | **未改动** |
| `src/kinematics/ik.py` | D | 改路径解析 + 加 `ik()` 包装 + 加 `ik_dls_limited()`；原 `ik_dls()` 未动 |
| `src/control/controller.py` | **F 主交付** | 新写 |
| `src/control/traj_adapter.py` | **F 临时**（E 职责） | 新写，E 交付后删除 |
| `src/control/fallback_render.py` | **F** | 新写 |
| `src/controller.py` | F 旧占位示例 | 改为转发入口（原 30 行示例已被完整取代，且其依赖的玩具场景已被 B 的真实场景替换，原代码已无法运行） |
| `README.md` | F | 重写（原为 13 行占位说明） |
| `docs/F模块运行说明.md` | **F** | 新写（本文件） |
| `docs/给E模块的接口说明.md` | **F** | 新写，供 E 同学对齐轨迹接口与采样周期约定 |
| `requirements.txt` | F | 新写 |
| `tests/test_traj_interface.py` | **F** | 新写，E 轨迹接口 + 时间尺度对齐自测 33 项 |
| `tests/test_video_output.py` | **F** | 新写，演示视频有效性自测 15 项 |
| `tests/test_weld_grasp.py` | **F** | 新写，weld 瞬移陷阱与修复回归测试 10 项 |
| `results/**` | F 运行产出 | 日志、图表、视频 |

**上游 B / C 的交付物一个字节都没改。** D 的 `ik.py` 改动已在上文逐项说明，
桌面原始文件未被触碰（复制到工程内后才修改）。

---

## 十二、复现命令

```bash
cd F_mujoco_pick_place_demo
pip install -r requirements.txt

# 1. 快速验证（约 1 秒，不出视频）
python src/control/controller.py --cycles 1 --no-video --no-figures

# 2. 主验收运行（约 6 秒仿真 + 视频编码）
python src/control/controller.py --cycles 3 --mode reset

# 3. 鲁棒性补充运行（不复位往返）
python src/control/controller.py --cycles 4 --mode pingpong --no-video

# 4. 控制策略对比（写进报告）
python src/control/controller.py --cycles 1 --no-video --no-ff     # 纯位置反馈
python src/control/controller.py --cycles 1 --no-video             # 前馈补偿

# 5. 实时观察（需桌面环境）
python src/control/controller.py --cycles 1 --viewer --no-video

# 6. E 模块交付后接入真实轨迹
python src/control/controller.py --cycles 3 --traj-file results/logs/e_traj.npz

# 7. 自测（验证交付物本身，只读 results/，不修改交付物）
python tests/test_traj_interface.py    # 33 项：三种格式加载、异常回退、五次多项式性质、时间尺度对齐
python tests/test_video_output.py      # 15 项：可解码、非黑、画面随时间变化
python tests/test_weld_grasp.py        # 10 项：weld 瞬移陷阱复现与修复、携带刚性、脱钩
```

成功标志：控制台末尾输出

```
验收结论：✓ 通过 —— 连续完成 3/3 次搬运
```

三个自测脚本末尾输出 `结果：N 通过 / 0 失败`，退出码 0。

实测自测结果（2026-10-17）：

| 自测 | 结果 |
|---|---|
| `tests/test_traj_interface.py` | **33 通过 / 0 失败** |
| `tests/test_video_output.py` | **15 通过 / 0 失败** |
| `tests/test_weld_grasp.py` | **10 通过 / 0 失败** |

`test_weld_grasp.py` 把 §2 的瞬移陷阱固化成了回归测试，关键实测数据：

| 测量项 | 数值 |
|---|---|
| 抓取姿态下真实 relpose 与编译值偏差 | 0.7183 m → 直接激活必然瞬移 |
| 直接激活 weld 的工件瞬移（陷阱复现） | **745.4 mm** |
| 同步 relpose 后激活的工件位移 | **0.0983 mm**（改善 **7582 倍**） |
| 携带中工件相对 TCP 偏移极差 | [1.87, 2.66, 0.45] mm，无滑脱 |
| 垂直偏移均值（抓取间隙） | 23.9 mm |
| 释放退让后 TCP 移动 / 工件移动 | 55.4 mm / 1.09 mm（确认脱钩） |

注意：陷阱复现的瞬移量在不同初始位姿下数值不同（在 keyframe `home` 位姿
实测约 1157 mm，在抓取位姿实测 745 mm），量级一致，都远超 2 mm 阈值。
测试中断言用的是"存在性"（>100 mm），不锁死具体数值。

自测中有两处断言阈值初版写错过，均已修正（如实记录，供 H 模块汇总 AI 使用记录）：

1. **速度边界容差**：初版对加载器重建的 `dq` 首末值用了 `1e-6` 容差，
   实测 `5.30e-05` 而判失败。原因是加载器的 `dq` 由 `np.gradient` 从 `q_traj`
   数值微分重建，边界为单侧差分，本就存在 O(dt) 误差。
   解析边界条件（严格为 0）改为直接校验 `quintic_joint_traj()` 的返回值，
   加载器路径容差放宽到 `1e-3` 并注明误差来源。

2. **位置连续性阈值**：初版写死 `单步变化 < 0.01 rad`，
   实测 `0.02385 rad` 而判失败。但该测试轨迹峰值速度为 2.3855 rad/s，
   `0.02385 = 2.3855 × 0.01` 正好等于 `峰值速度 × dt`，属正常。
   写死常数会在高速轨迹上误报，同时可能放过低速轨迹上的真实跳变。
   改为由峰值速度推导上界 `峰值速度 × dt × 1.5`，
   并补充"峰值速度与理论值 `1.875·Δq/T` 吻合"的交叉验证
   （实测 2.3855 vs 理论 2.3855，完全一致）。

---

## 十三、AI 使用记录（供 H 模块汇总）

任务书要求"记录 AI 使用（所用工具、主要提示词、人工验证方式）"。
F 模块的情况如下，供 H 直接取用：

**使用的工具**：千问工作助理（Qwen）

**AI 承担的部分**：
- 解析任务书、梳理 F 模块的交付项与验收标准
- 探测 B 场景的结构（关节/执行器/equality/site/keyframe 的真实索引与取值）
- 定位 weld 瞬移的根因（eq_data 布局与编译期 relpose）并给出同步方案
- 编写控制器状态机、前馈补偿、后备渲染器
- 编写日志/图表/摘要输出与文档

**人工验证方式（关键数据均为实测，非 AI 推测）**：
1. **weld 瞬移**：实测直接激活位移 1157.46 mm，同步后 0.783 mm，两组数字均来自脚本输出
2. **IK 收敛性**：6 个目标点逐一点验，位置误差 0.00098–0.00704 mm
3. **前馈效果**：同一设置下 `--no-ff` 与默认模式各跑一次，
   11.851° → 0.147°、4.85 mm → 0.86 mm
4. **验收运行**：3/3 次成功（reset）+ 4/4 次成功（pingpong）
5. **视频有效性**：不止看文件大小，做了逐帧抽样（6 帧非黑占比 99.8%）
   与帧间差异统计（平均像素差 4.40、静止帧 0%）两层校验，
   确认动画真实变化而非静止图重复
6. **坐标系差异**：实测 C 的 `fk(q)` 与 MuJoCo `wrist_3_link` 误差 0.866 mm / 0°，
   与 `attachment_site` 差 99.5 mm / 90°，据此确定日志采用 TCP 定义
7. **图表逐像素核查**：人工读 `tcp_and_workpiece_path.png` 时发现工件轨迹两端
   有短斜线、TCP 轨迹拱形外凸，一度怀疑存在厘米级滑脱。
   回到执行日志逐行核对后确认：抓取后 `wp − ee` 偏移极差仅
   [3.8, 3.9, 1.0] mm（动态微振），斜线实为关节空间段的笛卡尔弧线
   与 reset 瞬移的重叠绘制，700 mm 步移只出现在循环边界。
   结论是读图误区而非控制缺陷，已把四点读图提示写进 README 供答辩参考

**过程中的错误与修正（如实记录）**：
- 初版放置判定用了 keyframe 初始高度 0.03 而非静止高度 0.035，
  导致误报 4.83 mm 偏差；修正后为 0.174 mm
- `mj_fullM` 参数顺序写错（传成 `(m, dst, d)`），
  前馈开启时首步即崩溃；改用 `mj_mulM(m, d, res, vec)` 修复
- 五次多项式求值时 `tau (N,)` 与系数 `(6,)` 广播失败；
  用 `tau[:, None]` 修复
- 首次做前馈对比时两条命令都误加了 `--no-ff`，导致"对比"结果完全相同、
  该次对比无效；重跑后才得到 §3.3 的有效数据
- `test_weld_grasp.py` 初版让携带段直接在抓取位姿（工件底面离盘仅 0.2 mm）
  做振荡，首段向下运动使工件顶到托盘，接触力顶回 weld 造成 22 mm 的
  "假滑脱"判失败；真实流程 descend 到位后立刻 lift 向上，不存在该接触。
  改为先抬升到搬运高度再振荡后，极差降到 [1.87, 2.66, 0.45] mm。
  这是测试场景设计与控制器状态机不一致造成的，不是控制器缺陷
- 同测试的脱钩判据初版用"TCP 与工件绝对距离 >100 mm"，但 retreat 只把 TCP
  升到 CLEAR_Z=0.12，两者本就相距约 85 mm，判据永远不满足；
  改为比较释放前后位移（TCP 移动 55.4 mm 而工件仅 1.09 mm）
- **接入外部轨迹时的时间尺度缺陷**（真实 bug，非测试问题）：
  `_run_traj` 按"一个轨迹点 = 一次 mj_step(dt=0.002)"执行，
  但未考虑外部轨迹的采样周期可能不同。用 dt=0.01、N=301 的模拟轨迹实测，
  声明 3.00 s 的轨迹实际只执行 0.602 s，即 **5 倍速播放**，
  关节速度、加速度同步放大 5 倍，前馈补偿的 `kv·q̇`、`M·q̈` 项随之失真。
  该缺陷在首次 `--traj-file` 测试时**未被发现**——因为当时只核对了
  "放置成功"，没有核对执行时长；是后续为写给 E 的接口说明做核查时
  才通过日志逐段时长比对暴露出来。
  修复方式见 §4.4（真实时间轴三次样条重采样），
  并补了 10 项自测用例覆盖：重采样生效、dt 一致不重采样、
  偏差 <1% 容差生效、端点值不变、导数不被缩放。
  **教训**：验证"跑通了"不等于验证"跑对了"，
  凡是涉及时间/尺度的接口，必须核对实际时长与声明值是否一致。
