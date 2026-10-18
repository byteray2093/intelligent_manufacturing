# 集成版公共接口

保持 `src/contracts.py` 的 `Pose`、`IKResult`、`JointTrajectory`、`ExecutionLog` 不变。关节角为6维数组、位姿为4×4齐次矩阵，单位统一为 m / rad / s。

| 接口 | 输出及约定 |
|---|---|
| `src.kinematics.fk.fk_flange(q)` | C的独立DH法兰位姿 |
| `src.kinematics.fk.fk_tcp(q)` | C的独立DH+工具变换TCP位姿 |
| `src.kinematics.ik.get_fk(q)` | B的MuJoCo参考TCP位姿 |
| `src.kinematics.ik.ik(T, q_init)` | 公共 `IKResult` |
| `ik_dls` / `ik_dls_limited` | 兼容D/E/F：`(q, success, info)`，误差为 `pos_err` m、`rot_err` rad |
| `plan_pick_and_place(...)` | E轨迹对象，可 `to_npz`、`to_csv` |
| `load_e_trajectory(path)` | 校验时间/维度/有限值；保留NPZ速度与加速度 |
| `PickPlaceController.run(...)` | `summary`、`cycles`、`events`、`log_rows` |

执行用NPZ必需包含 `time[N]`、`q_traj[N,6]`，可含 `dq_traj`、`ddq_traj`；整条搬运轨迹还需 `trajectory_scope='full_pick_place'`、`segment_starts[7]`、`segment_durations[7]`、`segment_names[7]`。CSV支持 E 的 `t,q1..q6`，用于分析；它没有阶段元数据，不能直接作为完整搬运输入。

阶段固定为接近、下降、抬升、水平搬运、下放、退让、回home。在下降后插入吸附及保持，在下放后释放，不能把整条轨迹当成transfer段。整条外部轨迹当前仅支持reset模式，往返模式需要重新规划相反方向。

时间数组是真值，严格递增、从0起始，支持按时间重采样；时长须能由仿真步长整除。执行 N 个采样点对应 N−1 个时间间隔。状态日志在 `mj_step` 后刷新运动学，使 `q_actual` 和 `ee_*` 同帧。

数值失败不静默回退：显式模型/轨迹缺失或不合法抛出异常，笛卡尔段遇到IK失败终止规划。图形渲染后备模式在摘要中明确标识。
