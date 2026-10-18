# 模型与坐标冻结

模型由 B 交付，来源为 Google DeepMind MuJoCo Menagerie 的 Universal Robots UR5e；保留原 `universal_robots_ur5e/LICENSE` 及资源来源说明。本次未从网络替换组员模型。

- 独立机器人模型：`universal_robots_ur5e/ur5e.xml`；完整搬运场景：`scenes/pick_place.xml`。
- 世界坐标系与机器人基座坐标系重合，长度 m，角度 rad。
- 关节顺序：`shoulder_pan_joint`、`shoulder_lift_joint`、`elbow_joint`、`wrist_1_joint`、`wrist_2_joint`、`wrist_3_joint`。
- 限位读取模型 `jnt_range`；第三轴约 ±π，其余约 ±2π。仿真步长 0.002 s。
- 法兰为 `wrist_3_link`；TCP 为 `attachment_site`。TCP 在法兰局部坐标中的平移为 `[0,0.1,0]`，旋转为绕 X 轴 −90°。世界位姿满足 `T_world_tcp = T_world_flange @ T_flange_tcp`。
- C 的 `fk` / `fk_flange` 保留原有法兰语义；C 的 `fk_tcp` 加显式工具变换。D 的 `get_fk`、规划、执行日志统一为TCP。
- C 标准 DH 参数与 Menagerie 的连杆尺寸存在毫米级差别，验收保留并报告，不将两者作为机器精度一致的模型。

完整场景中工件质量0.1 kg，边长0.05 m，初始中心 `[-0.35,0.4,0.035]`，下料区 `[0.35,0.4]`。原地同步相对位姿后激活 weld，模拟理想吸附；解除后自由落下。未建实体吸盘/夹爪，不能把结果解释为真实接触抓取性能。

集成时修正了初始工件嵌入托盘、场景keyframe维度、mesh路径与工作台/地面视觉重叠。机械臂模型的连杆、惯量、执行器参数保留组员交付。
