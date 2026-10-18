# 原始交付及历史参考

`D/ik_original.py` 是 D 交付的原始源码（含原机路径，不能直接运行）；当前修复版在 `src/kinematics/ik.py`。

`original_logs/` 是 G 交付中的原始F日志，包含曾把完整E轨迹放进transfer执行的复跑。它们只用于追溯旧问题，不是当前版本验收结果。

`e_plan/`、`e_traj/`、`d_ik_cases.csv` 来自 G 的上游参考归档。当前 E 源码在 `src/planning/`，新生成轨迹在 `data/`。

`historical_scripts/` 与 `historical_tests/` 保留旧G分析流程。旧测试会断言“日志滞后一帧”“存在约0.4 rad指令跳变”等已修复问题，且依赖固定历史CSV/JSON和点数，故不作为当前回归测试。当前验收入口为 `python scripts/verify_all.py`，测试范围在 `pytest.ini` 明确指定为 `tests/`。

`docs/B_original`、`E_original`、`F_original`、`G_original` 保存组员原说明，其中的路径、进度和数字以原交付时刻为准。当前说明以README、INTERFACES及INTEGRATION_REPORT为准。
