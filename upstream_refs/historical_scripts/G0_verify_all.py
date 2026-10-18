"""G0 一键验收：把 G 组所有检查跑一遍，最后给一个"能交/不能交"的结论。

用途：你（或批改的人）不想逐条敲命令时，在这个目录下执行

    python scripts/G0_verify_all.py            # 全量：测试 + G1~G9 + 产物清单（44 CSV/JSON + 26 图）
    python scripts/G0_verify_all.py --no-figs  # 不生成图（没有 matplotlib 也能跑）
    python scripts/G0_verify_all.py --quick    # 只跑语法检查、测试与产物清单，跳过 G1~G9

三组实验的区别：
- G1~G4＝第一阶段（占位 DH 模型 + 自写参考轨迹），只需要 NumPy/SciPy，验证"分析算法"；
- G5~G8＝第二阶段（B 的真实 MuJoCo 模型 + C/D/F 的真实交付 + F 的真实执行日志），
  需要 mujoco 与 `universal_robots_ur5e/ur5e.xml`、`results/logs/exec_log_latest.csv`；
- G9＝第三阶段（接入 E 的规划轨迹），需要 `upstream_refs/e_traj/pick_place_traj.npz`
  与 `upstream_refs/e_plan/`（E 的轨迹模块副本）；集成复跑还需要
  `results/logs/exec_log_e_traj_1cycle.csv`，缺失时脚本会自动跳过该节。

退出码：0 = 全绿；1 = 有失败项（失败明细会打印出来）。

这个脚本**只读**已有产物、只调用同伴脚本，不新增任何公共接口。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# matplotlib 默认会去用户目录建缓存，在受限环境里会报 WinError 5；
# 子进程统一指到仓库内的 results/.mplconfig（与 src/analysis/plots.py 一致）。
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".mplconfig"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PASS, FAIL = "[ OK ]", "[FAIL]"


def run(cmd: list[str], title: str) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    print(f"{PASS if ok else FAIL} {title}")
    if not ok:
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-12:]
        for ln in tail:
            print("        | " + ln)
    return ok, out


def main() -> int:
    ap = argparse.ArgumentParser(description="G 组一键验收")
    ap.add_argument("--no-figs", action="store_true", help="透传给 G1/G3/G5/G6/G8，不生成图表")
    ap.add_argument("--quick", action="store_true", help="只跑测试与产物清单")
    args = ap.parse_args()

    print("=" * 78)
    print("G 组一键验收（第一阶段：分析算法；第二阶段：真实模型与真实日志；第三阶段：E 的规划轨迹）")
    print("=" * 78)
    print(f"目录    : {ROOT}")
    print(f"Python  : {sys.version.split()[0]}  ({PY})")
    for name in ("numpy", "scipy", "mujoco"):
        try:
            mod = __import__(name)

            print(f"{name:<9}: {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<9}: 不可用 -> {exc}")
            if name == "numpy":
                print("\n请先执行：python -m pip install -r requirements.txt")
                return 1
    try:
        import matplotlib

        print(f"matplotlib: {matplotlib.__version__}（有图）")
        has_mpl = True
    except Exception:  # noqa: BLE001
        print("matplotlib: 未安装（跳过图表，仍出 CSV 与结论）")
        has_mpl = False
    print("-" * 78)

    results: list[tuple[str, bool]] = []

    results.append(("G1~G9 脚本可编译（语法检查）", run(
        [PY, "-m", "compileall", "-q", "src", "scripts", "tests"], "语法检查 src/scripts/tests")[0]))

    if not args.quick:
        nofig = ["--no-figs"] if (args.no_figs or not has_mpl) else []
        stage1 = [
            ("G1 轨迹指标（占位模型）", ["scripts/G1_trajectory_metrics.py", *nofig]),
            ("G2 工作空间与可达性（占位模型）", ["scripts/G2_workspace_reachability.py"]),
            ("G3 奇异点与可达性案例（占位模型）", ["scripts/G3_singularity_reachability.py", *nofig]),
            ("G4 接口复用性冒烟", ["scripts/G4_interface_smoke.py"]),
        ]
        stage2 = [
            ("G5 上游交付一致性验证（C/D/F）", ["scripts/G5_model_validation.py", *nofig]),
            ("G6 真实模型奇异点分析", ["scripts/G6_real_singularity.py", *nofig]),
            ("G7 执行日志跟踪误差分析", ["scripts/G7_exec_tracking.py", *nofig]),
            ("G8 真实模型可达性分析", ["scripts/G8_real_reachability.py", *nofig]),
        ]
        stage3 = [
            ("G9 E 组轨迹验证与 E→F 集成实验", ["scripts/G9_e_trajectory.py", *nofig]),
        ]
        for title, cmd in stage1 + stage2 + stage3:
            results.append((title, run([PY, *cmd], title)[0]))

    ok, out = run([PY, "-m", "pytest", "tests", "-q"], "pytest 全量测试")
    results.append(("pytest 全量测试", ok))
    if ok:
        line = [ln for ln in out.splitlines() if "passed" in ln or "failed" in ln]
        if line:
            print(f"        {line[-1].strip()}")

    print("-" * 78)
    print("产物清单")
    csvs = sorted((ROOT / "results" / "csv").glob("G*"))
    figs = sorted((ROOT / "results" / "figures").glob("G*.png"))
    print(f"  CSV/JSON : {len(csvs)} 个")
    for p in csvs:
        print(f"    - {p.relative_to(ROOT)}  ({p.stat().st_size:,} B)")
    print(f"  图       : {len(figs)} 个")
    for p in figs:
        print(f"    - {p.relative_to(ROOT)}  ({p.stat().st_size:,} B)")
    expect_csv, expect_fig = 44, 26
    results.append((f"CSV/JSON 数量 == {expect_csv}", len(csvs) == expect_csv))
    results.append((f"图表数量 == {expect_fig}", len(figs) == expect_fig))

    print("=" * 78)
    bad = [name for name, okk in results if not okk]
    for name, okk in results:
        print(f"{PASS if okk else FAIL} {name}")
    print("=" * 78)
    if bad:
        print(f"结论：{len(bad)} 项未通过 -> 先不要提交，逐个看上面的 [FAIL] 明细。")
        return 1
    print("结论：全部通过，可以交给人检。")
    print("下一步：人工看 docs/G_结论说明.md 里的数字与 results/csv/ 是否对得上，再决定是否上传。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
