"""CSV / JSON 输出工具：所有 G 组数据都带"可复现元数据"。

交付准则要求：图表必须能追溯到原始参数、时间戳与生成脚本。因此每次运行都写一份
`*_meta.json`，记录：时间戳、git 提交（可用时）、Python/NumPy 版本、随机种子、
采样条件、模型来源、以及本次运行生成的图与数据文件清单。

**CSV 编码约定**：G 组 CSV 一律写成 UTF-8 with BOM（`utf-8-sig`），这样 Excel 双击打开
中文列名不乱码。读取时必须用 `utf-8-sig`——用普通 `utf-8` 会把 BOM 读进**第一列的列名**
（如 `'\ufefflabel'`），这是本模块唯一容易被下游踩到的坑，因此提供
`read_csv_rows()` 作为配套读取入口。
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def write_csv(path: str | Path, rows: list[dict], field_order: list[str] | None = None) -> Path:
    """写 CSV（UTF-8 with BOM，便于直接用 Excel 打开中文列名）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8-sig")
        return p.resolve()
    fields = field_order or list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fields})
    return p.resolve()


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(_fmt(v)) for v in value) + "]"
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _fmt(value.item())
        return "[" + ",".join(str(_fmt(v)) for v in value.ravel()) + "]"
    return value


def read_csv_rows(path: str | Path) -> list[dict]:
    """读回 G 组 CSV（用 `utf-8-sig`，避免 BOM 混进第一列的列名）。

    与 `write_csv()` 配对：写 `utf-8-sig`、读 `utf-8-sig`。下游（H 组报告、批改脚本、
    其它组做横向对比）请优先用这个入口，而不是自己 `open(..., encoding="utf-8")`。
    """
    with Path(path).open("r", newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_series_csv(path: str | Path, columns: dict[str, np.ndarray]) -> Path:
    """把等长的几列数组写成一个 CSV（逐采样点数据）。"""
    length = len(next(iter(columns.values())))
    rows = []
    for i in range(length):
        rows.append({k: (v[i] if np.ndim(v[i]) == 0 else ";".join(f"{x:.10g}" for x in np.ravel(v[i])))
                     for k, v in columns.items()})
    return write_csv(path, rows)


def git_commit() -> str:
    """尽力获取当前 git 提交；没有 git 时返回说明字符串（不影响分析）。"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - 仅用于元数据，任何失败都不应中断分析
        pass
    return "not-a-git-checkout"


def write_meta_json(path: str | Path, extra: dict | None = None) -> Path:
    """写运行元数据（时间戳、环境、种子、产物清单）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "G / singularity-reachability-error analysis",
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "script": sys.argv[0],
    }
    if extra:
        meta.update(extra)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return p.resolve()
