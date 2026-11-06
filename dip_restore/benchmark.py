"""批量实验、严格配对、全参考质量、CPU时间与隔离进程内存测量。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .datasets import load, verify, write_rgb
from .metrics import diagnostics, measure
from .registry import filename, parse, restore

ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = ("psnr_db", "ssim_rgb", "delta_e76", "luma_mae")
DIAG_KEYS = ("dark_ratio", "bright_ratio", "mean_luma", "luma_entropy")
DEFAULT_METHODS = ("identity", "agcwd", "dcp", "gamma>dcp", "dcp>gamma", "model_global_e", "model_full", "model_light_only", "model_haze_only", "side2020", "zhang2014")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        for level in (row["severity"], "ALL"):
            groups[(row["method"], row["kind"], level)].append(row)
    summary = []
    for (method, kind, level), group in sorted(groups.items()):
        good = [r for r in group if r["status"] == "ok"]
        line = dict(method=method, kind=kind, severity=level, n_expected=len(group), n_ok=len(good),
                    n_failed=len(group)-len(good), n_sources=len({r["source_id"] for r in group}))
        for key in METRIC_KEYS + DIAG_KEYS + ("wall_ms", "cpu_ms"):
            vals = [r[key] for r in good if r[key] != ""]
            line[key] = float(np.mean(vals)) if vals else ""
        summary.append(line)
    return summary


def memory_profile(method, image_path, timeout=120):
    """每个方法新开一个进程，记录进程峰值 RSS；包含解释器、库和图像缓冲。"""
    command = [sys.executable, "-m", "dip_restore.memory_worker", "--method", method, "--input", str(image_path)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode:
        return dict(method=method, status="failed", peak_rss_mib="", error=proc.stderr.strip()[-500:])
    result = json.loads(proc.stdout.strip())
    return dict(method=method, status="ok", peak_rss_mib=result["peak_rss_mib"], error="")


def run(dataset="synthetic", split="test", kind="joint", methods=DEFAULT_METHODS,
        output=Path("results/run"), input_dir=None, reference_dir=None, input_suffix="", reference_suffix="",
        repeats=1, limit=None, compare_count=2, save_images=False, measure_memory=False, data_root=None):
    if repeats < 1 or compare_count < 0 or (limit is not None and limit < 1):
        raise ValueError("repeats、limit 必须为正数，compare_count 不能为负")
    if len(set(methods)) != len(methods):
        raise ValueError("方法列表中有重复")
    for spec in methods:
        parse(spec)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空；请指定新的目录")
    cases = load(dataset, split, kind, input_dir, reference_dir, input_suffix, reference_suffix, data_root)
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("没有符合条件的图像")
    output.mkdir(parents=True, exist_ok=True)
    # macOS GCD 构建中 setNumThreads(1) 无效，0 才强制为 1。
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    rows, errors = [], []
    for method in methods:
        warmed = set()
        shown = 0
        for case in cases:
            row = dict(id=case.id, source_id=case.source_id, kind=case.kind, severity=case.severity,
                       method=method, status="ok", wall_ms="", cpu_ms="",
                       **dict.fromkeys(METRIC_KEYS + DIAG_KEYS, ""), error="")
            try:
                image, reference = verify(case)
                if image.shape not in warmed:
                    restore(image, method)
                    warmed.add(image.shape)
                wall, cpu = [], []
                for _ in range(repeats):
                    start_wall, start_cpu = time.perf_counter(), time.process_time()
                    result = restore(image, method)
                    cpu.append((time.process_time() - start_cpu) * 1000)
                    wall.append((time.perf_counter() - start_wall) * 1000)
                row["wall_ms"], row["cpu_ms"] = float(np.median(wall)), float(np.median(cpu))
                row.update(diagnostics(result))
                if reference is not None:
                    row.update(measure(reference, result))
                if save_images:
                    write_rgb(output / "images" / filename(method) / f"{case.id}.png", result)
                if shown < compare_count:
                    panels = [image, result] + ([reference] if reference is not None else [])
                    write_rgb(output / "comparisons" / filename(method) / f"{case.id}.png", np.concatenate(panels, axis=1))
                    shown += 1
            except Exception as exc:
                row["status"], row["error"] = "failed", f"{type(exc).__name__}: {exc}"
                errors.append(dict(row))
            rows.append(row)
        print(f"{method}: {len(cases)-sum(e['method']==method for e in errors)}/{len(cases)}", flush=True)
    save_csv(output / "per_image.csv", rows)
    save_csv(output / "summary.csv", aggregate(rows))
    save_json(output / "failures.json", errors)
    memory_rows = []
    if measure_memory:
        for method in methods:
            memory_rows.append(memory_profile(method, cases[0].input_path))
        save_csv(output / "memory.csv", memory_rows)
    code_hash = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((ROOT / "dip_restore").glob("*.py"))}
    manifest = (Path(data_root) if data_root is not None else ROOT / "data/synthetic") / "manifest.json"
    save_json(output / "run.json", dict(
        dataset=dataset, split=split if dataset == "synthetic" else None,
        data_root=str(Path(data_root).resolve()) if data_root is not None else None,
        kind=kind if dataset == "synthetic" else None, n_cases=len(cases), n_sources=len({c.source_id for c in cases}),
        methods=list(methods), repeats=repeats, limit=limit, compare_count=compare_count,
        python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__, platform=platform.platform(),
        opencv_threads=cv2.getNumThreads(), opencl=cv2.ocl.useOpenCL(),
        code_sha256=code_hash,
        joint_model_defaults=dict(white_anchor=0.85, min_exposure=0.025, omega=0.6,
                                  max_veil_fraction=0.85, noise_sigma=0.03,
                                  exposure_shape_clip=[0.75, 1.25]),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest() if dataset == "synthetic" else None,
        failures=len(errors), memory_failures=sum(r["status"] == "failed" for r in memory_rows),
        timing="预热后每图重复运行中位数；wall_ms=墙钟，cpu_ms=进程CPU；读写、哈希、指标不计入",
        memory="每法独立子进程处理首张图，报告进程峰值RSS；含库/图像内存，不是纯算法增量" if measure_memory else None,
        metrics="sRGB RGB uint8 PSNR; 11x11 Gaussian sigma1.5 valid RGB SSIM; 不自动缩放参考图"))
    return 1 if errors or any(r["status"] == "failed" for r in memory_rows) else 0
