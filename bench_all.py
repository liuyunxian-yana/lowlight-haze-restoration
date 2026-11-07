"""所有现有 RGB uint8 方法的统一批量入口；算法本身保持独立。"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import json
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ceef import restore as ceef_restore
from dip_restore.core import validate
from dip_restore.datasets import Case, IMAGE_EXTS, load, read_rgb, verify, write_rgb
from dip_restore.metrics import diagnostics, measure
from dip_restore.registry import ALL as REGISTERED, restore as registry_restore

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "benchmark"
IMAGE_SUFFIXES = {s.lower() for s in IMAGE_EXTS}
PAIR_NAMES = (("degraded", "GT"), ("degraded", "gt"), ("input", "gt"),
              ("input", "GT"), ("foggy", "clean"), ("hazy", "clear"),
              ("images", "GT"))
METRIC_COLUMNS = ("psnr_db", "ssim_rgb", "delta_e76", "luma_mae",
                  "mean_luma", "dark_ratio", "bright_ratio", "luma_entropy")


@dataclass(frozen=True)
class Sample:
    dataset: str
    id: str
    input_path: Path
    reference_path: Path | None
    case: Case | None = None
    source_id: str = ""
    split: str = ""
    kind: str = ""
    severity: str = ""


def methods_catalog() -> dict[str, callable]:
    """汇总项目注册方法、未注册的 CEEF 和已实现的两种串联顺序。"""
    methods = {name: (lambda image, spec=name: registry_restore(image, spec))
               for name in sorted(REGISTERED)}
    methods["ceef"] = ceef_restore
    methods["gamma>dcp"] = lambda image: registry_restore(image, "gamma>dcp")
    methods["dcp>gamma"] = lambda image: registry_restore(image, "dcp>gamma")
    return dict(sorted(methods.items()))


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _stem(path: Path, folder: Path, suffix: str = "") -> str:
    relative = path.relative_to(folder).with_suffix("").as_posix()
    if suffix and relative.endswith(suffix):
        relative = relative[:-len(suffix)]
    return relative


def _index(folder: Path, suffix: str = "") -> dict[str, Path]:
    found = {}
    for path in image_files(folder):
        key = _stem(path, folder, suffix)
        if not key or key in found:
            raise ValueError(f"图像名称重复或为空：{path}")
        found[key] = path
    return found


def _known(name: str, dataset_root: Path) -> list[Sample]:
    if name == "synthetic":
        cases = (load("synthetic", "val", "all", data_root=dataset_root) +
                 load("synthetic", "test", "all", data_root=dataset_root))
    else:
        cases = load(name)
    return [Sample(name, c.id, c.input_path, c.reference_path, c,
                   c.source_id, c.split, c.kind, c.severity) for c in cases]


def discover_metrics(root: Path) -> tuple[dict[str, list[Sample]], list[dict]]:
    """每个 data 子目录作为一个数据集；参考目录存在时按相对文件名严格配对。"""
    found, issues = {}, []
    if not root.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{root}")
    for dataset in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "new"):
        name = dataset.name
        try:
            if (name == "synthetic" and (dataset / "manifest.json").is_file()) or (
                name in {"nthazy", "gta5"} and root.resolve() == DATA.resolve()
            ):
                samples = _known(name, dataset)
            else:
                pair = next(((dataset / a, dataset / b) for a, b in PAIR_NAMES
                             if (dataset / a).is_dir() and (dataset / b).is_dir()), None)
                if pair:
                    inputs, references = pair
                    input_suffix, reference_suffix = (("_NTHazy", "_GT") if name == "nthazy" else ("", ""))
                    x, y = _index(inputs, input_suffix), _index(references, reference_suffix)
                    samples = [Sample(name, key, path, y.get(key), source_id=key)
                               for key, path in sorted(x.items())]
                    for key in sorted(x.keys() - y.keys()):
                        issues.append(dict(dataset=name, id=key, error="参考图缺失"))
                    for key in sorted(y.keys() - x.keys()):
                        issues.append(dict(dataset=name, id=key, error="输入图缺失"))
                    # 有 GT 目录时，不把缺少 GT 的样本当作无参考图像。
                    samples = [s for s in samples if s.reference_path is not None]
                else:
                    reference_folders = [dataset / n for n in ("GT", "gt", "clean", "clear", "reference")
                                         if (dataset / n).is_dir()]
                    if reference_folders:
                        raise ValueError(f"发现参考目录但没有识别到输入目录对：{reference_folders}")
                    input_folder = next((dataset / n for n in ("degraded", "input", "images", "foggy", "hazy")
                                         if (dataset / n).is_dir()), dataset)
                    samples = [Sample(name, _stem(path, input_folder), path, None,
                                      source_id=_stem(path, input_folder)) for path in image_files(input_folder)]
            if samples:
                found[name] = samples
            else:
                issues.append(dict(dataset=name, id="", error="未找到可评估图像"))
        except Exception as exc:
            issues.append(dict(dataset=name, id="", error=f"发现数据集失败：{type(exc).__name__}: {exc}"))
    return found, issues


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(id_: str) -> str:
    return id_.replace("/", "__").replace("\\", "__")


def _check_output(output: np.ndarray, shape: tuple[int, ...]) -> None:
    validate(output, shape)


def _comparison_grid(panels: list[tuple[str, np.ndarray | None]], columns: int = 4) -> np.ndarray:
    """把原图和各算法输出缩入等尺寸画格，并在画格顶部写名称。"""
    tile_w, tile_h, label_h = 320, 272, 28
    rows = (len(panels) + columns - 1) // columns
    canvas = np.full((rows * tile_h, columns * tile_w, 3), 245, np.uint8)
    for index, (name, image) in enumerate(panels):
        top, left = divmod(index, columns)
        y0, x0 = top * tile_h, left * tile_w
        cv2.putText(canvas, name, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (25, 25, 25), 1, cv2.LINE_AA)
        if image is None:
            cv2.putText(canvas, "FAILED", (x0 + 8, y0 + 64), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (180, 30, 30), 2, cv2.LINE_AA)
            continue
        h, w = image.shape[:2]
        scale = min(tile_w / w, (tile_h - label_h) / h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        shown = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        y = y0 + label_h + (tile_h - label_h - size[1]) // 2
        x = x0 + (tile_w - size[0]) // 2
        canvas[y:y + size[1], x:x + size[0]] = shown
    return canvas


def _preview(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, 320 / w, 244 / h)
    if scale == 1:
        return image.copy()
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _summaries(rows: list[dict], methods: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    result = []
    for method in methods:
        group = grouped[method]
        good = [r for r in group if r["status"] == "ok"]
        item = dict(method=method, n_expected=len(group), n_ok=len(good),
                    n_failed=len(group)-len(good), n_sources=len({r["source_id"] for r in group}))
        for key in (*METRIC_COLUMNS, "cpu_ms", "wall_ms"):
            values = [float(r[key]) for r in good if r[key] != ""]
            item[key] = float(np.mean(values)) if values else ""
        result.append(item)
    return result


def run_visual(data_root: Path, out: Path, methods: dict[str, callable], limit: int | None) -> list[dict]:
    folder = data_root / "new"
    if not folder.is_dir():
        raise FileNotFoundError(f"visual 需要目录：{folder}")
    images = image_files(folder)
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError(f"未找到图像：{folder}")
    failures = []
    for path in images:
        relative = path.relative_to(folder).with_suffix(".png")
        try:
            original = read_rgb(path)
            write_rgb(out / "original" / relative, original)
        except Exception as exc:
            failures.append(dict(mode="visual", dataset="new", image=str(path), method="original",
                                 error=f"{type(exc).__name__}: {exc}"))
            continue
        panels = [("original", _preview(original))]
        for name, method in methods.items():
            try:
                output = method(original.copy())
                _check_output(output, original.shape)
                write_rgb(out / name.replace(">", "__then__") / relative, output)
                panels.append((name, _preview(output)))
            except Exception as exc:
                failures.append(dict(mode="visual", dataset="new", image=str(path), method=name,
                                     error=f"{type(exc).__name__}: {exc}"))
                panels.append((name, None))
        try:
            write_rgb(out / "comparison" / relative, _comparison_grid(panels))
        except Exception as exc:
            failures.append(dict(mode="visual", dataset="new", image=str(path), method="comparison",
                                 error=f"{type(exc).__name__}: {exc}"))
    return failures


def run_metrics(data_root: Path, out: Path, methods: dict[str, callable], limit: int | None) -> list[dict]:
    datasets, issues = discover_metrics(data_root)
    if not datasets:
        raise ValueError("未找到评测数据集")
    failures = [dict(mode="metrics", method="discovery", image="", **issue) for issue in issues]
    for name, samples in datasets.items():
        selected = samples[:limit] if limit is not None else samples
        rows = []
        for method_name, method in methods.items():
            for sample in selected:
                row = dict(dataset=name, id=sample.id, source_id=sample.source_id,
                           split=sample.split, kind=sample.kind, severity=sample.severity,
                           method=method_name, has_gt=sample.reference_path is not None,
                           status="ok", cpu_ms="", wall_ms="",
                           **dict.fromkeys(METRIC_COLUMNS, ""), error="")
                try:
                    image, reference = verify(sample.case) if sample.case else (
                        read_rgb(sample.input_path), read_rgb(sample.reference_path) if sample.reference_path else None)
                    if reference is not None and image.shape != reference.shape:
                        raise ValueError(f"输入/GT 尺寸不一致：{image.shape} / {reference.shape}")
                    wall0, cpu0 = time.perf_counter(), time.process_time()
                    result = method(image.copy())
                    row["cpu_ms"] = (time.process_time()-cpu0)*1000
                    row["wall_ms"] = (time.perf_counter()-wall0)*1000
                    _check_output(result, image.shape)
                    row.update(measure(reference, result) if reference is not None else diagnostics(result))
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    failures.append(dict(mode="metrics", dataset=name, image=str(sample.input_path),
                                         method=method_name, error=row["error"]))
                rows.append(row)
        columns = ("dataset", "id", "source_id", "split", "kind", "severity", "method",
                   "has_gt", "status", "cpu_ms", "wall_ms", *METRIC_COLUMNS, "error")
        summary_columns = ("method", "n_expected", "n_ok", "n_failed", "n_sources",
                           *METRIC_COLUMNS, "cpu_ms", "wall_ms")
        _write_csv(out / name / "per_image.csv", rows, columns)
        _write_csv(out / name / "summary.csv", _summaries(rows, list(methods)), summary_columns)
        print(f"{name}: {len(selected)} 张 × {len(methods)} 法，失败 {sum(r['status'] == 'failed' for r in rows)}", flush=True)
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="全部算法的视觉与指标批量实验")
    parser.add_argument("--mode", choices=("visual", "metrics"), required=True)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--tag", default="run", help="结果子目录名，默认 run")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正数")
    if args.tag in {"", ".", ".."} or "/" in args.tag or "\\" in args.tag:
        parser.error("--tag 只能是单个目录名")
    catalog = methods_catalog()
    names = list(catalog) if args.methods == ["all"] else args.methods
    unknown = sorted(set(names) - set(catalog))
    if unknown:
        parser.error(f"未知算法：{unknown}；可选：{list(catalog)}")
    if len(names) != len(set(names)):
        parser.error("方法列表重复")
    methods = {name: catalog[name] for name in names}
    print(f"算法清单（{len(methods)}）：{', '.join(methods)}", flush=True)
    out = RESULTS / args.mode / args.tag
    if out.exists() and any(out.iterdir()):
        parser.error(f"输出目录非空：{out}；请更换 --tag")
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    try:
        failures = (run_visual(args.data_root, out, methods, args.limit) if args.mode == "visual"
                    else run_metrics(args.data_root, out, methods, args.limit))
    except Exception as exc:
        failures = [dict(mode=args.mode, dataset="", image="", method="setup",
                         error=f"{type(exc).__name__}: {exc}")]
        print(traceback.format_exc(), flush=True)
    (out / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "run.json").write_text(json.dumps(dict(mode=args.mode, methods=names, limit=args.limit,
        data_root=str(args.data_root.resolve()), failures=len(failures),
        timing="单次进程 CPU 与墙钟时间；只计算法调用，不计图像读写和指标",
        no_reference="mean_luma、dark_ratio、bright_ratio、luma_entropy 为无参考诊断量"),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出：{out}；失败 {len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
