"""新项目唯一运行入口。"""
import os

# 必须在 NumPy/OpenCV 导入前设置；默认在 MacBook Air 上轻负载执行。
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
from pathlib import Path

from dip_restore.benchmark import DEFAULT_METHODS, run
from dip_restore.registry import ALL


def main():
    p = argparse.ArgumentParser(description="CPU 微光有雾图像恢复与对比实验")
    p.add_argument("--dataset", choices=("synthetic", "nthazy", "gta5", "external"), default="synthetic")
    p.add_argument("--data-root", type=Path, help="自生成的 synthetic 清单目录")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--kind", choices=("lowlight", "haze", "joint", "all"), default="joint")
    p.add_argument("--input", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--input-suffix", default="")
    p.add_argument("--reference-suffix", default="")
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                   help=f"可用单项: {', '.join(ALL)}；串联如 gamma>dcp")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--compare-count", type=int, default=2)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--memory-sample", action="store_true", help="每个方法另开子进程测首图进程峰值RSS")
    args = p.parse_args()
    try:
        return run(dataset=args.dataset, split=args.split, kind=args.kind, methods=args.methods,
                   output=args.out, input_dir=args.input, reference_dir=args.reference,
                   input_suffix=args.input_suffix, reference_suffix=args.reference_suffix,
                   repeats=args.repeats, limit=args.limit, compare_count=args.compare_count,
                   save_images=args.save_images, measure_memory=args.memory_sample, data_root=args.data_root)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
