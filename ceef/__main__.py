"""CEEF 单图命令行入口；OpenCV 仅负责 PNG/JPEG 读写。"""
import argparse
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
from .core import restore


def main():
    parser = argparse.ArgumentParser(description="CEEF 2021 Python 复现")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error(f"读取失败：{args.input}")
    start = time.process_time()
    output = restore(bgr[..., ::-1].copy())
    cpu_ms = (time.process_time() - start) * 1000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output[..., ::-1]):
        parser.error(f"写入失败：{args.output}")
    print(json.dumps({"output": str(args.output), "cpu_ms": round(cpu_ms, 2),
                      "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
