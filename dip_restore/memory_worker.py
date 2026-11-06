"""独立进程的进程峰值 RSS 测量。"""
import argparse
import json
import resource
import sys
import cv2

from .datasets import read_rgb
from .registry import restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    image = read_rgb(args.input)
    restore(image, args.method)  # 与计时一致先预热
    restore(image, args.method)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mib = peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024
    print(json.dumps({"peak_rss_mib": round(float(mib), 3)}))


if __name__ == "__main__":
    main()
