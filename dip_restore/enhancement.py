"""只做低照度增强:输入输出统一 RGB uint8、原尺寸。

包含 Gamma、AGCWD、CLAHE 方法。
共用 core 的图像契约与滤波工具
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import u8, unit, validate


def gamma(image, exponent=0.6):
    """全局幂律提亮；用于基线，不抑制暗部噪声（当 0<\gamma<1 时，暗像素往往会被明显提亮，而亮像素相对变化较小。）。"""
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("exponent 必须为正数")
    return u8(np.power(unit(image), exponent))


def agcwd(image, alpha=0.5):
    """Huang 等的加权直方图 CDF 自适应 Gamma，作用于 HSV 的 V。"""
    validate(image)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha 必须为正数")
    # 把 RGB 转换为 HSV（色相、饱和度、明度）
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    # 统计亮度直方图
    hist = np.bincount(v.ravel(), minlength=256).astype(np.float64)
    if np.count_nonzero(hist) <= 1:
        return image.copy()
    pdf = hist / v.size
    lo, hi = float(pdf.min()), float(pdf.max())
    if hi <= lo:
        return image.copy()
    # 对直方图进行加权
    weighted = hi * ((pdf - lo) / (hi - lo)) ** alpha
    # 计算 CDF（加权后亮度小于或等于当前亮度值的像素累计占多少比例）并构造 Gamma
    cdf = np.cumsum(weighted)
    cdf /= cdf[-1]
    levels = np.arange(256, dtype=np.float64) / 255.0
    # LUT 映射 V 通道并再转换回 RGB
    lut = np.rint(255 * levels ** (1 - cdf)).astype(np.uint8)
    hsv[..., 2] = lut[v]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)



def clahe(image, clip_limit=2.0, grid=8):
    """CLAHE：分块、直方图裁剪、CDF 映射、双线性插值。"""
    validate(image)
    if not np.isfinite(clip_limit) or clip_limit <= 0 or not isinstance(grid, (int, np.integer)) or grid < 1:
        raise ValueError("CLAHE 参数无效")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # 明度
    L = lab[..., 0]
    if L.min() == L.max():
        return image.copy()

    h, w = L.shape
    # 向上取整 th/tw：每个小块的高度/宽度
    th, tw = (h + grid - 1) // grid, (w + grid - 1) // grid
    # 反射填充
    padded = np.pad(L, ((0, grid*th-h), (0, grid*tw-w)), mode="reflect")
    luts = np.empty((grid, grid, 256), np.uint8)
    # 裁剪阈值
    limit = max(1, int(clip_limit * th * tw / 256))

    # 每块计算直方图、裁剪、重新分配、CDF
    for i in range(grid):
        for j in range(grid):
            # 遍历 64 个小块
            tile = padded[i*th:(i+1)*th, j*tw:(j+1)*tw]
            # 统计直方图
            hist = np.bincount(tile.ravel(), minlength=256)
            # 裁剪直方图
            excess = int(np.maximum(hist - limit, 0).sum())
            hist = np.minimum(hist, limit)
            # 把裁剪掉的计数重新分配
            q, r = divmod(excess, 256)
            hist += q
            if r:
                hist[np.arange(r) * (256 // r)] += 1
            # CDF 构建 LUT
            luts[i, j] = np.rint(255 * np.cumsum(hist) / (th*tw)).astype(np.uint8)

    # 相邻四块的 LUT 进行双线性插值（距离哪个块中心更近，就更多参考哪个块的 LUT）
    y, x = np.arange(h)/th - 0.5, np.arange(w)/tw - 0.5
    i, j = np.floor(y).astype(int), np.floor(x).astype(int)
    wy, wx = (y-i)[:, None], (x-j)[None, :]
    i0, i1 = np.clip(i, 0, grid-1)[:, None], np.clip(i+1, 0, grid-1)[:, None]
    j0, j1 = np.clip(j, 0, grid-1)[None, :], np.clip(j+1, 0, grid-1)[None, :]
    top = (1-wx)*luts[i0,j0,L] + wx*luts[i0,j1,L]
    bottom = (1-wx)*luts[i1,j0,L] + wx*luts[i1,j1,L]

    lab[..., 0] = np.rint((1-wy)*top + wy*bottom).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


ENHANCERS = {
    "gamma": gamma,
    "agcwd": agcwd,
    "clahe": clahe,
}
