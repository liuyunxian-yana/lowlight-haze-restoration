"""只做低照度增强:输入输出统一 RGB uint8、原尺寸。

包含 Gamma、AGCWD、CLAHE、MSRCP、LIME 五种方法。
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



def _fft_gaussian_even(signal, sigma):
    """镜像扩展后的全分辨率 Gaussian 频域卷积，避免循环边界接缝。"""
    h, w = signal.shape
    mirror = np.pad(signal, ((0, h), (0, w)), mode="symmetric")
    spectrum = np.fft.rfft2(mirror)
    yy = np.fft.fftfreq(2 * h)[:, None]
    xx = np.fft.rfftfreq(2 * w)[None, :]
    transfer = np.exp(-2 * np.pi * np.pi * sigma * sigma * (xx * xx + yy * yy))
    return np.fft.irfft2(spectrum * transfer, s=mirror.shape)[:h, :w]


def msrcp(image, scales=(15.0, 80.0, 150.0), low=1.0, high=99.0):
    """MSRCP：强度 Retinex + 百分位色彩平衡 + RGB 共同增益。内存较高。"""
    x = unit(image).astype(np.float64) * 255.0
    if not scales or any(s <= 0 for s in scales) or not 0 <= low < high <= 100:
        raise ValueError("MSRCP 参数无效")
    intensity = x.mean(axis=2)
    surround = np.zeros_like(intensity)
    for sigma in scales:
        surround += np.log1p(np.maximum(_fft_gaussian_even(intensity, float(sigma)), 0))
    response = np.log1p(intensity) - surround / len(scales)
    p_lo, p_hi = np.percentile(response, (low, high))
    if p_hi - p_lo < 1e-6:
        return image.copy()
    target = np.clip((response - p_lo) * 255 / (p_hi - p_lo), 0, 255)
    peak = x.max(axis=2)
    gain = np.minimum(255 / np.maximum(peak, 1), target / np.maximum(intensity, 1))
    return u8((x * gain[..., None]) / 255.0)


def lime(image, alpha=0.15, gamma=0.8, epsilon=0.01, sigma=2.0,
         tolerance=1e-5, max_iterations=1000):
    """LIME 的结构加权照明估计与 PCG 求解；不含可选 BM3D。CPU/内存较重。"""
    x = unit(image)
    if alpha < 0 or gamma <= 0 or epsilon <= 0 or sigma <= 0 or tolerance <= 0 or max_iterations < 1:
        raise ValueError("LIME 参数无效")
    initial = x.max(axis=2).astype(np.float64)
    if initial.min() == initial.max():
        return image.copy()
    dx = np.diff(initial, axis=1)
    dy = np.diff(initial, axis=0)
    sx = cv2.GaussianBlur(dx, (0, 0), sigma) if dx.size else dx
    sy = cv2.GaussianBlur(dy, (0, 0), sigma) if dy.size else dy
    wx = alpha / ((np.abs(dx) + epsilon) * (np.abs(sx) + epsilon))
    wy = alpha / ((np.abs(dy) + epsilon) * (np.abs(sy) + epsilon))

    def apply(field):
        out = field.copy()
        horizontal = wx * (field[:, :-1] - field[:, 1:])
        vertical = wy * (field[:-1, :] - field[1:, :])
        out[:, :-1] += horizontal
        out[:, 1:] -= horizontal
        out[:-1, :] += vertical
        out[1:, :] -= vertical
        return out

    diagonal = np.ones_like(initial)
    diagonal[:, :-1] += wx
    diagonal[:, 1:] += wx
    diagonal[:-1, :] += wy
    diagonal[1:, :] += wy
    estimate = initial.copy()
    residual = initial - apply(estimate)
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    rz = float(np.sum(residual * preconditioned))
    threshold = tolerance * max(np.linalg.norm(initial), 1e-12)
    for _ in range(max_iterations):
        if np.linalg.norm(residual) <= threshold:
            break
        denominator = float(np.sum(direction * apply(direction)))
        if denominator <= 0:
            raise RuntimeError("LIME PCG 分母非正")
        step = rz / denominator
        estimate += step * direction
        residual -= step * apply(direction)
        preconditioned = residual / diagonal
        rz_next = float(np.sum(residual * preconditioned))
        direction = preconditioned + (rz_next / max(rz, 1e-30)) * direction
        rz = rz_next
    else:
        raise RuntimeError("LIME 照明求解未在 max_iterations 内收敛")
    illumination = np.clip(estimate, 0, 1) ** gamma
    return u8(x / np.maximum(illumination[..., None], epsilon))


ENHANCERS = {
    "gamma": gamma,
    "agcwd": agcwd,
    "clahe": clahe,
    "msrcp": msrcp,
    "lime": lime,
}
