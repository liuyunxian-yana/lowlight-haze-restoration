"""低照度单项基线；输入输出统一 RGB uint8、原尺寸。

来源：原项目 B 的 gamma/AGCWD/CLAHE/Retinex 变体/MSRCP/LIME/VEViD-Lite。
这里按各自数学定义重新实现，共用 core 的图像契约与滤波工具；Retinex
是受启发的单尺度增益法，LIME 不含论文可选 BM3D 去噪。
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import smooth, u8, unit, validate


def gamma(image, exponent=0.6):
    """全局幂律提亮；用于廉价基线，不抑制暗部噪声。"""
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("exponent 必须为正数")
    return u8(np.power(unit(image), exponent))


def agcwd(image, alpha=0.5):
    """Huang 等的加权直方图 CDF 自适应 Gamma，作用于 HSV 的 V。"""
    validate(image)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha 必须为正数")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    hist = np.bincount(v.ravel(), minlength=256).astype(np.float64)
    if np.count_nonzero(hist) <= 1:
        return image.copy()
    pdf = hist / v.size
    lo, hi = float(pdf.min()), float(pdf.max())
    if hi <= lo:
        return image.copy()
    weighted = hi * ((pdf - lo) / (hi - lo)) ** alpha
    cdf = np.cumsum(weighted)
    cdf /= cdf[-1]
    levels = np.arange(256, dtype=np.float64) / 255.0
    lut = np.rint(255 * levels ** (1 - cdf)).astype(np.uint8)
    hsv[..., 2] = lut[v]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def clahe(image, clip_limit=2.0, grid=8):
    """限制对比度的局部直方图均衡，只改变 Lab 明度。"""
    validate(image)
    if clip_limit <= 0 or grid < 1:
        raise ValueError("clip_limit 和 grid 必须为正")
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    if lab[..., 0].min() == lab[..., 0].max():
        return image.copy()
    lab[..., 0] = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def retinex_lite(image, sigma=30.0, strength=0.6, max_gain=4.0):
    """B 的自定义单尺度照明增益，不称为标准 SSR/MSR。"""
    x = unit(image)
    if sigma <= 0 or not 0 <= strength <= 1 or max_gain < 1:
        raise ValueError("Retinex 参数无效")
    illumination = smooth(np.max(x, axis=2), sigma)
    gain = np.minimum((illumination + 1 / 255) ** (-strength), max_gain)
    return u8(x * gain[..., None])


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


def vevid_lite(image, b=0.16, gain=1.4):
    """VEViD-Lite 的 HSV 明度相位映射。"""
    validate(image)
    if b < 0 or gain <= 0:
        raise ValueError("VEViD 参数无效")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32) / 255
    phase = np.arctan2(-gain * (v + b), v)
    lo, hi = float(phase.min()), float(phase.max())
    if hi - lo < 1e-7:
        return image.copy()
    hsv[..., 2] = np.rint(255 * (phase - lo) / (hi - lo)).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


ENHANCERS = {
    "gamma": gamma,
    "agcwd": agcwd,
    "clahe": clahe,
    "retinex_lite": retinex_lite,
    "msrcp": msrcp,
    "lime": lime,
    "vevid_lite": vevid_lite,
}
