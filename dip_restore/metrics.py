"""统一 RGB uint8 指标。全参考指标仅在可信、同尺寸参考图存在时计算。"""
from __future__ import annotations

import cv2
import numpy as np

from .core import validate

_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """D65 CIE Lab；rgb 是 sRGB [0,1] 的 float64 数组。"""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.stack([sum(linear[..., c] * _RGB_TO_XYZ[row, c] for c in range(3))
                    for row in range(3)], axis=-1) / _D65_WHITE
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.stack((116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def diagnostics(output: np.ndarray) -> dict[str, float]:
    """无参考曝光诊断；数值不代表恢复质量，也不能替代 PSNR/SSIM。"""
    validate(output)
    total, dark, bright, luma_sum = output.shape[0] * output.shape[1], 0, 0, 0.0
    histogram = np.zeros(256, dtype=np.int64)
    for r0 in range(0, output.shape[0], 128):
        x = output[r0:r0 + 128].astype(np.float64) / 255.0
        luma = np.sum(x * _LUMA_WEIGHTS, axis=2)
        luma_sum += float(luma.sum())
        dark += int(np.count_nonzero(luma < 0.05))
        bright += int(np.count_nonzero(luma > 0.95))
        histogram += np.bincount(np.rint(luma * 255).astype(np.uint8).ravel(), minlength=256)
    probs = histogram[histogram > 0].astype(np.float64) / total
    return {"mean_luma": luma_sum / total,
            "dark_ratio": dark / total,
            "bright_ratio": bright / total,
            "luma_entropy": float(-np.sum(probs * np.log2(probs)))}


def measure(reference: np.ndarray, output: np.ndarray) -> dict[str, float]:
    """PSNR/SSIM/DeltaE76 与亮度误差；C 旧评估的同一数值口径。"""
    validate(reference)
    validate(output, reference.shape)
    if min(reference.shape[:2]) < 11:
        raise ValueError("SSIM 需要宽高均至少 11 像素")
    h, w = reference.shape[:2]
    squared_sum = delta_sum = luma_error_sum = 0.0
    for r0 in range(0, h, 128):
        x = reference[r0:r0 + 128].astype(np.float64) / 255.0
        y = output[r0:r0 + 128].astype(np.float64) / 255.0
        squared_sum += float(np.square(x - y).sum())
        delta_sum += float(np.linalg.norm(_lab(x) - _lab(y), axis=2).sum())
        luma_x = np.sum(x * _LUMA_WEIGHTS, axis=2)
        luma_y = np.sum(y * _LUMA_WEIGHTS, axis=2)
        luma_error_sum += float(np.abs(luma_x - luma_y).sum())
    ssim_sum = 0.0
    count = 0
    # SSIM 按通道和行带计算；每带保留 5 行上下文，valid 区与整图卷积完全一致。
    for channel in range(3):
        for r0 in range(5, h - 5, 128):
            r1 = min(h - 5, r0 + 128)
            x = reference[r0 - 5:r1 + 5, :, channel].astype(np.float64) / 255.0
            y = output[r0 - 5:r1 + 5, :, channel].astype(np.float64) / 255.0
            cut = (slice(5, 5 + r1 - r0), slice(5, w - 5))
            mx, my = cv2.GaussianBlur(x, (11, 11), 1.5)[cut], cv2.GaussianBlur(y, (11, 11), 1.5)[cut]
            vx = np.maximum(cv2.GaussianBlur(x * x, (11, 11), 1.5)[cut] - mx * mx, 0)
            vy = np.maximum(cv2.GaussianBlur(y * y, (11, 11), 1.5)[cut] - my * my, 0)
            cov = cv2.GaussianBlur(x * y, (11, 11), 1.5)[cut] - mx * my
            ssim = ((2 * mx * my + 0.0001) * (2 * cov + 0.0009) /
                    ((mx * mx + my * my + 0.0001) * (vx + vy + 0.0009)))
            ssim_sum += float(ssim.sum())
            count += ssim.size
    mse = squared_sum / (h * w * 3)
    result = {"psnr_db": float(-10 * np.log10(mse)) if mse else float("inf"),
              "ssim_rgb": ssim_sum / count,
              "delta_e76": delta_sum / (h * w),
              "luma_mae": luma_error_sum / (h * w)}
    result.update(diagnostics(output))
    return result
