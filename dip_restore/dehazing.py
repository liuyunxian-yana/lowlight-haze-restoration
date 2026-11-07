"""只做去雾:A 组 DCP/CAP/反转 DCP 的 RGB 重构实现。"""
from __future__ import annotations

import cv2
import numpy as np

from .core import dark_channel, guided, u8, unit


def _gray_601(rgb):
    """原始 OpenCV 灰度引导图对应的 RGB 权重，保持去雾先验含义。"""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _dcp_float(x, patch=15, omega=0.95, t0=0.2, a_min=0.3, scale=0.5,
               gray_airlight=False, transmission_mix=0.0):
    """暗通道先验：估大气光、局部透射率，按散射模型反演。"""
    if patch < 1 or patch % 2 == 0 or not 0 <= omega <= 1 or not 0 < t0 <= 1 or not 0 < scale <= 1 or not 0 <= transmission_mix <= 1:
        raise ValueError("DCP 参数无效")
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    dark = dark_channel(small, patch)
    count = max(1, round(dark.size * 0.001))
    candidate = np.argpartition(dark.ravel(), -count)[-count:]
    airlight = small.reshape(-1, 3)[candidate].mean(axis=0)
    if gray_airlight:
        airlight[:] = airlight.mean()
    airlight = np.maximum(airlight, max(a_min, 1e-3))
    coarse_t = 1 - omega * dark_channel(small / airlight, patch)
    transmission = np.clip(guided(_gray_601(small), coarse_t.astype(np.float32), radius=20), 0, 1)
    if transmission.shape != (h, w):
        transmission = cv2.resize(transmission, (w, h), interpolation=cv2.INTER_LINEAR)
    if transmission_mix:
        transmission = (1 - transmission_mix) * transmission + transmission_mix * np.median(transmission)
    return np.clip((x - airlight) / np.maximum(transmission[..., None], t0) + airlight, 0, 1)


def dcp(image, **kwargs):
    """He 等暗通道去雾；采用大气光下限和缩图透射率估计。"""
    return u8(_dcp_float(unit(image), **kwargs))


def invert_dcp(image, **kwargs):
    """夜间反转图像思路：反转→DCP→反转。"""
    params = {"a_min": 0.0, **kwargs}
    return u8(1 - _dcp_float(1 - unit(image), **params))


def cap(image, beta=1.0, t0=0.2, a_min=0.3):
    """确定性 CAP-inspired 亮度/饱和度深度代理"""
    x = unit(image)
    if beta < 0 or not 0 < t0 <= 1 or not 0 <= a_min <= 1:
        raise ValueError("CAP 参数无效")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    depth = 0.121779 + 0.959710 * hsv[..., 2] - 0.780245 * hsv[..., 1]
    depth = cv2.erode(depth, np.ones((7, 7), np.uint8))
    depth = np.maximum(guided(_gray_601(x), depth, radius=20), 0)
    transmission = np.maximum(np.exp(-beta * depth), t0)
    count = max(1, round(depth.size * 0.001))
    candidates = np.argpartition(depth.ravel(), -count)[-count:]
    airlight = np.maximum(x.reshape(-1, 3)[candidates].mean(axis=0), max(a_min, 1e-3))
    recovered = (x - airlight) / transmission[..., None] + airlight
    return u8(recovered)


DEHAZERS = {"dcp": dcp, "invert_dcp": invert_dcp, "cap": cap}
