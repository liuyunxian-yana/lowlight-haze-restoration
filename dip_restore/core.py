"""公共图像处理算子：全项目唯一图像契约、颜色域转换和边缘保持公共算子。"""
from __future__ import annotations

import cv2
import numpy as np

# 检查输入图像是否合法
def validate(image, shape=None):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("图像必须是非空 H×W×3 RGB uint8")
    if shape is not None and image.shape != shape:
        raise ValueError(f"图像尺寸变化：{shape} -> {image.shape}")

# 把像素从 0～255 转换到 0～1
def unit(image):
    validate(image)
    return image.astype(np.float32) / 255.0

# 裁剪、乘以 255、四舍五入并转换回 uint8
def u8(image):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("算法输出必须为有限 H×W×3 数组")
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def gray(x):
    return np.sum(x * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=2)


def smooth(x, sigma):
    """低分辨率估大尺度平滑场，再回到原尺寸；只处理估计图。"""
    h, w = x.shape[:2]
    factor = max(1, int(max(h, w) / 320))
    if factor > 1:
        small = cv2.resize(x, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
        filtered = cv2.GaussianBlur(small, (0, 0), max(sigma / factor, 1))
        return cv2.resize(filtered, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(x, (0, 0), max(float(sigma), 1))


def guided(guide, source, radius=12, eps=1e-3):
    """共享灰度引导滤波，输入 float32 二维图。"""
    guide, source = guide.astype(np.float32), source.astype(np.float32)
    kernel = (2 * radius + 1, 2 * radius + 1)
    box = lambda z: cv2.boxFilter(z, -1, kernel)
    mg, ms = box(guide), box(source)
    a = (box(guide * source) - mg * ms) / (box(guide * guide) - mg * mg + eps)
    b = ms - a * mg
    return box(a) * guide + box(b)


def dark_channel(x, patch=11):
    if patch < 1 or patch % 2 == 0:
        raise ValueError("暗通道窗口须为正奇数")
    return cv2.erode(np.min(x, axis=2), np.ones((patch, patch), np.uint8))
