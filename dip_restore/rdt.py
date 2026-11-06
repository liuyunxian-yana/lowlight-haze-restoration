"""RDT 思路的 CPU 实现：颜色校正、Retinex、透射率估计与融合。

保留其颜色校正、McCann 式 Retinex、暗通道透射率上下界、潜在无雾图、
初始图融合与颜色迁移的研究结构。为适应边缘 CPU，以共享 guided 滤波
代替原实现的 225 邻域联合双边滤波，并限制内部处理边长。
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import dark_channel, gray, guided, u8, unit

_EPS = 1e-4
_RGB_LMS = np.array([[.3811, .5783, .0402], [.1967, .7244, .0782], [.0241, .1288, .8444]], np.float64)
_LMS_LAB = np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]], np.float64)
_LMS_LAB *= np.array([1 / np.sqrt(3), 1 / np.sqrt(6), 1 / np.sqrt(2)])[:, None]


def _shades_gray(x, order):
    # 使用有限阶 Minkowski 光源估计；对全黑图用 epsilon 保持稳定。
    e = np.mean(np.power(np.maximum(x.astype(np.float64), 1e-6), order), axis=(0, 1)) ** (1 / order)
    e /= max(float(np.linalg.norm(e)), 1e-12)
    return np.clip(x / np.maximum(e * np.sqrt(3), 1e-4), 0, 1).astype(np.float32)


def _color_correct(x):
    versions = [_shades_gray(x, order) for order in (4, 8, 16)]
    total = versions[0] + versions[1] + versions[2] + _EPS
    return np.clip(sum(z * z for z in versions) / total, 0, 1).astype(np.float32)


def _neighbor_step(estimate, log_image, shift, axis):
    shifted = estimate.copy()
    if axis == 1:
        shifted[:, shift:] = estimate[:, :-shift] + log_image[:, shift:] - log_image[:, :-shift]
    else:
        shifted[shift:, :] = estimate[:-shift, :] + log_image[shift:, :] - log_image[:-shift, :]
    return 0.5 * (estimate + np.minimum(shifted, 0))


def _mccann(x, passes=3):
    """有限次多尺度邻域比较的 Retinex 近似；无优化器或外部模型。"""
    h, w = x.shape[:2]
    out = np.empty_like(x)
    start = 1 << max(0, int(np.floor(np.log2(max(1, min(h, w))))) - 1)
    for c in range(3):
        log_i = np.log(np.maximum(x[..., c].astype(np.float32), 1 / 255))
        log_i -= log_i.max()
        estimate = np.zeros((h, w), np.float32)
        shift = start
        while shift:
            for _ in range(passes):
                estimate = _neighbor_step(estimate, log_i, shift, 1)
                estimate = _neighbor_step(estimate, log_i, shift, 0)
            shift //= 2
        out[..., c] = np.exp(estimate)
    return np.clip(out, 0, 1)


def _transmission(scene, radius=12):
    h, w = scene.shape[:2]
    dark = scene.min(axis=2)
    minima = []
    for divisor in (10, 20, 40):
        patch = max(1, (min(h, w) // divisor) | 1)
        minima.append(dark_channel(scene, patch))
    total = sum(minima) + _EPS
    dark_hat = np.clip(sum(z * z for z in minima) / total, 0, 1)
    lower = 1 - dark_hat
    upper = 1 - dark_hat * dark_hat
    weight = np.clip((upper - lower) / (lower + _EPS), 0, 1)
    coarse = weight * upper + (1 - weight) * lower
    return np.clip(guided(gray(scene), coarse.astype(np.float32), radius=radius, eps=1e-3), 0.12, 1)


def _fuse(initial, potential):
    # 以亮度梯度构造显著性，guided 平滑权重后融合两种候选图。
    grads = []
    for candidate in (initial, potential):
        y = gray(candidate)
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        grads.append(np.sqrt(gx * gx + gy * gy))
    weight = grads[1] / (grads[0] + grads[1] + 1e-3)
    weight = np.clip(guided(gray(potential), weight.astype(np.float32), radius=7, eps=1e-2), 0, 1)
    return np.clip((1 - weight[..., None]) * initial + weight[..., None] * potential, 0, 1)


def _linear_transform(x, matrix):
    return np.einsum("...c,dc->...d", x, matrix, optimize=False)


def _lab_color_transfer(source, target):
    """稳定的 Reinhard 式 lαβ 通道统计迁移，输入均为 RGB 0..1。"""
    def to_lab(rgb):
        lms = np.log10(np.maximum(_linear_transform(rgb.astype(np.float64), _RGB_LMS), 1e-4))
        return _linear_transform(lms, _LMS_LAB)

    a, b = to_lab(source), to_lab(target)
    adjusted = (b - b.mean(axis=(0, 1))) * (a.std(axis=(0, 1)) / np.maximum(b.std(axis=(0, 1)), 1e-4)) + a.mean(axis=(0, 1))
    lms = np.power(10, _linear_transform(adjusted, np.linalg.inv(_LMS_LAB)))
    rgb = _linear_transform(lms, np.linalg.inv(_RGB_LMS))
    return np.clip(rgb, 0, 1).astype(np.float32)


def restore_pair(image, max_side=512, retinex_passes=3, theta=1.05):
    """返回 (潜在无雾 Source, 融合 Target)，同尺寸 RGB uint8。"""
    original = unit(image)
    h0, w0 = original.shape[:2]
    if original.min() == original.max():
        return image.copy(), image.copy()
    if max_side is not None and max_side < 16:
        raise ValueError("max_side 至少为 16")
    if retinex_passes < 1 or theta <= 0:
        raise ValueError("RDT 参数无效")
    if max_side is not None and max(h0, w0) > max_side:
        factor = max_side / max(h0, w0)
        image_f = cv2.resize(original, (max(1, round(w0 * factor)), max(1, round(h0 * factor))), interpolation=cv2.INTER_AREA)
    else:
        image_f = original
    corrected = _color_correct(image_f)
    scene = np.maximum(corrected, _mccann(corrected, retinex_passes))
    atmosphere = corrected / np.maximum(scene, _EPS)
    inverse = 1 - scene
    init = 1 - np.maximum(inverse, _mccann(inverse, retinex_passes))
    transmission = _transmission(scene)
    potential = np.clip((scene - theta * (1 - transmission[..., None])) /
                        (transmission[..., None] + 1e-2), 0, 1)
    fused = _fuse(init, potential)
    refl = np.clip(fused * atmosphere, 0, 1)
    target = _lab_color_transfer(refl, potential)
    target /= max(float(target.max()), 1e-6)
    if potential.shape[:2] != (h0, w0):
        potential = cv2.resize(potential, (w0, h0), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return u8(potential), u8(target)


def rdt_approx_source(image, **kwargs):
    return restore_pair(image, **kwargs)[0]


def rdt_approx_target(image, **kwargs):
    return restore_pair(image, **kwargs)[1]


RDT_METHODS = {"rdt_approx_source": rdt_approx_source, "rdt_approx_target": rdt_approx_target}
