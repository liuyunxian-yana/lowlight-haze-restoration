"""已有联合恢复思路的独立工程重构。

A illumination + DCP：照明图提亮后用全局空气光暗通道去雾。
B Zhang 2014 / SIDE 2020：保留关键处理阶段；照明和局部空气光的
照明和局部空气光由本项目的公共算子估计。
所有方法只接受、返回同尺寸 RGB uint8；共用 core 的滤波与颜色接口。
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import dark_channel, gray, guided, smooth, u8, unit, validate
from .dehazing import dcp
from .enhancement import clahe, gamma


def _local_atmosphere_restore(
    rgb: np.ndarray,
    *,
    patch: int,
    airlight_radius: int,
    guide_radius: int,
    omega: float = 0.85,
    transmission_floor: float = 0.2,
) -> np.ndarray:
    """空间变化空气光的 DCP 工程模型；供两条论文处理链共用。

    局部形态学上包络近似空气光，并据此归一化后估计透射率。
    这并不等价于 Zhang/SIDE 原文中未展开或基于优化的全部细节。
    """
    if patch < 1 or patch % 2 != 1 or airlight_radius < 1 or guide_radius < 1:
        raise ValueError("patch 必须为正奇数，滤波半径必须为正")
    kernel = np.ones((2 * airlight_radius + 1, 2 * airlight_radius + 1), np.uint8)
    airlight = cv2.dilate(rgb, kernel)
    airlight = smooth(airlight, max(1.0, airlight_radius / 3.0))
    normalized = rgb / np.maximum(airlight, 0.05)
    coarse_t = 1.0 - omega * dark_channel(normalized, patch)
    t = np.clip(guided(gray(rgb), coarse_t, guide_radius, 1e-3), transmission_floor, 1.0)
    return np.clip((rgb - airlight) / t[..., None] + airlight, 0, 1)


def a_illum_dcp(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """A 原 `illum_dcp` 思路：光照图增亮，再以全局 DCP 去雾。

    使用新项目共同 DCP 核心，参数及量化细节不保证与原脚本逐像素一致。
    """
    validate(image)
    if not 0 < gamma <= 1:
        raise ValueError("gamma 必须属于 (0,1]")
    rgb = unit(image)
    value = rgb.max(axis=2)
    light = np.clip(guided(value, value, radius=30, eps=1e-2), 0.02, 1.0)
    bright = np.clip(rgb * (light ** (gamma - 1.0))[..., None], 0, 1)
    return dcp(u8(bright))


def zhang2014(image: np.ndarray, gamma: float = 1 / 3, color_gamma: float = 1 / 1.2) -> np.ndarray:
    """Zhang 等 ICIP 2014 的三阶段思路，工程近似：照明补偿、颜色校正、局部去雾。

    原文的照明估计引用其他方法；这里以粗尺度平滑加引导滤波实现，
    局部空气光由 `_local_atmosphere_restore` 估计。
    """
    validate(image)
    if not 0 < gamma <= 1 or not 0 < color_gamma <= 1:
        raise ValueError("gamma 和 color_gamma 必须属于 (0,1]")
    if image.min() == image.max():
        return image.copy()
    rgb = unit(image)
    value = rgb.max(axis=2)
    # 先估计空间光照，取其比值补偿所有颜色通道，尽量保持色比。
    light = np.clip(guided(value, smooth(value, 16), radius=32, eps=1e-2), 0.08, 1.0)
    adjusted_light = np.power(light, gamma)
    compensated = np.clip(rgb * (adjusted_light / light)[..., None], 0, 1)
    # 颜色下界来自局部最大值，再按平均色强进行温和校正。
    ratio = compensated / np.maximum(adjusted_light[..., None], 0.08)
    lower = cv2.dilate(ratio, np.ones((11, 11), np.uint8))
    guide = gray(compensated)
    incident_color = np.stack([guided(guide, lower[..., c], 32, 1e-2) for c in range(3)], axis=2)
    mean_color = np.clip(incident_color.mean(axis=2), 0.35, 1.0)
    scale = np.power(mean_color, color_gamma) / mean_color
    incident_color = np.clip(incident_color * scale[..., None], 0.35, 1.0)
    corrected = np.clip(compensated / incident_color, 0, 1)
    recovered = _local_atmosphere_restore(corrected, patch=11, airlight_radius=5, guide_radius=32)
    return u8(recovered)


def side2020(image: np.ndarray, halo_radius: int = 9, halo_strength: float = 0.65,
             illumination_gamma: float = 0.7) -> np.ndarray:
    """SIDE 2020 的处理链工程近似：光晕分离、照明/场景分解、局部去雾、重组。

    这里用引导滤波估计照明，并在局部去雾后重组场景。
    """
    validate(image)
    if halo_radius < 1 or not 0 <= halo_strength <= 1 or not 0 < illumination_gamma <= 1:
        raise ValueError("halo_radius/halo_strength/illumination_gamma 参数非法")
    if image.min() == image.max():
        return image.copy()
    rgb = unit(image)
    value = rgb.max(axis=2)
    # 亮区的低频扩散作为光晕近似，避免把这部分直接当成空气光。
    source_mask = np.clip((value - 0.55) / 0.45, 0, 1)
    halo = np.minimum(smooth(source_mask, halo_radius), value) * halo_strength
    scene = np.clip(rgb - halo[..., None], 0, 1)
    scene_value = scene.max(axis=2)
    light = np.clip(guided(scene_value, smooth(scene_value, 24), radius=24, eps=1e-2), 0.08, 1)
    normalized_scene = np.clip(scene / light[..., None], 0, 1)
    radiance = _local_atmosphere_restore(normalized_scene, patch=11, airlight_radius=5, guide_radius=24)
    return u8(radiance * np.power(light, illumination_gamma)[..., None])


def a_fusion(image: np.ndarray) -> np.ndarray:
    """A 的多曝光 Mertens 融合工程基线；与本文物理联合模型分开比较。

    三张候选图共用新项目的 Gamma、CLAHE、DCP 实现。Mertens 在 BGR
    色彩约定下融合，再转回统一 RGB；因此这是原思路的重构，非逐像素移植。
    """
    validate(image)
    bright = gamma(image, exponent=0.5)
    candidates = (bright, clahe(bright), dcp(bright))
    merged = cv2.createMergeMertens().process([c[..., ::-1].copy() for c in candidates])
    return u8(np.clip(merged[..., ::-1], 0, 1))


PRIOR_JOINT = {
    "a_illum_dcp": a_illum_dcp,
    "a_fusion": a_fusion,
    "zhang2014": zhang2014,
    "side2020": side2020,
}
