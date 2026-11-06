"""自行设计的联合恢复方法：共享曝光与空气光系数的线性域联合反演。

观测模型 Y=e[t J+(1-t) A]+n = p J+q A+n，p=e-q，q=e(1-t)。
用亮度锚点、缓变照明和局部暗通道约束单图分解。
"""
from __future__ import annotations

import numpy as np

from .core import dark_channel, gray, guided, linear_to_srgb, smooth, srgb_to_linear, u8, unit


def exposure_field(y, spatial=True, white_anchor=0.85, min_exposure=0.025):
    """用线性亮度高分位锚定全局曝光；仅允许小幅缓变空间修正。"""
    value = y.max(axis=2)
    global_e = float(np.clip(np.percentile(value, 99) / white_anchor, min_exposure, 1.0))
    if not spatial:
        return np.full(value.shape, global_e, dtype=np.float32)
    coarse = smooth(value, sigma=max(8, min(value.shape) / 12))
    anchor = max(float(np.percentile(coarse, 75)), 1e-5)
    shape = np.clip(coarse / anchor, 0.75, 1.25)
    return np.clip(global_e * shape, min_exposure, 1.0).astype(np.float32)


def airlight_and_veil(y, e, omega=0.9, patch=11, max_veil_fraction=0.85):
    """先在曝光校正图中估 A，再用原观测暗通道估 q=e(1-t)。"""
    corrected = np.clip(y / e[..., None], 0, 1)
    dark = dark_channel(corrected, patch)
    n = max(1, int(dark.size * 0.001))
    positions = np.argpartition(dark.ravel(), -n)[-n:]
    a = np.clip(corrected.reshape(-1, 3)[positions].mean(axis=0), 0.25, 1).astype(np.float32)
    q_coarse = omega * dark_channel(y / a[None, None, :], patch)
    q = guided(gray(corrected), q_coarse.astype(np.float32), radius=12, eps=1e-3)
    q = np.clip(q, 0, max_veil_fraction * e).astype(np.float32)
    return a, q


def inverse_scene(y, e, q, a, noise_sigma=0.03):
    """给定共享参数反演 J；零正则时退化为物理逆式。"""
    # q 已限制为小于 e；保持 p+q=e，正则项负责抑制小 p 时的噪声增益。
    p = np.maximum(e - q, 1e-6)
    prior = np.clip(y / np.maximum(e[..., None], 0.025), 0, 1)
    lam = noise_sigma ** 2
    residual = y - p[..., None] * prior - q[..., None] * a
    gain = p / (p * p + lam)
    scene = np.clip(prior + gain[..., None] * residual, 0, 1)
    return scene, gain


def recover(image, mode="full", noise_sigma=0.03, omega=0.6, return_fields=False):
    """返回 RGB uint8；模式 full/global_e/light_only/haze_only/unregularized 为消融。"""
    # 退化模型：Y = e[tJ + (1-t)A] + n
    if mode not in ("full", "global_e", "light_only", "haze_only", "unregularized"):
        raise ValueError(f"未知联合模型模式：{mode}")
    if noise_sigma < 0:
        raise ValueError("noise_sigma 不能为负")
    y = srgb_to_linear(unit(image))
    e = (np.ones(y.shape[:2], np.float32) if mode == "haze_only" else
         exposure_field(y, spatial=mode != "global_e"))
    if mode == "light_only":
        a = np.zeros(3, np.float32)
        q = np.zeros(y.shape[:2], np.float32)
    else:
        a, q = airlight_and_veil(y, e, omega=omega)
    # 正则逆解：在低 e·t 下向曝光校正先验收缩，防止读噪被无限放大。
    j, gain = inverse_scene(y, e, q, a, 0.0 if mode == "unregularized" else noise_sigma)
    result = u8(linear_to_srgb(j))
    if return_fields:
        return result, {"exposure": e, "veil_coefficient": q,
                        "transmission": 1 - q / np.maximum(e, 1e-6), "airlight": a,
                        "recovery_gain": gain}
    return result


def fusion_anchor(image, blend=0.3):
    """以 AGCWD 保留暗部亮度，加入 Mertens 联合恢复的雾与局部对比度修正。

    blend 在验证集固定；融合结果仍包含 DCP 去雾候选，不单独改变颜色通道。
    """
    from .enhancement import agcwd
    from .prior_joint import a_fusion

    if not np.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError("blend 必须在 [0, 1] 内")
    enhanced = agcwd(image).astype(np.float32)
    restored = a_fusion(image).astype(np.float32)
    return np.rint(np.clip((1 - blend) * enhanced + blend * restored, 0, 255)).astype(np.uint8)


JOINT_MODEL = {"model_full": lambda x: recover(x, "full"),
               "model_global_e": lambda x: recover(x, "global_e"),
               "model_light_only": lambda x: recover(x, "light_only"),
               "model_haze_only": lambda x: recover(x, "haze_only"),
               "model_unregularized": lambda x: recover(x, "unregularized"),
               "model_fusion_anchor": fusion_anchor}
