"""RDT 2021 的 CPU 论文流程：Retinex 分解、Taylor 透射、GFF 与颜色迁移。

依据原课程项目 dehaze/rdt.py 和 Tang 等 CVIU 2021 的公式重写为统一 RGB 接口。
联合双边滤波按论文所述保留，默认不缩图。
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import validate

EPS = 1e-4          # 式4-5、式9、式14、式24 中的“无穷小”
T_EPS = 1e-2        # 【假设】式27 分母的稳定项，论文写的是无穷小，太小会在辉光处数值爆炸


# ======================================================================= 基础工具
def luminance(img):
    """BGR 转灰度。"""
    return 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]


def box_filter(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r, eps):
    """引导滤波（单通道），用于图像融合中的权重细化。"""
    mean_g, mean_s = box_filter(guide, r), box_filter(src, r)
    cov = box_filter(guide * src, r) - mean_g * mean_s
    var = box_filter(guide * guide, r) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return box_filter(a, r) * guide + box_filter(b, r)


def joint_bilateral_filter(src, guide, radius=7, sigma_s=5.0, sigma_r=0.1):
    """
    联合双边滤波：用 guide（边缘引导图 Ig）的边缘去平滑 src（tUL），得到最终透射率 t。
    自己用平移叠加实现，不依赖 opencv-contrib。
    【假设】论文没给 radius/sigma，这里取 7 / 5 / 0.1。
    """
    h, w = src.shape
    pad_s = cv2.copyMakeBorder(src, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    pad_g = cv2.copyMakeBorder(guide, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    num = np.zeros_like(src)
    den = np.zeros_like(src)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = np.exp(-(dy * dy + dx * dx) / (2 * sigma_s ** 2))
            s = pad_s[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            g = pad_g[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            wgt = ws * np.exp(-((g - guide) ** 2) / (2 * sigma_r ** 2))
            num += wgt * s
            den += wgt
    return num / np.maximum(den, 1e-8)


# ======================================================================= 3.1 大气光分解
def shades_of_gray(img, p):
    """
    shades-of-gray 色彩恒常算法（Finlayson & Trezzi 2004）：
    光源颜色 e_c = (mean(I_c^p))^(1/p)，归一化后用 I_c / (e_c*sqrt(3)) 校正。
    p 就是论文中的 Minkowski 范数阶数 N。
    """
    img64 = img.astype(np.float64)
    e = np.power(np.mean(np.power(np.maximum(img64, 1e-8), p), axis=(0, 1)), 1.0 / p)
    e = e / (np.linalg.norm(e) + 1e-12)
    out = img64 / (e.reshape(1, 1, 3) * np.sqrt(3.0) + 1e-12)
    return np.clip(out, 0, 1).astype(np.float32)


def fuse_three(a1, a2, a3):
    """式(4)(5)：用 r_i = I_i/(I1+I2+I3+eps) 作为权重，把三幅图加权融合。"""
    s = a1 + a2 + a3 + EPS
    return (a1 * a1 + a2 * a2 + a3 * a3) / s


def color_correct(img):
    """式(3)-(5)：k∈{10,20,40}，N=floor(min(m,n)/k)，三种阶数校正后融合得到 Ic。"""
    h, w = img.shape[:2]
    outs = []
    for k in (10, 20, 40):
        n = max(int(min(h, w) // k), 1)
        outs.append(shades_of_gray(img, n))
    return np.clip(fuse_three(*outs), 0, 1)


def _mccann_step(op, L, s, axis):
    """Frankle-McCann Retinex 的一步：与偏移 s 处的邻居比较（比值->乘积->重置->平均）。"""
    ip = op.copy()
    if axis == 1:
        if s > 0:
            ip[:, s:] = op[:, :-s] + L[:, s:] - L[:, :-s]
        else:
            t = -s
            ip[:, :-t] = op[:, t:] + L[:, :-t] - L[:, t:]
    else:
        if s > 0:
            ip[s:, :] = op[:-s, :] + L[s:, :] - L[:-s, :]
        else:
            t = -s
            ip[:-t, :] = op[t:, :] + L[:-t, :] - L[t:, :]
    ip = np.minimum(ip, 0.0)                 # 重置：不允许超过最大值（白点）
    return 0.5 * (ip + op)                   # 与上一次输出取平均


def retinex_mccann(img, n_iter=8):
    """
    Frankle–McCann Retinex（Funt et al. 2004 的 MATLAB 实现思路），逐通道处理。
    n_iter 即论文的 nIterations，默认 8。输出范围 (0,1]，每个通道最亮处为 1。
    """
    h, w = img.shape[:2]
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        L = np.log(np.maximum(img[..., c].astype(np.float64), 1.0 / 255.0))
        L = L - L.max()                       # 最大值归零，对应“白点”
        op = np.zeros_like(L)
        shift = 2 ** max(0, int(np.log2(max(1, min(h, w)))) - 1)   # 初始偏移（最大）
        while abs(shift) >= 1:
            for _ in range(n_iter):
                op = _mccann_step(op, L, shift, 1)    # 水平
                op = _mccann_step(op, L, shift, 0)    # 垂直
            shift = -int(shift / 2)                   # 偏移减半并交替方向
        out[..., c] = np.exp(op)
    return out


def decompose_atmospheric_light(img, n_iter=8):
    """式(4)-(9)：返回色彩校正图 Ic、新夜间有雾图 In、大气光图 Ac。"""
    Ic = color_correct(img)                               # 式(4)
    Inw = retinex_mccann(Ic, n_iter)                      # 式(7)
    In = np.maximum(Inw, Ic)                              # 式(8)
    Ac = Ic / (In + EPS)                                  # 式(9)
    return Ic, In, Ac


# ======================================================================= 3.2 透射率估计
def min_filter_fused(dark, h, w):
    """式(15)后：对暗通道做 N×N 最小值滤波（N 由 k=10,20,40 决定），三幅结果按式(4)(5)融合。"""
    res = []
    for k in (10, 20, 40):
        n = max(int(min(h, w) // k), 1)
        res.append(cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))))
    return fuse_three(*res)


def estimate_transmission(In, n_iter=8, jbf_radius=7, jbf_sigma_s=5.0, jbf_sigma_r=0.1):
    """
    返回：最终透射率 t，以及中间量（Jinit、tL、tU 用于调试）。
    """
    h, w = In.shape[:2]
    # ---- 初始无雾图 Jinit 与粗透射率 tcoar（式11-14）
    Jne = retinex_mccann(1.0 - In, n_iter)                # 式(11)
    Jne = np.maximum(1.0 - In, Jne)                       # 式(12)
    Jinit = 1.0 - Jne                                     # 式(13)
    tcoar = (1.0 - In) / (Jne + EPS)                      # 式(14)

    # ---- 暗通道 + 最小值滤波（式15-16）
    dark = np.min(In, axis=2)                             # 式(15)
    dark_hat = min_filter_fused(dark, h, w)               # 式(16) 中的 Î^dark

    # ---- 下界 / 泰勒展开得到的上界（式17、22）
    tL = 1.0 - dark_hat                                   # 式(17)
    tU = 1.0 - dark_hat ** 2                              # 式(22)：对 f(ρ) 一阶泰勒展开(式20)后的上界

    # ---- 像素级融合（式23-24）
    omega = np.clip((tU - tL) / (tL + EPS), 0.0, 1.0)     # 式(24)，截断到[0,1]保证是凸组合
    tUL = omega * tU + (1.0 - omega) * tL                 # 式(23)

    # ---- 边缘引导图 Ig（式25-26 + 式4-5）
    tl = 1.0 - dark                                       # 式(25)
    tcoar = np.maximum(tcoar, tl[..., None])              # 式(26)，对三个通道分别取最大
    tcoar = np.clip(tcoar, 0, 1)
    s = tcoar.sum(axis=2) + EPS
    Ig = (tcoar ** 2).sum(axis=2) / s                     # 式(4)(5)：三通道当作 I1,I2,I3 融合

    # ---- 联合双边滤波得到最终 t
    t = joint_bilateral_filter(tUL.astype(np.float32), Ig.astype(np.float32),
                               jbf_radius, jbf_sigma_s, jbf_sigma_r)
    return np.clip(t, 0, 1), Jinit


def potential_haze_free(In, t, theta=1.05):
    """式(27)：Jpote = (In - θ(1-t)) / (t + eps)，θ=1.05（论文经验值）。"""
    J = (In - theta * (1.0 - t[..., None])) / (t[..., None] + T_EPS)
    return np.clip(J, 0, 1)


# ======================================================================= 3.3 融合与恢复
def gff_fuse(imgs):
    """
    引导滤波图像融合（Li et al. 2013, GFF）：
    基础层(31×31均值滤波) + 细节层，用 Laplacian 显著性生成权重图，再用引导滤波细化。
    r1=45, eps1=0.3, r2=7, eps2=1e-6 取自 GFF 原文的默认值（【假设】论文只说“采用 Li et al. 2013”）。
    """
    grays = [luminance(i).astype(np.float32) for i in imgs]
    bases = [cv2.blur(i, (31, 31)) for i in imgs]
    details = [i - b for i, b in zip(imgs, bases)]
    sal = []
    for g in grays:
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=1))
        sal.append(cv2.GaussianBlur(lap, (11, 11), 5))
    idx = np.argmax(np.stack(sal), axis=0)
    wb, wd = [], []
    for n, g in enumerate(grays):
        p = (idx == n).astype(np.float32)
        wb.append(np.clip(guided_filter(g, p, 45, 0.3), 1e-6, None))
        wd.append(np.clip(guided_filter(g, p, 7, 1e-6), 1e-6, None))
    sb, sd = sum(wb), sum(wd)
    B = sum((w / sb)[..., None] * b for w, b in zip(wb, bases))
    D = sum((w / sd)[..., None] * d for w, d in zip(wd, details))
    return np.clip(B + D, 0, 1)


# Reinhard 颜色迁移（lαβ 空间）
_RGB2LMS = np.array([[0.3811, 0.5783, 0.0402],
                     [0.1967, 0.7244, 0.0782],
                     [0.0241, 0.1288, 0.8444]])
_LMS2LAB = np.diag([1 / np.sqrt(3), 1 / np.sqrt(6), 1 / np.sqrt(2)]) @ \
           np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])


def _to_lab(bgr):
    rgb = bgr[..., ::-1].reshape(-1, 3).astype(np.float64)
    lms = np.log10(np.maximum(np.einsum("nc,dc->nd", rgb, _RGB2LMS), 1e-4))
    return np.einsum("nc,dc->nd", lms, _LMS2LAB)


def _from_lab(lab, shape):
    lms = 10 ** np.einsum("nc,dc->nd", lab, np.linalg.inv(_LMS2LAB))
    rgb = np.einsum("nc,dc->nd", lms, np.linalg.inv(_RGB2LMS))
    return np.clip(rgb.reshape(shape)[..., ::-1], 0, 1).astype(np.float32)


def color_transfer(source, target):
    """Reinhard 2001：把 source 的颜色统计（均值、标准差）迁移给 target。"""
    s, t = _to_lab(source), _to_lab(target)
    out = (t - t.mean(0)) * (s.std(0) / (t.std(0) + 1e-6)) + s.mean(0)
    return _from_lab(out, target.shape)


# ======================================================================= 总入口
def rdt_dehaze_all(bgr_uint8, n_iter=8, theta=1.05, max_side=None):
    """返回 (src, tgt, mid_dict)，mid_dict 为中间结果（uint8 BGR）。"""
    h0, w0 = bgr_uint8.shape[:2]
    img = bgr_uint8.astype(np.float32) / 255.0
    if max_side is not None and max(h0, w0) > max_side:
        s = max_side / max(h0, w0)
        img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)

    Ic, In, Ac = decompose_atmospheric_light(img, n_iter)
    t, Jinit = estimate_transmission(In, n_iter)
    Jpote = potential_haze_free(In, t, theta)
    Jfuse = gff_fuse([Jinit.astype(np.float32), Jpote.astype(np.float32)])
    Jrefe = Jfuse * Ac
    Jct = color_transfer(Jrefe, Jpote)
    J = Jct / max(float(Jct.max()), 1e-6)

    def to_u8(x):
        x = np.clip(x, 0, 1)
        if x.shape[:2] != (h0, w0):
            x = cv2.resize(x, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return (x * 255).astype(np.uint8)

    src, tgt = to_u8(Jpote), to_u8(J)
    mid = dict(Ic=Ic, In=In, Ac=Ac, Jinit=Jinit, t=np.dstack([t] * 3), Jpote=Jpote,
               Jfuse=Jfuse, Jrefe=Jrefe, J=J)
    return src, tgt, {k: to_u8(v) for k, v in mid.items()}


def rdt_dehaze(bgr_uint8, n_iter=8, theta=1.05, max_side=None):
    """返回 (RDT-Source, RDT-Target)，均为 uint8 BGR。"""
    src, tgt, _ = rdt_dehaze_all(bgr_uint8, n_iter, theta, max_side)
    return src, tgt




def restore_pair(image, n_iter=8, theta=1.05, max_side=None):
    """返回 RDT-Source、RDT-Target，均为原尺寸 RGB uint8。"""
    validate(image)
    if n_iter < 1 or theta <= 0 or (max_side is not None and max_side < 16):
        raise ValueError("RDT 参数无效")
    if image.min() == image.max():
        return image.copy(), image.copy()
    source, target = rdt_dehaze(image[..., ::-1].copy(), n_iter, theta, max_side)
    return source[..., ::-1].copy(), target[..., ::-1].copy()


def rdt_source(image, **kwargs):
    return restore_pair(image, **kwargs)[0]


def rdt_target(image, **kwargs):
    return restore_pair(image, **kwargs)[1]


RDT_METHODS = {"rdt_source": rdt_source, "rdt_target": rdt_target}
