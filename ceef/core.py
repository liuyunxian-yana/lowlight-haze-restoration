"""CEEF: CP-AHE 与 FSPD 多尺度曝光融合的独立 NumPy 实现。"""
from __future__ import annotations

import numpy as np


def box_sum(x: np.ndarray, radius: int) -> np.ndarray:
    """零延拓窗口求和，对应作者 boxfilter.m 的边界计数。"""
    if radius < 0:
        raise ValueError("radius 不能为负")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("box_sum 需要二维数组")
    h, w = x.shape
    y = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    yy = np.arange(h)
    xx = np.arange(w)
    top, bottom = np.maximum(yy-radius, 0), np.minimum(yy+radius+1, h)
    left, right = np.maximum(xx-radius, 0), np.minimum(xx+radius+1, w)
    return y[bottom[:, None], right[None, :]] - y[top[:, None], right[None, :]] - y[bottom[:, None], left[None, :]] + y[top[:, None], left[None, :]]


def _tile_maps(image: np.ndarray, tiles=(8, 8), clip_limit=.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape
    ny, nx = min(tiles[0], h), min(tiles[1], w)
    ys = np.linspace(0, h, ny+1, dtype=int)
    xs = np.linspace(0, w, nx+1, dtype=int)
    maps = np.empty((ny, nx, 256), np.float64)
    for i in range(ny):
        for j in range(nx):
            patch = image[ys[i]:ys[i+1], xs[j]:xs[j+1]]
            hist = np.bincount(patch.ravel(), minlength=256).astype(np.int64)
            area = patch.size
            # ClipLimit 作用于每块的像素数，截断量均匀回填到 256 个灰度级。
            cap = max(int(np.ceil(clip_limit * area)), int(np.ceil(area / 256)))
            excess = int(np.maximum(hist-cap, 0).sum())
            hist = np.minimum(hist, cap)
            hist += excess // 256
            rem = excess % 256
            if rem:
                hist[np.linspace(0, 255, rem, dtype=int)] += 1
            maps[i, j] = np.cumsum(hist) * (255.0 / area)
    return maps, (ys[:-1]+ys[1:]-1)/2, (xs[:-1]+xs[1:]-1)/2


def cp_ahe(image: np.ndarray) -> np.ndarray:
    """RGB 通道交错后，对共享的 8×8 CLAHE 网格作直方图映射。"""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("图像须为 RGB uint8")
    h, w, _ = image.shape
    combined = image.reshape(h, w*3)
    maps, cy, cx = _tile_maps(combined)
    yy = np.arange(h)
    xx = np.arange(w*3)
    iy1 = np.searchsorted(cy, yy, side='right').clip(1, len(cy)-1) if len(cy)>1 else np.zeros(h, int)
    ix1 = np.searchsorted(cx, xx, side='right').clip(1, len(cx)-1) if len(cx)>1 else np.zeros(w*3, int)
    iy0, ix0 = np.maximum(iy1-1, 0), np.maximum(ix1-1, 0)
    wy = np.clip((yy-cy[iy0]) / np.maximum(cy[iy1]-cy[iy0], 1), 0, 1) if len(cy)>1 else np.zeros(h)
    wx = np.clip((xx-cx[ix0]) / np.maximum(cx[ix1]-cx[ix0], 1), 0, 1) if len(cx)>1 else np.zeros(w*3)
    v = combined
    a = maps[iy0[:,None], ix0[None,:], v]
    b = maps[iy0[:,None], ix1[None,:], v]
    c = maps[iy1[:,None], ix0[None,:], v]
    d = maps[iy1[:,None], ix1[None,:], v]
    out = (1-wy[:,None])*((1-wx[None,:])*a+wx[None,:]*b)+wy[:,None]*((1-wx[None,:])*c+wx[None,:]*d)
    return np.rint(np.clip(out,0,255)).astype(np.uint8).reshape(h,w,3)


def _detail_weights(stack: np.ndarray, p: int, r: int):
    """局部结构强度 C、最大强度 Cmax 与 C^(p-1)/sum(C^p)。"""
    h,w = stack.shape[:2]
    n = stack.shape[-1]
    count = box_sum(np.ones((h,w)), r)
    means = np.empty((h,w,n), np.float64)
    strength = np.empty_like(means)
    channels = 3 if stack.ndim == 4 else 1
    for i in range(n):
        x = stack[...,i]
        if channels == 3:
            means[...,i] = sum(box_sum(x[...,c],r) for c in range(3))/(3*count)
            second = sum(box_sum(x[...,c]**2,r) for c in range(3))/(3*count)
        else:
            means[...,i] = box_sum(x,r)/count
            second = box_sum(x*x,r)/count
        strength[...,i] = np.sqrt(np.maximum(second-means[...,i]**2,0))*np.sqrt(channels)*(2*r+1)+1e-12
    weights = strength**(p-1)/np.sum(strength**p+1e-12,axis=2,keepdims=True)
    return weights * strength.max(axis=2,keepdims=True), means


def _base_weights(stack: np.ndarray) -> np.ndarray:
    """作者 weight_cal_base.m 中启用的梯度与 arctan 曝光权重。"""
    padded=np.pad(stack,((1,1),(1,1),(0,0)),mode='edge')
    gx=(padded[:-2,2:]+2*padded[1:-1,2:]+padded[2:,2:]
        -padded[:-2,:-2]-2*padded[1:-1,:-2]-padded[2:,:-2])/8
    gy=(padded[2:,:-2]+2*padded[2:,1:-1]+padded[2:,2:]
        -padded[:-2,:-2]-2*padded[:-2,1:-1]-padded[:-2,2:])/8
    grad=np.hypot(gx,gy)
    exposed = (2/np.pi)*np.arctan(20*np.minimum(stack,1-stack))
    weights = grad*exposed+1e-12
    return weights/weights.sum(axis=2,keepdims=True)


def _mask(shape):
    mask = np.zeros(shape, np.float64)
    mask[::2,::2] = 1
    return mask


def _scale(stack: np.ndarray, r: int, p: int, coarse=False):
    h,w=stack.shape[:2]
    n=stack.shape[-1]
    count=box_sum(np.ones((h,w)),r)
    sparse=_mask((h,w))
    sparse_count=box_sum(sparse,r)
    wd, means=_detail_weights(stack,p,r)
    down=means[::2,::2,:]
    detail=np.zeros_like(stack[...,0])
    base=np.zeros_like(detail) if coarse else None
    wb=_base_weights(stack) if coarse else None
    for i in range(n):
        weighted_mean=box_sum(means[...,i]*wd[...,i],r)/count
        weighted=box_sum(wd[...,i],r)/count
        if stack.ndim == 4:
            detail += weighted[...,None]*stack[...,i]-weighted_mean[...,None]
        else:
            detail += weighted*stack[...,i]-weighted_mean
        if coarse:
            base += box_sum(means[...,i]*wb[...,i],r)/count
    return (detail+base if coarse else detail),down,sparse_count


def _upsample_and_smooth(coarse: np.ndarray, shape, radius, count):
    sparse=np.zeros(shape,np.float64)
    sparse[::2,::2]=coarse
    return box_sum(sparse,radius)/count


def fuse(exposures: np.ndarray) -> np.ndarray:
    """FSPD: 细尺度、逐级中尺度、粗尺度，再以稀疏盒滤波逐层重建。"""
    if exposures.ndim!=4 or exposures.shape[2]!=3 or exposures.shape[3]!=5:
        raise ValueError("预期 H×W×3×5 曝光序列")
    h,w=exposures.shape[:2]
    if min(h,w)<16:
        raise ValueError("最短边须至少为 16 像素")
    fine,level,count=_scale(exposures,5,4)
    nlev=int(np.floor(np.log2(min(h,w))))-3
    layers=[]
    r=4
    for _ in range(nlev):
        detail,next_level,sparse_count=_scale(level,r,4)
        layers.append((detail,sparse_count,r))
        level=next_level
        r=max(r-1,1)
    coarse,_,_=_scale(level,max(r-1,1),6,coarse=True)
    for detail,sparse_count,radius in reversed(layers):
        coarse=_upsample_and_smooth(coarse,detail.shape,radius,sparse_count)+detail
    base=_upsample_and_smooth(coarse,(h,w),5,count)
    return base[...,None]+fine


def restore(image: np.ndarray) -> np.ndarray:
    """输入与输出均为同尺寸 RGB uint8。"""
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError("图像须为 RGB uint8")
    x=image.astype(np.float64)/255
    exposures=np.empty((*image.shape,5),np.float64)
    exposures[...,0]=cp_ahe(image)/255
    for exponent in range(2,6):
        exposures[...,exponent-1]=x**exponent
    result=fuse(exposures)
    return np.rint(np.clip(result,0,1)*255).astype(np.uint8)
