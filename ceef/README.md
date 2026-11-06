# CEEF 2021 Python 复现

独立目录，入口 `python -m ceef 输入.png 输出.png`；算法接口 `ceef.restore(rgb_uint8)`。图像处理仅用 NumPy；命令行使用项目现有的 OpenCV 读写图片，不调用 OpenCV 的增强、去雾或融合算法。

## 对应作者实现

- [论文](https://doi.org/10.1109/TMM.2021.3110483)：Xiaoning Liu、Hui Li、Ce Zhu，IEEE TMM 2021。
- [作者 MATLAB 代码](https://github.com/xiaohuiben/CEEF-TMM-2021)：`CEEF.m`、`scale_fine.m`、`scale_interm.m`、`scale_coarse.m`、`weight_cal_detail.m`、`weight_cal_base.m`、`boxfilter.m`、`meanfun.m`。

`CEEF.m` 生成五张候选：对 RGB 通道交错后的图应用 CP-AHE，以及输入的 2、3、4、5 次幂。FSPD 用局部均值和方差构造结构强度 `C`；细节权重按作者代码 `C^(p−1)/Σ(C^p+ε) × Cmax`，细尺度及中尺度取 `p=4`，粗尺度取 `p=6`。每一层将候选图与局部均值的差按结构权重融合；粗尺度另以 Sobel 梯度乘作者的 arctan 曝光函数产生基底权重。重建时把粗层样本放回偶数网格，用稀疏盒滤波及计数图逐层插值，最后加细层 RGB 细节。尺度半径与作者代码相同：细层 5、中层从 4 逐级减至 1、粗层为最终半径减 1 且最低 1。

CP-AHE 使用 8×8 网格、256 灰度级、clip limit 0.01，逐块截断直方图、均匀回填溢出计数并在相邻块之间双线性插值。MATLAB `adapthisteq` 的直方图截断计数、边界插值和像素量化细节没有公开在作者仓库中，因此本 Python 实现尚未与 MATLAB 达到逐像素一致。作者源码中的 `retinex_filter.m` 没有被 `CEEF.m` 调用，故未纳入主算法。

## 运行

```sh
python -m unittest ceef.test_ceef -v
python -m ceef data/synthetic/degraded/val/bsds_159029_joint_mild.png results/ceef.png
```

命令行只需项目已有 NumPy 和 OpenCV；`core.py` 不依赖 OpenCV。输入至少 16×16、RGB uint8。输出按 `clip(结果,0,1)` 转 RGB uint8。实测 CPU 时间和进程峰值 RSS 在命令行 JSON 中输出；峰值包含解释器与图像缓冲。

## 小样本检查

在合成验证集的两张联合退化图上检查，样本来自两张不同清晰图。每张图以对应清晰图为参考，得到：

| 方法 | PSNR dB | SSIM | CPU ms/张 |
|---|---:|---:|---:|
| CEEF Python | 14.24 | 0.635 | 186.8 |
| DCP | 10.56 | 0.425 | 5.4 |
| `a_fusion` | 17.50 | 0.699 | 45.0 |
| `model_fusion_anchor` | 19.55 | 0.736 | 25.3 |
| Zhang 2014 实现 | 15.22 | 0.516 | 22.4 |

同一张 481×321 图的独立命令行运行：CPU 190.08 ms，进程峰值 RSS 141.0 MB。上述质量指标只描述这两张样本，不能代表整套数据。作者 MATLAB 图像没有在本机 MATLAB/Octave 环境运行，因此尚无逐像素误差对照；当前已核对候选图数量、幂次、尺度、权重与重建代码路径。
