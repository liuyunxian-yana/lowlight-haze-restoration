# 微光有雾图像恢复：统一设计的 CPU 项目

本目录是独立实现的新项目。运行代码、数据、测试和实验结果都在这里；**不导入、不软链接、不运行原课程项目的任何代码**。原项目用于研究算法和数据来源。所有恢复函数统一接受和返回同尺寸的 `H×W×3 RGB uint8`；批量入口是 `run.py`。

## 1. 整体逻辑

```text
内置/外部图像 → 严格配对与哈希校验 → RGB uint8
                                   ↓
       单项增强 / 单项去雾 / 串联 / 旧联合基线 / 新联合模型
                                   ↓
          同尺寸输出 → PSNR、SSIM、颜色误差、CPU 时间、进程内存
                                   ↓
                  CSV、失败清单、运行元数据、对比图
```

`dip_restore/core.py` 是唯一的图像域和公共算子层；`enhancement.py`、`dehazing.py`、`prior_joint.py`、`joint_model.py` 实现算法；`registry.py` 管理方法及串联顺序；`datasets.py`、`metrics.py`、`benchmark.py` 组成单一实验流程。`synthesis.py` 可离线生成新样本。`tests/` 检查成像逆式、数据、接口及评估行为。

## 2. 自主联合恢复方法

课程数据的合成顺序来自原项目 `C_数据集与联合方案/prepare_data.py`：在线性 RGB 中先成雾，再乘曝光与缓变照明，最后加入传感器噪声。我们据此建立

\[
Y(x)=e(x)\{t(x)J(x)+[1-t(x)]A\}+n(x)=p(x)J(x)+q(x)A+n(x),
\quad p=e-q,\ q=e(1-t).
\]

`joint_model.py` 先将输入 sRGB 转为线性 RGB；从高分位亮度估计曝光尺度，从大尺度平滑图估计可选的空间变化；在曝光校正图上用暗通道选取大气光候选，再用原观测的暗通道估空气光系数 `q`，经引导滤波和 `q≤0.85e` 约束。最终**只使用同一个成像方程反演一次**：

\[
J_0=Y/e,\qquad
\hat J=J_0+\frac{p}{p^2+\lambda}\big[Y-pJ_0-qA\big].
\]

`λ=σ²` 是低信号下的稳定项。亮度估计直接参与雾系数估计和反演分母；模型并不混合两张串联输出。`model_global_e` 使用稳健的全局曝光，`model_full` 用受限空间曝光场；`model_light_only` 置 `q=0`，`model_haze_only` 置 `e=1`，`model_unregularized` 置 `λ=0`，用于消融。

**可辨识性与假设：** 单图无法唯一估出 `e,t,A,J`。高分位“白点”锚定、缓变曝光和局部暗通道均是先验，不是测量真值；亮物体、夜间彩色灯、光晕与空间变化大气光都可能破坏它们。为避免利用测试集调参，在验证集 18 张联合图上比较去雾强度与稳定项后，冻结 `white_anchor=0.85`、`omega=0.6`、`noise_sigma=0.03` 等默认值；测试结果保留原貌。

## 3. 已有成果如何被重新实现

| 新方法 | 参考成果 | 实现性质 |
|---|---|---|
| `gamma`, `agcwd`, `clahe`, `retinex_lite`, `msrcp`, `lime`, `vevid_lite` | 原 B `lowlight_project/lowlight.py` | 重新组织于统一图像域；Retinex 是自定义单尺度变体，MSRCP 使用镜像 FFT，LIME 使用 PCG 且不含 BM3D；不是把旧文件复制进来 |
| `dcp`, `invert_dcp`, `cap` | 原 A `dehaze/dehaze.py` | 暗通道、反转 DCP、颜色衰减先验工程版；`cap` 不宣称统计学习版 CAP |
| `a_illum_dcp`, `a_fusion` | 原 A 的光照增强+DCP与多曝光融合 | 完整联合基线，**不**作为“纯去雾”步骤重复叠加 |
| `zhang2014`, `side2020` | 原 B `joint_papers/` | 沿论文处理链重构的工程近似；局部空气光、光晕和照明估计不是论文优化器的逐像素复现 |
| `rdt_approx_source`, `rdt_approx_target` | 原 A `rdt.py` | 保留 Retinex、Taylor 透射与融合结构，但用引导滤波替代 225 邻域联合双边滤波；**只称工程近似**，默认不批量运行 |
| `model_*` | 本项目的线性成像模型 | 自主设计的先验约束联合反演；结果需用实测检验 |

原 A 的 RDT 有较重的联合双边滤波与颜色迁移；原实现还在当前 NumPy 环境发出数值警告。新项目提供明确标注的轻量 RDT 工程近似，并只在 1 张验证图上试跑，不能把它的输出称作原论文复现或完整质量评估。Zero-DCE++ 仅为原项目的 `NotImplementedError` 占位且无权重，也没有列入可运行方法。原 C 的 B 副本比顶层 B 版本旧，历史 MSRCP/LIME 时间与效果不能直接代表本项目。

## 4. 数据与依赖

- `data/synthetic/` 内置 18 张 BSDS 清晰图与 162 张退化图，按原文件字节复制了 `clean/`、`degraded/`、清单和来源元数据。验证集 54 张、测试集 108 张；测试联合退化 36 张来自 **12 张独立原图**。原 `maps/` 是审计场，不参与恢复和评估，故未复制。数据清单中的参考图、合成参数、随机种子只供评分和审计，恢复算法只接收输入 RGB。
- `data/nthazy/` 内置 25 对，`data/gta5/` 内置 77 对，均为本项目内的实际副本。它们主要检验有雾场景，不能作为真实低照度加雾的联合真值。
- `--dataset external` 支持无参考或同名配对参考图；有后缀时可用 `--input-suffix`、`--reference-suffix`。参考图必须同尺寸，不自动缩放；没有可靠参考时 PSNR/SSIM 为空。
- 仅依赖 Python 3.9+、NumPy、OpenCV，安装：`python -m pip install -r requirements.txt`。无需 GPU、云端 API 或模型权重。入口在导入数值库前约束 BLAS/OMP，并将本机 OpenCV 设为单线程、关闭 OpenCL。

## 5. 运行与复现

先做轻量检查：

```sh
python -m unittest discover -s tests -v
python run.py --dataset synthetic --split val --kind joint --limit 2 --methods identity agcwd dcp model_global_e side2020 --repeats 1 --memory-sample --out results/check
```

验证集消融与正式测试（每次用新的输出目录）：

```sh
python run.py --dataset synthetic --split val --kind joint --methods identity agcwd 'gamma>dcp' 'dcp>gamma' model_global_e model_full model_light_only model_haze_only model_unregularized --repeats 1 --out results/val_joint
python run.py --dataset synthetic --split test --kind joint --methods identity gamma agcwd clahe dcp cap 'gamma>dcp' 'dcp>gamma' model_global_e model_full model_light_only model_haze_only model_unregularized a_illum_dcp a_fusion side2020 zhang2014 --repeats 1 --memory-sample --out results/test_joint_new
python run.py --dataset synthetic --split test --kind all --methods identity agcwd dcp model_global_e --repeats 1 --out results/test_all_new
```

其他数据和任意组合：

```sh
python run.py --dataset gta5 --limit 2 --methods dcp cap model_global_e --out results/gta5_check
python run.py --dataset nthazy --limit 1 --methods dcp --out results/nthazy_check
python run.py --dataset external --input /path/to/input --reference /path/to/paired_ref --methods 'agcwd>cap' model_global_e --out results/external
```

`--save-images` 保存全部输出；默认每法保存前 2 张输入｜结果｜参考横向对比图。`per_image.csv` 是逐图结果，`summary.csv` 按方法、退化类别和强度分层，`failures.json` 记录失败，`run.json` 保存环境、方法参数、代码/清单 SHA、计时口径。`--memory-sample` 为每法另开子进程，在首图上采集**整个进程峰值 RSS**到 `memory.csv`；它包含解释器、库和图像缓冲，不是纯算法增量。`wall_ms` 与 `cpu_ms` 取预热后重复次数的中位数，排除读写、哈希和指标。若需要稳定时间估计可改 `--repeats 3`，但会增加本机负载。

离线生成新样本（新生成器采用 OpenCV 插值，**不保证与内置旧数据逐字节相同**）：

```sh
python -m dip_restore.synthesis --clean data/synthetic/clean --limit-sources 1 --out results/new_synthetic
python run.py --dataset synthetic --data-root results/new_synthetic --split val --kind joint --methods model_global_e --out results/new_synthetic_eval
```

## 6. 本次实测与结论

环境：macOS arm64、Python 3.9.6、NumPy 2.0.2、OpenCV 4.14.0，OpenCV 实际报告 1 线程，OpenCL 关闭。新项目 6 项测试通过；验证集和测试集记录分别在 `results/val_release/`、`results/test_joint_release/`、`results/test_all_release/`；可选 RDT 试跑在 `results/optional_methods_check/`。联合测试集 36 图 × 17 方法＝612 次处理，失败 0。下表为 `summary.csv` 的 joint/ALL 行（每图只计时 1 次，时间是本机初测）：

| 方法 | PSNR dB ↑ | SSIM ↑ | CPU ms ↓ |
|---|---:|---:|---:|
| 原图 `identity` | 11.22 | 0.377 | 0.0 |
| `agcwd` | 13.34 | 0.345 | 0.81 |
| `clahe` | 12.51 | 0.403 | 1.14 |
| `gamma>dcp` | 12.98 | 0.312 | 7.26 |
| `dcp>gamma` | 12.82 | 0.321 | 7.60 |
| 自主 `model_global_e` | 11.01 | 0.309 | 17.18 |
| 自主 `model_full` | 9.81 | 0.299 | 23.18 |
| `a_fusion` 工程版 | 13.59 | 0.338 | 17.15 |
| `side2020` 工程版 | 12.23 | 0.316 | 18.89 |
| `zhang2014` 工程版 | 11.97 | 0.258 | 23.62 |

**结论：** 新模型的成像逻辑与合成退化顺序一致，但当前曝光/空气光先验在联合测试集上估计不稳，质量低于 AGCWD、CLAHE 等基线；空间曝光估计尤其容易把场景明暗误当照明变化。额外对测试集三类各 36 张检查发现，`model_global_e` 在低照度单项上 PSNR 19.52、SSIM 0.468，但在纯雾单项上 11.08/0.624，也低于 DCP 的 16.54/0.762。此结果说明**模型有理论约束，并不代表已解决单图欠定估计**。真实微光有雾照片尚无可靠成对真值，需人工检查与独立数据验证；不得把合成数据分数称为真实场景表现。
