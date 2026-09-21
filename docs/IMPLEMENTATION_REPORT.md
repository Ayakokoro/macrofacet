# Macrofacet 实现报告

报告日期：2026-09-18

## 1. 完成状态

本工程从空目录建立，已实现并编译 A（论文去相关 Macrofacet）、B（最近完整交点条件下的式 (29) hazard）、C（同一中点筛选的局部闭合）和 F27（联合条件 GP 的有限网格首次相交参考）。实现不是接口空壳；测试和本报告中的数据均来自本机实际命令。

核对的本地 `macrofacet.pdf` 元数据为 2026-08-07，标题为 *Macrofacet Theory for Gaussian Process Statistical Surfaces*。同时用 arXiv v2（2026-05-17）交叉核对了 SE 核、密度、式 (29) crossing/exterior 比率、梯度 PDF 到 NDF 的式 (36)–(38) 和式 (40) 混合 proposal。公开版本的公式编号与本地版本不同，因此实现按公式内容而非编号硬匹配。

## 2. 架构和逻辑文件映射

| 规格项 | 实际路径 | 内容 |
|---|---|---|
| F01–F04 | `include/macrofacet/mathutility/{NumericPolicy,Quadrature,RootFinding,Gaussian1D}.h` | 状态传播、自适应 GK15、单调反演、深尾正态函数 |
| F05–F08 | `GaussianMoments1D`, `SmallGaussian`, `BivariateGaussian`, `GaussianScreenIntegral` | 截断矩、PSD 条件化、二维正交概率、筛选 crossing 通量 |
| F09–F12 | `include/macrofacet/gpss/`、`src/gpss/` | 均值场、SE 核、GPSS 局部统计、完整 `F0,G0` 条件射线 |
| F13–F16 | `include/macrofacet/macrofacet/`、`src/macrofacet/` | Generalized Gaussian/Beckmann/GGX NDF、A 消光、conductor 相函数和式 (40) proposal |
| F17–F24 | `include/macrofacet/transport/`、`src/transport/` | 状态、A/B/C kernel、光学深度与 null tracking、完整梯度 mark |
| F25 | `src/integrator/MacrofacetPathTracer.cpp` | 有限 AABB、环境照明、analog 多次散射、Russian roulette |
| F26 | `src/experiments/FlightCurveExperiment.cpp` | 同一 surface birth 的 A/B/C hazard、模型生存和距离直方图 |
| F27 | `src/experiments/ConditionalGPReference.cpp` | 联合路径首穿参考和多检查点公式估计 |
| F28–F29 | `RenderExperiment.cpp`, `ExperimentMain.cpp` | 白炉/方向环境 PFM 与 `curves/gp-reference/render/all` CLI |
| F30–F34 | `tests/test_*.cpp` | 数学、GP、A、hazard 和采样验收 |
| F35 | `configs/*.json` | 发布、CI 和白炉配置 |
| F36 | `CMakeLists.txt`, `vcpkg.json` | C++17 构建和 vcpkg manifest |
| F37 | 本文件 | 实际结果与限制 |

公共向量、AABB、光线和 RNG 位于 `include/macrofacet/core/`。CSV 的无第三方依赖 SVG 后处理位于 `scripts/plot_experiments.py`。

## 3. 关键实现决定

- 所有 travel direction 均指向光线实际前进方向；穿入 crossing 满足 `w·G<0`，反射后满足 `w_new·G>0`。
- `U`/`U1` 只作为局部 exterior screen probability 输出。模型透射率始终是同一 hazard 的 `exp(-integral h)`。
- `ConditionedRay` 观测完整 `F(x0)=0,G(x0)=g0`。近起点的值/沿线导数协方差使用 `expm1` 和级数，不加物理方差地板。
- PSD 条件化使用标准化后的特征空间伪逆，确定观测会检查支持一致性；不添加固定 jitter。
- C 的 numerator 和 denominator 使用同一 `Z`、同一 `t/2`。碰撞 mark 给定完整梯度时使用 `P(Y>0|F_t=0,G_t=g,Z)` rejection，而不是错误复用只给定 `K` 的筛选概率。
- F27 对每个联合 Gaussian 只分解一次，再生成所有样本；等式事件 `F_t=0` 通过 Gaussian 条件化实现，不做零概率 rejection。
- 路径追踪的距离使用 24-cell 分段 hazard 表；每格光学深度来自同一原 hazard 的自适应积分，采样和返回 PDF 都对应这张表。它是明确的渲染近似；曲线和数学测试仍直接积分原 hazard。
- Classic 渲染使用 F16 的 phase proposal；`uniform` 使用 `R=0`，`paper_mixture` 使用配置的式 (40) 权重。B/C 的 External 第一段仍用 A hazard，但通过 F24 采完整梯度以启动下一段条件模型。

## 4. 依赖、构建和测试

新增依赖均通过 vcpkg 安装：

- Eigen 5.0.1：矩阵、Cholesky、对称特征分解；
- nlohmann-json 3.12.0：配置解析与 resolved config 输出。

实际环境为 MSVC 19.44.35228、CMake 3.31.12、Windows SDK 10.0.26100。实际运行命令：

```powershell
vcpkg install eigen3:x64-windows nlohmann-json:x64-windows
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
build\Release\macrofacet_experiments.exe all --config configs\macrofacet_ci.json
build\Release\macrofacet_experiments.exe curves --config configs\macrofacet_experiments.json
build\Release\macrofacet_experiments.exe gp-reference --config configs\macrofacet_experiments.json
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_white_validation.json
py scripts\plot_experiments.py outputs\macrofacet_experiments
```

最终测试结果：`checks=14198 failures=0`，CTest 为 `100% tests passed, 0 tests failed`。

覆盖的独立验收包括深尾 log-CDF/分位数、M0–M3 数值积分、PSD 条件化、核有限差分、近起点 PSD、`integral n D(n) dOmega=E[G]`、投影面积、VNDF、GGX、F=1 phase/proposal 归一化、解析平面光学深度、null tracking 逃逸频率、B/C 退化关系、F27 的 N=0/1 对照、距离质量加逃逸原子归一化和完整梯度 crossing/reflection 支持。

## 5. 实际实验结果

### 5.1 默认固定段（65 点、每模式 20,000 个表模型样本）

输出位于 `outputs/macrofacet_experiments/`：

- `flight_curves.csv` / `flight_curves.svg`
- `flight_histograms.csv`
- `gp_reference_paths.csv` / `gp_reference.svg`
- `screened_formula.csv`

在实际截断距离 `b=1.5`：

| 模式 | H(b) | T_model(b) | h(b) |
|---|---:|---:|---:|
| Classic A | 0.493886 | 0.610250 | 0.0167891 |
| Conditional29 B | 0.620369 | 0.537746 | 0.0167891 |
| Midpoint C | 0.648695 | 0.522727 | 0.0167971 |

距离直方图各 bin 的最大 `|observed-expected|` 分别为 A `0.00249`、B `0.00455`、C `0.00448`。这里 expected 是采样所用同一张分段 hazard 表的质量，因此该比较验证概率一致性，不声称表已经等于连续 hazard。

### 5.2 整段条件 GP 参考

4,096 条联合路径在 `t=1.5` 的 prefix-positive 生存估计：

| 网格区间 | T_grid(1.5) | 标准误 |
|---:|---:|---:|
| 64 | 0.514160 | 0.007809 |
| 128 | 0.519287 | 0.007807 |
| 256 | 0.521729 | 0.007805 |

这些独立 Monte Carlo 估计在置信区间内相容；有限网格仍可能漏掉同号端点间的双穿越，不能称为连续精确值。

在 `t=1.5`，8,192 样本的多检查点公式给出：

| N | U_N | h_N | h 标准误 |
|---:|---:|---:|---:|
| 0 | 0.994873 | 0.0167320 | 0.000343 |
| 1 | 0.900635 | 0.0172617 | 0.000382 |
| 4 | 0.621826 | 0.0170233 | 0.000495 |
| 8 | 0.549805 | 0.0165873 | 0.000518 |
| 16 | 0.532837 | 0.0159993 | 0.000512 |
| 32 | 0.519165 | 0.0174867 | 0.000541 |

`N=0` 是 B，`N=1` 且检查点为中点是 C。较大 N 的变化与估计标准误同量级，当前预算不能据此断言单调收敛或 C 必然优于 B。

### 5.3 渲染和白炉

`outputs/macrofacet_ci/` 含三模式 8×8×2 spp 的 unit-white 与 directional-gradient PFM。六次运行均为 0 数值失败、0 safety-cap termination；低 spp Classic uniform proposal 白炉方差很大，因此另运行了 8×8×256 spp 的 `outputs/macrofacet_white_validation/render_classic_white.pfm`。

该高预算 Classic 白炉共有 16,384 paths、49,216 real collisions、0 numerical failures，线性 RGB 样本总平均为 `1.02174`（像素/通道范围 `0.7290–1.2616`）。测试中的确定性球面积分对 F=1 energy、uniform proposal PDF 和混合 proposal PDF 均在各自容限内归一到 1；白炉剩余偏差是有限样本/高方差，不是通过 exposure 隐藏的误差。

## 6. 规格问题逐项回答

1. **A：**NDF 有向面积、投影面积、VNDF、GGX、phase/proposal、解析平面自由程、null tracking 和白炉均有实际验收。
2. **B：**来自给定 `F0=0` 和完整 `G0` 的四维观测条件 Gaussian；未用单位法线替代梯度。
3. **C：**分子和分母严格使用同一个中点及同一条件协方差。
4. **整段公式：**F27 同时实现联合路径首穿和 N=0/1/4/8/16/32 多检查点公式；它们明确标记为有限网格/Monte Carlo 参考。
5. **距离 PDF：**A/B/C 曲线均由各自 hazard 积分；实验采样使用同一分段表定义的 PDF 和逃逸原子。
6. **反射 mark：**B 使用 `G|F_t=0,Z` 的 flux weighting；C 额外使用给定完整 `g` 的中点 rejection。
7. **近似：**External 首段用 A；域外真空；跨 bounce 只记最近交点；C 一个中点；F27 有限检查点；渲染用 24-cell hazard 表。
8. **误差来源：**报告中分别列出了积分容限、表离散、GP 网格漏检、Monte Carlo 标准误和低 spp 图像噪声。
9. **式 (40)：**混合 proposal 及其 full-mixture PDF 已实现并通过归一化测试；默认配置为 uniform (`R=0`)，只有 `paper_mixture` 才使用配置 R。
10. **未证明结论：**没有宣称 B/C 恢复完整 GP ensemble、互易性成立、C 必然更准，或有限网格已经是连续首次相交精确解。

## 7. 已知限制

- 这是独立解析场景实现，不含论文私有 PBRT 场景/资产，因此不声称逐像素复刻论文图片或作者运行时间。
- 渲染器目前只有凸 AABB 活跃域、环境光和 analog camera paths；B/C 没有 NEE。
- `paper_mixture` 的 Beckmann proposal 目前要求世界 Z 对齐的平面/核 frame；任意旋转 frame 可使用正确的 uniform proposal，旋转 Beckmann 专用采样尚未接入。
- Generalized Gaussian 支持 SPD 梯度协方差；显式支持的奇异面积密度只有对齐的 Beckmann heightfield。其他 PSD 梯度可采样，但不冒充已有普通 NDF 密度。
- Sphere mean 在球心按不可微输入报错。非平稳核、dielectric、完整路径历史条件、连续首次穿越 rare-event 求解和生产级优化不在当前范围。
- 默认 64×64×256 发布渲染配置已提供但未在本次会话执行；实际图像预算和时间均按上节记录，没有把 CI 图冒充默认预算结果。

