# Medium 按射线返回分段消光上界：实现方案

## 目标与数学契约

让 medium 对**一条已确定起点、方向和出生观测的 flight**，按射线顺序返回
`[t_begin,t_end,M]`，并保证区间内的当前模型消光系数 `Sigma(t|H)<=M`。
这里 `t` 始终是从 flight 出生位置量起的距离，`t_end` 不超过活动域出口。
`M=0` 的真空区间允许直接跨过；其余区间执行 delta tracking：在区间内
抽 `Exp(M)` 距离，在候选点以 `Sigma/M` 接受。

这与 pbrt-v4 的 `Medium::SampleRay()` 返回 `RayMajorantIterator`、
迭代 `RayMajorantSegment{tMin,tMax,sigma_maj}` 的做法对应。
pbrt 的网格介质用 DDA 逐格产生区间；迭代器可在发生碰撞时停止，
不用提前计算射线后面所有区间。
参考：[pbrt 第四版 Media](https://pbr-book.org/4ed/Volume_Scattering/Media)。

本项目的 conditional flight 保留推导文件采用的“终点 `F_t>0` 近似”。
分段上界必须覆盖[统一符号推导](../src/第一份推导核验_统一符号_NDF_VNDF_Phase.md)
的式（7）所定义的局部消光系数；改进 majorant 不会把它变成严格的
GP 首次穿越 hazard。

## 建议的接口

```cpp
struct ExtinctionSegment {
    double beginAge;
    double endAge;
    double majorant;
};

// 每个具体 medium 返回自己的轻量游标；next() 按年龄升序给出下一段。
class ConditionalMajorantCursor {
public:
    std::optional<ExtinctionSegment> next();
};

class ConditionalMedium {
public:
    ConditionalFlightKernel beginFlight(const FlightState& birth) const;
    ConditionalMajorantCursor sampleRay(const ConditionalFlightKernel& flight,
                                        double maximumAge) const;
    FlightSample sample(const ConditionalFlightKernel& flight, Random& rng,
                        DdaTrackingDiagnostics* diagnostics = nullptr) const;
};
```

优先采用**具体类型游标**，让 classic DDA 与 conditional 区间算法共享
`ExtinctionSegment` 和采样循环。当前模式在配置加载后就已确定，无需先引入
堆分配的多态 iterator。若未来确实要在一条路径中动态切换介质，再加类型擦除层。

现有代码的直接对应关系：

| 现状 | 调整 |
| --- | --- |
| `DensityMajorantGrid::segments()` 一次返回 classic DDA 的 `vector` | 提取逐格 `next()` 游标，输出 `densityMaximum * areaMajorant`；`NarrowBandMedium` 保持 classic 模式入口 |
| `ConditionalFlightKernel::twoSegmentMajorants()` 同时计算近段和整条后段 | kernel 负责 `evaluate(age)` 与 `intervalMajorant(a,b)`；新 `ConditionalMedium`/游标决定何时请求下一段 |
| `ConditionalNullTracking.cpp` 自己遍历近段与后段数组 | 提取一个消费 `ExtinctionSegment` 的 delta tracking 循环 |
| `MacrofacetPathTracer.cpp` 直接创建 conditional kernel 并调用函数 | 从 conditional medium 开始 flight 并调用 `sample()`；真实碰撞后生成新的出生观测 |

首个相机 flight 的 `F_0>0` 和完整梯度由现有 `startConditionalExterior()` 采样；
后续真实碰撞沿用 `F_0=0` 与采到的完整梯度。空碰撞、跨区间以及从非零年龄继续
追踪都不能重采出生观测或重置年龄。相位、Fresnel 与梯度分布维持当前模式定义。

## Conditional 分段上界

1. **出生段 `[0,h]`**：保留现有 Taylor 余项与稳定的 `E2/E4` 公式。
   因为 `Var(F_t|H)=O(t^4)`，把零点直接代入普通区间式会得到无穷上界。
   表面起点 `F_0=0` 使用推导式（9）的衰减包络；外部起点 `F_0>0`
   使用现有外部出生包络。`h` 限于首个光滑插值区间和已证明的正均值范围。
2. **后续有限段 `I=[a,b]`，`a>0`**：用现有 `MeanField::rayBounds()` 与
   NanoVDB 单元界求 `m_F`、方向导数和出生修正项的区间值。记
   `a_w=w^T A w`，从稳定的条件矩求
   `v_F,min=v_F(a)`、`z_min`、`mu_min`、`s_max`，按推导式（8）计算

   $$
   M_I=\frac{\phi(z_{\min})}{\sqrt{v_{F,\min}}\Phi(z_{\min})}
       \left[\frac{s_{\max}}{\sqrt{2\pi}}+(-\mu_{\min})_+\right].
   $$

   当 `mu_min>=0`，可用现有更紧的
   `s_max*phi(mu_min/s_max)` 替换括号中的粗界。
   逆 Mills 比、方差和正尾/负尾都按现有稳定分支计算。
3. **区间边界**：先在 NanoVDB 三线性插值单元边界和有限活动域边界处分段；
   再按光线相关长度限制宽度。若 `M_I*(b-a)` 过大，则二分当前区间并
   重新求界。这个量仅控制求界与空碰撞开销，不是消光截断阈值。
   游标按需处理当前区间；碰撞后的区间无需构造。

每段都必须有 `0<=M_I<infinity`、`a<b` 且最终覆盖所需 flight 范围；
若场无法给出所需区间界或浮点计算溢出，应明确失败。已有代码使用解析不等式
和 double 舍入裕量；若产品要求数学意义上**逐条可认证**的上界，
需再接入带向外舍入的区间算术处理 `exp`、正态尾函数、区间端点和插值运算。
有限 `M_I` 不能从少量消光采样值乘经验系数来代替。

## 落地顺序与验收

1. 提取共享的 `ExtinctionSegment` 和消费游标的 delta tracking 循环。
   先用适配器接入现有 `DensityMajorantGrid`，确认 classic local/global 图像
   与逃逸统计保持一致。
2. 新建 conditional medium 的游标，把现有近段证明和
   `ConditionalFlightKernel::intervalMajorant()` 接入 `next()`；将两段常数的
   快路径保留为可选优化。只在当前区间开销过大时做局部细分。
3. 用条件消光逐点数值核查每个返回段的上界，测试表面、外部、掠射起点、
   非零恢复年龄、跨插值单元以及高峰窄峰。再用独立的消光积分比较
   大样本首程逃逸率；该积分只用于测试，不进入渲染器。
4. 用 16×16、8 spp 做三模式图像和透射率曲线回归，再跑原有
   256×256、8 spp 的 shader ball 场景。记录分段数、求界次数、
   空碰撞数、越界次数和耗时，检查是否仍有数值失败。

这里移植的是 pbrt 的**分段上界接口和按需遍历**。
当前路径只需要碰撞位置与接受概率，继续保持直接使用消光系数的 delta tracking；
pbrt 为体积多次散射权重返回的 `T_maj` 不必加入当前的 conditional 权重流程。

## 落地实现与验证（分段游标）

当前实现使用 `ExtinctionSegment{beginAge,endAge,majorant}` 与共享的 `sampleSegmentedDeltaTracking()`。每段在内部用 `Exp(majorant)` 抽候选点，并以 `Sigma(t|H)/majorant` 接受；渲染路径不显式积分透射率。共享循环检查每段的有限非负上界、连续覆盖、候选点消光系数越界，并使用补偿求和保存小于当前年龄一个 ULP 的候选步长。

`NarrowBandMedium::sampleRay()` 返回逐格 DDA 游标。密度网格仍在构造时认证每格密度上界；沿光线只取下一格，再乘方向相关的投影面积上界。原 `DensityMajorantGrid::segments()` 保留为收集游标结果的兼容接口。局部与全局 classic 继续使用同一模型和采样入口。

`ConditionalMedium::sampleRay()` 返回 conditional 游标。首先按 Taylor 余项求出生段 `[0,h]`，随后从 `max(h,currentAge)` 开始按需证明当前有限区间的消光上界。后段先受相关长度和 NanoVDB 插值单元边界限制；若 `M*(b-a)>64`，游标把当前区间二分并重新证明子区间。该阈值只控制计算开销，不截断消光系数。出生段之后但仍在首个光滑单元内的区间，使用同一 Taylor 范围公式，避免 `t` 很小时普通区间算式丢失相消精度。游标在真实碰撞处停止，因此碰撞后区间不再计算。`ConditionalFlightKernel::twoSegmentMajorants()` 留作旧测试与外部调用的兼容接口；渲染器及透射率曲线已改走 `ConditionalMedium`。

Release 编译与 `ctest --test-dir build -C Release --output-on-failure` 通过。测试覆盖 DDA 游标与旧段序列一致、conditional 段的连续覆盖和内部消光探针、NanoVDB 跨单元区间、恢复年龄、小步长、窄峰以及独立积分的首程存活率。低分辨率输出在 [球体三模式渲染和透射率曲线](../outputs/segmented_sphere_validation/) 与 [shader ball 三模式渲染](../outputs/segmented_shader_ball_validation/)：均为 16×16、8 spp，六张图各场景均无数值失败。球体曲线仍使用同一批 1024 条平行光线比较三模式。另以 [shader ball 256×256、8 spp](../outputs/segmented_shader_ball_256/) 复验 conditional：白环境与方向环境分别约 19.86 s、20.01 s，均为 524288 条相机路径，数值失败为 0。以上时间是单次运行记录。

当前上界沿用解析不等式与 double 舍入裕量；若需要形式化逐条可认证的浮点上界，还需要将相关运算接入向外舍入的区间算术。

性能观察：球体低分辨率 conditional 每种环境约计算 62.7 万个有限区间上界，16×16、8 spp 的单次耗时约 0.43–0.55 s；此场景的 NanoVDB 单元边界较密，区间证明成为主要开销。shader ball 同规格耗时约 0.07–0.08 s。两组数字只反映上述配置和本机单次运行，不代表普遍加速。
