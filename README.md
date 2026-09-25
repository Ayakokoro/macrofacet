# Macrofacet / correlated free-flight reference implementation

这是《Macrofacet Theory for Gaussian Process Statistical Surfaces》的独立 CPU/double 复现。渲染支持 Classic 和 conditional29 两种传输模式：

- `classic`：论文的 Gaussian 高度密度、Generalized Gaussian/Beckmann/GGX NDF、方向消光和 conductor 多次散射；
- `conditional29`：保留起点场值与完整梯度，使用式 (29) 的 crossing/exterior 比率作为 hazard，以自适应积分和受保护 Newton 反演连续自由程；支持正场值首段初始化；
- `gp-reference`：联合采样条件 GP，并用逐步加密的检查点近似整段首次相交公式。它是数值参考，不会被描述成低成本精确算法。

工程使用 C++17、Eigen 和 nlohmann/json。依赖由 `vcpkg.json` 声明。

## 构建

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

如果你的 vcpkg 根目录没有保存在 `VCPKG_ROOT`，把 toolchain 路径替换成实际位置。

## 运行

```powershell
build\Release\macrofacet_experiments.exe curves --config configs\macrofacet_experiments.json
build\Release\macrofacet_experiments.exe gp-reference --config configs\macrofacet_experiments.json
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json
py scripts\plot_experiments.py outputs\macrofacet_experiments
```

`configs/macrofacet_experiments.json` 是规格预算；`macrofacet_ci.json` 是快速端到端配置；`macrofacet_white_validation.json` 是实际运行过的 256 spp Classic 白炉验证。

当前场管线会先把解析均值场烘焙为 NVDB；直接读入 NVDB 时，追踪域自动取自网格范围。窄带输入以 `density` 网格定义传输带：Classic 在带内进行 DDA null tracking，conditional29 在带内计算条件 hazard、带外按真空处理。使用说明见 [文档导航](docs/README.md)，尤其是
[NVDB 窄带传输](docs/NVDB_TRACING.md) 和
[配置参考](docs/CONFIGURATION_REFERENCE.md)。历史设计与当时的验证数据
保留在 [实现报告](docs/IMPLEMENTATION_REPORT.md) 及 `docs/archive/`。

conditional29 快速运行：

```powershell
build\Release\macrofacet_experiments.exe render --config configs\conditional29_validation.json
build\Release\macrofacet_experiments.exe render --config configs\conditional29_sphere_validation.json
```
