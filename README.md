# Macrofacet / correlated free-flight reference implementation

这是《Macrofacet Theory for Gaussian Process Statistical Surfaces》的独立 CPU/double 复现，并在论文去相关模型（A）之外实现两种显式相关闭合：

- `classic`：论文的 Gaussian 高度密度、Generalized Gaussian/Beckmann/GGX NDF、方向消光和 conductor 多次散射；
- `conditional29`：保留最近真实交点的 `F=0` 与完整梯度，使用论文式 (29) 的 crossing/exterior 比率作为 hazard；
- `midpoint`：在同一条件统计中，对分子和分母同时加入一个中点正值筛选；
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

详细的文件映射、实际测试数据、近似和限制见 [docs/IMPLEMENTATION_REPORT.md](docs/IMPLEMENTATION_REPORT.md)。

