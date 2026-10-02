# Macrofacet NanoVDB → sparse-conv-gpis-tungsten

这个工具只接收 `field.mean_type: "nanovdb"` 的 Macrofacet 渲染配置。它会：

1. 从 `.nvdb` 中读取名为 `sdf` 的 float grid；
2. 原样复制 source active voxel 的整数坐标和值，并保留 NanoVDB 的 `dx` 和 origin；
3. 在原始 active bbox 外生成默认 3 voxel 的正 SDF halo；
4. 生成 `sparse-conv-gpis-tungsten` 可读取的 `sparse_conv_noise` 场景配置。

halo 使用最近原始边界节点的 SDF，并按向外的欧氏距离继续增大。原始边界必须全部是
active、有限且严格为正；否则转换器会拒绝生成可能引入伪零等值面的 VDB。

生成场景中的 process box 始终使用加 halo **之前**的原始 active-node 世界空间包围盒，
不会随 halo 扩大。这样 Tungsten 原有的 `activeMin + 2 / activeMax - 2` 边界 clamp
会落在 halo 内，不再把 process box 中的 SDF 查询推向物体。

工具不会翻转轴、重采样、归一化模型，也不会修改 Tungsten 源码中的边界 clamp。

## 构建与使用

启用默认的 field 构建后，目标名为 `macrofacet_nvdb_to_tungsten`：

```powershell
cmake --build build --config Release --target macrofacet_nvdb_to_tungsten

.\build\Release\macrofacet_nvdb_to_tungsten.exe `
  --config configs\render_shader_ball_nanovdb_global.json `
  --output outputs\tungsten_shaderball\scene.json
```

默认把 SDF 写到输出 JSON 旁边的 `scene_sdf.vdb`。可用 `--vdb <path>` 指定其他位置。
`--halo-voxels <n>` 可以增加 halo 宽度，但 `n` 不得小于 3。

## 转换约束

- 有 sidecar 时必须声明 `coverage: "full_domain"`；surface-band 场会被拒绝。
- 原始 bbox 的所有边界节点必须 active，且 SDF 必须严格为正。
- NanoVDB transform 必须轴对齐、正向且三轴体素尺寸相同。
- 当前预设支持 conductor 材质与各向同性 squared-exponential covariance。
  `material.roughness` 按 `lengthScale = sqrt(2) * sigma / roughness` 转换；也接受三个值
  相同的 `field.correlation_lengths`。
- `step_size` 设为半个 voxel；这是 Tungsten 的过零 ray-marching 参数，不是 SDF 重采样。
- Macrofacet 的非白色环境目前转换为常量白色 infinite sphere，并在命令行打印警告。

VDB grid 使用 `normalize_size: false`、identity config transform 和 `linear` 插值，所以
世界坐标由 OpenVDB transform 逆变换回原始 voxel index。halo 只扩大 VDB active bbox，
不改变原始 SDF 数据、世界变换或 process box。
