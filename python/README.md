# Matérn 5/2 first-passage surrogate

这个目录包含 collision-state first-passage 累计消光模型的完整 Python 训练端。C++ 只负责生成 Monte Carlo 区间计数；Python 负责独立样条诊断、神经网络训练、验证、导出和推理。

## 模型

对固定的 Matérn 5/2 kernel，使用无量纲变量

\[
q=L/\ell,\qquad
\eta=(\beta_0,\beta_a,\beta_g)
=\left(\frac{m_b}{\sigma},\frac{a_b\ell}{\sigma},
\frac{g_d\ell}{\sigma}\right).
\]

网络只预测 32 个固定 I-spline 基的系数：

\[
(\beta_0,\beta_a,\log\beta_g)
\xrightarrow{3\to32\to32\to31}\tilde a_i,
\qquad a_0=0,\quad a_i=\operatorname{softplus}(\tilde a_i).
\]

累计 hazard、透射率和物理消光系数为

\[
H(q\mid\eta)=\sum_i a_i(\eta)I_i(q),\qquad
T(L\mid\eta)=e^{-H(L/\ell\mid\eta)},
\]

\[
\Sigma(L\mid\eta)=\frac1\ell\frac{\partial H}{\partial q}
=\frac1\ell\sum_i a_i(\eta)M_i(q).
\]

默认使用 32 个二次 M-spline（积分后为 I-spline）。端点各重复 3 次，29 个内部 knot 为

\[
q_j=q_{\max}(j/30)^{1.5},\qquad j=1,\ldots,29.
\]

这个参数化直接保证 `H` 非减、`T` 非增且位于 `(0,1]`。训练目标不是 Monte Carlo hazard 的数值导数，而是每个区间的原始风险集和事件计数。若第 `j` 个区间有 `n_j=at_risk`、`d_j=events`，则

\[
\Delta H_j=\sum_i a_i[I_i(q_{j+1})-I_i(q_j)],
\qquad p_j=1-e^{-\Delta H_j},
\]

\[
\mathcal L=-\sum_j d_j\log p_j+(n_j-d_j)\Delta H_j.
\]

这是 grouped interval first-passage 数据的二项 likelihood，避免了先求 hazard 再平滑的噪声放大。

## 环境

当前机器使用默认的 `B:\Python\Python310`，不再创建项目内虚拟环境。CUDA 13.0 版本的依赖可从项目根目录安装：

```powershell
python -m pip install -r python\requirements-cuda.txt
python -m pip install -e python --no-deps --no-build-isolation
```

如果没有把仓库包做 editable 安装，也可以在当前 PowerShell 会话中设置 `$env:PYTHONPATH="$PWD\python"` 后运行下列命令。

## 1. 用 C++ 生成正式数据

```powershell
build\Release\macrofacet_experiments.exe first-passage `
  --config configs\collision_state_matern52_training.json
```

默认配置使用 512 个 Latin-hypercube 状态、每状态 4096 条 realization、两个步长和 200 个输出区间。最细步长的行带有 `training_resolution=1`，Python 加载器只使用这些行；较粗步长只用于检查 crossing 离散收敛。`write_raw_samples=false` 会省略很大的逐轨迹 CSV，因为当前 likelihood 只需要 `first_passage_curves.csv` 中的 `at_risk/events`。

正式运行计算量较大。先检查管线可用：

```powershell
build\Release\macrofacet_experiments.exe first-passage `
  --config configs\collision_state_matern52_training.json `
  --trials 8 --bins 20 --threads 4 `
  --output outputs\collision_state_matern52_training_smoke
```

冒烟数据只用于检查程序，不适合评价模型误差。

## 2. 先测表示误差

在训练 MLP 前，每个状态独立拟合一组非负系数：

```powershell
python -m macrofacet_fpt.fit_splines `
  --data outputs\collision_state_matern52_training\first_passage_curves.csv `
  --kernel matern52 `
  --output outputs\models\matern52_ispline32_independent
```

输出 `independent_coefficients.csv` 和 `independent_fit_metrics.json`。若独立拟合误差已经很大，应先调整 knot、`q_max` 或 basis 数量；此时增加 MLP 容量没有意义。

## 3. 训练状态到系数的 MLP

```powershell
python -m macrofacet_fpt.train `
  --config python\configs\matern52_ispline32.json
```

训练/验证/测试按完整 `state_id` 切分，输入归一化只由训练集合计算。默认使用 Adam、batch 64、学习率 `1e-3`、weight decay `1e-6` 和 patience 50 的 early stopping。

主要输出位于 `outputs/models/matern52_ispline32`：

- `checkpoint.pt`：可恢复的 PyTorch checkpoint；
- `coefficient_net.pt`：只做系数预测的 TorchScript 模型；
- `model_bundle.json`：网络权重、归一化、kernel、有效参数范围、knots 和指标，供后续 C++ 实现；
- `ispline_basis.npz`：样条定义；
- `history.csv`、`metrics.json`：训练历史与三组指标；
- `coefficients.csv`、`predictions.csv`：逐状态系数和逐区间预测。

快速训练管线测试可添加 `--epochs 3 --data ... --output ... --device cpu`。

## 4. 评价与查询

```powershell
python -m macrofacet_fpt.evaluate `
  --checkpoint outputs\models\matern52_ispline32\checkpoint.pt `
  --data outputs\collision_state_matern52_training\first_passage_curves.csv

python -m macrofacet_fpt.predict `
  --checkpoint outputs\models\matern52_ispline32\checkpoint.pt `
  --beta-0 0 --beta-a 0 --beta-g 1 `
  --q 0,0.5,1,2,4,8 --ell 0.1

python -m macrofacet_fpt.predict `
  --checkpoint outputs\models\matern52_ispline32\checkpoint.pt `
  --beta-0 0 --beta-a 0 --beta-g 1 `
  --optical-depth 0.1,0.5,1.0 --ell 0.1
```

渲染时，在一次碰撞产生次级射线后只计算一次 `eta` 和系数。对沿该次级射线的任意距离使用同一个 birth state，并令 `q=L/ell`。透射查询返回 `exp(-H(q))`；采样时取 `xi=-log(1-u)` 并求解 `H(q)=xi`。实际还应先比较 `xi` 与 ray exit 处的 `H(q_exit)`：若更大，则该段无碰撞。不要在每个 marching 点重新定义 birth state，也不要在 `q>q_max` 或训练参数 box 外静默外推。

## 测试

```powershell
$env:PYTHONPATH="$PWD\python"
python -m unittest discover -s python\tests -v
```
