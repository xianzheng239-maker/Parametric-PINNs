# 参数化稳态概率密度反馈控制：电磁悬浮系统第一个目标密度

本项目使用物理信息神经网络（PINNs）学习稳态概率密度及反馈控制，提供 Soft（软约束）、Penalty（罚方法）和 Barrier（障碍方法）三种实现。

## 论文信息

- **标题**：Parametric Feedback Control for Shaping Stationary Probability Density Using Physics-Informed Neural Networks
- **作者**：Wantao Jia、Xianzheng Li、Xiaotong Feng
- **单位**：西北工业大学数学与统计学院；航空航天复杂系统科学教育部重点实验室。
- **对应内容**：随附论文第 4.2.1 节 *The first target PDF*，即第二个动力学系统（电磁悬浮系统）的第一个目标密度。

目标密度在计算域上归一化：

\[
p_\mathrm{target}(x_1,x_2;\omega)
=\frac{\exp[-\tfrac12\omega(500x_1^2+x_2^2)]}{Z(\omega)},
\qquad
Z(\omega)=\int_\Omega\exp[-\tfrac12\omega(500x_1^2+x_2^2)]\,dx_1dx_2.
\]

代码中的 `Q1_lb/Q1_ub` 实际控制目标参数 `omega` 的范围；指数中的系数 `500` 写在 `mytargetPDF.py` 中。

研究中使用本项目时，请引用上述论文。此引用建议不构成 MIT 许可证之外的使用限制。

## 文件结构

```text
.
├── README.md
├── LICENSE
├── barrier method equation2-target1/
├── penalty method equation2-target1/
├── soft method equation2-target1/
└── LXZ_Parametric_... (4).pdf
```

三个方法目录各自包含一套独立代码：

| 文件 | 作用 |
| --- | --- |
| `main.py` | 生成数据、建立网络、训练并导出结果 |
| `parameters.py` | 物理参数、计算域、网络结构和训练配置 |
| `myPDEs.py` | 稳态 FPK 方程残差 |
| `mytargetPDF.py` | 目标密度及归一化积分 |
| `myClass/` | 网络层、采样、积分、损失函数及优化器封装 |
| `utilities.py` | 训练流程、预测及结果保存 |
| `main1_G+P_non.py` | 附带的模拟脚本；见下文限制说明 |
| `pyjob.sh`、`jhsub.sh` | 原环境的运行/作业脚本，使用前需调整环境及集群配置 |

Barrier 目录另含 `1.py`、`myClass/myPINNFile_fu.py` 等辅助或备用实现；默认训练入口仍为 `main.py`。

## 环境依赖

主训练依赖 Python、TensorFlow、TensorFlow Probability、NumPy 和 SciPy。无需外部训练数据，程序自行采样。MATLAB 仅用于可选的 `.mat` 结果分析，不是训练依赖。

建议在独立环境中使用以下兼容组合；这是依据代码接口整理的安装基线，**不是已经完成端到端训练验证的环境**：

| 依赖 | 建议版本 | 用途 |
| --- | --- | --- |
| Python | 3.10，64 位 | 运行代码 |
| TensorFlow | 2.15.0 | 自动微分、网络及 Adam |
| TensorFlow Probability | 0.23.0 | L-BFGS |
| NumPy | 1.26.4 | 数值计算、采样及积分节点 |
| SciPy | 1.11.4 | 目标密度积分、MAT 文件读写 |

[TensorFlow Probability 0.23.0 发布说明](https://github.com/tensorflow/probability/releases/tag/v0.23.0)明确其与 TensorFlow 2.15.0 配套。代码使用旧版 Keras 的目录式 SavedModel 保存接口，以及 `numpy.product`，因此不要直接升级到 Keras 3 或 NumPy 2 后假定无需改动即可运行。

Windows 原生环境可使用 CPU。TensorFlow 2.10 之后的原生 Windows 版本不支持 CUDA GPU；GPU 环境可按 [TensorFlow 官方安装说明](https://www.tensorflow.org/install/pip)配置 Linux/WSL2，同时保持上述 TensorFlow/TFP 版本配套。三个方法当前源代码使用 `float64`，训练时间和内存需求取决于硬件。

## 安装与运行

### 1. 创建环境

以下命令从项目根目录执行。使用 Conda，命令适用于 PowerShell 或 Bash：

```shell
conda create -n pinn-levitation python=3.10 -y
conda activate pinn-levitation
python -m pip install "tensorflow==2.15.0" "tensorflow-probability==0.23.0" "numpy==1.26.4" "scipy==1.11.4"
python -m pip check
python -c "import tensorflow as tf, tensorflow_probability as tfp, numpy as np, scipy; print(tf.__version__, tfp.__version__, np.__version__, scipy.__version__); print(tf.config.list_physical_devices())"
```

### 2. 选择方法并训练

**必须进入对应方法目录再运行**，因为导入模块及输出路径使用当前工作目录。目录名称包含空格，需要保留引号。

运行 Barrier：

```shell
cd "barrier method equation2-target1"
python main.py
cd ..
```

运行 Penalty：

```shell
cd "penalty method equation2-target1"
python main.py
cd ..
```

运行 Soft：

```shell
cd "soft method equation2-target1"
python main.py
cd ..
```

每次执行依次进行数据生成、Adam 训练、L-BFGS 优化、预测和保存。默认 Adam 为 10000 轮，L-BFGS 最大迭代数为 5000；后者写在各目录的 `myClass/myMLFile.py` 中。日志通常每 100 个 Adam epoch 输出一次，初始化及积分计算也需要时间。

如仅检查运行流程，可在一份代码副本中减少 `parameters.py` 的 `epochs` 和 `myMLFile.py` 的 L-BFGS 最大迭代数。短训练结果不代表论文精度。

重复运行会写入相同输出位置，请先备份需要保留的结果。Barrier 旧有单精度保存模型不会因源码改为双精度而自动转换；使用当前双精度流程时需重新训练生成模型。

### 3. 查看输出

结果写入各自方法目录：

| 输出 | 内容 |
| --- | --- |
| `initial.mat` | 边界采样点及其指定密度值 |
| `pdf_mat_original.mat` | 预测网格、目标参数和预测密度 |
| `control_mat_original.mat` | 控制分量及其相加结果 |
| `history_original.mat` | 训练损失记录；Barrier 还记录障碍项 |
| `time_spend_original.mat` | 训练及预测流程计时 |
| `saved/NN/myPDF/` | 密度网络 SavedModel |
| `saved/NN/myControl/` | 控制网络 SavedModel |
| `my_control/` | Barrier 主程序额外导出的控制模型 |

例如在运行完成的方法目录中查看密度结果：

```shell
python -c "from scipy.io import loadmat; d=loadmat('pdf_mat_original.mat'); print([k for k in d if not k.startswith('__')]); print(d['pdf_mat'].shape)"
```

## 当前默认配置

| 配置 | 三种方法的共同值 |
| --- | --- |
| 计算及预测域 | `[-0.2, 0.2] × [-4, 4]` |
| 目标参数范围 | `[4.8, 5.3]` |
| 预测参数个数 | 6 |
| 训练点 / 边界点 | 10000 / 1000 |
| 预测网格步长 | `x1: 0.005`，`x2: 0.02` |
| 每方向积分子区间数 / 每子区间阶数 | 3 / 3，即每参数 81 个二维积分点 |
| 归一化积分 / 目标损失的参数采样数 | 10 / 20 |
| 浮点精度 | `float64` |
| 高斯白噪声强度 | `DG = 1.0` |
| 学习率调度边界 | `[5000, 10000, 15000]` |
| 学习率调度值 | `[0.0005, 0.0001, 0.00005, 0.00005]` |

`weight_u=0` 是边界密度均方误差的权重，当前不将该项加入优化目标；`weight_n=1` 控制归一化误差，`weight_c=5` 控制目标跟踪损失，`weight_f=1` 控制残差损失，`weight_uf=1` 为基础损失整体乘数。具体组合及方法特有项见各自 `myClass/myPINNFile.py`。

物理噪声通过 `myPDEs.py` 中的扩散项体现，主训练不需要生成白噪声时间序列。训练点随机采样及网络初始化属于另一个来源的随机性。

## 实现差异与复现范围

- 三种方法的 FPK 残差处理方式不同。Penalty 包含乘子；Barrier 包含松弛变量、乘子及障碍参数。这些是方法特有设置。
- Barrier 的 L-BFGS 额外加入系数为 `1e-10` 的变量平方和正则化；Soft、Penalty 没有该项。比较实验时应记录这一差异。
- 当前没有统一设置随机种子。参数一致不代表采样点或初始权重一致；严谨对比应记录种子、环境和多次独立实验结果。
- Penalty、Barrier 的 `myNN_finetuning` 仍读取未配置的 `omega_lb/omega_ub`。默认 `main.py` 不启用微调；启用前需与 `Q1_lb/Q1_ub` 的参数命名对齐。
- 附带的 `main1_G+P_non.py` 使用振子形式的动力学以及固定参数 `1.1`，与当前悬浮系统及训练参数范围不匹配。它不是此算例可直接使用的闭环验证入口，不能将其输出作为论文该算例的复现结果。Barrier 的 `1.py` 同样属于另行核对的辅助脚本。
- 本 README 根据当前源码和随附文稿整理。已进行静态检查，但编写文档的环境未安装 TensorFlow，未验证完整训练或论文图表复现。

## 许可证

本项目代码及配套代码文档采用 [MIT License](LICENSE)。允许使用、复制、修改、合并、发布、分发、再许可及销售代码，包括商业使用；分发软件副本或实质性部分时须保留版权声明和许可声明。软件按原样提供，不提供担保。完整条款以 `LICENSE` 为准，亦可参见 [MIT 标准文本](https://opensource.org/license/mit)。

随附论文 PDF 不属于上述代码许可的授权范围；论文及第三方依赖各自保留其版权和许可条件。
