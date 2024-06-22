# NeuralODE
本库用于存储关于 neural ODE 的相关代码和文档。主要分为三个部分：
1. 使用 neural ODE 进行简单时间序列的动力学学习
2. 使用 neural ODE 实现连续化的标准化流 (normalizing flow)
3. 将 [Zhang, 2022](https://www.mdpi.com/1099-4300/25/1/26) 中模型的各部分替换为 neural ODE，使用 neural ODE 进行心率信号的分类

## Part I
## Part II 标准化流
> [`normalizing flow`](./normalizingFlow) 文件夹中存放了使用 neural ODE 实现的标准化流的代码
> [`utils`](./normalizingFlow/utils) 文件夹中存放了一些用于训练工具函数，如数据集的生成，模型的保存等

### Planar Flow
- [`PlanarFlow_training.ipynb`](./normalizingFlow/PlanarFlow_training.ipynb) 中构建了一个 planar flow 的模型，并进行训练
- [`PlanarFlow_validation.ipynb`](./normalizingFlow/PlanarFlow_validation.ipynb) 中使用训练好的 planar flow 模型进行验证，所有的模型参数都保存在 [`Planar`](./normalizingFlow/Planar) 文件夹中

### OT Flow
- [`Phi.py`](./normalizingFlow/Phi.py) 中定义了 OT flow 中所需要的 Phi 函数，以及 Phi 函数的梯度以及 Hessian 矩阵的迹。
- [`OTFlow_training.ipynb`](./normalizingFlow/OTFlow_training.ipynb) 中构建了一个 OT flow 的模型，并进行训练
- [`OTFlow_validation.ipynb`](./normalizingFlow/OTFlow_validation.ipynb) 中使用训练好的 OT flow 模型进行验证，所有的模型参数都保存在 [`OT-Flow`](./normalizingFlow/OT-Flow) 文件夹中

