# NeuralODE
本库用于存储关于 neural ODE 的相关代码和文档。主要分为三个部分：
1. 使用 neural ODE 进行简单时间序列的动力学学习
2. 使用 neural ODE 实现连续化的标准化流 (normalizing flow)
3. 将 [Zhang, 2022](https://www.mdpi.com/1099-4300/25/1/26) 中模型的各部分替换为 neural ODE，使用 neural ODE 进行心率信号的分类

## Part I 简单动力学学习 @[申晓达](https://github.com/lancesxdlance)
详细见[ode.ipynb](./ode.ipynb)文件
## Part II 标准化流 @[郑盼盼](https://github.com/PanpanZheng-BNU)
> [`normalizing flow`](./normalizingFlow) 文件夹中存放了使用 neural ODE 实现的标准化流的代码
> [`utils`](./normalizingFlow/utils) 文件夹中存放了一些用于训练工具函数，如数据集的生成，模型的保存等

### Planar Flow
- [`PlanarFlow_training.ipynb`](./normalizingFlow/PlanarFlow_training.ipynb) 中构建了一个 planar flow 的模型，并进行训练
- [`PlanarFlow_validation.ipynb`](./normalizingFlow/PlanarFlow_validation.ipynb) 中使用训练好的 planar flow 模型进行验证，所有的模型参数都保存在 [`Planar`](./normalizingFlow/Planar) 文件夹中

### OT Flow
- [`Phi.py`](./normalizingFlow/Phi.py) 中定义了 OT flow 中所需要的 Phi 函数，以及 Phi 函数的梯度以及 Hessian 矩阵的迹。
- [`OTFlow_training.ipynb`](./normalizingFlow/OTFlow_training.ipynb) 中构建了一个 OT flow 的模型，并进行训练
- [`OTFlow_validation.ipynb`](./normalizingFlow/OTFlow_validation.ipynb) 中使用训练好的 OT flow 模型进行验证，所有的模型参数都保存在 [`OT-Flow`](./normalizingFlow/OT-Flow) 文件夹中

## Part III NIS @[龙裕坤](https://github.com/Thedeephunter)
- [`Neural_Information_Squeezer.ipynb`](./NIS&Classification/Neural_Information_Squeezer.ipynb) 中实现了用neural ODE结构替换了宏观动力学学习器的NIS的复现
- [`Neural_Ode_Squeezer (1).ipynb`](./NIS&Classification/Neural_Ode_Squeezer%20(1).ipynb) 中实现了所有结构均替换为neural ODE结构的NIS的复现
- [`Real_NVP_400_steps.gif`](./NIS&Classification/Real_NVP_400_steps.gif) 中展示了用neural ODE结构替换了的宏观动力学学习器的NIS在400个时间步上进行预测的动态效果
- [`Ode_400_steps.gif`](./NIS&Classification/Ode_400_steps.gif) 中展示了用neural ODE结构替换了所有结构的NIS在400个时间步上进行预测的动态效果


## Part IV Classification @[龙裕坤](https://github.com/Thedeephunter)
- [`ODE4classification.ipynb`](./NIS&Classification/ODE4classification.ipynb) 中实现了利用基于neural ODE特征提取的分类模型实现了对MIT-BIH数据库的ECG数据的分类
