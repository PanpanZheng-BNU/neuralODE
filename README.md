# NeuralODE

本库用于存储“面向复杂系统的人工智能”课程期末作业的相关代码和文档。本大作业分为单个部分：

1. 给定任意时间序列数据，利用Neural ODE进行动力学学习
    - [x] 直接简单的动力系统进行动力学的学习，即学习动力系统的 $f$ 
    - 结合RNN和Variational Autoencoder的思想：学习隐变量 (latent vriable) $z$ 的动力学演化方程。
        - [x] VAEs: 不同于传统的 Autoencoder, 其也由编码器 (encoder) 和解码器 (decoder) 组成，但其编码的结果并非一个确定的变量，而是一个服从特定分布（一般为高斯分布）的随机变量（我们可以简单的通过均值向量 $\boldsymbol{\mu}$ 和协方差矩阵 $\Sigma$ 来描述一个多维高斯分布）
        - [ ] RNNs: GRU, LSTM
    - [ ] 基于 "[Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907)" 这篇文章中所提及的 ODE-RNN 方法，对于不规则采样的时间序列数据进行动力学的学习。（大致思想：对于为观测到的隐变量 $h_{t_{i}}$  我们可以使用 $\text{ODESolver}(f_{\theta}, h_{t_{i-1}}, (t_{i-1}, t_{i}))$  的方法来估计得到 $h_{t_{i}}$, 然后将估计到的 hidden variable $h'_{t_i}$ 放入下一个 $\text{RNNCell}$  中获得下一个隐变量和输出）。然后我们根据之前生成的分布随机抽样得到一个隐变量 $z$，然后将其放入解码器中得到输出。
2. 实现利用Neural ODE实现的Continuous Normalizing Flow
    - [x] 标准化流的基本思想：离散流，连续流（可以参考 ResNet 流的方法）
    - [ ] 通过ODE的方法来实现连续流
    - [ ] [OT 流](https://arxiv.org/abs/2006.00104)
    - [ ] [FFJORD](https://arxiv.org/abs/1810.01367)
3. 利用Neural ODE实现复杂动力学学习（涌现动力学），将该论文中的所有部件全部替换为Neural ODE。

# test
- [x] Venus
- [x] Earth (Orbit/Moon)
- [x] Mars
- [ ] Jupiter
- [ ] Saturn
- [ ] Uranus
- [ ] Neptune
- [ ] Comet Haley
