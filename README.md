本项目是基于原Pytorch-Cifar项目的天津大学2024-2025学年第一学期神经网络与深度学习课程实验。

选取Cifar10和MNIST作为数据集，实验目的如下：

模型架构分析(模型深度与宽度，激活函数，归一化)：

- I. 对模型的层数/滤波器大小/滤波器数量等方面改进（至少改进2种）；

- II. 比较至少两种不同的激活函数（例如ReLU , GELU, Tanh, RELU等）；

- III. 在至少两个建议的数据集分析模型改进和激活函数的影响。

- I. 归一化方法分析与比较；

- II. 比较至少两种不同的归一化方法（例如无归一化 , BN, LN, GN等）；

- III. 在至少两个建议的数据集上分析比较不同归一化方法在不同Batchsize下的性能。

针对目的设计以下实验：

- 实验一：ResNet18和ResNet34性能对比

- 实验二：ResNet18卷积核大小为3和ResNet卷积核大小为5性能对比

- 实验三：激活函数分别为Relu、Gelu、Leaky Relu性能对比

本实验环境：ubuntu20.04，PyTorch框架版本为1.10.0，Cuda版本为11.3，Python版本为3.8，使用一张24g的RTX4090D训练。

对于本项目各文件说明如下：

- plots文件夹用于保存各实验生成的图片，其中，3_5_开头为实验二训练过程可视化，18_34_开头为实验一，R_G_L为实验三。

- 3_5_Comp.py文件为训练实验二的脚本，18_34_Comp.py文件为训练实验一的脚本，Relu_Gelu_Leaky.py文件为训练实验三的脚本。

- 实验二和实验三对于不同模型架构的修改可见于models文件夹中的resnet.py文件

以下为原项目ReadMe内容：

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01


