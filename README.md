# CNN 训练与测试 - CIFAR-10

## 1. 项目简介
本项目实现了一个 **卷积神经网络 (CNN)**，用于在 **CIFAR-10** 数据集上进行图像分类。项目包含数据预处理、模型构建、训练、验证和测试功能。

## 2. 环境要求
- Python 3.7 及以上
- PyTorch
- torchvision
- tqdm
- matplotlib

### 依赖安装
建议使用 Conda 创建虚拟环境，并安装依赖：
```bash
conda create -n cnn_env python=3.9 -y
conda activate cnn_env
pip install torch torchvision tqdm matplotlib
```

## 3. 使用方法

### 训练模型  
直接**运行** `my_CNN.py` 文件即可开始训练：  
- 在 **PyCharm** 或 **VS Code** 中打开 `my_CNN.py`，点击**运行按钮**。
- 代码会自动加载数据集、训练模型，并保存效果最好的模型 (`best_cnn_model.pth`)。

### 测试模型  
训练完成后，代码会自动在测试集上评估模型，无需额外运行命令。  
如果需要**单独测试**，可以重新运行 `my_CNN.py`，代码会加载 `best_cnn_model.pth` 进行测试。

## 4. 代码结构
- **数据加载**：使用 `torchvision.datasets.CIFAR10` 进行数据预处理和划分。
- **CNN 模型**：包含多个卷积层、BatchNorm、ReLU、MaxPooling 和 Dropout。
- **训练**：使用 `Adam` 进行优化，并加入 `StepLR` 进行学习率衰减。
- **验证 & 测试**：在验证集上评估模型，并在测试集上计算最终准确率。
- **可视化**：绘制训练和验证损失曲线。

## 5. 参考
- CIFAR-10 数据集: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch 官方文档: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
