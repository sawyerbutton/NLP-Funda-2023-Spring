import torch.nn as nn
import torch
import math


class GELU(nn.Module):
    """
    定义 GELU 激活函数
    """

    def forward(self, x):
        # GELU 激活函数的公式解释为：0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        # GELU 比 RELU 效果更好，但是计算速度更慢，之所以使用 GELU 是因为其在负轴上的值更小，更适合 NLP 任务
        # GELU 的论文：https://arxiv.org/abs/1606.08415
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
