import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    "构建 LayereNorm 层"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # nn.Parameter 是一种特殊的 Variable，会被自动添加到 Module 的 参数列表中
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps # epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 求均值
        std = x.std(-1, keepdim=True) # 求标准差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 # 归一化,通过仿射变换将数据映射到另一个空间
        # 仿射变换是一种线性变换，保持了原有的“平直性”，并且保持了原有的“平行性”
