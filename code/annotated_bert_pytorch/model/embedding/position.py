import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # 在对数空间中计算位置编码
        pe = torch.zeros(max_len, d_model).float() # 初始化位置编码
        pe.require_grad = False # 不需要梯度

        position = torch.arange(0, max_len).float().unsqueeze(1) # 位置向量
        # 位置编码的方式是：pos / (10000^(2i/d_model))
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # 位置编码的分母

        pe[:, 0::2] = torch.sin(position * div_term) # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数位置

        pe = pe.unsqueeze(0) # 增加一个维度
        self.register_buffer('pe', pe) # 将pe注册为buffer

    def forward(self, x):
        # 因为这里的x是一个batch的数据，所以需要将位置编码的矩阵扩展为batch_size个
        return self.pe[:, :x.size(1)] # 返回位置编码
    
