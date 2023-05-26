import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    一个残差连接，后跟一个层归一化
    简化代码的目的来说，将归一化放在子层之前，而不是在子层之后
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 残差就是将输入直接加到子层的输出上，而不是将输入与子层的输出进行拼接
        return x + self.dropout(sublayer(self.norm(x)))
