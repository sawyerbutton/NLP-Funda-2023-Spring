import torch.nn as nn
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    "实现 FeedForward Neural Networks，两层全连接网络"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # 第一层全连接网络
        self.w_2 = nn.Linear(d_ff, d_model) # 第二层全连接网络
        self.dropout = nn.Dropout(dropout) # dropout层
        self.activation = GELU() # 使用 GELU 作为激活函数

    def forward(self, x):
        # 先经过第一层全连接网络，再经过dropout层，再经过激活函数，最后经过第二层全连接网络
        return self.w_2(self.dropout(self.activation(self.w_1(x)))) 