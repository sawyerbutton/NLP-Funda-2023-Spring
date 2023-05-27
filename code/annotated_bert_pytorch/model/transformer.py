import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    TransformerBlock = MultiHead_Attention + Feed_Forward with sublayer connection + normalization
    多头注意力层 + FFNN 层 + 残差连接 + 归一化
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: transformer block 中 attention 层的隐藏层维度
        :param attn_heads: 多头注意力层的头的数量
        :param feed_forward_hidden: FFNN 层的隐藏层维度，通常是 attention 层隐藏层维度的 4 倍
        :param dropout: dropout 概率
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)) # self-attention层进行计算后送入残差连接层
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
