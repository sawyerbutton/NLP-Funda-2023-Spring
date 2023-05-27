import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    计算注意力权重，使用query和key计算权重，然后用权重对value加权求和
    注意这里是计算单头的得分，还需要进行多头的合并计算
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1)) # query和key的转置相乘，除以sqrt(d_k), d_k是query和key的维度也就是 hidden_size / num_heads
        # scores的维度是[batch_size, num_heads, seq_len, seq_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # mask为0的地方填充为-1e9
            # 对于mask为0的地方，经过softmax后，就会变成0，这样就不会对value产生影响，注意先进行mask操作再进行softmax操作

        p_attn = F.softmax(scores, dim=-1) # 对最后一个维度进行softmax操作，也就是对key的维度进行softmax操作
        # p_attn的维度是[batch_size, num_heads, seq_len, seq_len]
        if dropout is not None:
            p_attn = dropout(p_attn) # dropout操作

        return torch.matmul(p_attn, value), p_attn # 返回加权后的value和注意力权重
