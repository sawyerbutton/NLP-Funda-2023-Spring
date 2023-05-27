import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    接收模型的维度和头的数量作为参数
    """

    def __init__(self, h, d_model, dropout=0.1): # h是头的数量，d_model是模型的维度， dropout是dropout的概率
        super().__init__()
        assert d_model % h == 0 # 模型的维度必须是头的数量的整数倍

        # We assume d_v always equals d_k
        self.d_k = d_model // h # d_k是每个头的维度
        self.h = h # 头的数量

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)]) # 3个线性层，分别是query，key，value
        # nn.ModuleList是一个储存不同模块的列表，这些模块的参数会被自动添加到整个网络中
        # 这里的3个线性层的输入和输出维度都是d_model且初始化的权重都是随机的
        self.output_linear = nn.Linear(d_model, d_model) # 输出的线性层，维度是d_model
        self.attention = Attention() # 初始化Attention

        self.dropout = nn.Dropout(p=dropout) # dropout层

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0) # batch_size

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 将query，key，value分别输入到3个线性层中，得到3个输出分别对应的query，key，value
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # 将query，key，value输入到Attention中，得到输出结果向量和注意力分数
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # 将多头的输出结果向量拼接起来，然后输入到输出的线性层中
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x) # 返回输出的线性层的输出结果
