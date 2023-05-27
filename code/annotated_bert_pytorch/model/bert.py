import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    定义 Bert 模型，继承自 nn.Module
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: 文本字典的大小
        :param hidden: Bert 模型的隐藏层维度
        :param n_layers: Bert 模型的层数
        :param attn_heads: Bert 模型的头的数量
        :param dropout: Bert 模型的 dropout 概率
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        # 基于论文中的描述，Bert 模型中的前馈网络的隐藏层维度是 attention层 隐藏层维度的 4 倍
        self.feed_forward_hidden = hidden * 4

        # Bert 模型的 embedding 层，包括位置 embedding，segment embedding，token embedding
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # 定义 Bert 模型的多层 transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # 对于padding 的部分，不参与 attention 的计算
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        '''
        x > 0 会返回一个布尔类型的张量，其中大于 0 的位置为 True，否则为 False
        unsqueeze(1) 会在第 1 维上增加一个维度，将张量从形状 (batch_size, seq_len) 变为 (batch_size, 1, seq_len)
        repeat(1, x.size(1), 1) 会将张量在第 2 维上复制 x.size(1) 次，即将张量从形状 (batch_size, 1, seq_len) 变为 (batch_size, seq_len, seq_len)
        unsqueeze(1) 会在第 1 维上增加一个维度，将张量从形状 (batch_size, seq_len, seq_len) 变为 (batch_size, 1, seq_len, seq_len)
        最终得到的张量 mask 是一个形状为 (batch_size, 1, seq_len, seq_len) 的张量
        其中第 2 维表示每个位置是否为 padding，True 表示不是 padding，False 表示是 padding
        '''
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # 将输入的文本序列转换为向量序列
        x = self.embedding(x, segment_info)

        # 对于每个 transformer block，进行 forward 操作
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
