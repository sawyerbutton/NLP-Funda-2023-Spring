import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding 由以下特征组成：
    TokenEmbedding：普通的嵌入矩阵
    PositionalEmbedding：使用 sin 和 cos 添加位置信息
    SegmentEmbedding：添加句子段信息，(sent_A:1, sent_B:2)
    所有这些特征的加和是 BERTEmbedding 的输出
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: 字典的大小
        :param embed_size: token embedding 的维度
        :param dropout: dropout 的概率
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label) # 将三个特征相加
        return self.dropout(x) # 对相加的结果进行dropout
