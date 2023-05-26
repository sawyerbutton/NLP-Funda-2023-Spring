import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0) # 调用父类的构造函数
        # 直接使用 nn.Embedding 的构造函数，padding_idx=0 表示填充的索引为0
        # 这里的 3 表示有三种句子段，分别是句子A、句子B和填充
