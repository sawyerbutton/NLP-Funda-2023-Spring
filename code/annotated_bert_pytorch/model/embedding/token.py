import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0) # 调用父类的构造函数
        # 直接使用 nn.Embedding 的构造函数，padding_idx=0 表示填充的索引为0
        # nn.Embedding 的构造函数的参数：num_embeddings (int) – size of the dictionary of embeddings
        # nn.Embedding 的作用是使用一个随机初始化的矩阵，将每个token映射为一个向量
