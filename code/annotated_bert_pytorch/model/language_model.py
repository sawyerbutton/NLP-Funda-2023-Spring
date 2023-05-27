import torch.nn as nn

from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    对 NSP 和 MLM 两个任务的结果进行拼接，得到最终的 BERT 模型
    重点在下面两个类
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    NSP 任务的模型会判断第一句和第二句是否是上下句关系
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2) # 2-class classification
        self.softmax = nn.LogSoftmax(dim=-1) # log softmax

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0])) # x[:, 0] 表示第一个 token 的输出也就是 CLS 的输出


class MaskedLanguageModel(nn.Module):
    """
    MLM 任务的模型会根据输入的 token 序列预测每个位置的 token，也就是说，模型会预测输入序列中被 mask 掉的 token
    是一个N分类任务，N为词表大小
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: Bert 模型的输出大小
        :param vocab_size: 词表大小
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size) # 线性层的输出是词表大小的向量，每个位置对应一个词
        self.softmax = nn.LogSoftmax(dim=-1) # 对每个位置的向量进行 softmax，得到每个位置的词的概率

    def forward(self, x):
        return self.softmax(self.linear(x))
