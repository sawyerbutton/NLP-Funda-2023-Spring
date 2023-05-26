from torch.utils.data import Dataset
import tqdm
import torch
import random


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=0, on_memory=True):
        self.vocab = vocab # 词表
        self.seq_len = seq_len # 句子长度

        self.on_memory = on_memory # 是否将数据加载到内存中
        self.corpus_lines = corpus_lines # 数据集的行数
        self.corpus_path = corpus_path # 数据集路径
        self.encoding = encoding # 数据集编码

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory: # 如果数据集行数为None并且不将数据加载到内存中
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1 # 统计数据集行数

            if on_memory: # 如果将数据加载到内存中
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)] # 将数据集加载到内存中
                self.corpus_lines = len(self.lines) # 统计数据集行数

        if not on_memory: # 如果不将数据加载到内存中
            self.file = open(corpus_path, "r", encoding=encoding) # 打开数据集
            self.random_file = open(corpus_path, "r", encoding=encoding) # 打开数据集
            # 上面的代码可以简化为 self.file, self.random_file = open(corpus_path, "r", encoding=encoding), open(corpus_path, "r", encoding=encoding)
            # randomint 用于生成一个指定范围内的整数，randomint(a, b) 用于生成一个 [a, b] 之间的整数
            # 源码中没有声明左边界的值，可能会导致错误, 因此加上一个左边界1
            for _ in range(random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)): # 随机读取1000行数据或者数据集行数
                self.random_file.__next__() # 读取一行数据

    def __len__(self):
        return self.corpus_lines # 返回数据集行数

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split() # 将句子分割成单词, 其实就是x
        output_label = [] # 用于存储标签，其实就是y
        # 在MLM任务中，需要对x进行替换，然后用y来表示替换的结果

        for i, token in enumerate(tokens): # 遍历句子中的每个单词
            prob = random.random() # 生成一个随机数
            if prob < 0.15: # 15%的概率进行替换
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index # 将单词替换成mask的索引

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab)) # 将单词替换成随机单词的索引

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index) # 将单词替换成当前单词的索引
                    # self.vocab.stoi.get(token, self.vocab.unk_index) 用于获取单词的索引，如果单词不存在则返回unk_index

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                # self.vocab.stoi.get(token, self.vocab.unk_index) 用于获取单词的索引，如果单词不存在则返回unk_index
                # 用于记录替换前的单词索引

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index) # 将单词替换成当前单词的索引
                output_label.append(0) # 0 表示不替换

        return tokens, output_label # 返回替换后的句子和标签

    def random_sent(self, index): 
        # 用于创建NSP任务所需要的相连句子与非相连句子
        t1, t2 = self.get_corpus_line(index) # 获取第 index 行数据

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5: # 50%的概率返回两个句子
            return t1, t2, 1 # 1 表示两个句子是相邻的
        else:
            return t1, self.get_random_line(), 0 # 0 表示两个句子不是相邻的

    def get_corpus_line(self, item):
        if self.on_memory: # 如果将数据加载到内存中
            return self.lines[item][0], self.lines[item][1] # self.lines[item][0] 表示第 item 行的第一个数据，self.lines[item][1] 表示第 item 行的第二个数据
        else:
            line = self.file.__next__() # 尝试读取一行数据
            if line is None: # 如果读取到的数据为空
                self.file.close()  # 关闭数据集
                self.file = open(self.corpus_path, "r", encoding=self.encoding) # 重新打开数据集
                line = self.file.__next__() # 读取一行数据

            t1, t2 = line[:-1].split("\t") # line[:-1] 表示去掉最后一个字符，即换行符, 然后按照 \t 进行分割
            return t1, t2 # 返回分割后的数据

    def get_random_line(self):
        if self.on_memory: # 如果将数据加载到内存中
            return self.lines[random.randrange(len(self.lines))][1] # 随机返回一行数据

        line = self.file.__next__() # 读取一行数据
        if line is None: # 如果读取到的数据为空
            self.file.close() # 关闭数据集
            self.file = open(self.corpus_path, "r", encoding=self.encoding) # 打开数据集
            # 源码中没有声明左边界的值，可能会导致错误, 因此加上一个左边界1
            for _ in range(random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]
