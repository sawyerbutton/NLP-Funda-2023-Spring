import pickle
import tqdm
from collections import Counter # Counter 是一个简单的计数器，例如，统计字符出现的个数


class TorchVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """
        self.freqs = counter # collections.Counter 是一个简单的计数器，例如，统计字符出现的个数
        counter = counter.copy() # 返回一个浅拷贝，即只拷贝了容器内元素的地址，而不拷贝元素本身
        min_freq = max(min_freq, 1) # min_freq 为 传入值和1相比的最大值

        self.itos = list(specials) # 将传入的特殊字符列表转换为列表
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        # 特殊字符的频率在构建词汇表时不计算次数
        for tok in specials: # 遍历特殊字符列表
            del counter[tok] # 删除特殊字符列表中的字符

        max_size = None if max_size is None else max_size + len(self.itos) # max_size 为 None 或者 设定的max_size + 特殊字符列表长度

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0]) # 按照字母顺序排序
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True) # 按照词频排序

        for word, freq in words_and_frequencies: # 遍历排序后的词频
            if freq < min_freq or len(self.itos) == max_size: # 如果词频小于设定的最小词频或者词表长度等于设定的最大词表长度
                break # 跳出循环，因为单词与词频已经进行了排序的操作，因此一旦出现不满足条件的词频，代表后面的words_and_frequencies中的词频都不满足条件
            self.itos.append(word) # 将词频大于设定的最小词频的词加入到词表中

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)} # 将词表中的词和对应的索引组成字典

        self.vectors = None # 初始化词向量为None
        if vectors is not None:  # 如果传入了词向量
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache) # 调用load_vectors方法
        else:
            assert unk_init is None and vectors_cache is None # 如果没有传入词向量，那么unk_init和vectors_cache必须为None

    def __eq__(self, other): # 重写__eq__方法
        if self.freqs != other.freqs: # 如果两个词频不相等
            return False
        if self.stoi != other.stoi: # 如果两个词表不相等
            return False
        if self.itos != other.itos: # 如果两个索引不相等
            return False
        if self.vectors != other.vectors: # 如果两个词向量不相等
            return False
        return True

    def __len__(self): 
        return len(self.itos) # 返回词表长度

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)} # 重新对词表进行索引

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos # 如果sort为True，那么对词表进行排序
        for w in words: # 遍历词表
            if w not in self.stoi: # 如果词表中没有该词
                self.itos.append(w) # 将该词加入到词表中
                self.stoi[w] = len(self.itos) - 1 # 将该词和对应的索引加入到词表中


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq) # 调用父类的__init__方法

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list: # 将句子转换为索引
        pass

    def from_seq(self, seq, join=False, with_pad=False): # 将索引转换为句子
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab': # 加载词表
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path): # 保存词表
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter() # 初始化Counter
        # tqdm是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
        for line in tqdm.tqdm(texts): # 遍历文本
            if isinstance(line, list): # 如果文本是列表
                words = line #  将文本赋值给words
            else:
                words = line.replace("\n", "").replace("\t", "").split() # 如果文本不是列表，那么将文本中的换行符和制表符替换为空格，然后按照空格进行切分

            for word in words:
                counter[word] += 1 # 统计词频
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False): # 将句子转换为索引
        if isinstance(sentence, str): # 如果句子是字符串
            sentence = sentence.split() # 将句子按照空格进行切分

        seq = [self.stoi.get(word, self.unk_index) for word in sentence] # 将句子中的词转换为索引

        if with_eos: # 如果with_eos为True
            seq += [self.eos_index]  # 将句子末尾加上eos索引
        if with_sos: # 如果with_sos为True
            seq = [self.sos_index] + seq # 将句子开头加上sos索引

        origin_seq_len = len(seq) # 获取句子长度

        if seq_len is None: # 如果seq_len为None
            pass
        elif len(seq) <= seq_len: # 如果句子长度小于等于seq_len
            seq += [self.pad_index for _ in range(seq_len - len(seq))] # 将句子末尾加上pad索引
        else:
            seq = seq[:seq_len] # 如果句子长度大于seq_len，那么将句子截断

        return (seq, origin_seq_len) if with_len else seq # 如果with_len为True，那么返回句子和句子长度，否则只返回句子

    def from_seq(self, seq, join=False, with_pad=False): # 将索引转换为句子
        words = [self.itos[idx] # 将索引转换为词
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words # 如果join为True，那么返回句子，否则返回词列表

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab': # 加载词表
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build():
    import argparse

    parser = argparse.ArgumentParser() # 初始化参数处理器
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)
