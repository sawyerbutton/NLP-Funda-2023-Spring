# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import # 为了在Python 2.x中启用Python 3.x的模块导入方式，使得在不同目录下的同名模块可以被正确导入
from __future__ import division # 为了在Python 2.x中启用Python 3.x的除法语义，即使得除法总是返回一个浮点数，如果要得到整除结果，使用"//"即可
from __future__ import print_function # 为了在Python 2.x中启用Python 3.x的打印函数，使得可以使用print()函数来打印输出，而不是print语句,print语句太过于反直觉

import collections
import re
import unicodedata # unicodedata模块提供了对Unicode字符数据库的访问
import six #兼容python2和python3
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """
  init_checkpoint 是模型文件路径
  用于在加载预训练BERT模型的时候
  检查用户传入的大小写配置是否与预训练模型的实际配置一致。
  其功能主要基于BERT模型有大小写敏感（cased）和大小写不敏感（uncased）两种情况，
  并且这个配置并没有被存储在模型配置文件中，
  所以需要一种方式来检查并保证预训练模型和用户指定的大小写配置是否一致。
  如果检查出现错误，那么将抛出ValueError异常
  """

  # 如果没有初始化模型路径，则不需要进行检查，直接返回
  if not init_checkpoint:
    return

  # 尝试从模型文件路径中匹配出模型名称，如果匹配不出则直接返回
  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  if m is None:
    return

  # 获取模型名称
  model_name = m.group(1)

  # 定义已知的不区分大小写的模型名称
  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  # 定义已知的区分大小写的模型名称
  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  # 初始化一个变量用于标记配置是否有误
  is_bad_config = False
  
  # 如果模型是不区分大小写的，但用户传入的配置为区分大小写，则标记配置有误
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  # 如果模型是区分大小写的，但用户传入的配置为不区分大小写，则标记配置有误
  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  # 如果配置有误，则抛出ValueError异常
  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))



def convert_to_unicode(text):
  """
  将输入的文本（text）转换为Unicode字符串，这样可以保证后续处理的一致性
  因为Unicode是全球字符集编码，可以支持几乎所有语言的字符
  函数会检查输入的文本是否已经是Unicode，如果不是，就尝试以utf-8编码将其转换为Unicode
  """
  if six.PY3:
    # 如果是 Python 3.x 版本
    if isinstance(text, str):
      # 如果text已经是字符串，直接返回
      return text
    elif isinstance(text, bytes):
      # 如果text是字节串，将其解码为字符串
      return text.decode("utf-8", "ignore")
    else:
      # 如果text既不是字符串也不是字节串，抛出异常
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    # 如果是 Python 2.x 版本
    if isinstance(text, str):
      # 如果text是字节串，将其解码为字符串
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      # 如果text已经是unicode，直接返回
      return text
    else:
      # 如果text既不是字符串也不是字节串，抛出异常
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    # 如果Python版本既不是2.x也不是3.x，抛出异常
    raise ValueError("Not running on Python2 or Python 3?")



def printable_text(text):
  """
  将输入的文本（text）转换为适合打印或者用于tf.logging的格式
  函数会检查输入的文本的类型，根据Python版本的不同进行适当的处理
  如果输入的文本不是字符串或者字节串，会抛出一个异常
  """

  # 对于Python 2和Python 3，这些函数都需要 `str` 类型，
  # 但在一个情况下它是一个 Unicode 字符串，在另一个情况下它是一个字节串。
  if six.PY3:
    # 如果是 Python 3.x 版本
    if isinstance(text, str):
      # 如果text已经是字符串，直接返回
      return text
    elif isinstance(text, bytes):
      # 如果text是字节串，将其解码为字符串
      return text.decode("utf-8", "ignore")
    else:
      # 如果text既不是字符串也不是字节串，抛出异常
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    # 如果是 Python 2.x 版本
    if isinstance(text, str):
      # 如果text是字节串，直接返回
      return text
    elif isinstance(text, unicode):
      # 如果text是unicode，将其编码为字节串
      return text.encode("utf-8")
    else:
      # 如果text既不是字符串也不是字节串，抛出异常
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    # 如果Python版本既不是2.x也不是3.x，抛出异常
    raise ValueError("Not running on Python2 or Python 3?")



def load_vocab(vocab_file):
  """
  从词汇文件中加载词汇到一个字典中
  其中每个词汇（token）对应一个索引值（index）
  这样就可以通过词汇快速找到其对应的索引
  也可以通过索引找到对应的词汇
  """
  # 创建一个有序字典
  vocab = collections.OrderedDict()
  index = 0
  # 用tensorflow的gfile打开词汇文件
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      # 读取一行，将其转换为unicode字符串
      token = convert_to_unicode(reader.readline())
      # 如果读到文件末尾，结束循环
      if not token:
        break
      # 删除字符串两端的空白符
      token = token.strip()
      # 将token和其对应的索引添加到字典中
      vocab[token] = index
      # 索引加1
      index += 1
  # 返回词汇字典
  return vocab



def convert_by_vocab(vocab, items):
  """
  将输入的items（可以是一个字符串，也可以是一个字符串列表）转化成对应的索引列表
  """
  # 初始化一个空列表
  output = []
  # 遍历 items 中的每一个元素
  for item in items:
    # 在vocab中查找item对应的值，然后添加到output列表中
    output.append(vocab[item])
  # 返回转换后的列表
  return output


def convert_tokens_to_ids(vocab, tokens):
  """
  将 tokens 转换为 ids
  也可以理解成基于tokens找ids
  """
  # 调用 convert_by_vocab 函数进行转换
  return convert_by_vocab(vocab, tokens)

def convert_ids_to_tokens(inv_vocab, ids):
  """
  将 ids 转换为 tokens
  也可以理解成基于ids找tokens
  """
  # 调用 convert_by_vocab 函数进行转换
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """
  对一段文本进行基本的空白字符清理和分割，最基本的文本预处理
  将一个完整的句子分割为一个个单词
  """
  # 删除字符串两端的空白符
  text = text.strip()
  # 如果清理后的字符串为空，直接返回空列表
  if not text:
    return []
  # 使用split()函数将字符串分割为列表，split()默认以空白字符（包括空格、换行、制表符等）为分割符
  tokens = text.split()
  # 返回分割后的词列表
  return tokens



class FullTokenizer(object):
  """
  进行端到端的分词
  需要先了解BasicTokenizer和WordpieceTokenizer这两个类的作用
  """

  def __init__(self, vocab_file, do_lower_case=True):
    """初始化 FullTokenizer 类"""
    # 加载词汇表
    self.vocab = load_vocab(vocab_file)
    # 创建词汇表的逆映射，即 {id: token}
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    # 创建一个基础分词器
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    # 创建一个词片分词器
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    """将文本进行分词"""
    split_tokens = []
    # 首先进行基础分词
    for token in self.basic_tokenizer.tokenize(text):
      # 对每个基础词进行词片分词，将整词拆解为subwords
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    """将分词结果转换为id"""
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    """将id转换为分词结果，找逆映射"""
    return convert_by_vocab(self.inv_vocab, ids)



class BasicTokenizer(object):
  """
  负责基础的文本分词，包括标点符号分割、转小写等
  对英文文本进行处理时，会将文本转换为小写，对标点符号进行分割
  对中文文本处理时，会为每个中文字符的前后添加空格，然后进行分词
  """

  def __init__(self, do_lower_case=True):
    """构造一个 BasicTokenizer类"""
    # 基于输入参数决定是否将输入转换为小写
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """对一段文本进行分词。"""
    # 首先将文本转换为unicode
    text = convert_to_unicode(text)
    # 清理文本中的无效字符和多余的空格
    text = self._clean_text(text)
    # 对中文字符进行分词
    text = self._tokenize_chinese_chars(text)
    # 使用空格进行分词
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      # 如果设置为转小写，就将token转换为小写
      if self.do_lower_case:
        token = token.lower()
        # 去除token中的重音符号
        token = self._run_strip_accents(token)
      # 根据标点符号进行分词
      split_tokens.extend(self._run_split_on_punc(token))

    # 重新使用空格进行分词，确保分词结果中的每个token都没有前后空格
    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """去除一段文本中的重音符号"""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn": # Mn表示重音符号
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """根据标点符号对一段文本进行分词"""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """为任何中文字符周围添加空格"""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """检查 CP 是否是 CJK 字符的码点"""
    # 这里定义的“中文字符”是指在 CJK Unicode 块中的字符
    # 参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # 注意，CJK Unicode 块不包括所有日本和韩国字符
    # 尽管它的名称是这样。现代韩语的 Hangul 字母是一个不同的块，
    # 日语的平假名和片假名也是如此。这些字母用于书写空格分隔的单词，
    # 因此它们不被特殊处理，像其他所有语言一样处理
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """对一段文本执行无效字符移除和空格清理"""
    output = []
    for char in text:
      cp = ord(char) # ord()函数返回对应的 ASCII 数值，或者 Unicode 数值
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """
  执行 WordPiece 分词
    Args:
      vocab: 一个字典，用于进行 WordPiece 分词的词汇表。
      unk_token: 表示未知令牌（也就是不在词汇表中的词）的字符串
      max_input_chars_per_word: 每个词的最大字符数，如果一个词的字符数超过了这个值，那么这个词将被视为未知令牌，中文场景中不存在这样的案例
  """

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """
    将一段文本分词成其 WordPiece
    这使用一个贪婪的最长匹配优先算法来使用给定的词汇表进行分词
    例如：
      输入 = "unaffable"
      输出 = ["un", "##aff", "##able"]
    Args:
      text: 一个单独的token或由空格分隔的token
      这个text已经通过 `BasicTokenizer` 进行过处理
    Returns:
      一个 WordPiece token的列表
    """

    text = convert_to_unicode(text) #将文本转换为unicode编码

    output_tokens = []
    for token in whitespace_tokenize(text): #对文本进行分词
      chars = list(token) #将分词后的词转换为字符列表，比如英文单词"unaffable"转换为["u","n","a","f","f","a","b","l","e"]
      if len(chars) > self.max_input_chars_per_word: #如果分词后的词长度大于最大字符数，则将该词视为未知token
        output_tokens.append(self.unk_token) #将该词视为未知token放入输出并跳过本轮次循环
        continue

      is_bad = False
      start = 0
      sub_tokens = [] #初始化数组用于存放分词后的词
      # 举例来说，对于输入“unaffable”，如果词汇表中有“un”，“aff”，“able”，则分词后的词为["un","aff","able"]
      # 在第一轮循环中，start=0，end=9，cur_substr=None，substr="unaffable"，由于substr不在词汇表中，所以end减1，即end=8
      # 在第二轮循环中，start=0，end=8，cur_substr=None，substr="unaffabl"，由于substr不在词汇表中，所以end减1，即end=7
      # 在第三轮循环中，start=0，end=7，cur_substr=None，substr="unaffab"，由于substr不在词汇表中，所以end减1，即end=6
      # 在第四轮循环中，start=0，end=6，cur_substr=None，substr="unaffa"，由于substr不在词汇表中，所以end减1，即end=5
      # 在第五轮循环中，start=0，end=5，cur_substr=None，substr="unaff"，由于substr不在词汇表中，所以end减1，即end=4
      # 在第六轮循环中，start=0，end=4，cur_substr=None，substr="unaf"，由于substr不在词汇表中，所以end减1，即end=3
      # 在第七轮循环中，start=0，end=3，cur_substr=None，substr="una"，由于substr不在词汇表中，所以end减1，即end=2
      # 在第八轮循环中，start=0，end=2，cur_substr=None，substr="un"，由于substr在词汇表中，所以cur_substr="un"，并跳出内部循环
      # 在第九轮循环中，start=2，end=9，cur_substr="un"，substr="affable"，由于substr不在词汇表中，所以end减1，即end=8
      # 在第十轮循环中，start=2，end=8，cur_substr="un"，substr="affabl"，由于substr不在词汇表中，所以end减1，即end=7
      # 在第十一轮循环中，start=2，end=7，cur_substr="un"，substr="affab"，由于substr不在词汇表中，所以end减1，即end=6
      # 在第十二轮循环中，start=2，end=6，cur_substr="un"，substr="affa"，由于substr不在词汇表中，所以end减1，即end=5
      # 在第十三轮循环中，start=2，end=5，cur_substr="un"，substr="aff"，由于substr在词汇表中，所以cur_substr="aff"，并跳出内部循环
      # 在第十四轮循环中，start=5，end=9，cur_substr="aff"，substr="able"，由于substr在词汇表中，所以cur_substr="able"，并跳出内部循环
      # start的值等于上一轮循环中end的值，即start=9，由于start等于len(chars)，所以跳出外部循环
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end]) #将字符列表转换为一部分字符串，会用这个字符串去字典里找对应
          if start > 0:
            substr = "##" + substr #如果不是第一个词，则在词前加上"##"，表示该词是一个词的一部分
          if substr in self.vocab: #如果分词后的词在词汇表中
            cur_substr = substr #将该词放入输出
            break  # 如果分词后的词在词汇表中，则将该词放入输出并跳出内部循环
          end -= 1
        if cur_substr is None: #如果分词后的词不在词汇表中
          is_bad = True #将is_bad置为True
          break #如果分词后的词不在词汇表中，则将该词视为未知token放入输出并跳过本轮次外部循环
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token) #将该词视为未知token放入输出
      else:
        output_tokens.extend(sub_tokens) #将分词后的词放入输出
    return output_tokens


def _is_whitespace(char):
    """
    检查`char`是否为空白字符
    '\t', '\n', 和 '\r'在技术上是控制字符，但我们将它们视为空白字符
    因为它们通常被视为这类字符
    Args:
      char: 需要检查的字符
    Returns:
      如果字符是空白字符则返回True，否则返回False。
    """
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs": # Zs:空格分隔符
        return True
    return False



def _is_control(char):
    """
    检查`char`是否为控制字符
    '\t', '\n', 和 '\r'在技术上是控制字符，但我们将它们视为空白字符
    Args:
      char: 需要检查的字符
    Returns:
      如果字符是控制字符则返回True，否则返回False
    """
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"): # Cc:其他控制字符， Cf:格式控制字符
        return True
    return False



def _is_punctuation(char):
    """
    检查`char`是否为标点符号
    我们将所有非字母/数字的ASCII字符视为标点符号。像 "^", "$", 和 "`" 这样的字符虽然不在Unicode的标点类别中，
    但是我们仍将其视为标点符号，以保持一致性
    Args:
      char: 需要检查的字符
    Returns:
      如果字符是标点符号则返回True，否则返回False
    """
    cp = ord(char) # ord()函数返回对应的 ASCII 数值，或者 Unicode 数值
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char) # category()函数返回字符在 Unicode 中的类别
    if cat.startswith("P"): # P:标点字符
        return True
    return False

