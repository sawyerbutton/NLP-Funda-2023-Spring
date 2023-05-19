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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """
  表示简单序列分类任务的单个训练/测试示例的类
  属性：
  guid：示例的唯一ID
  text_a：第一个序列的未分词文本。对于单序列任务，只需指定这个序列
  text_b：（可选）第二个序列的未分词文本。只有在序列对任务中才需要指定这个序列
  label：（可选）示例的标签。这应该对训练和开发示例指定，但对于测试示例不应指定
  """
  def __init__(self, guid, text_a, text_b=None, label=None):
    """
    构造一个 InputExample 实例
    参数：
    guid：示例的唯一id。
    text_a：字符串。第一个序列的未分词文本。对于单序列任务，只需指定此序列。
    text_b：（可选）字符串。第二个序列的未分词文本。只在序列对任务中需要指定。
    label：（可选）字符串。示例的标签。对于训练和开发示例，应指定此选项，但对于测试示例，不应指定。
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label



class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """
  表示数据单一特征集的类
  属性：
  input_ids：输入序列的ID列表
  input_mask：掩码列表，标记哪些输入ID是真实输入，哪些是填充
  segment_ids：标记输入序列中的各个部分的列表,对于单序列任务，只需指定这个序列,对于NSP任务，需要分开指定序列对的标记
  label_id：标签的ID
  is_real_example：布尔值，标记此特征是否来自真实的样本
  """

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    """
    构造一个 InputFeatures 实例

    参数：
    input_ids：输入序列的ID列表
    input_mask：掩码列表，标记哪些输入ID是真实输入，哪些是填充
    segment_ids：标记输入序列中的各个部分的列表
    label_id：标签的ID
    is_real_example：布尔值，标记此特征是否来自真实的样本，默认为 True
    """
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example



class DataProcessor(object):
  """
  用于序列分类数据集的数据转换器的基类
  注意是基类，代表着需要继承该类，然后实现其中的方法
  """

  def get_train_examples(self, data_dir):
    """
    为训练集获取一组 `InputExample`
    参数:
    data_dir: 数据的目录路径
    抛出:
    NotImplementedError: 这是一个抽象方法，需要在子类中实现
    """
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """
    为验证集获取一组 `InputExample`
    参数:
    data_dir: 数据的目录路径
    抛出:
    NotImplementedError: 这是一个抽象方法，需要在子类中实现
    """
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """
    为测试集获取一组 `InputExample`
    参数:
    data_dir: 数据的目录路径
    抛出:
    NotImplementedError: 这是一个抽象方法，需要在子类中实现
    """
    raise NotImplementedError()

  def get_labels(self):
    """
    获取此数据集的标签列表
    抛出:
    NotImplementedError: 这是一个抽象方法，需要在子类中实现
    """
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """
    读取一个制表符分隔的值文件
    参数:
    input_file: 输入文件的路径
    quotechar: 用于包装字段的字符
    返回:
    文件的行列表，每行是一个字段列表
    """
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines



class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

# 处理MRPC数据集（GLUE版本）的数据转换器，继承自DataProcessor类
# MRPC (Microsoft Research Paraphrase Corpus)是一个用于文本匹配任务的数据集
# 该数据集包含了5,801对句子，其中每一对句子都被标注为是“相似”或“不相似”
# 每一行数据都以制表符分隔，而不是空格
# 每一行数据都由四个字段组成，以制表符分隔。这四个字段分别是
# 序号、第一个句子、第二个句子、标签
# 序号（index）：每一行数据的唯一标识符
# 第一个句子（#1 String）：第一个句子
# 第二个句子（#2 String）：第二个句子
# 标签（Quality）：标签，取值为“1”或“0”，分别表示“相似”和“不相似”
# 如：
'''
index	sentence1	sentence2	label
1	The bird is bathing in the sink.	Birdie is washing itself in the water basin.	1
2	In May 2010, the troops attempted to invade Kabul.	The US army invaded Kabul on May 7th, 2010.	0
3	John said he is considered a witness but not a suspect.	"He is not a suspect anymore." John said.	1
...	...	...	...
'''

class MrpcProcessor(DataProcessor):
  """
  用于处理MRPC数据集（GLUE版本）的数据转换器
  """

  def get_train_examples(self, data_dir):
    """
    通过读取训练集tsv文件并创建训练样例
    参数:
    data_dir: 数据的目录路径
    返回:
    一个InputExample的列表，每个InputExample包含一个训练样例
    """
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """
    通过读取验证集tsv文件并创建验证样例
    参数:
    data_dir: 数据的目录路径
    返回:
    一个InputExample的列表，每个InputExample包含一个验证样例。
    """
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """
    通过读取测试集tsv文件并创建测试样例
    参数:
    data_dir: 数据的目录路径
    返回:
    一个InputExample的列表，每个InputExample包含一个测试样例
    """
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """
    返回:
    此数据集的标签列表，这里是["0", "1"]
    """
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """
    为训练集和验证集创建样例
    参数:
    lines: 文件的行列表，每行是一个字段列表。
    set_type: 数据集的类型（"train"，"dev"，"test"）。
    返回:
    一个InputExample的列表，每个InputExample包含一个样例
    """
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3]) # 第一个句子
      text_b = tokenization.convert_to_unicode(line[4]) # 第二个句子
      if set_type == "test": # 测试集的标签为"0", 待预测结过进行修改
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0]) # line[0]在数据集中是label，因此是直接将正确的label进行赋值操作
        # 这里要注意下你使用的MRPC的数据的格式，如果是自己的数据集，那么这里的label的位置可能不是line[0]，而是line[1]或者其他位置
        # 需要对此进行调整
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples



class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  '''
  将输入的InputExample对象转换成InputFeatures对象
  InputExample对象是一个基本的数据样本，包括一个或两个句子及其关联的标签
  InputFeatures对象则是为模型准备的输入特性，包括输入的句子（以其标记ID表示）、输入掩码、段落ID以及标签ID
  '''

  # 检查输入的example是否是PaddingInputExample的实例
  # 如果是，返回一个填充的InputFeatures对象，其中所有的特性都是0
  # paddingInput 代表其是一个填充的样本，不具备任何意义
  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)
  
  # 创建一个标签映射（label_map），将所有可能的标签映射到唯一的整数ID
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  # 将输入的句子（text_a和text_b）分词为（tokens_a和tokens_b）
  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None # 这里是因为某些数据集中只有一个句子，因此tokens_b为None
  if example.text_b: # 如果有第二个句子，那么进行分词
    tokens_b = tokenizer.tokenize(example.text_b)
  # 根据最大序列长度对分词结果进行截断或填充
  # 在Bert中NSP任务需要增加三个特殊标签 CLS SEP SEP, 因此需要将最大长度减去3
  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Bert模型非NSP任务中只需要增加两个特殊标签 CLS SEP, 因此需要将最大长度减去2
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)] # 长度超过了最大长度，进行截断

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  # 基于Bert的模型输入特性
  tokens = []
  segment_ids = []
  tokens.append("[CLS]") # 无论是NSP任务还是非NSP任务，都需要在句首增加[CLS]标签
  segment_ids.append(0) # [CLS]标签的segment_id为0
  for token in tokens_a: # 对于句子1，其segment_id也为0
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]") # 句子1的结尾增加[SEP]标签
  segment_ids.append(0) # [SEP]标签的segment_id为0

  if tokens_b: # 如果有句子2，那么其segment_id为1，以下同理
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens) # 将分词结果转换为对应的ID

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids) # 对于真实的token，其input_mask为1，否则为0

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length: # 对于长度不足的序列，进行填充操作。填充补充为0
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length # 确保输入的序列长度是最大长度
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length 

  label_id = label_map[example.label] # 将标签转换为对应的ID
  if ex_index < 5: # 打印范例样本的信息
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures( # 构建InputFeatures对象
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """
  将一组InputExample对象转换成一个TFRecord文件
  InputExample对象是一个基本的数据样本，包括一个或两个句子及其关联的标签
  TFRecord文件是一种用于存储大量数据的二进制文件，可以高效地读取和处理
  """
  # 创建一个TFRecordWriter，用于向输出文件写入数据
  writer = tf.python_io.TFRecordWriter(output_file)

  # 遍历输入的examples
  # 对于每个InputExample，它首先调用convert_single_example函数，将InputExample转换为InputFeatures对象
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0: # 每处理10000个样本，打印一次日志
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)
    # 定义一个函数create_int_feature
    # 函数将一个整数列表转换为tf.train.Feature对象，这是TensorFlow用于在TFRecord文件中表示特性的标准格式
    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    # 创建一个有序字典features
    # 其中包含五个键值对：input_ids，input_mask，segment_ids，label_ids和is_real_example
    # 每个键的值都是一个通过调用create_int_feature函数得到的tf.train.Feature对象
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])
    # 使用这个features字典创建一个tf.train.Example对象
    # 然后将其序列化为一个字符串，并写入TFRecord文件
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """
  创建一个input_fn函数
  这个函数在TPUEstimator（Tensor Processing Unit Estimator，一个用于运行在TPUs上的TensorFlow模型的Estimator）中被用来读取输入数据
  """
  # 定义一个名为name_to_features的字典，这个字典中的键值对定义了数据集中每个样本应包含的特性及其类型
  # 在这个例子中，一个样本应包含五个特性："input_ids"，"input_mask"，"segment_ids"，"label_ids"和"is_real_example"
  # 它们的类型都是定长的整数向量（除了"label_ids"和"is_real_example"是标量）
  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }
  # 该函数将一个TFRecord文件中的一个记录解码为一个tf.Example对象
  # 该函数会将所有的整数类型从int64转换为int32，因为TPU只支持int32类型
  def _decode_record(record, name_to_features):
    example = tf.parse_single_example(record, name_to_features) # 解析一个样本

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  # 定义input_fn函数，这个函数是被返回的函数
  # 这个函数接受一个参数params，该参数是一个字典，包含一些控制数据输入的参数（例如batch_size）
  # 该函数从TFRecord文件中读取数据，将其转换为适合模型使用的形式
  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file) # 使用tf.data.TFRecordDataset创建一个数据集对象，该对象会从TFRecord文件中读取数据
    if is_training: # 如果is_training参数为True，那么数据集将被无限重复，而且在每个epoch后都会被随机打乱
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    # 使用tf.contrib.data.map_and_batch函数，将_decode_record函数应用到数据集中的每个元素上，并且将结果按批次组织
    # 如果drop_remainder参数为True，那么最后一个批次可能被舍弃，以保证所有批次的大小都是batch_size
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """用于在其组合长度超过给定的最大长度时截断一对token序列"""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  # 不断检查两个序列的总长度（即 tokens_a 和 tokens_b 的长度之和）
  # 如果总长度小于或等于 max_length，它会退出循环并停止截断
  # 如果总长度大于 max_length，它会从 tokens_a 和 tokens_b 中较长的序列末尾弹出（移除）一个token
  # 如果两个序列长度相等，则会从 tokens_b 中移除一个token
  # 这样做的目的是确保处理的序列长度不超过模型能处理的最大长度，且在确保长度满足要求的情况下尽可能地保留每个序列的信息
  # 之所以会选择更长的那个进行截取，是因为更短的seq可能具备更大的信息浓度，更值得被保留，直觉
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """创建一个文本分类模型"""
  # 初始化一个 BertModel 对象，将输入的数据（input_ids, input_mask, segment_ids）传递给这个模型
  # 这一步将把文本转换为 BERT 能理解的形式，即把文本转换为对应的嵌入向量
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  # 从 BertModel 对象中获取经过预训练的 BERT 模型输出
  # 在这个案例中，我们只关注模型的汇总输出（model.get_pooled_output()），也就是cls向量
  # 该输出对应于每个输入序列的整体语义信息
  # 如果想获取每个输入token的输出，应该使用 model.get_sequence_output() 方法，该方法返回一个形状为 [batch_size, seq_length, hidden_size] 的张量
  output_layer = model.get_pooled_output()
  
  hidden_size = output_layer.shape[-1].value # 获取输出的维度，这个维度大小是固定的，等于配置文件中的 hidden_size 参数

  # 定义输出层的权重（output_weights）和偏置项（output_bias）
  # 输出层的维度是 num_labels x hidden_size，其中 hidden_size 是 BERT 输出的维度，num_labels 是分类任务的类别数
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02)) # 使用截断正态分布初始化权重

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer()) # 使用0初始化偏置项

  with tf.variable_scope("loss"): 
    if is_training: # 如果当前是在训练模式下，对模型的输出使用 dropout，以防止过拟合
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    # 计算模型的 logits（也就是模型对每个类别的原始预测值）
    # 这个值通过将 BERT 的输出和 output_weights 进行矩阵乘法并加上 output_bias 而得到
    logits = tf.matmul(output_layer, output_weights, transpose_b=True) # 矩阵乘法
    logits = tf.nn.bias_add(logits, output_bias) # 加上偏置项
    probabilities = tf.nn.softmax(logits, axis=-1) # 使用 softmax 函数将 logits 转换为 probabilities
    log_probs = tf.nn.log_softmax(logits, axis=-1) #  使用 log_softmax 函数将 logits 转换为 log_probs

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32) # 将输入的label转换为one-hot向量
    # 交叉熵损失函数，将模型预测值log_probs和真实值one_hot_labels作为输入
    # 交叉熵损失函数是用来估量预测值和真实值之间的差距，交叉熵值越大，两者之间的差距越大，模型的性能也就越差
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) # 计算每个样本的交叉熵损失
    loss = tf.reduce_mean(per_example_loss) # 计算平均损失

    return (loss, per_example_loss, logits, probabilities) # 返回损失值，每个样本的损失值，logits和probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """
  构建 TPUEstimator 所需的模型函数 model_fn
  bert_config: BERT模型的配置信息；
  num_labels: 标签的数量；
  init_checkpoint: 初始化模型时的检查点路径；
  learning_rate: 学习率；
  num_train_steps: 训练步数；
  num_warmup_steps: 预热步数；
  use_tpu: 是否使用 TPU；
  use_one_hot_embeddings: 是否使用 one-hot 编码的词嵌入
  """

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """
    features: 一个包含输入特性的字典；
    labels: 一个包含标签的 Tensor；
    mode: 表示当前是否为训练、评估或预测模式的符号；
    params: 包含超参数的字典
    首先通过字典键将特性提取到各自的变量中
    然后调用 create_model 函数创建模型并获取损失和预测结果
    然后，如果提供了初始化检查点，将用其初始化模型的变量
    """

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()): # 打印特征信息
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features: # 如果提供了 is_real_example 特征，那么它就是一个布尔值的 Tensor
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else: # 否则就创建一个值为 1.0 的 Tensor，其长度与 label_ids 相同
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN) # 判断当前是否为训练模式

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings) # 调用 create_model 函数创建模型并获取损失和预测结果

    tvars = tf.trainable_variables() # 获取模型中所有的可训练变量
    initialized_variable_names = {} # 保存初始化变量的名字和值的字典
    scaffold_fn = None # 用于加载模型的 Scaffolding
    if init_checkpoint: # 如果提供了初始化检查点
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu: # 如果使用 TPU

        def tpu_scaffold(): # 创建一个 Scaffolding
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map) # 从检查点初始化模型变量
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map) # 从检查点初始化模型变量

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names: # 如果变量在初始化变量的名字和值的字典中
        init_string = ", *INIT_FROM_CKPT*" # 将 init_string 设置为 *INIT_FROM_CKPT*
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN: # 如果当前为训练模式
      # 创建一个优化器，并返回一个 TPUEstimatorSpec，该对象指示 TPU Estimator 应该进行训练操作
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec( # 创建一个 TPUEstimatorSpec
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL: # 如果当前为评估模式
      # 定义评估指标（如准确度和损失），并返回一个 TPUEstimatorSpec
      # 该对象指示 TPU Estimator 应该进行评估操作
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) # 获取预测结果
        accuracy = tf.metrics.accuracy( # 计算准确度
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example) # 计算损失
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example]) # 定义评估指标
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else: # 如果当前为预测模式
      # 返回一个 TPUEstimatorSpec，该对象指示 TPU Estimator 应该进行预测操作
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


'''
该函数的目的是创建一个可以传递给TPUEstimator的input_fn闭包
这个闭包主要用于指定如何从features生成用于模型训练或者预测的数据
'''
# 这个函数在这个文件中没有被使用，但仍被Colab和依赖它的人使用
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """创建一个可以传递给TPUEstimator的'input_fn'闭包"""

  # 初始化几个空的列表，用于存放所有输入数据
  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  # 遍历所有特征，将每个特征的各个部分添加到相应的列表中
  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """实际的输入函数"""
    # 从参数中获取批处理大小
    batch_size = params["batch_size"]

    # 获取样本数量
    num_examples = len(features)

    # 创建一个tf.data.Dataset对象，这是用于创建输入数据的首选方式
    # 对于大规模数据集，通常不使用Dataset.from_generator()，因为这个函数使用了tf.py_func，而这个函数并不兼容TPU
    # 更推荐的方式是使用TFRecordReader
    d = tf.data.Dataset.from_tensor_slices({ # 创建一个 Dataset
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    # 如果是在训练模式下，数据需要进行无限次的重复，并且需要进行随机打乱
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    # 将数据进行批处理，如果drop_remainder为True，那么会丢弃掉最后一个不完整的批次
    # 最后一个批次的大小可能会小于batch_size
    # 这对于TPU来说是必须的，因为TPU需要所有的批次都具有相同的大小
    # 对于训练和验证，批处理大小都是固定的，但是对于预测，批处理大小是可变的
    # 为了支持这一点，我们使用了 tf.contrib.tpu.InputPipelineConfig.per_host_input_for_training 来动态设置批处理大小
    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  # 返回内部定义的input_fn函数，这个函数在被调用时会返回一个用于模型输入的tf.data.Dataset对象
  return input_fn

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """用于将一组 InputExample 转换为一组 InputFeatures
  examples：包含输入示例的列表；
  label_list：标签列表；
  max_seq_length：最大序列长度；
  tokenizer：用于对文本进行标记化的分词器
  """

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0: # 每 10000 个样本输出一次日志
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer) # 将单个输入示例转换为单个输入特征

    features.append(feature)  # 将输入特征添加到输入特征列表
  return features


def main(_):
  # 设置TensorFlow日志的详细程度为INFO
  tf.logging.set_verbosity(tf.logging.INFO)

  # # 定义一个字典，保存各个任务的处理器
  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
  }

  # 验证预训练BERT模型的大小写设置是否和我们当前的设置一致
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # 至少需要选择一个任务，训练、评估或预测
  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  # # 从预训练BERT模型的配置文件中读取模型配置
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  # 确保设定的最大序列长度不超过预训练BERT模型的最大序列长度
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  # 创建输出目录，如果目录不存在
  tf.gfile.MakeDirs(FLAGS.output_dir)

  # 获取任务名，并将其转换为小写
  task_name = FLAGS.task_name.lower()

  # 如果任务名不存在于已定义的任务列表中，那么抛出错误
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  # 获取任务处理器
  processor = processors[task_name]()

  # 获取任务的标签列表
  label_list = processor.get_labels()

  # 创建一个全词库的分词器，用于文本的分词
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # 如果使用了TPU，并且设置了TPU名称，那么创建一个TPU解析器
  # 世界线分支，这里可以选择TPU或是GPU
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  # 创建一个TPU的运行配置
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  # 初始化训练示例、训练步数和预热步数
  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    # 获取训练示例
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    # 计算训练步数和预热步数
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  # 创建模型函数
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  # 创建一个TPU估计器，如果没有TPU，就会回滚到CPU或GPU的正常估计器
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    # 创建一个训练文件的路径
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    # 将训练示例转换为tf.record格式并保存
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    # 输出训练信息
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    # 创建输入函数
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    # 使用TPU估计器进行训练
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    # 获取评估示例
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU需要固定的批处理大小，因此我们需要添加一些虚假的示例以使示例数量可以被批处理大小整除
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())
    # 创建一个评估文件的路径
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    # 将评估示例转换为tf.record格式并保存
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    # 输出评估信息
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu: # 如果使用TPU，需要指定步数
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False # 如果使用TPU，需要设置eval_drop_remainder为True
    # 创建输入函数
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)
    # 使用TPU估计器进行评估
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    # 保存评估结果
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    # 获取预测示例
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples) # 获取预测示例的数量
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())
    # 创建一个预测文件的路径
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    # 将预测示例转换为tf.record格式并保存
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    # 创建输入函数
    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn) # 使用TPU估计器进行预测

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv") # 创建预测结果文件的路径
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples: # 如果预测的数量大于预测示例的数量，则跳出循环
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples # 断言预测的数量等于预测示例的数量


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
