def count_n_grams(data, n, start_token = "<s>", end_token = "<e>") -> 'dict':

  # 创建n-gram的字典
  n_grams = {}
 
  # 遍历数据集中的句子
  for sentence in data:
        
    # 在句子前和后分别加上 start token 和 end token
    sentence = [start_token]*n + sentence + [end_token]
    
    # 将句子转化为 tuple
    sentence = tuple(sentence)

    # 存储句子的长度
    # 1-gram时特化处理
    m = len(sentence) if n==1 else len(sentence)-1
    
    # 遍历句子长度，生成N-Gram对且存储进字典中
    for i in range(m):
        
      # 生成n-gram对
      n_gram = sentence[i:i+n]
    
      # 请填充如下代码
      # 将当前生成的N-gram对加入到N-Grams 字典中
      # 如果已存在当前的N-Gram则对计数+1
      
        
  return n_grams

# 计算单词的概率
def prob_for_single_word(word, previous_n_gram, n_gram_counts, nplus1_gram_counts, vocabulary_size, k = 1.0) -> 'float':

  # 将上一个 n-gram对转化为 tuple 
  previous_n_gram = tuple(previous_n_gram)
    
  # 计算上一个 n-gram 的频数
  previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0
  
  # 计算分母
  denom = previous_n_gram_count + k * vocabulary_size

  # 将上一个n-gram与当前词组成一个 tuple
  nplus1_gram = previous_n_gram + (word,)

  # 填充如下代码
  # 计算当前词与上一个n-gram组成词对的频数
  

  # 计算分子
  num = nplus1_gram_count + k

  # Final Fraction
  prob = num / denom
  return prob

# 计算文本的概率分布
def probs(previous_n_gram, n_gram_counts, nplus1_gram_counts, vocabulary, k=1.0) -> 'dict':

  # 上一个n-gram对转化为tuple
  previous_n_gram = tuple(previous_n_gram)

  # 向子典集中加入 UNK 和 end token
  vocabulary = vocabulary + ["<e>", "<unk>"]

  # 计算字典的大小
  vocabulary_size = len(vocabulary)

  # 构建概率字典
  probabilities = {}

  # 遍历所有单词
  for word in vocabulary:
    
    # 计算概率
    probability = prob_for_single_word(word, previous_n_gram, 
                                           n_gram_counts, nplus1_gram_counts, 
                                           vocabulary_size, k=k)
    # 填充概率分布
    probabilities[word] = probability

  return probabilities