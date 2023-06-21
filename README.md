# NLP-Funda-2023-Spring
2023 Sping NLP Fundamental Course
2023 春季 NLP 基础课程

Course Hour：Every Weekend 16:00pm-17:30pm

## Syllabus

1. NLP基础概念及评价标准
    - 语言模型及NLP的概念
    - NLP的度量指标
    - 语言模型生成文本的思路

2. 基于统计的NLP模型之 N-Gram
    - N-Gram概念
    - N-Gram 平滑优化

3. 基于统计的NLP模型之 概率图模型
    - Markov Model 马尔科夫模型
    - HMM 隐藏马尔科夫模型
    - MEMM 最大熵马尔科夫模型
    - CRF 条件随机场模型

4. NLP之文本预处理技术
    - 文本分词
    - 停用词过滤
    - 拼写纠错
    - 词性还原
    - 词干提取

5. NLP之 基于统计的文本表示技术
    - One Hot 表示
    - 词袋模型表示
    - TF-IDF模型表示

6. NLP之 基于神经网络的文本表示技术
    - Word2Vect概念
    - C-BOw模型
    - Skip-Gram 模型
    - 负采样与层次Softmax

7. NLP之 基于统计和神经网络的文本表示技术
    - 共现矩阵的概念
    - Glove 模型概念

8. NLP之 神经网络模型RNN
    - RNN 模型
    - RNN 的缺陷：梯度爆炸，梯度消失

9. NLP之 先进RNN模型
    - LSTM 模型
    - GRU 模型
    - 双向RNN模型
    - 多层RNN模型

10. NLP之 Seq2Seq模型
    - Encoder-Decoder 模型
    - Seq2Seq的优化演进

11. NLP之 Transformer模型
    - Transformer模型概述
    - Word Embedding 与 Positional Encoding
    - Encoder 编码器详解
    - Decoder 解码器详解

12. NLP之 Bert模型
    - Bert模型概述
    - Bert的预训练任务

13. NLP之 GPT模型
    - GPT模型概述
    - GPT1.0模型
    - GPT2.0模型
    - GPT3.0模型
    - Instruct GPT

14. NLP之 Prompt Learning
    - 预训练模型概述
    - Pre-Training 和 Fine Tunning 范式
    - Prompt 范式


## Code Samples

1. [文本生成的演进](code/NLP基础课第一节-文本生成.ipynb)
2. [N-gram文本生成](code/NLP%E5%9F%BA%E7%A1%80%E8%AF%BE%E7%AC%AC%E4%BA%8C%E8%8A%82-N_Gram%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E5%8F%8A%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90.ipynb)
3. [补充内容-使用Python创建一个NLP任务的Vocabulary](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/%E4%BD%BF%E7%94%A8Python%E6%90%AD%E5%BB%BAVocabulary.ipynb)
4. [文本预处理基础案例-推特客服服务信息](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/%E6%8E%A8%E7%89%B9%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86.ipynb)
5. [RNN_From_Scratch](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/RNN_From_Scratch_1.ipynb)
6. [GLove_implementation](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/Glove_Implementation.ipynb)
7. [ChatGPT-Conversation2Image](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/gpt-image.js)
8. [Transformer_Source_Code](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/Transformer_Source_Code.ipynb)
9. [GPT-API-Easy-To-Use](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/gpt_ez.ipynb)
10. [Bert1](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/Bert_Demo_1.ipynb)
11. [Bert2](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/Bert_Demo_2.ipynb)
12. [Bert3](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/Bert_Demo_3.ipynb)
13. [Annotated-Bert-Tokenization-Sourcecode](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/Bert-Tokenization-Annotated.py)
14. [Annotated-Bert-Run-Classifier-Sourcecode](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/Bert-Run-Classifier-Annotated.py)
15. [Annotated-Bert-Modeling-Sourcecode](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/code/Bert-Modeling-Annotated.py)
16. [Annotated-Bert-Pytorch-Series](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/tree/main/code/annotated_bert_pytorch)

## Homeworks

2. [N-Gram-Python实现snippet](Homework/NLP%E5%9F%BA%E7%A1%80%E7%AC%AC%E4%BA%8C%E8%8A%82%E8%AF%BE%E5%90%8E%E4%BD%9C%E4%B8%9A.py)
3. [基于NN的词向量表示](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/Homework/words_vetctorize.ipynb)


## Related Works

1. [METEOR](Related/METEOR.ipynb)
2. [构建基于Wiki的文本簇](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/Related/%E4%BD%BF%E7%94%A8Wiki%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E6%96%87%E6%9C%AC%E7%B0%87.ipynb)
3. [文本预处理Demo](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/Related/%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86Demo.ipynb)
4. [如何使用NLTK进行词性标注](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/Related/%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8NLTK%E8%BF%9B%E8%A1%8C%E8%AF%8D%E6%80%A7%E6%A0%87%E6%B3%A8.ipynb)
5. [后GPT4时代-什么是 Prompt](https://gist.github.com/sawyerbutton/2305cd60e134ffc06d8402fbfc4b7035)
6. [SVD分解_1](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/Related/SVD%E5%88%86%E8%A7%A320230410.pdf)
7. [LangChain-01](https://gist.github.com/sawyerbutton/412a6c1dbde2b112faa7f84493b9e28f)
8. [LangChain-usecases](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/Related/langchain_usecases.ipynb)
9. [Linux-Funda](https://gist.github.com/sawyerbutton/2328d829cb5d56d0b3f2d017f869d747)
10. [Positional Encoding in Transformer](https://gist.github.com/sawyerbutton/4b7cb394dc29c385adb6051b266570a4)
11. [浮点数类型初探](https://gist.github.com/sawyerbutton/40a62689511fd7c0adc7d52760c0d656)
12. [浮点数类型再探fp16与bf16](https://gist.github.com/sawyerbutton/a0f870200e26577ea9895f1c669ccf64)
13. [GPT模型详细解释大纲](https://github.com/sawyerbutton/NLP-Funda-2023-Spring/blob/main/Related/GPT%E6%A8%A1%E5%9E%8B%E8%AF%A6%E7%BB%86%E8%A7%A3%E9%87%8A.pdf)
