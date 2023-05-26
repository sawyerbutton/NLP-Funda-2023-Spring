import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import BERTLM, BERT
from .optim_schedule import ScheduledOptim

import tqdm


class BERTTrainer:

    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.
    Bert 的预训练任务分为 MLM任务和NSP任务
    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu") # 优先使用GPU

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device) # 用BERTLM来做预训练任务

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1: # 如果有多个GPU，那么使用并行计算
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            # nn.DataParallel()是并行计算，将模型放到多个GPU上

        # Setting the train and test data loader
        # train_dataloader 是一个DataLoader对象，里面包含了训练数据
        self.train_data = train_dataloader
        # test_dataloader 是一个DataLoader对象，里面包含了测试数据
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay) # 初始化优化器
        # betas是Adam优化器的参数，weight_decay是L2正则化的参数
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)
        # ScheduledOptim是一个类，用来调整学习率

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0) # NLLLoss是负对数似然损失函数，ignore_index=0表示忽略0的损失

        self.log_freq = log_freq # log_freq是日志频率

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()])) # 打印模型参数的数量

    def train(self, epoch):
        self.iteration(epoch, self.train_data) # 对训练数据进行迭代

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False) # 对测试数据进行迭代

    def iteration(self, epoch, data_loader, train=True):
        """
        循环遍历数据加载器以进行训练或测试
        如果处于训练状态，则激活反向操作，并在每个 epoch 自动保存模型

        参数 epoch：当前 epoch 索引
        data_loader：用于迭代的 torch.utils.data.DataLoader
        train：是训练流程还是测试流程的布尔值
        返回值：无
        """
        str_code = "train" if train else "test"

        # 设定tqdm进度条
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0 # 平均损失
        total_correct = 0 # 总正确数
        total_element = 0 # 总元素数

        for i, data in data_iter:
            # 将 batch_data 发送到设备(GPU 或 CPU)中
            data = {key: value.to(self.device) for key, value in data.items()}

            # 前向传播 next_sentence_prediction 和 masked_lm 模型
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # . is_next 分类结果的 NLL(negative log likelihood) 损失
            next_loss = self.criterion(next_sent_output, data["is_next"])

            # 预测掩码标记词的 NLL Loss
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 添加 next_loss 和 mask_loss
            loss = next_loss + mask_loss

            # 仅在训练时进行反向传播和优化
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # 下一个句子预测准确率
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            # 打印日志信息
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            # 每隔 log_freq 打印一次日志信息
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element) # 打印平均损失和总准确率
        
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        从文件中保存当前 BERT 模型到指定路径
        参数 epoch：当前 epoch 索引
        file_path：模型输出路径，将保存在 file_path+"ep%d" % epoch
        返回值：最终输出路径
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
