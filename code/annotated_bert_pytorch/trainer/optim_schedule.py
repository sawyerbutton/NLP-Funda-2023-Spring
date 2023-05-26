'''A wrapper class for optimizer '''
import numpy as np


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    # 一个简单的学习率调整的包装类

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer # 优化器
        self.n_warmup_steps = n_warmup_steps # warmup步数
        self.n_current_steps = 0 # 当前步数
        self.init_lr = np.power(d_model, -0.5) # 学习率初始化

    def step_and_update_lr(self): # 更新学习率
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step() # 更新参数

    def zero_grad(self): # 梯度清零
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self): # 获取学习率的缩放
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
        # 返回 warmup步数的-1.5次方乘以当前步数 和 当前步数的-0.5次方 的最小值

    def _update_learning_rate(self): # 更新学习率
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1 # 当前步数加1
        lr = self.init_lr * self._get_lr_scale() # 学习率 = 学习率初始化 * 学习率缩放

        for param_group in self._optimizer.param_groups: # 遍历优化器的参数组
            param_group['lr'] = lr # 更新学习率
