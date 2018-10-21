import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np


a = torch.rand(5, 4)
b = a.numpy()
print(b)

a = np.array([[3, 4], [3, 6]])
b = torch.from_numpy(a)
print(a)

x = Variable(torch.Tensor([3]), requires_grad=True)
y = Variable(torch.Tensor([5]), requires_grad=True)
z = 2*x + y + 4
z.backward()

print('dz/dx: {}', format(x.grad.data))
print('dz/dy: {}', format(y.grad.data))


class net_name(nn.Module):
    def __init__(self):
        super(net_name, self).__init__()
        # 可以添加各种网络层
        self.conv1 = nn.Conv2d(3, 10, 3)
        # 具体每种层的参数可以去查看文档

    def forward(self, x):
        # 定义向前传播
        out = self.conv1(x)
        return out