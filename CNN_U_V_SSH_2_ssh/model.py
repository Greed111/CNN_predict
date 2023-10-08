import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils import tensorboard
import numpy as np
class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
           # nn.BatchNorm1d(16),
            nn.MaxPool2d(3, stride=1, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        #    nn.MaxPool2d(2),
        #     nn.Flatten(),
        #   nn.Linear(32*26*5, 16)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return torch.squeeze(x)


    #验证网络正确性，非必要模块。当文件当作脚本运行时候，就执行代码；但是当文件被当做Module被import的时候，就不执行相关代码。
if __name__ == '__main__':
    testm = my_model()
    print(testm)
    x1 = torch.rand(10, 3, 30, 156)
    output = testm(x1)
    print(x1.shape,
          output.shape,
          )


    # writer = SummaryWriter('../logs')
    # writer.add_graph(testm, x1)
    # writer.close()