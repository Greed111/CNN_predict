from model import *
from load_data import *
from config_paras import *
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

train_data = my_dataset(mypara.path_data_u_train,
                        mypara.path_data_v_train,
                        mypara.path_data_ssh_train,
                        mypara.train_time)
test_data = my_dataset(mypara.path_data_u_test,
                       mypara.path_data_v_test,
                       mypara.path_data_ssh_test,
                       mypara.test_time)

train_data_size = len(train_data)
test_data_size = len(test_data)

# 添加tensorboard
writer = SummaryWriter("../logs_train")

#加载模型和数据集
CNN_u_v_2_ssh = my_model().cuda()
train_dataloader = DataLoader(train_data, batch_size=mypara.batch_size_train)
test_dataloader = DataLoader(test_data, batch_size=mypara.batch_size_test)

#损失函数
loss_fn = nn.CrossEntropyLoss().cuda()

# 优化器
# optimizer = torch.optim.SGD(CNN_u_v_2_ssh.parameters(), lr=0.002)
optimizer = torch.optim.Adam(CNN_u_v_2_ssh.parameters(), lr=0.001)

#optimizer = torch.optim.Adam([], lr=mypara.lr_rate)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0

# 开始调整网络
for i in range(mypara.num_epochs):
    print("--------------第 {} 轮训练开始--------------".format(i+1))

    # 训练步骤开始
    CNN_u_v_2_ssh.train()
    for data in train_dataloader:
        samples, targets = data
        samples = samples.cuda()
        targets = targets.cuda()
        outputs = CNN_u_v_2_ssh(samples)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录训练次数，每10次输出一次。
        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    CNN_u_v_2_ssh.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            samples, targets = data
            samples = samples.cuda()
            targets = targets.cuda()
            outputs = CNN_u_v_2_ssh(samples)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # accuracy = (outputs.argmax(1) == targets).sum()
            # total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
   # print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
   # writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

# np.save('last_x.np', outputs.cpu())
# np.save('last_y.np', targets.cpu())
writer.close()
torch.save(CNN_u_v_2_ssh, "sealevel_1.pth")
print("模型已保存")