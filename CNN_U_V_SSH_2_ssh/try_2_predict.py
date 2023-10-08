from load_data import *
import torch
from model import *




saved_model = torch.load('sealevel_new.pth')
data_u10_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/datafile/u10_test.npy'
data_v10_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/datafile/v10_test.npy'
data_ssh_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/datafile/ssh_test.npy'
data_u10_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/datafile/u10_train.npy'
data_v10_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/datafile/v10_train.npy'
data_ssh_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/datafile/ssh_train.npy'

train_data = my_dataset(data_u10_train, data_v10_train, data_ssh_train, 552)
# test_data = my_dataset(data_u10_test, data_v10_test, data_ssh_test, 120)

predict_data_x, truth_data_y = train_data.predict_ssh(482)# (mypara.single_pre_time)
print(truth_data_y.shape,
      predict_data_x.shape)
predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
predict_data_x = predict_data_x.cpu().detach().numpy()
print(truth_data_y.shape,
      predict_data_x.shape)
np.save('3y_3.npy', truth_data_y)
np.save('3x_3.npy', predict_data_x)
