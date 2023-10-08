import numpy as np
import xarray as xr
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch

# tensor_u10_test = torch.tensor(u10, dtype=float)
# tensor_v10_test = torch.tensor(v10, dtype=float)
# tensor_ssh_test = torch.tensor(ssh, dtype=float)

# tensor_u10_train = torch.tensor(u10, dtype=float)
# tensor_v10_train = torch.tensor(v10, dtype=float)
# tensor_ssh_train = torch.tensor(ssh, dtype=float)

# 主体类方法部分
class my_dataset(Dataset):
      def __init__(self, u10_path, v10_path, ssh_path, time_scale):
            super(my_dataset, self).__init__()
            self.data_u10 = np.load(u10_path)
            self.data_v10 = np.load(v10_path)
            self.data_ssh = np.load(ssh_path)
            self.time = time_scale
            progress_1 = np.concatenate((self.data_u10, self.data_v10, self.data_ssh), axis=1)
            self.x_axis = progress_1.reshape(self.time, 3, 31, 156)

      def __getitem__(self, index):
            # u10 = self.data_u10[index]
            # v10 = self.data_v10[index]
            targets_ssh = self.data_ssh[index]
            sample = self.x_axis[index]
           # final = progress2.swapaxes(0,1)
            return sample, targets_ssh

      def __len__(self):
            return len(self.data_u10)

      def predict_ssh(self, time2):
            x = self.x_axis[time2]
            y = self.data_ssh[time2]
            return x.reshape(1, 3, 31, 156), y.reshape(1, 31, 156)

if __name__ == '__main__':
      data_u10_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/datafile/u10_train.npy'
      data_v10_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/datafile/v10_train.npy'
      data_ssh_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/datafile/ssh_train.npy'
      testm = my_dataset(data_u10_train, data_v10_train, data_ssh_train, 552)
      x1, x2 =testm.predict_ssh(480)
      print(x1.shape, x2.shape)



# if __name__ == '__main__':
#       data_u10_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/u10_test.npy'
#       data_v10_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/v10_test.npy'
#       data_ssh_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/ssh_test.npy'
#       testdata = my_dataset(data_u10_test, data_v10_test, data_ssh_test, 552)
#       print(testdata)
#       x1,x2 = DataLoader(testdata)
#       print(x1.shape,
#             x2.shape)


