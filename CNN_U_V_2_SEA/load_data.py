from config_paras import *
import numpy as np
from torch.utils.data import Dataset


class my_dataset(Dataset):
    def __init__(self, u10_path, v10_path, ssh_path, time_scale):
        super(my_dataset, self).__init__()
        self.data_u10 = np.load(u10_path)
        self.data_v10 = np.load(v10_path)
        self.data_ssh = np.load(ssh_path)
        self.time = time_scale
        progress_1 = np.concatenate((self.data_u10, self.data_v10), axis=1)
        self.x_axis = progress_1.reshape(self.time, 2, 31, 156)

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
        return x.reshape(1, 2, 31, 156), y.reshape(1, 31, 156)

if __name__ == '__main__':
    testm = my_dataset(mypara.path_data_u_train,
                        mypara.path_data_v_train,
                        mypara.path_data_ssh_train,
                        mypara.train_time)
    x1, x2 =testm.predict_ssh(480)
    print(x1.shape, x2.shape)