from config_paras import *
import numpy as np
from torch.utils.data import Dataset


class my_dataset(Dataset):
    def __init__(self, u10_path, v10_path, ssh_path, t05, t25,
                 t45, t60, t90, t120, time_scale):
        super(my_dataset, self).__init__()
        self.data_u10 = np.load(u10_path)
        self.data_v10 = np.load(v10_path)
        self.data_ssh = np.load(ssh_path)
        self.data_t05 = np.load(t05)
        self.data_t25 = np.load(t25)
        self.data_t45 = np.load(t45)
        self.data_t60 = np.load(t60)
        self.data_t90 = np.load(t90)
        self.data_t120 = np.load(t120)
        self.time = time_scale

        progress_1 = np.concatenate((self.data_u10,
                                     self.data_v10,
                                     self.data_t05,
                                     self.data_t25,
                                     self.data_t45,
                                     self.data_t60,
                                     self.data_t90,
                                     self.data_t120,
                                     self.data_ssh), axis=1)

        self.x_axis = progress_1.reshape(self.time, 9, 31, 156)

    def __getitem__(self, index):

        targets_ssh = self.data_ssh[index]
        sample = self.x_axis[index]
        return sample, targets_ssh

    def __len__(self):
        return len(self.data_u10)

    def predict_ssh(self, time2):
        # x1 = self.data_u10(time),
        # x2 = self.data_v10(time),
        # x3 = self.data_t05(time),
        # x4 = self.data_t25(time),
        # x5 = self.data_t45(time),
        # x6 = self.data_t60(time),
        # x7 = self.data_t90(time),
        # x8 = self.data_t120(time),
        # y = self.data_ssh(time),
        # x = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,y), axis=0)
        x = self.x_axis[time2]
        y = self.data_ssh[time2]
        return x.reshape(1, 9, 31, 156), y.reshape(1, 31, 156)

if __name__ == '__main__':
    testm = my_dataset(mypara.path_data_u_train,
                        mypara.path_data_v_train,
                        mypara.path_data_ssh_train,
                        mypara.path_data_t05_train,
                        mypara.path_data_t25_train,
                        mypara.path_data_t45_train,
                        mypara.path_data_t60_train,
                        mypara.path_data_t90_train,
                        mypara.path_data_t120_train,
                        mypara.train_time)
    x1, x2 =testm.predict_ssh(480)
    print(x1.shape, x2.shape)

