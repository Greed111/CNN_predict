import numpy as np
from torch.utils.data import Dataset


class my_dataset(Dataset):
    def __init__(self, u10_path, v10_path, ssh_path, time_scale):
        super(my_dataset, self).__init__()
        self.data_u10 = np.load(u10_path)
        self.data_v10 = np.load(v10_path)
        self.data_ssh = np.load(ssh_path)
        self.time = time_scale
        progress_1 = np.concatenate((self.data_u10, self.data_v10, self.data_ssh), axis=1)
        self.x_axis = progress_1.reshape(self.time, 3, 30, 156)

    def __getitem__(self, index):
        # u10 = self.data_u10[index]
        # v10 = self.data_v10[index]
        targets_ssh = self.data_ssh[index]
        sample = self.x_axis[index]
        sample2 = self.data_v10[index]
        # final = progress2.swapaxes(0,1)
        return sample, targets_ssh

    def __len__(self):
        return len(self.data_u10)