import torch

class mypara:
    def __init__(self):
        pass

mypara = mypara()
mypara.single_pre_time = 480      # 这里是鱼粉问题
mypara.device = torch.device("cuda:0")
mypara.num_epochs = 40
mypara.lr_rate = 0.001
mypara.batch_size_train = 24
mypara.batch_size_test = 12
mypara.train_time = 540         # 选择与后续时间保持一致
mypara.test_time = 84           # 选择与后续时间保持一致

# 关于保存数据问题
mypara.path_adr_u = 'E:/BaiduNetDOWNLORD/ORAS3/ORAS3taux.nc'
mypara.name_u = 'TAUX'
mypara.path_adr_v = 'E:/BaiduNetDOWNLORD/ORAS3/ORAS3tauy.nc'
mypara.name_v = 'TAUY'
mypara.path_adr_sealevel = 'E:/BaiduNetDOWNLORD/ORAS3/ORAS3SSH.nc'
mypara.name_ssh = 'SSH'

#关于数据读取问题
mypara.path_data_u_train = 'datafile/trainu10.npy'
mypara.path_data_u_test = 'datafile/testu10.npy'
mypara.path_data_v_train = 'datafile/trainv10.npy'
mypara.path_data_v_test = 'datafile/testv10.npy'
mypara.path_data_ssh_train = 'datafile/trainssh.npy'
mypara.path_data_ssh_test = 'datafile/testssh.npy'

# 关于数据维度问题
mypara.lev_range_l = 1-1
mypara.lev_range_r = 156

mypara.lat_range_l = 1-1
mypara.lat_range_r = 31

mypara.time_range_train_l = 1-1
mypara.time_range_train_r = 540

mypara.time_range_test_l = 528
mypara.time_range_test_r = 612

#关于模型保存问题
mypara.model_savepath = "./model/"
mypara.model_name = "sealevel_1.pth"
mypara.seeds = 1

ls = [  mypara.lat_range_l, mypara.lat_range_r,
        mypara.lev_range_l, mypara.lev_range_r,
        mypara.time_range_train_l, mypara.time_range_train_r,
        mypara.time_range_test_l, mypara.time_range_test_r]
print(ls)