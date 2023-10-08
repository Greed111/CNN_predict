from config_paras import *
import numpy as np
import xarray as xr

# 将矩阵范围写为列表
ls = [  mypara.lat_range_l, mypara.lat_range_r,
        mypara.lev_range_l, mypara.lev_range_r,
        mypara.time_range_train_l, mypara.time_range_train_r,
        mypara.time_range_test_l, mypara.time_range_test_r]
def read_data(path, name):
    data_all = xr.open_dataset(path)
    # 转化为矩阵形式
    data_2_matrix = data_all[name].values
    # 设置mask
    mask = np.isnan(data_2_matrix)
    data_2_matrix[mask] = 0
    return data_2_matrix

def save_data(matrix, name, lat1, lat2, lon1 , lon2 , time_train_1, time_train_2, time_test_1, time_test_2):
    #train
    np.save('datafile/train'+name, matrix[time_train_1:time_train_2, lat1:lat2, lon1:lon2])
    #test
    np.save('datafile/test'+name, matrix[time_test_1:time_test_2, lat1:lat2, lon1:lon2])
    return '成功保存'


u10_all = read_data(mypara.path_adr_u, mypara.name_u)
v10_all = read_data(mypara.path_adr_v, mypara.name_v)
seal_all = read_data(mypara.path_adr_sealevel, mypara.name_ssh)
T05_all = np.squeeze(read_data(mypara.path_adr_t05, mypara.name_t05))
T25_all = np.squeeze(read_data(mypara.path_adr_t25, mypara.name_t25))
T45_all = np.squeeze(read_data(mypara.path_adr_t45, mypara.name_t45))
T60_all = np.squeeze(read_data(mypara.path_adr_t60, mypara.name_t60))
T90_all = np.squeeze(read_data(mypara.path_adr_t90, mypara.name_t90))
T120_all = np.squeeze(read_data(mypara.path_adr_t120, mypara.name_t120))

print(u10_all.shape)

save_data(u10_all, 'u10.npy', ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7])
save_data(v10_all, 'v10.npy', ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7])
save_data(seal_all, 'ssh.npy', ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7])
save_data(T05_all, 't05.npy', ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7])
save_data(T25_all, 't25.npy', ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7])
save_data(T45_all, 't45.npy', ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7])
save_data(T60_all, 't60.npy', ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7])
save_data(T90_all, 't90.npy', ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7])
save_data(T120_all, 't120.npy', ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7])