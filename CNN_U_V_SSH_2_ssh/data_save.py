import numpy as np
import xarray as xr

data_address_u = xr.open_dataset('E:/BaiduNetDOWNLORD/ORAS3/ORAS3taux.nc')
data_address_v = xr.open_dataset('E:/BaiduNetDOWNLORD/ORAS3/ORAS3tauy.nc')
data_address_ssh = xr.open_dataset('E:/BaiduNetDOWNLORD/ORAS3/ORAS3SSH.nc')

#读取数据，xarray中的矩阵
windu = data_address_u['TAUX'].values
windv = data_address_v['TAUY'].values
sealevel = data_address_ssh['SSH'].values

mask = np.isnan(windu)
windu[mask] = 0
windv[mask] = 0
sealevel[mask] = 0

u10_train = windu[: 552, :31, :]
v10_train = windv[: 552, :31, :]
ssh_train = sealevel[: 552, :31, :]

u10_test = windu[492:, :31, ]
v10_test = windv[492:, :31, ]
ssh_test = sealevel[492:, :31, ]

print(u10_test.shape, u10_train.shape, 1,
      v10_test.shape, v10_train.shape,
      ssh_test.shape, ssh_train.shape,
      )


np.save('datafile/u10_test.npy', u10_test)
np.save('datafile/v10_test.npy', v10_test)
np.save('datafile/ssh_test.npy', ssh_test)
np.save('datafile/u10_train.npy', u10_train)
np.save('datafile/v10_train.npy', v10_train)
np.save('datafile/ssh_train.npy', ssh_train)

x1 = np.load("datafile/u10_train.npy")
x2 = np.load('datafile/v10_train.npy')
x3 = np.load('datafile/ssh_train.npy')
X1 = np.load("datafile/u10_test.npy")
xx = np.concatenate((x1, x2, x3), axis=1)
xxx = xx.reshape(552, -1, 31, 156)
print(x1.shape,
      X1.shape,
      xx.shape,
      xxx.shape,
      )