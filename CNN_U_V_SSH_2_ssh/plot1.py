import numpy as np
import matplotlib.pyplot as plt

x = np.load("last_x.np.npy")  # 读取文件
y = np.load("last_y.np.npy")

# origin=y_test[0,:,:,0]
# predict=ypredeict[0,:,:,0]
# for i in range(1,8):
#     origin = origin+y_test[i, :, :, 0]
#     predict = predict+ypredeict[i, :, :, 0]
# origin=origin/8
# predict=predict/8

origin = y[0, :, :]
predict = x[0, :, :]
#plt.matshow()

#把画布分为三行五列，并设置figure标题
fig,axs=plt.subplots(2,1,figsize=(20,10),sharex=True,sharey=False)
A=axs[0].matshow(origin)
fig.colorbar(A, ax=[axs[0]], shrink=0.5)
B=axs[1].matshow(predict)
fig.colorbar(A, ax=[axs[1]], shrink=0.5)
plt.show()

