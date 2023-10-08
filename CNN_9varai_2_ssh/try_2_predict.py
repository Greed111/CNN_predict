from load_data import *
import torch
from model import *
from config_paras import  *

saved_model = torch.load('sealevel_gpu.pth')
train_data = my_dataset(mypara.path_data_u_train,
                        mypara.path_data_v_train,
                        mypara.path_data_ssh_train,
                        mypara.path_data_t05_train,
                        mypara.path_data_t25_train,
                        mypara.path_data_t45_train,
                        mypara.path_data_t60_train,
                        mypara.path_data_t90_train,
                        mypara.path_data_t120_train,
                        mypara.train_time)

predict_data_x, truth_data_y = train_data.predict_ssh(484)#mypara.single_pre_time)
predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
predict_data_x = predict_data_x.cpu().detach().numpy()

np.save('9y_5.npy', truth_data_y)
np.save('9x_5.npy', predict_data_x)
