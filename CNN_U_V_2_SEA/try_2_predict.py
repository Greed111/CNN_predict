from load_data import *
import torch
from model import *
from config_paras import *

saved_model = torch.load('sealevel_1.pth')
train_data = my_dataset(mypara.path_data_u_train,
                        mypara.path_data_v_train,
                        mypara.path_data_ssh_train,
                        mypara.train_time)

predict_data_x, truth_data_y = train_data.predict_ssh(484)# (mypara.single_pre_time)
print(truth_data_y.shape,
      predict_data_x.shape)
predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
predict_data_x = predict_data_x.cpu().detach().numpy()
print(truth_data_y.shape,
      predict_data_x.shape)
np.save('2y_5.npy', truth_data_y)
np.save('2x_5.npy', predict_data_x)
