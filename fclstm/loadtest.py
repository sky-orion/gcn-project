# import numpy as np
# npz=np.load(r"D:\QQ\1392776996\FileRecv\features.npy")
# # var = npz.files  #查看有哪些数据集
# print(npz.shape)
# # var = npz['dis']  #读取dis数据集
# seq_length_x, seq_length_y = 12, 12
# raw_data = npz
# print(raw_data.shape)
# data=raw_data
#
# x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
# y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
# print(x_offsets,y_offsets)
# num_samples, num_nodes, input_dim = data.shape
#
# x, y = [], []
# min_t = abs(min(x_offsets))
# max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
# print(min_t,max_t)
# for t in range(min_t, max_t):
#     x_t = data[t + x_offsets, ...]
#
#     y_t = np.expand_dims(data[t + y_offsets, :, -1], axis=-1)
#     print(np.array(x_t).shape,np.array(y_t).shape, t + x_offsets, t + y_offsets,data[t + x_offsets ,0, -1],data[t + y_offsets ,0, -1])
#     x.append(x_t)
#     y.append(y_t)
# print(np.array(x).shape,np.array(y).shape)
# x = np.stack(x, axis=0)
# y = np.stack(y, axis=0)
# import h5py
# import numpy as np
#
# # #HDF5的写入：
# # imgData = np.zeros((2,4))
# # f = h5py.File(r'D:\METR-LA\metr-la.h5','r')   #创建一个h5文件，文件指针是f
# # f['data'] = imgData                 #将数据写入文件的主键data下面
# # f['labels'] = np.array([1,2,3,4,5])            #将数据写入文件的主键labels下面
# # f.close()                           #关闭文件
#
# #HDF5的读取：
# f = h5py.File(r'D:\METR-LA\metr-la.h5', 'r')  #打开h5文件
# # 可以查看所有的主键
#
# for key in f.keys():
#     print(f.)
    # print(f[key].shape)
    # print(f[key].value)

# import numpy as np
#
# # .npy文件是numpy专用的二进制文件
# arr = np.array([[1, 2], [3, 4]])
# test=[arr,arr]
# # 保存.npy文件
# np.save("test.npy", test)
# print("save .npy done")
#
# # 读取.npy文件
# arr1=np.load("test.npy")
# print(arr1)
# print("load .npy done")
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import layer
from models import attention
from models.attention import ConvLSTMCell, AttentionStem
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.my_tensor = torch.randn(1) # 参数直接作为模型类成员变量
        self.register_buffer('my_buffer', torch.randn(1)) # 参数注册为 buffer
        self.my_param = nn.Parameter(torch.randn(1))
        self.fc = nn.Linear(2,2,bias=False)
        self.conv = nn.Conv2d(2,1,1)
        self.fc2 = nn.Linear(2,2,bias=False)
        self.f3 = self.fc
    def forward(self, x):
        return x

model = MyModel().to("cuda")
print(model.state_dict())
# >>>OrderedDict([('my_param', tensor([-0.3052])), ('my_buffer', tensor([0.5583])), ('fc.weight', tensor([[ 0.6322, -0.0255],
#         [-0.4747, -0.0530]])), ('conv.weight', tensor([[[[ 0.3346]],
#
#          [[-0.2962]]]])), ('conv.bias', tensor([0.5205])), ('fc2.weight', tensor([[-0.4949,  0.2815],
#         [ 0.3006,  0.0768]])), ('f3.weight', tensor([[ 0.6322, -0.0255],
#         [-0.4747, -0.0530]]))])