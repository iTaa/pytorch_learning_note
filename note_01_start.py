# 安装

# 官方网站 http://pytorch.org/

# Run this command:
# pip install http://download.pytorch.org/whl/torch-0.1.12.post2-cp35-cp35m-macosx_10_7_x86_64.whl
# pip install torchvision
# OSX Binaries dont support CUDA, install from source if CUDA is needed

# test

import torch

# run , no error ， 安装成功

import numpy as np

# 创建一个numpy_data
np_data = np.arange(6).reshape((2,3))

# numpy_data转换为torch_data
torch_data = torch.from_numpy(np_data)

# torch_data转换为numpy_data
torch2numpy = torch_data.numpy()


print(
    '\n numpy_data', np_data,
    '\n torch_data', torch_data,
    '\n torch2numpy', torch2numpy
)



data = [-1, -2, 1, 2]
# 32位
tensor = torch.FloatTensor(data)
# abs
np_abs = np.abs(data)
th_abs = torch.abs(tensor)
# sin
np_sin = np.sin(data)
th_sin = torch.sin(tensor)
# 均值
np_mean = np.mean(data)
th_mean = torch.mean(tensor)


print(
    '\n np_abs', np_abs,
    '\n th_abs', th_abs,
    '\n np_sin', np_sin,
    '\n th_sin', th_sin,
    '\n np_mean', np_mean,
    '\n th_mean', th_mean,
)


# matrix multiply 矩阵相乘的问题

mat_data = [[11, 22], [33, 44]]
mat_tensor = torch.FloatTensor(mat_data)

print(
    '\n numpy matrix multiply:', np.matmul(mat_data, mat_data),
    '\n torch matrix multiply:', torch.mm(mat_tensor, mat_tensor)
)

