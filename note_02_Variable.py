import torch
from torch.autograd import Variable


tensor = torch.FloatTensor([[1, 2], [3, 4]])
# requires_grad 一般为false
#
variable = Variable(tensor, requires_grad=True)

print(tensor)
# Variable containing:
print(variable)

th_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)

print(th_out)
# Variable containing:
print(v_out)

# 反向传递
v_out.backward()
# variable's grad variable 的梯度
print(variable.grad)
# out put
# Variable containing:
# 0.5000  1.0000
# 1.5000  2.0000

print(variable.data)

