import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# fake data
# torch 中只处理二维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

# 转换为torch可用的的Variable
x, y = Variable(x), Variable(y)

# 试用散点图画出来
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


# 继承 Module
class Net(torch.nn.Module):
    # 初始化信息
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        # n_output 其实就是1
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 前向传递的过程，主要是在这个地方搭建流程图
    def forward(self, x):
        # 试用 hidden layer 加工一下信息， 试用激励函数激活一下
        x = F.relu(self.hidden(x))
        # 这里 predict 为了不截断，可以不用 activation function
        x = self.predict(x)
        return x


# 1 个输入值，10个隐藏层，1个输出值
net = Net(1, 10, 1)
# 查看神经网络的层次结构
print(net)
# Net (
#   (hidden): Linear (1 -> 10)
#   (predict): Linear (10 -> 1)
# )

# 可视化
# 设置 matplotlib 变成一个实时的过程，不卡死
plt.ion()

# lr 是 learning rate 学习速率
# SGD 梯度下降法
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# MSELoss 均方差
loss_func = torch.nn.MSELoss()

for i in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    # 优化步骤
    optimizer.zero_grad()
    # 反向传递
    loss.backward()
    # 优化梯度
    optimizer.step()

    # 可视化结果
    if i % 5 == 0:
        # 画图
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
