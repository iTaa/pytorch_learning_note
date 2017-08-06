import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# make fake data
# n_data 是基数
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
# 64位 Integer
y = torch.cat((y0, y1), ).type(torch.LongTensor)


x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
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

# net 1
net = Net(2, 10, 2)
print(net)
# Net (
#   (hidden): Linear (2 -> 10)
#   (predict): Linear (10 -> 2)
# )

# create net method 2
# net 2
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print(net2)

# Sequential (
#   (0): Linear (2 -> 10)
#   (1): ReLU ()
#   (2): Linear (10 -> 2)
# )


optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# 交叉熵
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

for t in range(2):
    out = net(x)  # input x and predict based on x
    loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 10 == 0 or t in [3, 6]:
        # plot and show learning process
        plt.cla()
        _, prediction = torch.max(F.softmax(out), 1)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.show()
        plt.pause(0.5)


