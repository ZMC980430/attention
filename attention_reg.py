# %%
import matplotlib
import torch
import torch.nn as nn
import utils
import math
import matplotlib.pyplot as plt
from IPython import display

num_samples = 50
epoch = 10
lr = 1

x = torch.sort(torch.rand(num_samples))[0]*5
x = x.reshape(1, -1)

def f(x):
    return 2*torch.sin(x) + x/2 - torch.cos(x)
    # return x


y = f(x) + torch.randn(num_samples).reshape(1, -1)/10
# plt.scatter(x.numpy(), y.numpy(), marker='+', alpha=0.3)

# %% Nonparametric Attention Pooling based on distance
num_test = 100
x_test=torch.sort(torch.rand(num_test)*5)[0]
def nonparametric(inputs):
    inputs = inputs.repeat(num_samples, 1)
    weights = nn.functional.softmax(-(x-inputs.T)**2/2, dim=1)
    y_hats = torch.matmul(y, weights.T)
    return y_hats, weights

def test(net):
    if isinstance(net, nn.Module):
        net.eval()
    y_hats, weights = net(x_test)
    fig, ax = plt.subplots()
    ax.scatter(x.numpy(), y.numpy(), marker='+', alpha=0.3)
    ax.scatter(x_test.numpy(), y_hats.detach().numpy(),cmap='Reds', marker='+', alpha=0.5)
    utils.heatmap(weights.reshape((1,1,weights.shape[0],-1)), 'xxx', 'yyy')


# test(nonparametric)

# %% Parametric Attention Pooling
class NWKernelRegression(nn.Module):
    def __init__(self, keys, values):
        super().__init__()
        self.keys = keys
        self.values = values
        self.weight = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries):
        queries = queries.repeat(self.keys.shape[1], 1)
        weights = -self.weight*(self.keys-queries.T)**2/2
        weights = nn.functional.softmax(weights, dim=1)
        y_hats = torch.matmul(self.values, weights.T)
        return y_hats, weights

    def setKV(self, keys, values):
        self.keys = keys
        self.values = values


net = NWKernelRegression(x, y)
criterion = nn.MSELoss()
updater = torch.optim.SGD(net.parameters(), lr=1)

x_train = x.repeat(50, 1)[(1-torch.eye(x.shape[1])).type(torch.bool)].reshape(x.shape[1], -1)
y_train = y.repeat(50, 1)[(1-torch.eye(y.shape[1])).type(torch.bool)].reshape(y.shape[1], -1)


for epoch in range(400):
    net.setKV(x_train, y_train)
    updater.zero_grad()
    y_hats, weights = net(x)
    loss = criterion(y_hats, y)
    loss.backward()
    updater.step()
    # plt.cla()
    # display.clear_output(wait=True)
    # test(net)
    # plt.pause(0.1)
    print(f'epoch: {epoch}, loss: {loss}')

net.setKV(x, y)
test(net)

# %%
