import torch
import torch.nn as nn
import torchdiffeq as tdf
import matplotlib.pyplot as plt
from torch import nn, optim
import numpy as np


device = 'cpu'
class Lorenz(nn.Module):
    def __init__(self, *args):
        super(Lorenz, self).__init__()
        self.s = args[0]
        self.r = args[1]
        self.b = args[2]

    def forward(self, t, y):
        return torch.Tensor([self.s * (y[1] - y[0]), y[0] * (self.r - y[2]) - y[1], y[0] * y[1] - self.b * y[2]])

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50,3)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)



y0 = torch.Tensor([1,2,3])
t = torch.linspace(0., 20., 1000)

a = Lorenz(10, 28, 2.667)
with torch.no_grad():
    true_y = tdf.odeint(a, y0, t)


def get_batch(batch_size, batch_num, total_size=len(t)):
    s = torch.from_numpy(np.random.choice(np.arange(total_size-batch_size),batch_num,replace=False))
    batch_t = t[:batch_size]
    batch_y0 = true_y[s]
    batch_y = torch.stack([true_y[s+i] for i in range(batch_size)], dim=0)
    return batch_t.to(device), batch_y0.to(device), batch_y.to(device)
# print(true_y.squeeze().T)

print(get_batch(10,100))


func = ODEFunc().to(device)
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

ax = plt.figure().add_subplot(projection='3d')
for i in range(100):
    optimizer.zero_grad()
    batch_t, batch_y0, batch_y = get_batch(len(t)//20,5)
    pred_y = tdf.odeint_adjoint(func, batch_y0, batch_t)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    # if loss < 1e-3:
    #     break
    optimizer.step()
    with torch.no_grad():
        y_pred = tdf.odeint(func, y0, t)
    # ax.plot(*true_y.squeeze().T)
    ax.plot(*y_pred.squeeze().T, linestyle="--", color="r")
    plt.show()

    print(loss)

ax = plt.figure().add_subplot(projection='3d')
with torch.no_grad():
    y_pred = tdf.odeint(func, y0, t)

ax.plot(*true_y.squeeze().T)
ax.plot(*y_pred.squeeze().T, linestyle="--", color="r")
plt.show()
