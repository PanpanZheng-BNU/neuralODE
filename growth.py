import torch
from torch import nn, optim, Tensor
import torchdiffeq as tdf
import matplotlib.pyplot as plt
import numpy as np

device = 'cpu'
class growth_dynamic(nn.Module):
    def __init__(self, *args):
        super(growth_dynamic, self).__init__()
        self.K = args[0]
        self.r = args[1]
    def forward(self, t, y):
        dfdx = self.r * (1 - y/self.K) * y
        return Tensor([dfdx])


true_growth = growth_dynamic(100,0.75)
y0 = Tensor([0.1])
t = torch.linspace(0.,20.,1000)
with torch.no_grad():
    y_true = tdf.odeint(true_growth, y0, t)


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1,50),
            nn.Tanh(),
            nn.Linear(50,1)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 2)
                nn.init.constant_(m.bias, 2)
    def forward(self, t, y):
        return self.net(y)



func = ODEFunc()
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

def get_batch(batch_time, batch_num, total_time=len(t)):
    s = torch.from_numpy(np.random.choice(np.arange(total_time - batch_time), batch_num))
    batch_y0 = y_true[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([y_true[s+i] for i in range(batch_time)],dim=0)
    return batch_y.to(device), batch_y0.to(device), batch_t.to(device)




for i in range(1000):
    batch_y, batch_y0, batch_t = get_batch(100, 300)
    optimizer.zero_grad()
    pred_y = tdf.odeint_adjoint(func, batch_y0, batch_t)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()
    print(loss)

with torch.no_grad():
    y_pred = tdf.odeint(func, y0, t)


plt.plot(y_true.squeeze())
plt.plot(y_pred.squeeze(), linestyle="--")
plt.show()
