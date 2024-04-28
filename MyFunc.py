import torch
from torch import nn, optim, Tensor, tensor
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import progressbar

class ODEFunc(nn.Module):
    def __init__(self, model: nn.Module, autonomous: bool=True):
        super(ODEFunc, self).__init__()
        self.net = model
        self.autonomous = autonomous
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        if not self.autonomous:
            y = torch.cat((torch.ones_like(y[:, :1]) * t, y), dim=1)
        return self.net(y**3)

class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, solver: str = "dopri5", 
                 rtol:float=1e-4, atol:float=1e-4, adjoint:bool=True,
                 autonomous:bool=True):
        super(ODEBlock, self).__init__()
        self.odefunc = ODEFunc(odefunc, autonomous)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.use_adjoint = adjoint
        self.integration_time = torch.tensor([0, 1], dtype=torch.float32)

    @property
    def odeint_method(self):
        return odeint_adjoint if self.use_adjoint else odeint


    def forward(self, x:Tensor, adjoint:bool=True, integration_time=None):
        integration_time = self.integration_time if integration_time is None else integration_time
        out = self.odeint_method(self.odefunc, x, integration_time, rtol=self.rtol, atol=self.atol, method=self.solver, options={'dtype': torch.float32})
        return out



class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y**3, true_A)


def get_batch(batch_time, batch_size, total_time):
    s = torch.from_numpy(np.random.choice(np.arange(total_time - batch_time), batch_size))
    batch_y0 = true_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def update(num):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size, t.size(0))
    pred_y = odeblock(batch_y0, integration_time=batch_t)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    y_pred = odeblock(true_y0, integration_time=t)
    ax.clear()
    l1, =ax.plot(*true_y.squeeze().T)
    l2, =ax.plot(*y_pred.cpu().detach().numpy().squeeze().T, linestyle="--")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    ax.legend([l1,l2], ['True Trajectory', 'Predicted Trajectory'], loc="lower right")
    ax.title.set_text('Epoch: {}, Loss: {:.5f}'.format(num,loss.item()))
    ax.title.set_fontweight('bold')
    optimizer.step()

if __name__=="__main__":
    device = 'cpu'
    model = nn.Sequential(nn.Linear(2, 50), nn.Tanh(), nn.Linear(50, 2))
    odeblock = ODEBlock(model).to(device)
    true_y0 = Tensor([[2., 0.]]).to(device)
    t = torch.linspace(0., 25., 10000).to(device)
    true_A = torch.tensor([[-0.1,2.0], [-2.0, -0.1]], dtype=torch.float32).to(device)
    with torch.no_grad():
        true_y = odeint(Lambda(), true_y0, t, method='dopri5', options={'dtype': torch.float32})


    optimizer = optim.Adam(odeblock.parameters(), lr=1e-3)
    max_epoch = 1000
    batch_time = 20
    batch_size = 2000

    # for epoch in progressbar.progressbar(range(max_epoch), redirect_stdout=True):
        # optimizer.zero_grad()
        # batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size, t.size(0))
        # pred_y = odeblock(batch_y0, integration_time=batch_t)
        # loss = torch.mean(torch.abs(pred_y - batch_y))
        # loss.backward()
        # optimizer.step()
        # print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))

    fig, ax = plt.subplots(
            figsize=(10, 8)
            )

    ani = animation.FuncAnimation(fig, update, frames=200, repeat=False)
    ani.save('animation_drawing2.gif', writer='imagemagick', fps=20)


    # y_pred = odeblock(true_y0, integration_time=t)
    # figure = plt.figure()
    # ax = figure.add_subplot(111)
    # ax.plot(*true_y.squeeze().T)
    # ax.plot(*y_pred.cpu().detach().numpy().squeeze().T, linestyle="--")
    # plt.show()

    # print(out.shape)
