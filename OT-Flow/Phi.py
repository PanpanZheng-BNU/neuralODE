import torch
import torch.nn as nn


def antidrivTanh(x): # defined the activation function of ResNet, which is the antiderivative of the tanh function
    return torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))

def derivTanh(x): # define the derivative of the tanh function.
    return 1 - torch.pow(torch.tanh(x), 2) 


class ResNN(nn.Module):
    def __init__(self, d: int, m: int, nTh=2):
        """
            Corresponding to the N of Phi: a ResNet with nTh layers
        :param d:   int, dimension of the space input.
        :param m:   int, hidden dimension of ResNet.
        :param nTh: int, number of ResNet layers.
        """
        super().__init__()

        assert nTh >= 2, "nTh must be an integer >= 2"

        self.d = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList(
                [nn.Linear(d + 1, m, bias=True)] + [nn.Linear(m, m, bias=True) for _ in range(nTh - 1)]
                )

        self.act = antidrivTanh
        self.h = 1.0 / (self.nTh - 1)

    def forward(self, x):
        """
        :param x:   tensor nex-by-d+1, inputs
        """
        x = self.act(self.layers[0](x))

        for i in range(1, self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x


class Phi(nn.Module):
    def __init__(self, nTh, m, d, r=10, alph=[1.0] * 5):
        """
            Phi function in as Eq. (10) in the paper
            Phi([x,t]) = w'*ResNet([x,t]) + 0.5 * [x' t] * A'A * [x;t] + b'*[x;t] + c
        :param nTh:     int, number of ResNet layers.
        :param m:       int, hidden dimension of ResNet.
        :param d:       int, dimension of the space input.
        :param r:       int, the rank r of the A matrix.
        :param alph:    list of floats, the coefficients for the optimization problem
        """
        super().__init__()

        self.m = m
        self.nTh = nTh
        self.d = d
        self.alph = alph

        r = min(r, d + 1)

        self.A = nn.Parameter(torch.zeros(r, d+1), requires_grad=True)
        self.A = nn.init.xavier_uniform_(self.A)
        self.c = nn.Linear(d+1, 1, bias=True)
        self.w = nn.Linear(m, 1, bias=False)

        self.N = ResNN(d, m, nTh=nTh)
        
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)

    def forward(self, x):
        symA = torch.mm(self.A.t(), self.A)

        return self.w(self.N(x)) + 0.5 * torch.sum(torch.mm(x, symA) * x, dim=1, keepdim=True) + self.c(x)

    def trHess(self, x, justGrad=False):
        """
            computate  gradient of Phi wrt x and trace (Hessian of Phi): corresponding to the Eq. (12) and Eq.(14) in the paper.
        """

        N    = self.N
        m    = N.layers[0].weight.shape[0]
        nex  = x.shape[0] # number of examples in the batch
        d    = x.shape[1]-1
        symA = torch.matmul(self.A.t(), self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = N.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( N.layers[0].weight.t() , tanhopen.t() * z[1] )
        grad = z[0] + torch.mm(symA, x.t() ) + self.c.weight.t()

        if justGrad:
            return grad.t()


        Kopen = N.layers[0].weight[:, 0:d] # indexed version of Kopen = torch.m(N.layers[0].weight, E)
        temp = derivTanh(opening.t()) * z[1] 

        trH = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2),2), dim = (0,1))

        temp = tanhopen.t()
        Jac = Kopen.unsqueeze(2) * temp.unsqueeze(1)

        for i in range(1, N.nTh):
            KJ = torch.mm(N.layers[i].weight, Jac.reshape(m,-1))
            KJ = KJ.reshape(m,-1,nex)
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            temp = N.layers[i].forward(u[i-1]).t()
            t_i = torch.sum((derivTanh(temp) * term).reshape(m,-1,nex) * torch.pow(KJ, 2), dim = (0,1))
            trH = trH + N.h * t_i

            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ

        return grad.t(), trH + torch.trace(symA[0:d,0:d])

                

if __name__ == "__main__":
    import pandas as pd
    d = 2
    m = 5

    net = Phi(nTh=2, m=m, d=d)
    net.N.layers[0].weight.data  = 0.1 + 0.0 * net.N.layers[0].weight.data
    net.N.layers[0].bias.data    = 0.2 + 0.0 * net.N.layers[0].bias.data
    net.N.layers[1].weight.data  = 0.3 + 0.0 * net.N.layers[1].weight.data
    net.N.layers[1].weight.data  = 0.3 + 0.0 * net.N.layers[1].weight.data

    # number of samples-by-(d+1)
    x = torch.Tensor([[1.0 ,4.0 , 0.5],[2.0,5.0,0.6],[3.0,6.0,0.7],[0.0,0.0,0.0]])
    y = net(x)
    g,h = net.trHess(x )
    print(g)
    print(h)

