import fire
import numpy as np
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from streamvis import Client
import util

def swissroll():
    # Make the swiss roll dataset
    N = 1000
    phi = np.random.rand(N) * 3*np.pi + 1.5*np.pi
    x = phi * np.cos(phi)
    y = phi * np.sin(phi)
    return np.stack((x, y), axis=1)

def reshape(ten, *dims):
    # reshape a tensor, using the current shape where dims[i] is None
    dims = [tdim if dim is None else dim for tdim, dim in zip(ten.shape, dims)]
    return ten.reshape(*dims)

class RBFNetwork(nn.Module):
    def __init__(self, D=2, T=40, H=16):
        super().__init__()
        norm = t.distributions.Normal(0.5, 1.0)
        self.centers = nn.Parameter(norm.sample((D, H)))
        self.mu_alphas = nn.Parameter(t.full((T, H, D), 1.0))
        self.sigma_alphas = nn.Parameter(t.full((T, H, D), 1.0))

    def forward(self, xcond):
        # x: B, T, D
        # dists: B, H
        # returns: B, T, D
        # B, T, D, H -> B, T, H
        dists = ((xcond.unsqueeze(-1) - self.centers) ** 2).sum(dim=2)

        # B, T, H, D -> B, T, D
        mu = (dists.unsqueeze(-1) * self.mu_alphas).sum(dim=2)
        sigma = (dists.unsqueeze(-1) * self.sigma_alphas).sum(dim=2)
        sigma = t.sigmoid(sigma)
        return mu, sigma 

def main(batch_size, sample_size, lr):
    client = Client('localhost', 8080, 'swissroll')
    client.init('swiss_roll')
    # client.update('swiss_roll', 0, { 'x': x.tolist(), 'y': y.tolist() })

    T = 40
    betas = t.linspace(1e-4, 0.1, T)
    Q = util.QDist(sample_size, betas)
    P = util.PDist(RBFNetwork())
    data = swissroll()
    dataset = util.TensorDataset(data)
    sampler = util.LoopingRandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    opt = Adam(P.parameters(), lr)

    iloader = iter(loader)
    for step in range(10000):
        batch = next(iloader)
        x = Q.sample(batch)
        log_dens = P(x)
        loss = - log_dens.mean()
        P.zero_grad()
        loss.backward()
        opt.step()
        print(f'loss = {loss.item():5.3f}')

if __name__ == '__main__':
    fire.Fire(main)


