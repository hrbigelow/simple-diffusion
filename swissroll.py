import fire
import numpy as np
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions.normal import Normal
from streamvis import Client, ColorSpec, GridSpec
import util
import models

def swissroll(n):
    # Make the swiss roll dataset with zero mean, unit sd
    phi = np.random.rand(n) * 2.6*np.pi + 1.2*np.pi
    x = phi * np.cos(phi)
    y = phi * np.sin(phi)
    data = np.stack((x, y), axis=1) + np.random.rand(n,2) * 0.01
    data -= data.mean(axis=0)
    data /= data.std()
    return data

def main(batch_size, sample_size, lr, port=8080):
    client = Client(f'localhost:{port}', 'swissroll')
    # client.update('swiss_roll', 0, { 'x': x.tolist(), 'y': y.tolist() })

    T = 40
    betas = t.linspace(1e-4, 0.1, T)
    Q = models.QDist(sample_size, betas)
    rbf = models.RBFNetwork()
    P = models.PDist(rbf)
    data = swissroll(100000)
    dataset = util.TensorDataset(data)
    sampler = util.LoopingRandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    opt = Adam(P.parameters(), lr)
    sched = ExponentialLR(opt, gamma=0.9)

    iloader = iter(loader)
    inspect_layer = 5

    grid_map = dict(
            mu=(0,0,1,1),
            rbf_centers=(0,1,1,1),
            mu_alphas=(0,2,1,1),
            loss=(1,0,1,1),
            sigma_alphas=(1,1,1,1),
            psamples=(1,2,1,1)
            )
    client.set_layout(grid_map)
    inspect_layers = list(range(0, T))

    for step in range(10000):
        batch = next(iloader)
        x = Q.sample(batch)
        log_dens = P(x)
        if t.any(t.isnan(log_dens)):
            pdb.set_trace()

        loss = - log_dens.mean()
        P.zero_grad()
        loss.backward()
        opt.step()

        if step > 0 and step % 100 == 0:
            sched.step()

        # T,1
        loss_vis = -log_dens.mean(dim=0)

        client.tandem_lines('loss', step, loss_vis, 'Viridis256')
        if step % 10 == 0:
            last_lr = sched.get_last_lr()[0]
            print(f'step = {step}, loss = {loss.item():5.3f}, lr = {last_lr:5.4f}')

        client.scatter('rbf_centers', rbf.basis_centers, spatial_dim=1, append=False)
        client.scatter('mu_alphas', rbf.mu_alphas, spatial_dim=2, append=False,
                color=ColorSpec('Viridis256', 0))
        client.scatter('sigma_alphas', rbf.sigma_alphas, spatial_dim=2, append=False,
                color=ColorSpec('Viridis256', 0))
        # client.update('rbf_sigmas', step, util.to_dict(rbf.basis_sigmas, key_dim=1))

        with t.no_grad():
            ntick, limit = 20, 2.3
            ls = t.linspace(-limit, limit, ntick)
            
            # B,D  (D=2)
            pts0 = t.stack(t.meshgrid(ls, ls), dim=2).flatten(0, 1)

            # B,T,D
            pts0 = pts0.expand(T, *pts0.shape).transpose(1, 0)
            mu, sigma = P.model(pts0)
            pts1 = pts0 + mu

            # B,T,D,L  (L is the 'line points dimension', = 2)
            vis = t.stack((pts0, pts1), dim=3)
            client.multi_lines('mu', vis, line_dims=(0,1), spatial_dim=2, append=False,
                    grid=GridSpec(1,8,1.2))

        if step % 100 == 0:
            # B,T,D
            samples = P.sample(1000)
            client.scatter('psamples', samples, spatial_dim=2, append=False,
                    color=ColorSpec('Viridis256', 1),
                    grid=GridSpec(dim=1, num_columns=8, padding_factor=1.2))
                

if __name__ == '__main__':
    fire.Fire(main)


