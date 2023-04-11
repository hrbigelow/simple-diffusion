import fire
import numpy as np
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions.normal import Normal
from streamvis import Client
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

def main(batch_size, sample_size, lr, port=8081):
    client = Client('localhost', port, 'swissroll')
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

    client.init('loss', 'rbf_centers', 'rbf_sigmas', 'mu', 'log_sigma', 'sigma_alphas',
            'mu_alphas', 'psamples')
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
        loss_vis = -log_dens.mean(dim=0).unsqueeze(-1)
        # T,3
        loss_vis = t.cat((t.full((T,1), step), loss_vis), dim=1)
        loss_vis = util.to_dict(loss_vis, key_dim=1, val_dims=())
        client.update('loss', step, loss_vis) 
        if step % 10 == 0:
            last_lr = sched.get_last_lr()[0]
            print(f'step = {step}, loss = {loss.item():5.3f}, lr = {last_lr:5.4f}')

        # T,H,D+1 
        sigma_alphas = util.dim_to_data(rbf.sigma_alphas, 0) 
        sigma_alphas = util.to_dict(sigma_alphas, key_dim=2, val_dims=(), key_string='xyz')

        mu_alphas = util.dim_to_data(rbf.mu_alphas, 0)
        mu_alphas = util.to_dict(mu_alphas, key_dim=2, val_dims=(), key_string='xyz')

        client.update('rbf_centers', step, util.to_dict(rbf.basis_centers, key_dim=1)) 
        client.update('rbf_sigmas', step, util.to_dict(rbf.basis_sigmas, key_dim=1))
        client.update('sigma_alphas', step, sigma_alphas)
        client.update('mu_alphas', step, mu_alphas)

        with t.no_grad():
            ntick, limit = 30, 2.3
            ls = t.linspace(-limit, limit, ntick)
            
            # B,D  (D=2)
            pts0 = t.stack(t.meshgrid(ls, ls), dim=2).flatten(0, 1)

            # B,T,D
            pts0 = pts0.expand(T, *pts0.shape).transpose(1, 0)
            mu, sigma = P.model(pts0)
            pts1 = pts0 + mu

            # B,T,D,L  (L is the 'line points dimension', = 2)
            vis = t.stack((pts0, pts1), dim=3)
            vis = util.make_grid(vis, grid_dim=1, spatial_dim=2, ncols=8, pad_factor=1.4)
            vis = util.to_dict(vis, key_dim=2, val_dims=(3,))
            client.update('mu', step, vis)

            grid = util.make_grid(pts0, grid_dim=1, spatial_dim=2, ncols=8)
            log_det = t.log(sigma).sum(dim=2, keepdim=True)
            sigma_vis = t.cat((grid, log_det), dim=2)
            sigma_vis = util.to_dict(sigma_vis, key_dim=2, val_dims=(), key_string='xyz')
            client.update('log_sigma', step, sigma_vis)

        if step % 100 == 0:
            # B,T,D
            try:
                samples = P.sample(1000)
                samples = util.make_grid(samples, grid_dim=1, spatial_dim=2, ncols=8)
                samples = util.dim_to_data(samples, dim=1) 
                samples = util.to_dict(samples, key_dim=2, val_dims=(), key_string='xyz')
                client.update('psamples', step, samples)
            except ValueError as ex:
                print(f'Got error during P.sample: {ex}.  Skipping')
                

if __name__ == '__main__':
    fire.Fire(main)


