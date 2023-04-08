import sys
import fire
import torch as t
from torch import nn
import numpy as np
from torch.distributions import Normal
from torch.optim import Adam
from streamvis import Client 
from collections import defaultdict

class PCond(nn.Module):

    @staticmethod
    def normal_pdf(x, mu, sigma):
        scaled_dist = (x - mu) / sigma
        expon = -0.5 * scaled_dist ** 2
        sqrt_2pi = t.sqrt(t.tensor(2 * 3.1415927410125732))
        return (sigma * sqrt_2pi) ** -1 * t.exp(expon)

    def __init__(self, nbins, T):
        super().__init__()
        self.T = T
        self.bin_centers = t.linspace(-0.2, 1.2, nbins) 
        self.comp_sigma = 10.0 * (nbins ** -1)
        # T, nbins
        self.mu_components = nn.Parameter(t.zeros(T, nbins))

        vals = self.normal_pdf(0.0, self.bin_centers, self.comp_sigma)
        norm = vals.sum().item()
        print(f'norm = {norm}')

        # init_val = - t.log(norm).item()
        init_val = - 3.0 * norm ** -1
        # init_val = 1.0

        self.sigma_components = nn.Parameter(t.full((T, nbins), init_val))

    def cond_params(self, xcond, timestep=None):
        # compute mu, sigma for P(x^{t-1} | x^t)
        # xcond: batch, points

        # 1, 1, nbins
        comp_means = self.bin_centers.reshape(1, 1, -1)

        # batch, points, nbins 
        comp_vals = self.normal_pdf(xcond.unsqueeze(-1), comp_means, self.comp_sigma)

        # 1, nt, nbins
        if timestep is None:
            mu_components = self.mu_components.unsqueeze(0)
            sigma_components = self.sigma_components.unsqueeze(0)
        else:
            mu_components = self.mu_components[timestep].reshape(1, 1, -1)
            sigma_components = self.sigma_components[timestep].reshape(1, 1, -1)

        # B, nt
        p_residual_mean = t.sum(mu_components * comp_vals, dim=-1)
        p_log_sigma = t.sum(sigma_components * comp_vals, dim=-1)

        p_mean = xcond + p_residual_mean
        # return p_mean, t.full(tuple(p_mean.shape), -1.0)
        return p_mean, p_log_sigma

    def mu_grad_norm(self, timestep):
        # return the norm of the mu component gradient at given timestep
        return self.mu_components.grad[timestep].norm()

    def mu_curve(self, timestep, points):
        """
        points: 1D tensor
        """
        # B, 1
        with t.no_grad():
            p_mean, _ = self.cond_params(points.unsqueeze(0), timestep)
        mu = p_mean - points.unsqueeze(0)
        return mu[0,:]

    def sigma_curve(self, timestep, points):
        """
        evaluate the sigma curve across points at layer timestep
        """
        with t.no_grad():
            _, p_log_sigma = self.cond_params(points.unsqueeze(0), timestep)
        return t.exp(p_log_sigma[0,:])

    def sample(self, n):
        """
        Generate n samples from P
        """
        # B, 1
        with t.no_grad():
            xcond = Normal(0, 1).sample((n,))

            for ts in reversed(range(self.T)):
                p_mean, p_log_sigma = self.cond_params(xcond, ts)
                p_sigma = t.exp(p_log_sigma)
                xcond = Normal(p_mean, p_sigma).sample()
        return xcond

    def forward(self, x):
        """
        compute log p_t(x^{t-1} | x^t)
        x: B, T+1
        returns: B, T
        """
        # B, T
        xvar, xcond = x[:,:-1], x[:,1:]
        p_mean, p_log_sigma = self.cond_params(xcond)
        p_sigma = t.exp(p_log_sigma)
        offset = xvar - p_mean
        scaled_dist = (xvar - p_mean) / p_sigma
        p_expon = - 0.5 * scaled_dist ** 2
        log_sqrt_2pi = 0.5 * t.log(t.tensor(2 * 3.1415927410125732))
        p_log_dens = - p_log_sigma - log_sqrt_2pi + p_expon
        return p_log_dens

def train(host, port, run_name, lr, every, batch_size, nmix, T):
    data_client = Client('localhost', 8080, run_name)

    # num_timesteps = 1000
    # num_timesteps = 20
    inspect_layers = list(range(0, T, max(1, T//10)))
    # inspect_layers = [0,1,5,10,20,50,100,500,999]
    num_positions = 5
    data_client.clear()
    data_client.init('xbymu', 'xbysigma', 'psamples', 'loss')

    num_mixture_components = nmix
    num_samples = 2000
    # betas = t.linspace(1e-4, 0.02, T)
    betas = t.full((T,), 0.02)
    Q = QDist(batch_size, betas)
    P = PCond(num_mixture_components, T)
    # dataset = t.tensor([0.25, 0.37, 0.44, 0.98])
    dataset = t.tensor([0.3, 0.7])
    opt = Adam(P.parameters(), lr)

    xi = []
    for step in range(10000):
        # x: B, T+1
        x = Q.sample(dataset)

        # log_dens: B, T
        log_dens = P(x)
        sum_log_dens = t.sum(log_dens)
        loss = -1.0 * (batch_size ** -1) * sum_log_dens 
        P.zero_grad()
        loss.backward()
        opt.step()
        data_client.updatel('loss', step, { 'step': step, 'loss': loss.item() })

        if step % every == 0:
            print(f'{step}: {loss:2.3f}')
            # x, y = P.mu_curve(0, t.linspace(0, 1, 1000))
            # data_client.update('mu', step, { 'x': [x.numpy()], 'y': [y.numpy()] })
            timestep_space = 1
            domain_space = t.linspace(0, 0.5, num_positions) 
            mu_scale = 4
            sigma_space = 0.1

            xpoints = np.linspace(0, 1, 1000).tolist() 
            points = { 'x': step }
            xbymu = { 'x': xpoints }
            xbysigma = { 'x': xpoints }
            
            # for ti, time in enumerate([0, 1, 2, 10, 20, 50, 100, 500]):
            for ti, time in enumerate(inspect_layers):
                mus = P.mu_curve(time, t.tensor(xbymu['x'])) 
                mus = mus * mu_scale + (ti * timestep_space)
                sigmas = P.sigma_curve(time, t.tensor(xbysigma['x']))
                sigmas = sigmas + (ti * sigma_space)

                xbymu[f't{time}'] = mus.numpy().tolist()
                xbysigma[f't{time}'] = sigmas.numpy().tolist()
            # data_client.updatel('mu', step, points)
            data_client.update('xbymu', step, xbymu)
            data_client.update('xbysigma', step, xbysigma)

        if step % 50 == 0:
            psamples = P.sample(num_samples)
            histo, bins = np.histogram(psamples, 400)
            centers = (bins[1:] + bins[:-1]) / 2.0
            data = { 'x': centers.tolist(), 'y': histo.tolist() }
            data_client.update('psamples', step, data)

def main():
    func_map = { 'train': train, 'showq': showq }
    fire.Fire(func_map)

if __name__ == '__main__':
    main()

