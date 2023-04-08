import fire
import numpy as np
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.distributions.normal import Normal
from streamvis import Client
import util
import models

def swissroll():
    # Make the swiss roll dataset with zero mean, unit sd
    N = 1000
    phi = np.random.rand(N) * 3*np.pi + 1.5*np.pi
    x = phi * np.cos(phi)
    y = phi * np.sin(phi)
    data = np.stack((x, y), axis=1)
    data -= data.mean(axis=0)
    data /= data.std()
    return data

def main(batch_size, sample_size, lr):
    client = Client('localhost', 8080, 'swissroll')
    # client.update('swiss_roll', 0, { 'x': x.tolist(), 'y': y.tolist() })

    T = 40
    betas = t.linspace(1e-4, 0.1, T)
    Q = models.QDist(sample_size, betas)
    rbf = models.RBFNetwork()
    P = models.PDist(rbf)
    data = swissroll()
    dataset = util.TensorDataset(data)
    sampler = util.LoopingRandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    opt = Adam(P.parameters(), lr)

    iloader = iter(loader)
    inspect_layer = 5

    client.init('loss', 'centers', 'sigmas', 'mu_alphas10', 'sigma_alphas',
            'grid_mu10', 'grid_sigma10')
    inspect_layers = list(range(0, T))

    # client.init((f'q{t}' for t in inspect_layers))
    # client.init('loss', 'q')

    for step in range(10000):
        batch = next(iloader)
        x = Q.sample(batch)
        log_dens = P(x)
        loss = - log_dens.mean()
        P.zero_grad()
        loss.backward()
        opt.step()
        client.updatel('loss', step, dict(x=step,y=loss.item()))
        if step % 10 == 0:
            print(f'step = {step}, loss = {loss.item():5.3f}')

        """
        # B*P, T+1, D
        xgrid = util.make_grid(x, grid_dim=1, spatial_dim=2, ncols=8)
        xgrid = xgrid.flatten(0,1)
        client.update('q', step, xydict(xgrid))
        """
        sigma_alphas = util.to_dict(util.dim_to_data(rbf.sigma_alphas, 0), 'xyz')

        client.update('centers', step, util.to_dict(rbf.centers, 'xy')) 
        client.update('sigmas', step, util.to_dict(rbf.sigmas, 'xy'))
        # client.update('mu_alphas10', step, xydict(rbf.mu_alphas[inspect_layer]))
        client.update('sigma_alphas', step, sigma_alphas)

        with t.no_grad():
            ntick = 10
            ls = t.linspace(-1, 1, ntick)
            # T, B, D
            grid = t.dstack(t.meshgrid(ls, ls)).reshape(ntick*ntick, 2)
            grid_mu, grid_sigma = P.model(grid.expand(T, *grid.shape))

            # T, B, D+D
            mu = t.dstack((grid, grid + grid_mu))



        with t.no_grad():
            ntick = 10
            ls = t.linspace(-1, 1, ntick)
            grid = t.dstack(t.meshgrid(ls, ls)).reshape(ntick*ntick, 2)
            grid_mu, grid_sigma = P.model(grid, inspect_layer)
            mu = t.dstack((grid, grid + grid_mu))
            mu_x, mu_y = mu[:,0,:].tolist(), mu[:,1,:].tolist()
            sigma_x = grid[:,0].tolist()
            sigma_y = grid[:,1].tolist()
            sigma_z = grid_sigma[:,0].tolist() # only look at x dimension 0

            client.update('grid_mu10', step, dict(zip('xy', [mu_x, mu_y])))
            client.update('grid_sigma10', step, dict(zip('xyz', [sigma_x, sigma_y,
                sigma_z])))

        # if step % 1000 == 0:
            # client.update('psamples', step, xydict(P.sample(1000)))

if __name__ == '__main__':
    fire.Fire(main)


