import sys
import fire
from matplotlib import pyplot as plt
import torch as t
from torch import nn
from torch.distributions import Normal
from torch.optim import Adam
# import wandb
import signal
from streamvis import Sender

class QDist:
    """
    Generate samples from q(x^{0:T})

    """
    def __init__(self, batch_size, betas):
        self.batch_size = batch_size
        self.betas = betas
        self.alpha = (1.0 - self.betas) ** 0.5
        self.cond = [ Normal(0, beta) for beta in self.betas]

    def sample(self, x0):
        """
        x0: P
        returns: B * P, T+1  (batch, num_points, num_timesteps+1)
        """
        # sample a 'trajectory'
        # 1, P
        xi_pre = x0.repeat(self.batch_size)
        total_reps = xi_pre.shape[0]
        x = [xi_pre]
        for cond, alpha in zip(self.cond, self.alpha):
            xi = cond.sample((total_reps,)) + alpha * xi_pre  
            xi_pre = xi
            x.append(xi)
        return t.stack(x, dim=1)

class PCond(nn.Module):
    def __init__(self, nbins, T):
        super().__init__()
        self.T = T
        bins = t.linspace(0, 1, nbins+1)

        # nbins
        self.bin_centers = (bins[:-1] + bins[1:]) / 2 

        # scalar
        self.comp_sigma = nbins ** -1

        # T, nbins
        self.mu_components = nn.Parameter(t.zeros(T, nbins))
        self.sigma_components = nn.Parameter(t.full((T, nbins), 0.000001))

    def cond_params(self, xcond, timestep=None):
        # compute mu, sigma for P(x^{t-1} | x^t)
        # xcond: B, nt

        # 1, 1, nbins
        comp_means = self.bin_centers.reshape(1, 1, -1)
        scaled_dist = (comp_means - xcond.unsqueeze(-1)) / self.comp_sigma 

        # B, nt, nbins 
        expon = - 0.5 * scaled_dist ** 2
        comp_vals = (self.comp_sigma ** -1) * t.exp(expon)

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
        return p_mean, t.full(tuple(p_mean.shape), -1.0)
        # return p_mean, p_log_sigma

    def mu_grad_norm(self, timestep):
        # return the norm of the mu component gradient at given timestep
        return self.mu_components.grad[timestep].norm()

    def mu_curve(self, timestep, npoints, training_step):
        # B, 1
        points = t.linspace(0, 1, npoints)
        with t.no_grad():
            p_mean, _ = self.cond_params(points.unsqueeze(-1), timestep)
        mu = p_mean - points.unsqueeze(-1)
        return points, mu[:,0]

        # tab = t.stack((points, mu[:,0])).permute(1, 0)
        # tab = wandb.Table(['x', 'mu'], tab.numpy())
        # art = wandb.plot.line(tab, 'x', 'mu')
        # art = wandb.plot.line_series(points.numpy(), [mu[:,0].numpy()],
                # title=f'mu curve, step {training_step}', xname='x-value')
        # return art

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
        # print(f'p_sigma.max(): {p_sigma.max():2.3f}, p_expon.max(): {p_expon.max():2.3f}')
        p_log_dens = - p_log_sigma + p_expon
        # print(f'p_expon[:,0].max() = {p_expon[:,0].max():2.3f}, '
         #        f'offset = {offset[:,0].min():2.5f}, {offset[:,0].max():2.5f}')
        return p_log_dens

def train(lr, every, batch_size):
    """
    run = wandb.init(
            project='1d-diffusion',
            id=run_id,
            anonymous='never'
            )

    def cleanup(signal, frame):
        run.finish()
        exit(0)

    signal.signal(signal.SIGINT, cleanup)
    """

    blog = Sender(port=1234)

    num_timesteps = 1000
    num_mixture_components = 1000
    num_samples = 1000
    betas = t.linspace(1e-4, 0.02, num_timesteps)
    Q = QDist(batch_size, betas)
    P = PCond(num_mixture_components, num_timesteps)
    # dataset = t.tensor([0.25, 0.37, 0.44, 0.98])
    dataset = t.tensor([0.25])
    opt = Adam(P.parameters(), lr)

    xi = []
    for step in range(10000):
        # x: B, T+1
        x = Q.sample(dataset)
        # dists = ((x[:,1:] - x[:,:-1]) ** 2).sum(dim=0)
        # print('dists.max(): ', dists.max())

        # log_dens: B, T
        log_dens = P(x)
        sum_log_dens = t.sum(log_dens)
        loss = -1.0 * (batch_size ** -1) * sum_log_dens 
        # loss = -1.0 * (batch_size ** -1) * t.sum(log_dens[:,0])
        P.zero_grad()
        loss.backward()
        opt.step()
        """
        for name, par in P.named_parameters():
            if par.grad is None:
                continue
            if step % every == 0:
                print(f'{name} mu0: {P.mu_grad_norm(0):2.3f} mu990: {P.mu_grad_norm(990):2.3f}')
            with t.no_grad():
                par -= lr * par.grad
        run.log({
            'step': step,
            'loss': loss
            }, step=step)
        """
        blog.send(step, 'main', { 'loss': loss.item() })

        if step % every == 0:
            print(f'{step}: {loss:2.3f}')
            x, y = P.mu_curve(0, 100, step)
            blog.send(step, 'mu', { 'x': [x.numpy()], 'y': [y.numpy()] })
            # run.log({ 'mus': P.mu_curve(0, 100, step) }, step=step)

        # run.commit()

        if step % 1000 == 0 and step > 0:
            psamples = P.sample(num_samples)
            plt.hist(psamples.tolist(), bins=30, alpha=0.5, color='blue')
            plt.show()
    # run.finish()

def showq(timestep, num_timesteps, batch_size):
    betas = t.linspace(1e-4, 0.02, num_timesteps)
    Q = QDist(batch_size, betas)
    # dataset = t.tensor([0.25, 0.44, 0.98])
    dataset = t.tensor([0.001, 0.999])
    x = Q.sample(dataset)
    xt = x[:, timestep]

    plt.hist(xt.tolist(), bins=100, alpha=0.5, color='blue')
    plt.show()

    # for ts in range(1, num_timesteps, 10):
        # xt = x[:, ts]

def main():
    func_map = { 'train': train, 'showq': showq }
    fire.Fire(func_map)

if __name__ == '__main__':
    main()

