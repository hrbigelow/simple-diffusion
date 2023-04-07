from sys import stderr
import torch as t
from torch import nn
from torch.distributions.log_normal import Normal
from torch.utils.data import Dataset, DataLoader, Sampler

class TensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:]

class LoopingRandomSampler(Sampler):
    def __init__(self, dataset, num_replicas=1, rank=0, start_epoch=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = start_epoch
        print(f'LoopingRandomSampler with {self.rank} out of {self.num_replicas}', file=stderr)

    def __iter__(self):
        def _gen():
            while True:
                g = t.Generator()
                g.manual_seed(self.epoch * self.num_replicas + self.rank)
                n = len(self.dataset)
                vals = list(range(self.rank, n, self.num_replicas))
                perms = t.randperm(len(vals), generator=g).tolist()
                # print(f'LoopingRandomSampler: first 10 perms: {perms[:10]}', file=stderr)
                yield from [vals[i] for i in perms]
                self.epoch += 1

        return _gen()

    def __len__(self):
        return int(2**31)


class QDist:
    """
    Generate samples from q(x^{0:T})
    """
    def __init__(self, sample_size, betas):
        self.sample_size = sample_size
        self.betas = betas
        self.alpha = (1.0 - self.betas) ** 0.5
        self.cond = [ Normal(0, beta) for beta in self.betas]

    def sample(self, x0):
        """
        x0: P, D
        returns: B * P, T+1, D  (batch, num_points, num_timesteps+1)
        """
        # sample a 'trajectory'
        xi_pre = x0.repeat(self.sample_size, 1)
        total_reps = xi_pre.shape[0]
        x = [xi_pre]
        for cond, alpha in zip(self.cond, self.alpha):
            xi = cond.sample(xi_pre.shape) + alpha * xi_pre  
            xi_pre = xi
            x.append(xi)
        res = t.stack(x, dim=1)
        return res

class PDist(nn.Module):
    """
    Accepts a mu_sigma_model, which is an nn.Module which 
    """
    def __init__(self, mu_sigma_model):
        super().__init__()
        self.model = mu_sigma_model

    @staticmethod
    def log_gaussian_density(x, mu, sigma):
        """
        x: B, dims
        mu, sigma: dims
        return: B, dims
        """
        scaled_dist = (x - mu) / sigma
        expon = -0.5 * scaled_dist ** 2
        log_sqrt_2pi = 0.5 * t.log(t.tensor(2 * 3.1415927410125732))
        log_density = - t.log(sigma) - log_sqrt_2pi + expon
        return log_density

    def forward(self, x):
        """
        x: B, T, D
        returns: B, T
        """
        xvar, xcond = x[:,:-1], x[:,1:]
        mu, sigma = self.model(xcond)
        log_density = self.log_gaussian_density(xvar, xcond + mu, sigma)
        return log_density





