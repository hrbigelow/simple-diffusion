import torch as t
from torch import nn
from torch.distributions.log_normal import Normal
import funcs

class RBFNetwork(nn.Module):
    """
    Implements a 'normalized RBF network'
      https://arxiv.org/abs/1503.03585 (Sohl-Dickstein, Ganguli 2015) App D.1.1
      https://en.wikipedia.org/wiki/Radial_basis_function_network
      top layer weights are specific to each timestep
      all other weights shared across mu, sigma, and across all timesteps
    """
    def __init__(self, D=2, T=40, H=16):
        super().__init__()
        self.T = T
        self.D = D
        self.H = H
        self.centers = nn.Parameter(t.randn((H, D)))
        self.sigmas = nn.Parameter(t.sigmoid(t.full((H, D), -0.5)))
        self.mu_alphas = nn.Parameter(t.full((T, H, D), 0.0))
        self.sigma_alphas = nn.Parameter(t.full((T, H, D), -2.0))

    def xdim(self):
        return self.D 

    def forward(self, xcond, ts=None):
        """
        Runs in two modes:

        If ts is None:
           xcond: B,T,D
           returns mu, sigma: B,T,D
        Else:
           xcond: B,D
           returns mu, sigma: B,D 
        """
        B = xcond.shape[0]
        D, H = self.D, self.H
        if ts is None:
            if xcond.ndim != 3:
                raise RuntimeError(f'If ts is None, xcond.ndim must be 3.  Got {xcond.ndim}')
            Tdim = self.T
            mu_alphas = self.mu_alphas
            sigma_alphas = self.sigma_alphas
        else:
            if xcond.ndim != 2:
                raise RuntimeError(
                    f'If ts is not None, xcond.ndim must be 2.  Got {xcond.ndim}')
            xcond = xcond.unsqueeze(1)
            Tdim = 1
            mu_alphas = self.mu_alphas[ts].unsqueeze(0)
            sigma_alphas = self.sigma_alphas[ts].unsqueeze(0)

        # B, Tdim, H
        pdf = funcs.combo_isonormal_pdf(xcond, self.centers, self.sigmas)

        # B, Tdim, 1
        normalizer = pdf.sum(-1, True) ** -1

        # B, Tdim, H, D -> B, Tdim, D
        mu = (pdf.unsqueeze(-1) * mu_alphas).sum(dim=2) * normalizer
        sigma_logits = (pdf.unsqueeze(-1) * sigma_alphas).sum(dim=2) * normalizer
        sigma = t.sigmoid(sigma_logits)

        if ts is not None:
            mu = mu.squeeze(1)
            sigma = sigma.squeeze(1)

        return mu, sigma

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

    def forward(self, x):
        """
        x: B, T, D
        returns: B, T
        """
        xvar, xcond = x[:,:-1,:], x[:,1:,:]
        mu, sigma = self.model(xcond)
        log_density = funcs.isonormal_logpdf(xvar, xcond + mu, sigma)
        return log_density

    def sample(self, n):
        with t.no_grad():
            xcond = Normal(0, 1).sample((n,self.model.xdim()))
            for ts in reversed(range(self.model.T)):
                mu, sigma = self.model(xcond, ts)
                xcond = Normal(mu, sigma).sample()
        return xcond


