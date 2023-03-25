"""
Create a generative model q, a 1D gaussian with some mean and variance.

Train a second model, p initialized to a standard gaussian using KL divergence
objective.

"""
import sys
import torch as t
from torch import nn
from torch.distributions import Normal
from collections import deque


class LogNormalFunc(nn.Module): 
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(t.tensor([0.0]))
        self.log_sigma = nn.Parameter(t.tensor([0.0]))
        # self.sigma = nn.Parameter(t.tensor([1.0]))

    def forward(self, x):
        sigma = t.exp(self.log_sigma)
        return t.log(1 / sigma) - 0.5 * ((x - self.mu) / sigma) ** 2 

def main():
    target_mu = float(sys.argv[1])
    target_sigma = float(sys.argv[2])
    lr = float(sys.argv[3])
    print(f'{target_mu=}, {target_sigma=}, {lr=}')
    qdist = Normal(t.tensor([target_mu]), t.tensor([target_sigma]))
    pdist = LogNormalFunc()
    lr = 0.001
    running_sum = 0.0
    batch = deque()
    max_batch = 1000

    for step in range(100000):
        sample = qdist.sample()
        objective = pdist(sample) 
        val = objective.item()
        batch.append(val)
        running_sum += val 
        if len(batch) > max_batch:
            prev_val = batch.popleft()
            running_sum -= prev_val

        pdist.zero_grad()
        objective.backward()
        for par in pdist.parameters():
            with t.no_grad():
                par += lr * par.grad
        if step % 100 == 0:
            mean = running_sum / len(batch) 
            sigma = t.exp(pdist.log_sigma).item()
            print(f'{step}: {mean:2.3f} {pdist.mu.item():2.3f} '
                    f'{sigma:2.3f} norm={par.grad.norm():2.3f}')

if __name__ == '__main__':
    main()

