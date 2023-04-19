from sys import stderr
import torch as t
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

