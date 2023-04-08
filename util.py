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

def slice_min(data, slice_dim):
    """
    Find the min along the slice_dim
    Returns a tensor of shape data.shape[slice_dim] 
    """
    ans = data
    for d in range(data.ndim):
        if d == slice_dim:
            continue
        ans = ans.min(dim=0)[0]
    return ans 

def slice_max(data, slice_dim):
    ans = data
    for d in range(data.ndim):
        if d == slice_dim:
            continue
        ans = ans.max(dim=0)[0]
    return ans 

def make_grid(data, grid_dim, spatial_dim, ncols, pad_factor=1.2):
    """
    data: tensor with data.shape[spatial_dim] == 2 
    Transform data by spacing it out evenly across an nrows x ncols grid
    according to the index of grid_dim
    return: gridded data
    """
    if grid_dim >= data.ndim:
        raise RuntimeError(f'grid_dim = {grid_dim} must be < data.ndim (= {data.ndim})')
    if spatial_dim >= data.ndim or data.shape[spatial_dim] != 2:
        raise RuntimeError(
            f'spatial_dim = {spatial_dim} but data.shape[{spatial_dim}] either doesn\'t '
            f'exist or doesn\'t equal 2')
    
    strides = (slice_max(data, spatial_dim) - slice_min(data, spatial_dim)) * pad_factor
    G = data.shape[grid_dim]

    cols = (t.arange(G) % ncols)
    rows = (t.arange(G) // ncols)

    # G, 2
    bcast = [ 1 if i not in (grid_dim, spatial_dim) else d for i, d in enumerate(data.shape) ]
    grid = t.dstack((cols, rows)) * strides
    grid_data = data + grid.reshape(*bcast)
    return grid_data

def dim_to_data(data, dim):
    """
    Increase the size of the last dimension with the dim index
    """
    if dim >= data.ndim - 1:
        raise RuntimeError(
            f'Got dim = {dim}, data.ndim = {data.ndim}. dim must be < data.ndim-1')
    D = data.shape[dim]
    bcast = tuple(D if i == dim else 1 for i in range(data.ndim))
    vals = t.arange(D).reshape(*bcast).expand(*data.shape[:-1], 1)
    return t.cat((data, vals), data.ndim-1)

def to_dict(data, key_string):
    """
    data: bdims, D
    key_string: string of length D
    output: { key_string[i] => data[...,i] values as a 1D list } 
    """
    if data.shape[-1] != len(key_string):
        raise RuntimeError(
            f'data.shape[-1] (= {data.shape[-1]} must equal '
            f'len(key_string (= {key_string})')
    lol = data.flatten(0, data.ndim-2).permute(1, 0).tolist()
    return dict(zip(key_string, lol))




