import torch
import numpy as np

IMG_FEAT = 5184
IMG_DIM = 18
IMG_SIZE = torch.Size((16, IMG_DIM, IMG_DIM))


def grid_desc_to_tensor(grid_desc):
    grid = torch.Tensor(IMG_FEAT).fill_(0)
    grid.index_fill_(0, grid_desc.long(), 1)
    grid = grid.view(IMG_SIZE)
    return grid

def expand_grid_desc(grid_desc):
    # grid_desc: (K, T, D)
    K, T, D = grid_desc.shape
    grid = torch.zeros((K, T, IMG_FEAT), dtype=torch.float32)
    grid.scatter_(2, grid_desc, 1.0)
    grid = grid.view(K, T, *IMG_SIZE)
    return grid


def arr_desc(arr: np.ndarray):
    flat_tensor = arr.reshape(-1).astype(np.int32)
    idxs = flat_tensor.nonzero()[0].squeeze()
    # idxs.sort()
    return idxs

def parse_str_tensor(tensor_str):
    inp_grid_coord = []
    inp_grid_str = tensor_str
    for coord_str in inp_grid_str.split():
        idx, val = coord_str.split(':')
        inp_grid_coord.append(int(idx))
        assert(float(val)==1.0)
    inp_grid = torch.ShortTensor(inp_grid_coord)
    return inp_grid

idx2action = {
    0: 'move',
    1: 'turnLeft',
    2: 'turnRight',
    3: 'pickMarker',
    4: 'putMarker'
}
action2idx = {
    'move': 0,
    'turnLeft': 1,
    'turnRight': 2,
    'pickMarker': 3,
    'putMarker': 4
}