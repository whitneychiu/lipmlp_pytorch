import numpy as np
import torch

def sample_2D_grid(resolution, low = 0, high = 1):
    idx = np.linspace(low,high,num=resolution)
    x, y = np.meshgrid(idx, idx)
    V = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), 1)
    return torch.from_numpy(V).type(torch.FloatTensor)