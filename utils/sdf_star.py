import numpy as np
import torch

def sdf_star(x_torch, r = 0.22):
    """
    output the signed distance value of a star in 2D
    Inputs
    x: nx2 array of locations
    r: size of the star
    Outputs
    array of signed distance values at x
    Reference:
    https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
    """
    x = x_torch.detach().cpu().numpy()
    kxy = np.array([-0.5,0.86602540378])
    kyx = np.array([0.86602540378,-0.5])
    kz = 0.57735026919
    kw = 1.73205080757

    x = np.abs(x - 0.5)
    x -= 2.0 * np.minimum(x.dot(kxy), 0.0)[:,None] * kxy[None,:]
    x -= 2.0 * np.minimum(x.dot(kyx), 0.0)[:,None] * kyx[None,:]
    x[:,0] -= np.clip(x[:,0],r*kz,r*kw)
    x[:,1] -= r
    length_x = np.sqrt(np.sum(x*x, 1))
    return torch.from_numpy(length_x*np.sign(x[:,1])).type(torch.FloatTensor)