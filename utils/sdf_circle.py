import numpy as np
import torch

def sdf_circle(x_torch, r = 0.282, center = np.array([0.5,0.5])):
    """
    output the SDF value of a circle in 2D
    Inputs
    x: nx2 array of locations
    r: radius of the circle
    center: center point of the circle
    Outputs
    array of signed distance values at x
    """
    x = x_torch.detach().cpu().numpy()
    dx = x - center
    out = np.sqrt(np.sum((dx)**2, axis=1)) - r
    return torch.from_numpy(out).type(torch.FloatTensor)