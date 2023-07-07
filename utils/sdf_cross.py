import numpy as np
import torch

def sdf_cross(p_torch, bx=0.35, by=0.12, r=0.):
    p = p_torch.detach().cpu().numpy()
    p = np.array(p - 0.5)
    p = np.abs(p)
    p = np.sort(p,1)[:,[1,0]]
    b = np.array([bx, by])
    q = p - b
    k = np.max(q, 1)
    w = q
    w[k<=0,0] = b[1] - p[k<=0,0]
    w[k<=0,1] = -k[k<=0]
    w = np.maximum(w, 0.0)
    length_w = np.sqrt(np.sum(w*w, 1))
    out = np.sign(k) * length_w + r
    return torch.from_numpy(out).type(torch.FloatTensor)