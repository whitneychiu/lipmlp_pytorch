import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_sdf(u):
    u = u.detach().cpu().numpy()
    resolution = np.round(u[:].shape[0] ** (1./2)).astype(int)
    max_val = np.abs(u).max()
    min_val = -max_val
    levels = np.linspace(min_val, max_val, 31)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("SDF", [(0, "#eff3ff"), (0.5, "#3182bd"), (0.5, "#31a354"), (1, "#f7fcb9")], N=256)
    plt.contourf(u.reshape(resolution,resolution), levels=levels, cmap=colormap)
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')