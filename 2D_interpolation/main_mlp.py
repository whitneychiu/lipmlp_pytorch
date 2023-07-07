import sys
sys.path.append("../")
from models.mlp import mlp
from utils.sample_2D_grid import sample_2D_grid
from utils.sdf_circle import sdf_circle
from utils.sdf_cross import sdf_cross
from utils.sdf_star import sdf_star
from utils.plot_sdf import plot_sdf
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

device = 'cpu'
folder = './mlp/'

resolution = 32
V = sample_2D_grid(resolution) # |V|x2 
gt0 = sdf_cross(V)
gt1 = sdf_star(V)
latent0 = torch.tensor([0])
latent1 = torch.tensor([1])

# save ground truth
plt.figure(0)
plot_sdf(gt0)
plt.savefig(folder + "gt0.png")
plt.figure(1)
plot_sdf(gt1)
plt.savefig(folder + "gt1.png")

dims = [3, 256,  256, 256, 1]
model = mlp(dims)
model_str = "mlp" 

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 200000
pbar = tqdm(range(num_epochs))

V0 = torch.hstack((V,latent0.repeat(V.shape[0],1))).to(device) # nVx3
V1 = torch.hstack((V,latent1.repeat(V.shape[0],1))).to(device) # nVx3
gt0 = gt0.to(device)
gt1 = gt1.to(device)
model = model.to(device)

loss_history = []
for epoch in pbar: 
    sdf0 = model(V0).squeeze(1)
    sdf1 = model(V1).squeeze(1)
    loss = loss_func(sdf0, gt0) + loss_func(sdf1, gt1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    pbar.set_postfix({"loss": loss.item()})

    if (epoch+1) % 1000 == 0:
        plt.figure(0)
        plt.clf()
        plt.semilogy(loss_history)
        plt.title("loss history")
        plt.savefig(folder + "_loss_history.png")

        # save weights
        torch.save(model.state_dict(), folder + "mlp_params.pt")

        # save a bunch of snap shots of interpolation
        for t in np.linspace(0, 1, num=11):
            t_repeat = torch.tensor([t]).repeat(V.shape[0],1).type(torch.FloatTensor)
            Vt = torch.hstack((V,t_repeat)).to(device)
            sdft = model(Vt).squeeze(1)
            plt.figure(1)
            plt.clf()
            plot_sdf(sdft)
            t_str = "{:0.2f}".format(t)
            plt.savefig(folder + "_recon_" + t_str + ".png")
