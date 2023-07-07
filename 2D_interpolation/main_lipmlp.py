import sys
sys.path.append("../")
from models.lipmlp import lipmlp
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
folder = './lipmlp/'

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

dims = [3, 64, 64, 64, 64, 1]
model = lipmlp(dims)

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
loss_sdf_history = []
loss_lipschitz_history = []
lam = 1e-5
for epoch in pbar: 
    # forward
    sdf0 = model(V0).squeeze(1)
    sdf1 = model(V1).squeeze(1)

    # compute loss
    loss_sdf = loss_func(sdf0, gt0) + loss_func(sdf1, gt1)
    loss_lipschitz = lam * model.get_lipschitz_loss()
    loss = loss_sdf + loss_lipschitz

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    loss_sdf_history.append(loss_sdf.item())
    loss_lipschitz_history.append(loss_lipschitz.item())

    pbar.set_postfix({"loss": loss.item(), "sdf loss": loss_sdf.item(), "lip loss": loss_lipschitz.item()})

    if (epoch+1) % 5000 == 0:
        plt.figure(0)
        plt.clf()
        plt.semilogy(loss_history)
        plt.semilogy(loss_sdf_history)
        plt.semilogy(loss_lipschitz_history)
        plt.title("loss history")
        plt.legend(['total loss', 'sdf loss', 'lipschitz loss'])
        plt.savefig(folder + "loss_history.png")

        # save weights
        torch.save(model.state_dict(), folder + "lipmlp_params.pt")

        # save a bunch of snap shots of interpolation
        for t in np.linspace(0, 1, num=11):
            t_repeat = torch.tensor([t]).repeat(V.shape[0],1).type(torch.FloatTensor)
            Vt = torch.hstack((V,t_repeat)).to(device)
            sdft = model(Vt).squeeze(1)
            plt.figure(1)
            plt.clf()
            plot_sdf(sdft)
            t_str = "{:0.2f}".format(t)
            plt.savefig(folder + "recon_" + t_str + ".png")
