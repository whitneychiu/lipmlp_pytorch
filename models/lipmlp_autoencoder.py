import sys
sys.path.append("../")
import torch
import math
from utils.sample_2D_grid import sample_2D_grid 

class mlp_encoder(torch.nn.Module):
    def __init__(self, dims):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))

        self.layer_output = torch.nn.Linear(dims[-2], dims[-1])
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        x = self.layer_output(x)
        return self.tanh(x) * 100 # bound the latent range

class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)

class lipmlp_decoder(torch.nn.Module):
    def __init__(self, dims):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(LipschitzLinear(dims[ii], dims[ii+1]))

        self.layer_output = LipschitzLinear(dims[-2], dims[-1])
        self.relu = torch.nn.ReLU()

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, V_in, code_in):
        resolution = 28
        batch_size = code_in.shape[0]
        latent_size = code_in.shape[1]

        # reshape V and code for forward pass
        V = V_in.unsqueeze(0).repeat(batch_size,1,1)
        code = code_in.unsqueeze(1).repeat(1,resolution**2,1)
        V_code = torch.cat((V, code), dim=2)
        x = V_code.reshape(batch_size * resolution**2, latent_size+2)

        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        return self.layer_output(x).reshape(batch_size, resolution**2)