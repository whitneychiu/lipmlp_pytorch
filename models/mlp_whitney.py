import torch

class mlp(torch.nn.Module):
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
        self.relu = torch.nn.ReLU()

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            W = self.layers[ii].weight.data
            W_abs_row_sum = torch.abs(W).sum(1)
            lips_constant = torch.max(W_abs_row_sum)

            loss_lipc = loss_lipc * lips_constant

        W_layer_output = self.layer_output.weight.data
        W_layer_output_abs_row_sum = torch.abs(W_layer_output).sum(1)
        lips_constant_layer_output = torch.max(W_layer_output_abs_row_sum)

        loss_lipc = loss_lipc * lips_constant_layer_output
        
        return loss_lipc


    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        return self.layer_output(x)
    

        # loss_lipc_init = 1.0
        # self.layers = torch.nn.ModuleList()
        # for ii in range(len(dims)-2):
        #     self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))

        # get lipschitz_constant
        #     W = self.layers[ii].weight.data
        #     W_abs_row_sum = torch.abs(W).sum(1)
        #     lips_constant = torch.max(W_abs_row_sum)

        #     loss_lipc = loss_lipc_init * lips_constant

        # self.layer_output = torch.nn.Linear(dims[-2], dims[-1])
        # self.relu = torch.nn.ReLU()
        
        # W_layer_output = self.layer_output.weight.data
        # W_layer_output_abs_row_sum = torch.abs(W_layer_output).sum(1)
        # lips_constant_layer_output = torch.max(W_layer_output_abs_row_sum)

        # loss_lipc = loss_lipc * lips_constant_layer_output