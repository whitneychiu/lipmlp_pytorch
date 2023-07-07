import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def mnist_dataloader(mnist_sdf_npy_path, mnist_label_npy_path, num_data_in_used = None, batch_size = 32):
    """
    this load the mnist sdf data and turn it into size (batch_size, 784) 
    
    Usage:
    train_loader = mnist_dataloader(mnist_train_sdf_path, mnist_train_label_path, num_data_in_used, batch_size)
    for x, label in train_loader:
        x = x.to(device)
        ...
    """
    sdfs = np.load(mnist_sdf_npy_path)
    labels = np.load(mnist_label_npy_path)
    grid_size = 28
    
    # extract a subset for training
    if num_data_in_used is not None:
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        idx = idx[:num_data_in_used]
        sdfs = sdfs.transpose(0, 2, 1).reshape(-1, grid_size**2)
        sdfs = sdfs[idx,:]
        labels = labels[idx]
    else:
        sdfs = sdfs.transpose(0, 2, 1).reshape(-1, grid_size**2)
    print('number of data: %d' % (len(labels)))
    dataset = TensorDataset(torch.Tensor(sdfs), torch.Tensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader