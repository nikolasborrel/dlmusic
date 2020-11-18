import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce

# The digit classes to use
#classes = [3, 7]
#classes = [0, 1, 4, 9]

def load_mnist(path_train, path_test, classes):
    # Flatten the images into a vector
    flatten = lambda x: ToTensor()(x).view(28**2)

    # Define the train and test sets
    dset_train = MNIST(path_train, train=True,  transform=flatten, download=True)
    dset_test  = MNIST(path_test, train=False, transform=flatten)

    batch_size = 64
    eval_batch_size = 100
    
    # The loaders perform the actual work
    train_loader = DataLoader(dset_train, batch_size=batch_size,
                            sampler=stratified_sampler(dset_train.train_labels,classes))
    test_loader  = DataLoader(dset_test, batch_size=eval_batch_size, 
                            sampler=stratified_sampler(dset_test.test_labels,classes))

    return train_loader, test_loader

def stratified_sampler(labels,classes):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)