import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, inputs, targets, encoder_decoder):
        self.inputs = inputs
        self.targets = targets
        self.encoder_decoder = encoder_decoder

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        # One-hot encode input and target sequence
        X, _ = self.encoder_decoder.encode(self.inputs[index])  # LEAD
        _, y = self.encoder_decoder.encode(self.targets[index]) # ACCOMPANYING

        X = torch.Tensor(X)
        y = torch.LongTensor(y)

        return X, y