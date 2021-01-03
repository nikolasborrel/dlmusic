import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def eval_lstm(net, data_set, vocab_size, encoder_decoder):
    net.eval()
    outputs_all = []

    # For each sentence in validation set
    for data in data_set:
        # One-hot encode input and target sequence
        inputs_one_hot, _ = encoder_decoder.encode(data)  # MELODY
        
        # NBJ comment: 3 dimensional array needed (outer most of size 1). Not sure why pyTorch needs this structure
        inputs_one_hot_3D = [inputs_one_hot]

        # Convert input to tensor            
        inputs_one_hot_3D = torch.Tensor(inputs_one_hot_3D) # permute not needed
        
        # Forward pass
        outputs = net.forward(inputs_one_hot_3D).data.numpy()    
        outputs_all.extend(outputs)

    return outputs_all