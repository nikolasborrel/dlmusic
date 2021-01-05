import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.tools import timer

@timer
def train_lstm(net, num_epochs, training_set, validation_set, vocab_size, encoder_decoder):
    # Define a loss function and optimizer for this problem
    # YOUR CODE HERE!
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Track loss
    training_loss, validation_loss = [], []

    # For each epoch
    for i in range(num_epochs):
        
        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0
        
        net.eval()
            
        # For each sentence in validation set
        for inputs_one_hot, targets_idx in validation_set:
            optimizer.zero_grad()

            # DIMENSION inputs_one_hot
            # (batch_size, seq_len, input_size)
            batch_size, seq_len, input_size = inputs_one_hot.shape
            # DIMENSION targets_idx
            # (batch_size, seq_len)

            
            # # One-hot encode input and target sequence
            # inputs_one_hot, _ = encoder_decoder.get_inputs_batch(input_and_target[0])  # MELODY
            # _, targets_idx = encoder_decoder.encode(input_and_target[1])               # BASS
            
            # # NBJ comment: 3 dimensional array needed with dimension (batch_size, seq_len, input_size)
            # # TODO: batch size of one currently!
            # inputs_one_hot_3D = [inputs_one_hot]

            # # Convert input to tensor            
            # inputs_one_hot_3D = torch.Tensor(inputs_one_hot_3D) # permute not needed
            
            # # Convert target to tensor
            # targets_idx = torch.LongTensor(targets_idx)
            
            # Forward pass
            # YOUR CODE HERE!
            outputs = net.forward(inputs_one_hot)
            
            # Compute loss
            # YOUR CODE HERE!
            targets_idx = targets_idx.reshape(batch_size * seq_len)
            loss = criterion(outputs, targets_idx)
            
            # Update loss
            epoch_validation_loss += loss.detach().numpy()
        
        net.train()
        
        # For each sentence in training set
        for inputs_one_hot, targets_idx in training_set:
            optimizer.zero_grad()
            batch_size, seq_len, input_size = inputs_one_hot.shape

            # DIMENSION inputs_one_hot
            # (batch_size, seq_len, input_size)

            # DIMENSION targets_idx
            # (batch_size, seq_len)


            # # One-hot encode input and target sequence
            # inputs_one_hot, _ = encoder_decoder.get_inputs_batch(input_and_target[0])
            # _, targets_idx = encoder_decoder.encode(input_and_target[1])
            
            # # NBJ comment: 3 dimensional array needed with dimension (batch_size, seq_len, input_size)
            # # TODO: batch size of one currently!
            # inputs_one_hot_3D = [inputs_one_hot]

            # # Convert input to tensor
            # inputs_one_hot_3D = torch.Tensor(inputs_one_hot_3D) # permute not needed
            
            # # Convert target to tensor
            # targets_idx = torch.LongTensor(targets_idx)
            
            # Forward pass
            # YOUR CODE HERE!
            outputs = net.forward(inputs_one_hot)
            
            # Compute loss
            # YOUR CODE HERE!
            targets_idx = targets_idx.reshape(batch_size * seq_len)
            loss = criterion(outputs, targets_idx)
            
            # Backward pass
            # YOUR CODE HERE!
            # zero grad, backward, step...
            loss.backward()
            optimizer.step()
            
            # Update loss
            epoch_training_loss += loss.detach().numpy()
            
        # Save loss for plot
        training_loss.append(epoch_training_loss/len(training_set))
        validation_loss.append(epoch_validation_loss/len(validation_set))

        # Print loss every 10 epochs
        if i % 10 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

    return training_loss, validation_loss