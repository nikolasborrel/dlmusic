import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.hot_encoder import one_hot_encode_sequence

def train_lstm(net, num_epochs, training_set, validation_set, vocab_size, word_to_idx):
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
        for inputs, targets in validation_set:
            optimizer.zero_grad()

            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
            targets_idx = [word_to_idx[word] for word in targets]
            
            # Convert input to tensor
            inputs_one_hot = torch.Tensor(inputs_one_hot)
            inputs_one_hot = inputs_one_hot.permute(0, 2, 1)
            
            # Convert target to tensor
            targets_idx = torch.LongTensor(targets_idx)
            
            # Forward pass
            # YOUR CODE HERE!
            outputs = net.forward(inputs_one_hot)
            
            # Compute loss
            # YOUR CODE HERE!
            loss = criterion(outputs, targets_idx)
            
            # Update loss
            epoch_validation_loss += loss.detach().numpy()
        
        net.train()
        
        # For each sentence in training set
        for inputs, targets in training_set:
            optimizer.zero_grad()

            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
            targets_idx = [word_to_idx[word] for word in targets]
            
            # Convert input to tensor
            inputs_one_hot = torch.Tensor(inputs_one_hot)
            inputs_one_hot = inputs_one_hot.permute(0, 2, 1)
            
            # Convert target to tensor
            targets_idx = torch.LongTensor(targets_idx)
            
            # Forward pass
            # YOUR CODE HERE!
            outputs = net.forward(inputs_one_hot)
            
            # Compute loss
            # YOUR CODE HERE!
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