import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.tools import timer
from tqdm import tqdm
from models.model_lstm import DEVICE

@timer
def train_lstm(net, num_epochs, training_set, validation_set, vocab_size, encoder_decoder, lr=1e-4):
    # Define a loss function and optimizer for this problem
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.Adagrad(net.parameters(), lr=lr)
    #optimizer = optim.RMSprop(net.parameters(), lr=lr)
    #optimizer = optim.SparseAdam(net.parameters(), lr=lr)

    # Track loss
    training_loss, validation_loss = [], []
    clip = 5
    # For each epoch
    for i in tqdm(range(num_epochs)):
        
        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0
        '''
        if i in list(range(5,40,5)):
            lr = lr/2
            print('lr is decreased to: ',lr)
        #lr=0.001
        #Cyclic learning rate
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
        '''
        net.eval()
            
        # For each sentence in validation set
        for inputs_one_hot, targets_idx in validation_set:
            inputs_one_hot = inputs_one_hot.to(DEVICE)
            targets_idx = targets_idx.to(DEVICE)
            # DIMENSION inputs_one_hot
            # (batch_size, seq_len, input_size)
            batch_size, seq_len, input_size = inputs_one_hot.shape
            # DIMENSION targets_idx
            # (batch_size, seq_len)
            
            # Forward pass
            # YOUR CODE HERE!
            outputs = net.forward(inputs_one_hot).to(DEVICE)
            
            # Compute loss
            # YOUR CODE HERE!
            targets_idx = targets_idx.reshape(batch_size * seq_len)
            loss = criterion(outputs, targets_idx)
            
            # Update loss
            epoch_validation_loss += loss.cpu().detach().numpy()
        
        net.train()
        
        # For each sentence in training set
        for inputs_one_hot, targets_idx in training_set:
            
            inputs_one_hot = inputs_one_hot.to(DEVICE)
            targets_idx = targets_idx.to(DEVICE)
            
            batch_size, seq_len, input_size = inputs_one_hot.shape
            # Forward pass
            outputs = net.forward(inputs_one_hot)
            
            # Compute loss
            targets_idx = targets_idx.reshape(batch_size * seq_len)
            loss = criterion(outputs, targets_idx)
            
            # Backward pass
            # zero grad, backward, step
            optimizer.zero_grad()
            loss.backward()  # Calculates gradient
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step() #Update parameters
            
            # Update loss
            epoch_training_loss += loss.cpu().detach().numpy()
            
        # Save loss for plot
        training_loss.append(epoch_training_loss/len(training_set))
        validation_loss.append(epoch_validation_loss/len(validation_set))

        # Print loss every 10 epochs
        if i % 10 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

    return training_loss, validation_loss

@timer
def train_lstm_scheduled_lr(net, num_epochs, 
                training_set, validation_set, 
                vocab_size, encoder_decoder, 
                lr=1e-4, half_lr_schedule=list(range(5,40,5))):
    # Define a loss function and optimizer for this problem
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    clip = 5
    # Track loss
    training_loss, validation_loss = [], []

    # For each epoch
    for i in tqdm(range(num_epochs)):
        
        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0
        
        if i in list(half_lr_schedule):
            lr = lr/2
            print('lr is decreased to: ',lr)
        #Scheduled Learning rate
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
        
        net.eval()
            
        # For each sentence in validation set
        for inputs_one_hot, targets_idx in validation_set:
            
            #inputs_one_hot [batch_size, seq_len, input_size]
            #targets_idx [batch_size, seq_len]
            inputs_one_hot = inputs_one_hot.to(DEVICE)
            targets_idx = targets_idx.to(DEVICE)
            # DIMENSION inputs_one_hot
            # (batch_size, seq_len, input_size)
            batch_size, seq_len, input_size = inputs_one_hot.shape
            # DIMENSION targets_idx
            # (batch_size, seq_len)
            
            # Forward pass
            outputs = net.forward(inputs_one_hot).to(DEVICE)
            
            # Compute loss
            #targets_idx = targets_idx.reshape(batch_size * seq_len)
            targets_idx = targets_idx.view(batch_size*seq_len)
            loss = criterion(outputs, targets_idx) #.long()
            
            # Update loss
            epoch_validation_loss += loss.cpu().detach().numpy()
        
        net.train()
        
        # For each sentence in training set
        for inputs_one_hot, targets_idx in training_set:
            
            inputs_one_hot = inputs_one_hot.to(DEVICE)
            targets_idx = targets_idx.to(DEVICE)
            
            batch_size, seq_len, input_size = inputs_one_hot.shape

            # DIMENSION inputs_one_hot
            # (batch_size, seq_len, input_size)

            # DIMENSION targets_idx
            # (batch_size, seq_len)
            
            # Forward pass
            outputs = net.forward(inputs_one_hot)
            
            # Compute loss
            targets_idx = targets_idx.reshape(batch_size * seq_len)
            loss = criterion(outputs, targets_idx) #.long()
            
            # Backward pass
            # zero grad, backward, step...
            optimizer.zero_grad()
            loss.backward()  # Calculates gradient
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step() #Update parameters
            
            # Update loss
            epoch_training_loss += loss.cpu().detach().numpy()
            
        # Save loss for plot
        training_loss.append(epoch_training_loss/len(training_set))
        validation_loss.append(epoch_validation_loss/len(validation_set))

        # Print loss every 10 epochs
        if i % 10 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

    return training_loss, validation_loss

    def adjust_learning_rate(optimizer, iter, each):
        # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
        lr = args.lr * (0.1 ** (iter // each))
        state_dict = optimizer.state_dict()
        for param_group in state_dict['param_groups']:
            param_group['lr'] = lr
        optimizer.load_state_dict(state_dict)
        return lr