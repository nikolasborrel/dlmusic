import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        for input_and_target in validation_set:
            optimizer.zero_grad()

            # One-hot encode input and target sequence

            #### TODO - ONLY FOR DEBUG/TESTING ####

            # input_and_target is a tuple, first element is melody, second is bass
            
            inputs_one_hot, label_not_used = encoder_decoder.encode(input_and_target[0])        # MELODY
            inputs_one_hot_not_used, targets_idx = encoder_decoder.encode(input_and_target[1])  # BASS
            
            #inputs_one_hot, targets_idx = encoder_decoder.encode(input_and_target[0])

            #print(inputs_one_hot)
            #print(targets_idx)

            #inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
            #targets_idx = [word_to_idx[word] for word in targets]
            
            # NBJ comment: 3 dimensional array needed (outer most of size 1). Not sure why pyTorch needs this structure
            inputs_one_hot_3D = [inputs_one_hot]

            # Convert input to tensor            
            inputs_one_hot_3D = torch.Tensor(inputs_one_hot_3D) 
            #inputs_one_hot = inputs_one_hot.permute(0, 2, 1) # not needed
            
            # Convert target to tensor
            targets_idx = torch.LongTensor(targets_idx)
            
            # Forward pass
            # YOUR CODE HERE!
            outputs = net.forward(inputs_one_hot_3D)
            
            # Compute loss
            # YOUR CODE HERE!

            ## IMPORTANT: how to evaluate loss when input/output is different? See Seq2Seq paper
            loss = criterion(outputs, targets_idx)
            
            # Update loss
            epoch_validation_loss += loss.detach().numpy()
        
        net.train()
        
        # For each sentence in training set
        for input_and_target in training_set:
            optimizer.zero_grad()

            # One-hot encode input and target sequence
            inputs_one_hot, _ = encoder_decoder.encode(input_and_target[0])
            _, targets_idx = encoder_decoder.encode(input_and_target[1])

            # TODO - SAME COMMENTS as above regarding validation
            # inputs_one_hot, targets_idx = encoder_decoder.encode(input_and_target[0])

            # inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
            # targets_idx = [word_to_idx[word] for word in targets]
            
            # NBJ comment: 3 dimensional array needed (outer most of size 1). Not sure why pyTorch needs this structure
            inputs_one_hot_3D = [inputs_one_hot]

            # Convert input to tensor
            inputs_one_hot_3D = torch.Tensor(inputs_one_hot_3D)
            #inputs_one_hot = inputs_one_hot.permute(0, 2, 1) # not needed
            
            # Convert target to tensor
            targets_idx = torch.LongTensor(targets_idx)
            
            # Forward pass
            # YOUR CODE HERE!
            outputs = net.forward(inputs_one_hot_3D)
            
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