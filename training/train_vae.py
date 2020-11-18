from collections import defaultdict
import torch
import numpy as np
from utils.plotting import make_vae_plots
from models.model_vae import VariationalAutoencoder, VariationalInference

def train_vae(train_loader, test_loader, num_epochs, data_shape) -> None:    
    # define the models, evaluator and optimizer

    # VAE
    latent_features = 2
    vae = VariationalAutoencoder(data_shape, latent_features)

    # Evaluator: Variational Inference
    beta = 1
    vi = VariationalInference(beta=beta)

    # The Adam optimizer works really well with VAEs.
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # define dictionary to store the training curves
    training_data = defaultdict(list)
    validation_data = defaultdict(list)

    epoch = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">> Using device: {device}")

    # move the model to the device
    vae = vae.to(device)

    # training..
    while epoch < num_epochs:
        epoch+= 1
        training_epoch_data = defaultdict(list)
        vae.train()
        
        # Go through each batch in the training dataset using the loader
        # Note that y is not necessarily known as it is here
        for x, y in train_loader:
            x = x.to(device)
            
            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = vi(vae, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # gather data for the current bach
            for k, v in diagnostics.items():
                training_epoch_data[k] += [v.mean().item()]
                

        # gather data for the full epoch
        for k, v in training_epoch_data.items():
            training_data[k] += [np.mean(training_epoch_data[k])]

        # Evaluate on a single batch, do not propagate gradients
        with torch.no_grad():
            vae.eval()
            
            # Just load a single batch from the test loader
            x, y = next(iter(test_loader))
            x = x.to(device)
            
            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = vi(vae, x)
            
            # gather data for the validation step
            for k, v in diagnostics.items():
                validation_data[k] += [v.mean().item()]
        
        # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
        make_vae_plots(vae, x, y, outputs, training_data, validation_data, native_plot=True)

    