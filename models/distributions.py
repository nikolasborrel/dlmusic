import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
import matplotlib.pyplot as plt 
from utils.tools import flatten
import numpy as np
from collections import Counter

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        #m = torch.distributions.normal.Normal(self.mu, self.sigma * self.sample_epsilon())
        return self.mu + self.sigma*self.sample_epsilon() #m.sample()
        
    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        m = torch.distributions.normal.Normal(self.mu, self.sigma)    
        return m.log_prob(z)


def get_histograms_from_dataloader(DataLoader, vocab_size=14, plot=True):

    mel_stats = []
    bass_stats = []

    for inputs_one_hot, targets_idx in DataLoader:
        a = inputs_one_hot.detach().numpy()
        b = targets_idx.detach().numpy()

        batch_idxs = []
        for batch in a:
            note_idxs = []
            for note in batch:
                note_idxs.append(np.argmax(note))
            batch_idxs.append(note_idxs)
        mel_stats.append(batch_idxs)
        bass_stats.append(b.ravel())

    mel_notes = flatten(flatten(mel_stats))
    bass_notes = flatten(bass_stats)

    if plot:
        plt.figure()
        fig, axs = plt.subplots(nrows=1,ncols=2)
        axs[0].hist(mel_notes, bins=range(0,vocab_size))
        axs[0].set_title('Melody')
        axs[0].set_xlabel('')

        axs[1].hist(bass_notes, bins=range(0,vocab_size))
        axs[1].set_title('Bass')
        plt.show()
        print('Melody histogram: ', Counter(mel_notes))
        print('Bass histogram: ', Counter(bass_notes))
    
    return mel_notes, bass_notes

