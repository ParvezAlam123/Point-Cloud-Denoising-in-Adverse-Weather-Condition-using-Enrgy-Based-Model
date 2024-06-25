import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import SGD
from dataset import KITTI 
from torch.utils.data import DataLoader
from vae import VAE  
from Unet import UNet 
import matplotlib.pyplot as plt 
import numpy as np






class VAE_with_UNet(nn.Module):
    def __init__(self, capacity, latent_dim, n_channels, n_classes):
        super().__init__()

        self.capacity = capacity 
        self.latent_dim = latent_dim 
        self.n_channels = n_channels 
        self.n_classes = n_classes 

        self.vae = VAE(capacity = self.capacity, latent_dim=latent_dim)
        self.unet = UNet(n_channels = self.n_channels, n_classes=self.n_classes)

    def forward(self, x):
        x_recon, x_mu, x_log_var = self.vae(x)
        logit_output = self.unet(x_recon)
        return x_recon, x_mu, x_log_var, logit_output 
    




