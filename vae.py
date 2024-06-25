import torch 
import torch.nn as nn 
import torch.nn.functional as F 





class Encoder(nn.Module):
    def __init__(self, capacity, latent_dim):
        super().__init__() 

        self.c = capacity 
        self.latent_dim=latent_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c * 2, kernel_size=4, stride=2, padding=1)  # [B, 128, 16, 256]
        self.fc_mu = nn.Linear(self.c * 2 * 16 * 256, self.latent_dim)
        self.fc_log_var = nn.Linear(self.c * 2 * 16 * 256, self.latent_dim) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x_mu = self.fc_mu(x)
        x_log_var = self.fc_log_var(x)
        return x_mu, x_log_var 




class Decoder(nn.Module):
    def __init__(self, capacity, latent_dim):
        super().__init__()

        self.c = capacity
        self.latent_dim = latent_dim 
        self.fc = nn.Linear(self.latent_dim, self.c * 2 * 16 * 256)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c * 2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=3, kernel_size=4, stride=2, padding=1)


    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.c*2, 16, 256)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv1(x))
        return x 
    


class VAE(nn.Module):
    def __init__(self, capacity, latent_dim):
        super().__init__() 
        self.capacity = capacity 
        self.latent_dim = latent_dim 
        self.encoder = Encoder(capacity=self.capacity, latent_dim=self.latent_dim)
        self.decoder = Decoder(capacity=self.capacity, latent_dim=self.latent_dim)

    def reparametrization(self,mean, variance):
        # the reparameterization trick
            std = variance.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mean)
    
    def forward(self, x):
        latent_mu, latent_log_var = self.encoder(x)
        latent = self.reparametrization(latent_mu, latent_log_var)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_log_var 
    














    


