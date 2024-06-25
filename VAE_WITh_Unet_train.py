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
from vae_with_unet import VAE_with_UNet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



detection_train_path = "/home/parvez_alam/Data/Kitti/Object/data_object_velodyne/training/velodyne"
detection_test_path = "/home/parvez_alam/Data/Kitti/Object/data_object_velodyne/testing/velodyne"
tracking_train_path = "/home/parvez_alam/Data/Kitti/Tracking/data_tracking_velodyne/training/velodyne"
tracking_test_path = "/home/parvez_alam/Data/Kitti/Tracking/data_tracking_velodyne/testing/velodyne" 



train_data = KITTI(detection_train_path=detection_train_path, detection_test_path=detection_test_path, 
             tracking_train_path=tracking_train_path, tracking_test_path=tracking_test_path)
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=False)



capacity = 64 
latent_dim = 8 
n_channels = 3 
n_classes = 2 
T = 1
m_in = -15 




model = VAE_with_UNet(capacity=capacity, latent_dim=latent_dim, n_channels=n_channels, n_classes=n_classes)
model.to(device)

optimizer = SGD(model.parameters(), lr=0.0001)


training_loss = []

def train(model, train_loader, epochs):
   def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(torch.sigmoid(recon_x), torch.sigmoid(x), reduction='sum')
    kldivergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kldivergence 

   for i in range(epochs):
      running_loss = 0.0 

      for n, data in enumerate(train_loader):
          range_image = data["range_image"].to(device).float()
          range_image = range_image.permute(0, 3, 1, 2)
          x_recon, x_mu, x_log_var, logit_output = model(range_image)  
          vae_loss_ = vae_loss(x_recon, range_image, x_mu, x_log_var) / 1000
          
          exp_positive_score = torch.exp(-logit_output)
          exp_sum = exp_positive_score.sum(dim=1).unsqueeze(dim=1)
          negative_score_entropy = exp_positive_score / exp_sum 

          B, _, _, _ = logit_output.shape

          negative_ground_truth = torch.zeros(B, 2, 64, 1024)
          negative_ground_truth[:, 1, :, :] = torch.ones(64, 1024) 
          negative_ground_truth = negative_ground_truth.to(device).float()

          binary_cross_entropy_loss = F.binary_cross_entropy(negative_score_entropy, negative_ground_truth) 

          positive_energy = torch.log(torch.exp(logit_output / T).sum(dim=1))
          poistive_energy = -T * positive_energy 
          
          positive_energy_loss = torch.maximum(torch.zeros(B, 64, 1024).to(device), m_in - positive_energy)**2 
          
          energy_loss = torch.mean(positive_energy_loss) 
          
          loss = vae_loss_ + binary_cross_entropy_loss + energy_loss 
         
          optimizer.zero_grad() 
          loss.backward() 
          optimizer.step() 

          running_loss = running_loss + loss.item() 
    
      print("running_loss = {}, epoch={}".format(running_loss, i+1))
      training_loss.append(running_loss)
      checkpoint={
            "epoch_number": i+1,
            "model_state": model.state_dict()
        }
      torch.save(checkpoint, "VAE_with_Unet.pth") 



train(model=model,  train_loader=train_loader, epochs=200)


plt.plot(np.arange(200)+1, training_loss, 'g')
plt.xlabel("Number of epochs")
plt.ylabel("Training loss")
plt.title("Training Loss Curve")
plt.show() 











     
   

