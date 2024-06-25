import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import SGD
from dataset import KITTI 
from torch.utils.data import DataLoader
from vae import VAE   



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


def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(torch.sigmoid(recon_x), torch.sigmoid(x), reduction='sum')
    kldivergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kldivergence 


model = VAE(capacity=capacity, latent_dim=latent_dim)
model.to(device)

optimizer = SGD(model.parameters(), lr=0.0001)



def train(model, train_loader, epochs):
   for i in range(epochs):
      running_loss = 0.0 

      for n, data in enumerate(train_loader):
          range_image = data["range_image"].to(device).float()
          range_image = range_image.permute(0, 3, 1, 2)
          x_recon, x_mu, x_log_var = model(range_image)  
          loss = vae_loss(x_recon, range_image, x_mu, x_log_var) / 1000
         
          optimizer.zero_grad() 
          loss.backward() 
          optimizer.step() 

          running_loss = running_loss + loss.item() 
    
      print("running_loss = {}, epoch={}".format(running_loss, i+1))

      checkpoint={
            "epoch_number": i+1,
            "model_state": model.state_dict()
        }
      torch.save(checkpoint, "trained_VAE.pth") 



train(model=model, train_loader=train_loader, epochs=200)










     
   




