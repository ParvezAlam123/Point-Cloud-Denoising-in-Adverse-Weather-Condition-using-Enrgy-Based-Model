import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from torch.optim import SGD
from dataset import KITTI, KITTI_NOISE 
from torch.utils.data import DataLoader
from vae import VAE 
from Unet import  UNet 
import matplotlib.pyplot as plt 
from vae_with_unet import VAE_with_UNet 





device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



detection_train_path = "/home/parvez_alam/Data/Kitti/Object/data_object_velodyne/training/velodyne"
detection_test_path = "/home/parvez_alam/Data/Kitti/Object/data_object_velodyne/testing/velodyne"
tracking_train_path = "/home/parvez_alam/Data/Kitti/Tracking/data_tracking_velodyne/training/velodyne"
tracking_test_path = "/home/parvez_alam/Data/Kitti/Tracking/data_tracking_velodyne/testing/velodyne" 


capacity = 64 
latent_dim = 8 
batch_size = 8
T = 1 
m_in = -15 
m_out = -3 
lambda_parameter = 0.1 
n_channels = 3 
n_classes = 2 





 


#loaded_checkpoint = torch.load("trained_VAE_with_Unet.pth")
#model_parameters = loaded_checkpoint["model_state"]
#torch.save(model_parameters, "model_stae_vae_with_unet") 

#model_vae_with_unet = VAE_with_UNet(capacity=capacity, latent_dim=latent_dim, n_channels=n_channels, n_classes=n_classes)
#model_vae_with_unet.load_state_dict(torch.load("model_state_vae_with_unet"))
#model_vae_with_unet.to(device)
#model_vae_with_unet.eval() 



model_vae = VAE(capacity=capacity, latent_dim=latent_dim)
model_vae.load_state_dict(torch.load("model_state_vae.pth"))
model_vae.to(device)
model_vae.eval()



train_ds = KITTI_NOISE(generator_model=model_vae, detection_train_path=detection_train_path, 
                       detection_test_path=detection_test_path, tracking_train_path=tracking_train_path, tracking_test_path=tracking_test_path)

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)



model = UNet(n_channels=3, n_classes=2)
model.to(device)




optimizer = SGD(model.parameters(), lr=0.0001)





training_loss = []

def train(model, train_loader, epochs):
   for i in range(epochs):
      running_loss = 0.0 

      for n, data in enumerate(train_loader): 
          range_image = data["range_image"].to(device).float()
          negative_range_image = data["negative_range_image"].to(device).float()
          positive_score = model(range_image)
          negative_score = model(negative_range_image) 

          exp_positive_score = torch.exp(-positive_score)
          exp_sum = exp_positive_score.sum(dim=1).unsqueeze(dim=1)
          positive_score_entropy = exp_positive_score / exp_sum 

          exp_negative_score = torch.exp(-negative_score)
          exp_sum = exp_negative_score.sum(dim=1).unsqueeze(dim=1)
          negative_score_entropy = exp_negative_score / exp_sum 

          B, _, _, _ = positive_score.shape

          ground_truth_positive = torch.zeros(B, 2, 64, 1024)
          ground_truth_positive[:, 0, :, :] = torch.ones(64, 1024)

          ground_truth_positive = ground_truth_positive.to(device).float()  

          ground_truth_negative = torch.zeros(B, 2, 64, 1024)
          ground_truth_negative[:, 1, :, :] = torch.ones(64, 1024)
          ground_truth_negative = ground_truth_negative.to(device)
          
          

          binary_cross_entropy_loss_positive_sample = F.binary_cross_entropy(positive_score_entropy, ground_truth_positive)
          binary_cross_entropy_loss_negative_sample = F.binary_cross_entropy(negative_score_entropy, ground_truth_negative)
          cross_entropy_loss = binary_cross_entropy_loss_positive_sample + binary_cross_entropy_loss_negative_sample 

          positive_energy = torch.log(torch.exp(positive_score / T).sum(dim=1))
          negative_energy = torch.log(torch.exp(negative_score / T).sum(dim=1))
          poistive_energy = -T * positive_energy 
          negative_energy = -T * negative_energy  
          
          positive_energy_loss = torch.maximum(torch.zeros(B, 64, 1024).to(device), m_in - positive_energy)**2 
          negative_energy_loss =  torch.maximum(torch.zeros(B, 64, 1024).to(device), negative_energy - m_out)**2 
          energy_loss = torch.mean(positive_energy_loss) + torch.mean(negative_energy_loss) 
         
          loss = cross_entropy_loss + lambda_parameter * energy_loss

                
        
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
      torch.save(checkpoint, "trained_unet_with_vae_as_negative_sample.pth") 




train(model=model, train_loader=train_loader, epochs=40)

plt.plot(np.arange(40)+1, training_loss, 'g')
plt.xlabel("Number of epochs")
plt.ylabel("Training loss")
plt.title("Training Loss Curve")
plt.show() 













