import open3d as o3d 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import open3d as o3d
import time
import numpy as np
import os 
import matplotlib.pyplot as plt 
import struct 
from dataset import KITTI 
from vae import VAE 
from vae_with_unet import VAE_with_UNet
from projection import LaserScan 




device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


capacity = 64 
latent_dim = 8 
batch_size = 8
T = 1 
m_in = -15 
m_out = -3 
lambda_parameter = 0.1 
n_channels = 3 
n_classes = 2 


kitti360_path = "/media/parvez_alam/Expansion/KITTI360/3d data and labels/data_3d_test_slam/test_3/2013_05_28_drive_0002_sync/velodyne_points/data"
validation_pcd_path = "/media/parvez_alam/Expansion/Denoise validation data/test_3/pcd"
validation_label_path = "/media/parvez_alam/Expansion/Denoise validation data/test_3/label"




model_vae = VAE_with_UNet(capacity=capacity, latent_dim=latent_dim, n_channels=n_channels, n_classes=n_classes)
model_vae.load_state_dict(torch.load("model_state_vae_with_unet.pth"))
model_vae.to(device)
model_vae.eval()





pcd_files = sorted(os.listdir(kitti360_path)) 

for i in range(len(pcd_files)):
    file_name = pcd_files[i]
    file_path = os.path.join(kitti360_path, pcd_files[i])
    if file_name.endswith(".pcd"):
        continue 
    laserscan = LaserScan(project=True)  
    laserscan.open_scan(file_path)
    points = laserscan.proj_xyz 
    positive_points = points 

    points = torch.from_numpy(points).permute(2, 0, 1) 
    temp_points = points.unsqueeze(dim=0).to(device).float()
    x_return, x_mu, x_log_var, logit_output = model_vae(temp_points) 

    negative_points = x_return[0].permute(1,2,0).cpu().detach().numpy()
    
    labels = np.hstack((np.ones(32*1024), np.zeros(32*1024)))
    
    positive_points = positive_points[0:32, :, :].reshape(-1, 3)
    negative_points = negative_points[32:64, :, :].reshape(-1, 3)

    points = np.vstack((positive_points, negative_points))

    pcd_file_path = os.path.join(validation_pcd_path, file_name.split(".")[0]+".pcd")
    label_file_path = os.path.join(validation_label_path, file_name.split(".")[0]+"txt")

    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) 

   
    o3d.io.write_point_cloud(pcd_file_path, pcd)
    

    with open(label_file_path, "w") as f:
        for n in range(len(labels)):
            f.write(str(int(labels[n])))
            f.write(" ")
        f.close() 


    







    

    
    








    
