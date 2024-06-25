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




class NonBlockVisualizer:
    def __init__(self, point_size=1.5, background_color=[0, 0, 0]):
        self.__visualizer = o3d.visualization.Visualizer()
        self.__visualizer.create_window()
        opt = self.__visualizer.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt = self.__visualizer.get_render_option()
        opt.point_size = point_size

        self.__pcd_vis = o3d.geometry.PointCloud()
        self.__initialized = False
        
        self.view_control = self.__visualizer.get_view_control()
        
       
        
        
        
        
        

    def update_renderer(self, pcd, wait_time=0):
        self.__pcd_vis.points = pcd.points
        self.__pcd_vis.colors = pcd.colors

        if not self.__initialized:
            self.__initialized = True
            self.__visualizer.add_geometry(self.__pcd_vis)
        else:
            self.__visualizer.update_geometry(self.__pcd_vis)
            
            
        
        #self.view_control.set_up(np.array([0, -1, 1]))
        #self.view_control.set_front(np.array([0, -0.5, -1]))
        #self.view_control.set_lookat([0,0,5])
        #self.view_control.set_zoom(0.1)
        
        self.__visualizer.poll_events()
        self.__visualizer.update_renderer()
        
        



obj = NonBlockVisualizer()



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



detection_train_path = "/home/parvez_alam/Data/Kitti/Object/data_object_velodyne/training/velodyne"
detection_test_path = "/home/parvez_alam/Data/Kitti/Object/data_object_velodyne/testing/velodyne"
tracking_train_path = "/home/parvez_alam/Data/Kitti/Tracking/data_tracking_velodyne/training/velodyne"
tracking_test_path = "/home/parvez_alam/Data/Kitti/Tracking/data_tracking_velodyne/testing/velodyne" 



train_data = KITTI(detection_train_path=detection_train_path, detection_test_path=detection_test_path, 
             tracking_train_path=tracking_train_path, tracking_test_path=tracking_test_path)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)



capacity = 64 
latent_dim = 8 
n_channels=3 
n_classes = 2 



loaded_checkpoint = torch.load("VAE_with_Unet.pth") 
model_parameters = loaded_checkpoint["model_state"]
torch.save(model_parameters, "model_state_vae_with_unet.pth")




model = VAE_with_UNet(capacity=capacity, latent_dim=latent_dim, n_channels=n_channels, n_classes=n_classes)
model.load_state_dict(torch.load("model_state_vae_with_unet.pth"))
model.to(device)
model.eval()






def visualization(model): 
        for n, data in enumerate(train_loader):
           range_image = data["range_image"].to(device).float()
           range_image = range_image.permute(0, 3, 1, 2)
           x_recon, x_mu, x_log_var, logit_output = model(range_image)  
           x_recon = x_recon[0].permute(1,2,0).cpu()
           points = x_recon.detach().reshape(-1, 3).numpy()
           
       
           pcd = o3d.geometry.PointCloud()
           pcd.points = o3d.utility.Vector3dVector(points)
        
           obj.update_renderer(pcd)
           time.sleep(0.18)
        



visualization(model)


