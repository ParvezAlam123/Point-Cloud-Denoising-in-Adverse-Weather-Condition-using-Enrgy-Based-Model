import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import open3d as o3d 
import struct 
from dataset import SemanticSTF, RB_Data
from Unet import UNet 
import time 
import matplotlib.pyplot as plt 






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

T = 1


velo_test_path = "/media/parvez/Expansion/TiHAN AWDS/Fog/Scene_0"


train_ds = RB_Data(velo_test_path)
train_loader = DataLoader(dataset=train_ds, batch_size=1, shuffle=False)





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



#loaded_checkpoint = torch.load("trained_UNet_with_negative_points.pth") 
#model_parameters = loaded_checkpoint["model_state"]
#torch.save(model_parameters, "model_state_UNet_with_negative_points.pth")



model = UNet(n_channels=3, n_classes=2)

#model.load_state_dict(torch.load("model_state_unet.pth"))
model.to(device)
#model.eval()



for n, data in enumerate(train_loader):  
    range_image = data["range_image"].to(device).float() 
    range_image_input = range_image.permute(0, 3, 1, 2)
    energy_score = -T * (torch.log(torch.exp(model(range_image_input)) / T).sum(dim=1)) 
    energy_score = energy_score[0].reshape(1, -1)[0].cpu().detach().numpy()
       
    #plt.plot(np.arange(len(energy_score)), energy_score)
    #plt.xlabel("PCD Points")
    #plt.ylabel("Energy Score Value")
    #plt.show()


    mask_index = energy_score <= -2
    #mask_index = mask_index.cpu().detach().numpy()
    points = range_image[0].cpu().detach().reshape(-1, 3).numpy()
    filtered_points = points[mask_index] 

    range = np.sqrt((filtered_points[:, 0])**2+ (filtered_points[:, 1])**2 + (filtered_points[:, 2])**2)
    range_index = range>=2.5 
    filtered_points = filtered_points[range_index]



    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(points)
    obj.update_renderer(pcd)
    time.sleep(1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
    obj.update_renderer(pcd)
    time.sleep(1) 

    





   


    
    



     





 




    






    












        






