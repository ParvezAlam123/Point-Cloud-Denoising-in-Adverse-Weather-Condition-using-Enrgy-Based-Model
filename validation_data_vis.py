import os 
import numpy as np 
import torch 
import torch.nn as  nn 
import struct 
import open3d as o3d 
import time


pcd_path = "/media/parvez_alam/Expansion/Denoise validation data/test_0/pcd"
label_path = "/media/parvez_alam/Expansion/Denoise validation data/test_0/label"





class NonBlockVisualizer:
    def __init__(self, point_size=1, background_color=[0, 0, 0]):
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




pcd_files = sorted(os.listdir(pcd_path))
label_files = sorted(os.listdir(label_path))



for i in range(len(pcd_files)):
    pcd_file_path = os.path.join(pcd_path, pcd_files[i])
    label_file_path = os.path.join(label_path, label_files[i])

    # Load point cloud file
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points = np.asarray(pcd.points)
    
    labels_list = [] 
    with open(label_file_path, "r") as f :
        labels = f.read() 
        n = 0 
        while (n < len(labels)):
            labels_list.append(int(labels[n])) 
            n  = n + 2 

        
    labels = np.array(labels_list)
    

    N, _ = points.shape 
    colors = np.zeros((N, 3)) 
    for col in range(N):
       lab = labels[col]
       if lab == 1:
           color = [0, 0, 255]
       else:
           color = [255, 0, 0]
       colors[col] = color 
       
    
     
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) 
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors)) 
    
        
    obj.update_renderer(pcd)
    time.sleep(0.1) 
        

        
        




