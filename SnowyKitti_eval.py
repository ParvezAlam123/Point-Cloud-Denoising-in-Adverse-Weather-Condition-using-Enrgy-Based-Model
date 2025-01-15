#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs import point_cloud2 
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2  
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import SGD, Adam
from  vit_backbone import Network 
import matplotlib.pyplot as plt 
from Unet import UNet 

import open3d as o3d
import time
import os 
import struct 
import math

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2 
import time 

import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



embedding_dim = 256 
patch_height = 4 
patch_width = 8 
n_heads = 8 
centroid_1 = 50 
centroid_2 = 100
n_patches = int((64//4)*(1024//8)) 


model = Network(embedding_dim=embedding_dim,patch_height=patch_height, patch_width=patch_width,n_patches=n_patches, n_heads=n_heads)

model.load_state_dict(torch.load("model_state_Score.pth", map_location=device))
model.to(device)
model.eval()



model_UNet = UNet(n_channels=3, n_classes=2)

model_UNet.load_state_dict(torch.load("model_state_unet_with_vae_as_negative_sample.pth", map_location=device))
model_UNet.to(device)
model_UNet.eval()

T = 1

TP = 0 
TN = 0 
FP = 0 
FN = 0 
energy_threshold = -1.9
score_threshold = 70 
radius_threshold = 2.0 




class LaserScan2:
       """ class that contains laserscan x y, z, r"""

       EXTENSIONS_SCAN = [".bin"]

       def __init__(self, project=False, H=64, W=1024, fov_up=15, fov_down=-15):
           self.project = project 
           self.H = H 
           self.W = W 
           self.proj_fov_up = fov_up 
           self.proj_fov_down = fov_down 
           #self.num_features = num_features
           self.reset()



       def reset(self):
           """ Reset scan members""" 
           self.points = np.zeros((0, 3), dtype=np.float32)     # [N, 3]
           self.remissions = np.zeros((0, 1), dtype=np.float32) # [N, 1]

           # projected range image [H, W] (-1 is no data)
           self.proj_range = np.full((self.H, self.W), -1, dtype=np.float32)

           # unprojected range -(list of depth for each point)
           self.unproj_range = np.zeros((0, 1), dtype=np.float32)

           # projected point cloud xyz-[h,w,3] (-1 means no data)
           self.proj_xyz = np.full((self.H, self.W, 3), 0, dtype=np.float32)

           # projected remission - [H, W] (-1 means no data)
           self.proj_remission = np.full((self.H, self.W), -1, dtype=np.float32)

           # projected index (for each pixel in range image what I am in point cloud); [H, W] (-1 means no data)
           self.proj_idx = np.full((self.H, self.W), -1, dtype=np.int32) 

           # for each point where it is in range image [N, 1]
           self.proj_x = np.zeros((0, 1), dtype=np.float32)
           self.proj_y = np.zeros((0, 1), dtype=np.float32)


           # mask containing for each pixel, if it contains a point or not 
           self.proj_mask = np.zeros((self.H, self.W), dtype=np.int32)

    
       def size(self):
           """ return the size of the point cloud"""
           return self.points.shape[0]
    
       def __len__(self):
           return self.size() 
    

       def open_scan(self, velo_points):
           """ open raw scan and fill attributes values"""

           # reset just in case any  structure is open 
           self.reset() 

           # check the filename is proper string type 
           #if not isinstance(filename, str):
           #    raise TypeError("Filename should be string type but found {}".format(str(type(filename))))
        
           # check extension is a laser scan 
           #if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
           #    raise RuntimeError("Filename extension is not valid laser scan")
        
           # if all is well open laser scan 
           #scan = np.fromfile(filename, dtype=np.float32) 
           #scan = scan.reshape((-1, self.num_features))

           # put in  attribute 
           #points = scan[:, 0:3]
           remissions = self.remissions
           points = velo_points 

           #print("hello")
        
           self.set_points(points, remissions)


       def set_points(self,points, remissions=None):
           """ Set scan attribute instead of opening it"""

           # reset any open structure 
           self.reset() 

           # check scan makes sense 
           if not isinstance(points, np.ndarray):
               raise TypeError("Scan should be numpy array")
        
           # check remission make sense 
           if remissions is not None and not isinstance(remissions, np.ndarray):
               raise TypeError("Remissions should be numpy array")
        

           # put the attrubutes 
           self.points = points 
           if self.remissions is not None :
               self.remissions = remissions 
           else:
               self.remissions = np.zeros((points.shape[0]), dtype=np.float32)


           # if projection wanted 
           if self.project:
               self.do_range_projection() 

    
       def do_range_projection(self):
           """ Project a point cloud into a spherical projection image"""

           # laser parameters 
           fov_up = (self.proj_fov_up / 180.0) * np.pi
           fov_down = (self.proj_fov_down / 180.0) * np.pi 
           fov = abs(fov_up) + abs(fov_down)

           # get depth of all the points 
           depth = np.linalg.norm(self.points, 2, axis=1)

           # get scan components 
           scan_x = self.points[:, 0]
           scan_y = self.points[:, 1]
           scan_z = self.points[:, 2]

           # get angle of the projection 
           yaw = np.arctan2(scan_y, scan_x) 
           pitch = np.arcsin(scan_z / depth)

           # get normalized projections 
           proj_x = 0.5 * (yaw / np.pi + 1.0) 
           proj_y = (fov_up - pitch) / fov 
           #for i in range(len(proj_y)):
           #    if proj_y[i] < 0 :
           #        print("pitch = ", pitch[i]*180/np.pi)
           #        print("fov_up=", fov_up*180/np.pi)
           #        break

           # scale to image size using angular resolution 
           proj_x = proj_x * self.W 
           proj_y = proj_y * self.H 

           # round and clamp for use as index 
           proj_x = np.floor(proj_x)
           proj_x = np.minimum(self.W - 1, proj_x)
           proj_x = np.maximum(0, proj_x).astype(np.int32)
           self.proj_x = np.copy(proj_x)    # store a copy in the original order

           proj_y = np.floor(proj_y)
           proj_y = np.minimum(self.H - 1, proj_y)
           proj_y = np.maximum(0, proj_y).astype(np.int32)
           self.proj_y = np.copy(proj_y)   # store a copy in the original order 

           # copy of the depth in original order 
           self.unproj_range = np.copy(depth)


           # order in increaseing depth 
           indices = np.arange(depth.shape[0])
           order = np.argsort(depth)
        
           #print(self.points)
           depth = depth[order]
           indices = indices[order]
           points = self.points[order]
           #remissions = self.remissions[order]
           proj_x = proj_x[order]
           proj_y = proj_y[order]
        
           # assigns to images 
           self.proj_range[proj_y, proj_x] = depth 
           self.proj_xyz[proj_y[0:], proj_x[0:]] = points[0:] 
           #self.proj_remission[proj_y, proj_x] = remissions.flatten() 
           self.proj_idx[proj_y, proj_x] = indices 
           self.proj_mask = (self.proj_idx > 0).astype(np.float32)  


        



class NonBlockVisualizer:
    def __init__(self, point_size=8, background_color=[0, 0, 0]):
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
        
        

        
            
            
            
#obj = NonBlockVisualizer()  



def filter_and_segment_point_cloud(points, bounds, threshold=1):
    """
    Filters the point cloud within the given bounds and segments it into connected components.
    Removes connected components with a number of points below the threshold.

    Args:
        points (np.ndarray): Input point cloud as a numpy array of shape (N, 3).
        bounds (tuple): Bounds as (xmin, ymin, xmax, ymax).
        threshold (int): Minimum number of points in a connected component to keep.

    Returns:
        np.ndarray: Filtered and segmented point cloud.
    """
    xmin, ymin, xmax, ymax = bounds

    # Step 1: Filter points within the bounds
    filtered_points = points[(points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                             (points[:, 1] >= ymin) & (points[:, 1] <= ymax)]   
    

    comp_points = points[~((points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 1] >= ymin) & (points[:, 1] <= ymax))] 

    
    
    #print("filtered_shape = ", filtered_points.shape)
   
    # Step 2: Build a k-d tree for finding connected components
    tree = cKDTree(filtered_points) 

    # Step 3: Find connected components using a distance threshold
    distance_threshold = 0.3  # Adjust as needed for point density
    #print("before")
    connected_indices = tree.query_ball_tree(tree, r=distance_threshold) 
    #print("hello")

    # Step 4: Group points into connected components
    visited = set()
    components = []

    for idx in range(len(filtered_points)):
        if idx not in visited:
            stack = [idx]
            component = []
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(connected_indices[current])
            components.append(component)

    # Step 5: Filter out components with fewer points than the threshold
    valid_components = [comp for comp in components if len(comp) > threshold] 

    if valid_components == [] :
        result_points = np.array([[]]) 
        #print(result_points.shape)
    else:

       # Step 6: Collect the points from valid components
       result_points = np.vstack([filtered_points[comp] for comp in valid_components])

    return result_points, comp_points

        
            



SnowyKitti_path = "/home/parvez/snowyKITTI/dataset/sequences" 

light_TP = 0 
light_FP = 0 
light_FN = 0 
total_light = 0 

medium_TP = 0 
medium_FP = 0 
medium_FN = 0 
total_medium = 0 

heavy_TP = 0 
heavy_FP = 0 
heavy_FN = 0 
total_heavy = 0 



sequences = sorted(os.listdir(SnowyKitti_path))  

#print(len(sequences))


for seq in range(len(sequences)):
    seq_path = os.path.join(SnowyKitti_path, sequences[seq]) 
    data = os.listdir(seq_path) 
    #print("hello")
    time.sleep(5) 
    #print("enter") 
    print(data, seq)
    if 'info.txt' in data:
        info_file_path = os.path.join(seq_path, 'info.txt')
        with open(info_file_path, "r") as f:
            data = f.read().split("\n")
            snowfall_rate = float(data[1])
            terminal_velocity = float(data[4]) 
        print(snowfall_rate, terminal_velocity)
        if snowfall_rate >= 0.5 and snowfall_rate <1.5:
            snow_labels_path = os.path.join(seq_path, "snow_labels")
            snow_velodyne_path = os.path.join(seq_path, "snow_velodyne") 
            snow_labels = sorted(os.listdir(snow_labels_path)) 
            snow_velodyne = sorted(os.listdir(snow_velodyne_path))  
            #print("normal")
            for frame in range(len(snow_velodyne)):
                pcd_path = os.path.join(snow_velodyne_path, snow_velodyne[frame]) 
                label_path = os.path.join(snow_labels_path, snow_labels[frame])
                label = np.fromfile(label_path, dtype=np.int32) 

                #print("hello", f)

                size_float = 4
                list_pcd = []
   
   
                if pcd_path.endswith(".bin"):
                   
                   with open(pcd_path, "rb") as f:
                      byte = f.read(size_float * 4)
                      while byte:
                         x, y, z, intensity = struct.unpack("ffff", byte)
                         list_pcd.append([x, y, z])
                         byte = f.read(size_float * 4)
        
                points = np.array(list_pcd) 
                
                # get the noisy points from the data 
                snowy_points = points[label == 1]
                N, _ = snowy_points.shape 
                total_light = total_light + N  

                # apply ring radius filtering 

                range_cal = np.sqrt(snowy_points[:, 0]**2+snowy_points[:, 1]**2+snowy_points[:, 2]**2)
                range_mask = range_cal >= radius_threshold  

                filtered_points = snowy_points[range_mask] 

                light_TP = light_TP + (N - filtered_points.shape[0]) 

                N, _ = filtered_points.shape 

    
                # Convert to numpy array
                points_array = filtered_points   

    



                proj_obj = LaserScan2(project=True)  
                proj_obj.open_scan(points_array) 

    
                frame_tensor = torch.from_numpy(proj_obj.proj_xyz).unsqueeze(dim=0).to(device).float()  
                start_time = time.time() 
    

    

    
                range_image_input = frame_tensor.permute(0, 3, 1, 2)
                label = np.ones((64, 1024)) * -1  

                #start_time = time.time() 
                #with torch.no_grad():
                energy_score = -T * (torch.log(torch.exp(model_UNet(range_image_input)) / T).sum(dim=1)) 
    
                #print(np.max(energy_score[0].cpu().detach().numpy()))
                energy_values = []
                for i in range(64):
                   for j in range(1024):
                      energy_value = energy_score[0][i][j] 
                      if energy_value.cpu().detach().item() <= energy_threshold:
                         label[i][j] = 1 
                      else:
                         pass 
                      energy_values.append(energy_value.cpu().detach().item()) 
            
                mask = label == 1
    
                filtered_points = proj_obj.proj_xyz[mask].reshape((-1, 3))   

                # delete the points having [0, 0, 0]
                # Filtering points where coordinates are not [0, 0, 0]
                filtered_points = filtered_points[~np.all(filtered_points == [0, 0, 0], axis=1)]  
                #print("filtered_points.shape", filtered_points.shape)

                light_TP = light_TP + (N - filtered_points.shape[0]) 

                N, _ = filtered_points.shape  


                #range_cal = np.sqrt(filtered_points[:, 0]**2+filtered_points[:, 1]**2+filtered_points[:, 2]**2)
                #range_mask = range_cal >= radius_threshold 
    
                #filtered_points = filtered_points[range_mask]

                real_points = proj_obj.proj_xyz.reshape((-1, 3))
    
                start_time = time.time() 
                # Define the bounding box
                bounds = (-70, -40, 70, 40)
                

                
                # Process the point cloud
                filtered_points, comp_points = filter_and_segment_point_cloud(filtered_points, bounds, threshold=10) 
                if filtered_points.shape[1] == 0:
                    light_TP = light_TP + N 
                    light_FP = light_FP + 0 
                else:
                   light_TP = light_TP + (N - filtered_points.shape[0]) 
                   light_FP = light_FP + filtered_points.shape[0]
                

                #filtered_points = np.vstack((filtered_points, comp_points)) 

                #print("hello", frame)

        #print("after normal")
        if snowfall_rate >= 1.5 and snowfall_rate <2.5:
            snow_labels_path = os.path.join(seq_path, "snow_labels")
            snow_velodyne_path = os.path.join(seq_path, "snow_velodyne") 
            snow_labels = sorted(os.listdir(snow_labels_path)) 
            snow_velodyne = sorted(os.listdir(snow_velodyne_path))  
            #print("middle")
            for frame in range(len(snow_velodyne)):
                pcd_path = os.path.join(snow_velodyne_path, snow_velodyne[frame]) 
                label_path = os.path.join(snow_labels_path, snow_labels[frame])
                label = np.fromfile(label_path, dtype=np.int32) 

                size_float = 4
                list_pcd = []
   
   
                if pcd_path.endswith(".bin"):
                   
                   with open(pcd_path, "rb") as f:
                      byte = f.read(size_float * 4)
                      while byte:
                         x, y, z, intensity = struct.unpack("ffff", byte)
                         list_pcd.append([x, y, z])
                         byte = f.read(size_float * 4)
        
                points = np.array(list_pcd) 
                
                # get the noisy points from the data 
                snowy_points = points[label == 1]
                N, _ = snowy_points.shape 
                total_medium = total_medium + N  

                # apply ring radius filtering 

                range_cal = np.sqrt(snowy_points[:, 0]**2+snowy_points[:, 1]**2+snowy_points[:, 2]**2)
                range_mask = range_cal >= radius_threshold  

                filtered_points = snowy_points[range_mask] 

                medium_TP = medium_TP + (N - filtered_points.shape[0]) 

                N, _ = filtered_points.shape 

    
                # Convert to numpy array
                points_array = filtered_points   

    



                proj_obj = LaserScan2(project=True)  
                proj_obj.open_scan(points_array) 

    
                frame_tensor = torch.from_numpy(proj_obj.proj_xyz).unsqueeze(dim=0).to(device).float()  
                start_time = time.time() 
    

    

    
                range_image_input = frame_tensor.permute(0, 3, 1, 2)
                label = np.ones((64, 1024)) * -1  

                start_time = time.time() 

                #with torch.no_grad():
                energy_score = -T * (torch.log(torch.exp(model_UNet(range_image_input)) / T).sum(dim=1)) 
    
    
    
                energy_values = []
                for i in range(64):
                   for j in range(1024):
                      energy_value = energy_score[0][i][j] 
                      if energy_value.cpu().detach().item() <= energy_threshold:
                         label[i][j] = 1 
                      else:
                         pass 
                      energy_values.append(energy_value.cpu().detach().item()) 
            
                mask = label == 1
    
                filtered_points = proj_obj.proj_xyz[mask].reshape((-1, 3))  
                #delete zero poiints 

                filtered_points = filtered_points[~np.all(filtered_points == [0, 0, 0], axis=1)]   

                medium_TP = medium_TP + (N - filtered_points.shape[0]) 

                N, _ = filtered_points.shape 

                #range_cal = np.sqrt(filtered_points[:, 0]**2+filtered_points[:, 1]**2+filtered_points[:, 2]**2)
                #range_mask = range_cal >= radius_threshold 
    
                #filtered_points = filtered_points[range_mask]

                real_points = proj_obj.proj_xyz.reshape((-1, 3))
    

                start_time = time.time() 
                # Define the bounding box
                bounds = (-70, -40, 70, 40)

                # Process the point cloud
                filtered_points, comp_points = filter_and_segment_point_cloud(filtered_points, bounds, threshold=10)  
                if filtered_points.shape[1] == 0:
                    medium_TP = medium_TP + N 
                    medium_FP = medium_FP + 0 
                else:
                   medium_TP = medium_TP + (N - filtered_points.shape[0]) 
                   medium_FP = medium_FP + filtered_points.shape[0]


                #filtered_points = np.vstack((filtered_points, comp_points)) 
        
        #print("after_middle")
        if snowfall_rate >= 2.5 and snowfall_rate <=3.0:
            snow_labels_path = os.path.join(seq_path, "snow_labels")
            snow_velodyne_path = os.path.join(seq_path, "snow_velodyne") 
            snow_labels = sorted(os.listdir(snow_labels_path)) 
            snow_velodyne = sorted(os.listdir(snow_velodyne_path)) 
            #print("heavy") 
            for frame in range(len(snow_velodyne)):
                pcd_path = os.path.join(snow_velodyne_path, snow_velodyne[frame]) 
                label_path = os.path.join(snow_labels_path, snow_labels[frame])
                label = np.fromfile(label_path, dtype=np.int32) 

                size_float = 4
                list_pcd = []
   
   
                if pcd_path.endswith(".bin"):
                   
                   with open(pcd_path, "rb") as f:
                      byte = f.read(size_float * 4)
                      while byte:
                         x, y, z, intensity = struct.unpack("ffff", byte)
                         list_pcd.append([x, y, z])
                         byte = f.read(size_float * 4)
        
                points = np.array(list_pcd) 
                
                # get the noisy points from the data 
                snowy_points = points[label == 1]
                N, _ = snowy_points.shape 
                total_heavy = total_heavy + N  

                # apply ring radius filtering 

                range_cal = np.sqrt(snowy_points[:, 0]**2+snowy_points[:, 1]**2+snowy_points[:, 2]**2)
                range_mask = range_cal >= radius_threshold  

                filtered_points = snowy_points[range_mask] 

                heavy_TP = heavy_TP + (N - filtered_points.shape[0]) 

                N, _ = filtered_points.shape 

    
                # Convert to numpy array
                points_array = filtered_points   

    



                proj_obj = LaserScan2(project=True)  
                proj_obj.open_scan(points_array) 

    
                frame_tensor = torch.from_numpy(proj_obj.proj_xyz).unsqueeze(dim=0).to(device).float()  
                start_time = time.time() 
    

    

    
                range_image_input = frame_tensor.permute(0, 3, 1, 2)
                label = np.ones((64, 1024)) * -1  

                start_time = time.time() 

                #with torch.no_grad():
                energy_score = -T * (torch.log(torch.exp(model_UNet(range_image_input)) / T).sum(dim=1)) 
    
    
    
                energy_values = []
                for i in range(64):
                   for j in range(1024):
                      energy_value = energy_score[0][i][j] 
                      if energy_value.cpu().detach().item() <= energy_threshold:
                         label[i][j] = 1 
                      else:
                         pass 
                      energy_values.append(energy_value.cpu().detach().item()) 
            
                mask = label == 1
    
                filtered_points = proj_obj.proj_xyz[mask].reshape((-1, 3)) 
                # delet zero points
                filtered_points = filtered_points[~np.all(filtered_points == [0, 0, 0], axis=1)]  

                heavy_TP = heavy_TP + (N - filtered_points.shape[0]) 

                N, _ = filtered_points.shape 

                #range_cal = np.sqrt(filtered_points[:, 0]**2+filtered_points[:, 1]**2+filtered_points[:, 2]**2)
                #range_mask = range_cal >= radius_threshold 
    
                #filtered_points = filtered_points[range_mask]

                real_points = proj_obj.proj_xyz.reshape((-1, 3))
    

                start_time = time.time() 
                # Define the bounding box
                bounds = (-70, -40, 70, 40)

                # Process the point cloud
                filtered_points, comp_points = filter_and_segment_point_cloud(filtered_points, bounds, threshold=10)  
                
                if filtered_points.shape[1] == 0:
                    heavy_TP  = heavy_TP + N 
                    heavy_FP = heavy_FP + 0 
                else:
                   heavy_TP = heavy_TP + (N - filtered_points.shape[0]) 
                   heavy_FP = heavy_FP + filtered_points.shape[0]


                #filtered_points = np.vstack((filtered_points, comp_points))            
        #print("after heavy")
    else:
        continue  

    #print("hello")



IOU_light = light_TP / (light_TP + light_FN + light_FP) 
IOU_medium = medium_TP / (medium_TP + medium_FN + medium_FP) 
IOU_heavy = heavy_TP / (heavy_TP + heavy_FN + heavy_FP) 


print("IOU_light = ", IOU_light)
print("IOU_medium = ", IOU_medium) 
print("IOU_heavy = ", IOU_heavy) 




