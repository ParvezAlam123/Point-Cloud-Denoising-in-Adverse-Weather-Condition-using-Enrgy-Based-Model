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

model.load_state_dict(torch.load("model_state_Score.pth"))
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
energy_threshold = -1.5
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

    

    # Step 2: Build a k-d tree for finding connected components
    tree = cKDTree(filtered_points)
    
    # Step 3: Find connected components using a distance threshold
    distance_threshold = 0.3  # Adjust as needed for point density
    connected_indices = tree.query_ball_tree(tree, r=distance_threshold)

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

    # Step 6: Collect the points from valid components
    result_points = np.vstack([filtered_points[comp] for comp in valid_components])

    return result_points, comp_points

        
            
        


def numpy_to_pointcloud2(points, frame_id="base_link"):
    """
    Converts a numpy array of shape (N, 3) or (N, 4) into a PointCloud2 message.
    
    Args:
        points (np.ndarray): Nx3 or Nx4 array (x, y, z, [intensity]).
        frame_id (str): Frame of reference for the point cloud.
    
    Returns:
        sensor_msgs.msg.PointCloud2: PointCloud2 message.
    """
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    
    if points.shape[1] == 4:  # Add intensity if available
        fields.append(PointField('intensity', 12, PointField.FLOAT32, 1))
    
    header = rospy.Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now()
    
    return pc2.create_cloud(header, fields, points)






 
def callback(msg): 

    #path = "/media/parvez/Expansion/TiHAN AWDS/Rain/Scene_0/1652867705.663807869.pcd" 
    #pcd = o3d.io.read_point_cloud(path)
    #points_array = np.asarray(pcd.points)
    # Convert PointCloud2 message to list of points (x, y, z)
    pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    
    # Convert to numpy array
    points_array = np.array(list(pc_data)) 
    



    proj_obj = LaserScan2(project=True)  
    proj_obj.open_scan(points_array) 

    
    frame_tensor = torch.from_numpy(proj_obj.proj_xyz).unsqueeze(dim=0).to(device).float()  
    start_time = time.time() 
    

    """with open("/home/parvez/score.text", "w") as f:
        f.write(str(score.item())) 
        f.write("\n")"""

    
    range_image_input = frame_tensor.permute(0, 3, 1, 2)
    label = np.ones((64, 1024)) * -1  

    start_time = time.time() 

    #with torch.no_grad():
    energy_score = -T * (torch.log(torch.exp(model_UNet(range_image_input)) / T).sum(dim=1)) 
    
    print("UNet_Model= ", time.time() - start_time)
    
    energy_values = []
    for i in range(64):
        for j in range(1024):
            energy_value = energy_score[0][i][j] 
            if energy_value.cpu().detach().item() <= energy_threshold:
               label[i][j] = 1 
            else:
                pass 
            energy_values.append(energy_value.cpu().detach().item()) 
            #print(energy_value)
    mask = label == 1
    
    filtered_points = proj_obj.proj_xyz[mask].reshape((-1, 3))  
    range_cal = np.sqrt(filtered_points[:, 0]**2+filtered_points[:, 1]**2+filtered_points[:, 2]**2)
    range_mask = range_cal >= radius_threshold 
    
    filtered_points = filtered_points[range_mask]

    real_points = proj_obj.proj_xyz.reshape((-1, 3))
    

    start_time = time.time() 
    # Define the bounding box
    bounds = (-70, -40, 70, 40)

    # Process the point cloud
    filtered_points, comp_points = filter_and_segment_point_cloud(filtered_points, bounds, threshold=10)  

    filtered_points = np.vstack((filtered_points, comp_points)) 

    print("connected component time = ", time.time() - start_time)

    


  
    
    
    
    """colors = [] 
    for i in range(64):
        for j in range(1024):
            if mask[i][j] == True:
                colors.append([0, 1, 0])
            else:
                colors.append([0, 0, 1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points) 
    #pcd.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float32))
    #print("hello")
        
    #obj.update_renderer(pcd)
    #time.sleep(10)

    o3d.visualization.draw_geometries([pcd])"""   


    pub = rospy.Publisher('/filtered_points', PointCloud2, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz
    
    # Example numpy array: Nx3 or Nx4 (x, y, z, [intensity])
    #extracted_points = np.random.rand(100, 3)  # Replace with your actual extracted points
    #print(np.all(points)==0)
    #while not rospy.is_shutdown():
    pointcloud_msg = numpy_to_pointcloud2(filtered_points, frame_id="velodyne")
    pub.publish(pointcloud_msg)
    rate.sleep() 
    #print("hello")










def callback_real_points(msg): 
 



   # Convert PointCloud2 message to list of points (x, y, z)
   pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    
   # Convert to numpy array
   points_array = np.array(list(pc_data)) 

   proj_obj = LaserScan2(project=True) 
   proj_obj.open_scan(points_array) 
   proj_points = proj_obj.proj_xyz 

   


   

   # take the 16 channels dadta and visulized it 
   points = proj_points
   points = points.reshape((-1, 3)) 

  


   pub = rospy.Publisher('/real_points', PointCloud2, queue_size=10)
   rate = rospy.Rate(10)  # 10 Hz
    
   
   #while not rospy.is_shutdown(): 
   pointcloud_msg = numpy_to_pointcloud2(points, frame_id="velodyne") 
   pub.publish(pointcloud_msg)
   rate.sleep() 










    

     
def listener():
 
     # In ROS, nodes are uniquely named. If two nodes with the same
     # name are launched, the previous one is kicked off. The
     # anonymous=True flag means that rospy will choose a unique
     # name for our 'listener' node so that multiple listeners can
     # run simultaneously.
     rospy.init_node('listener', anonymous=True)
 
     rospy.Subscriber("/velodyne_points", PointCloud2, callback) 
     rospy.Subscriber("/velodyne_points", PointCloud2, callback_real_points)
 
     # spin() simply keeps python from exiting until this node is stopped
     rospy.spin()
 
if __name__ == '__main__':
       listener() 


       




       
