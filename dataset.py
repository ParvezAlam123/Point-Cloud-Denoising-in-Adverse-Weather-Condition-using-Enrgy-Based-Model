import torch 
import torch.nn as nn 
import numpy as np
import os 
from torch.utils.data import DataLoader, Dataset   
from projection import LaserScan 
import open3d as o3d 




device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



def image_to_patches(image, patch_size):
    """
    Divide an image into patches.
    
    Args:
        image(numpy array): numpy array of image [H, W, 3]
        patch_size (list) : size of the patch [4, 8]
    
    Returns:
        patches (np.array): array of the patches 
    """

    # get the imaget dimension 
    image_height, image_width, _ = image.shape 
    
    # get the dimension of the patche
    patch_height, patch_width = patch_size[0], patch_size[1]

    # get the total number of patces in each row and column 
    patches_per_row = image_width // patch_width 
    patches_per_column = image_height // patch_height 

    patches = [] 
    
    for i in range(patches_per_column):
        for j in range(patches_per_row):
            patch = image[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width]
            patches.append(patch)

    patches = np.array(patches)

    return patches 




class LaserScan2:
    """ class that contains laserscan x y, z, r"""

    EXTENSIONS_SCAN = [".bin"]

    def __init__(self, num_features, project=False, H=64, W=1024, fov_up=3, fov_down=-25):
        self.project = project 
        self.H = H 
        self.W = W 
        self.proj_fov_up = fov_up 
        self.proj_fov_down = fov_down 
        self.num_features = num_features
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
        self.proj_xyz = np.full((self.H, self.W, 3), -1, dtype=np.float32)

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
    

    def open_scan(self, filename):
        """ open raw scan and fill attributes values"""

        # reset just in case any  structure is open 
        self.reset() 

        # check the filename is proper string type 
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type but found {}".format(str(type(filename))))
        
        # check extension is a laser scan 
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid laser scan")
        
        # if all is well open laser scan 
        scan = np.fromfile(filename, dtype=np.float32) 
        scan = scan.reshape((-1, self.num_features))

        # put in  attribute 
        points = scan[:, 0:3]
        remissions = scan[:, 3:4]
        
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

        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remissions = self.remissions[order]
        proj_x = proj_x[order]
        proj_y = proj_y[order]
        
        # assigns to images 
        self.proj_range[proj_y, proj_x] = depth 
        self.proj_xyz[proj_y, proj_x] = points 
        self.proj_remission[proj_y, proj_x] = remissions.flatten() 
        self.proj_idx[proj_y, proj_x] = indices 
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)

        







class LaserScanRB:
    """ class that contains laserscan x y, z, r"""

    EXTENSIONS_SCAN = [".pcd"]

    def __init__(self, project=False, H=64, W=1024, fov_up=3, fov_down=-25):
        self.project = project 
        self.H = H 
        self.W = W 
        self.proj_fov_up = fov_up 
        self.proj_fov_down = fov_down 
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
        self.proj_xyz = np.full((self.H, self.W, 3), -1, dtype=np.float32)

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
    

    def open_scan(self, filename):
        """ open raw scan and fill attributes values"""

        # reset just in case any  structure is open 
        self.reset() 

        # check the filename is proper string type 
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type but found {}".format(str(type(filename))))
        
        # check extension is a laser scan 
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid laser scan")
        
        # if all is well open laser scan 
        pcd = o3d.io.read_point_cloud(filename)
        points = np.asarray(pcd.points) 
        
        
        self.set_points(points)


    def set_points(self,points):
        """ Set scan attribute instead of opening it"""

        # reset any open structure 
        self.reset() 

        # check scan makes sense 
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")
        
        
        

        # put the attrubutes 
        self.points = points 
        


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

        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        #remissions = self.remissions[order]
        proj_x = proj_x[order]
        proj_y = proj_y[order]
        
        # assigns to images 
        self.proj_range[proj_y, proj_x] = depth 
        self.proj_xyz[proj_y, proj_x] = points 
        #self.proj_remission[proj_y, proj_x] = remissions.flatten() 
        self.proj_idx[proj_y, proj_x] = indices 
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)

        
















class KITTI(Dataset):
    def __init__(self, detection_train_path, detection_test_path, tracking_train_path, tracking_test_path):
        self.detection_train_path = detection_train_path 
        self.detection_test_path = detection_test_path 
        self.tracking_train_path = tracking_train_path 
        self.tracking_test_path = tracking_test_path 

        self.files = [] 

        # add detection train files path 
        detection_train_files = sorted(os.listdir(self.detection_train_path)) 
        for file in detection_train_files:
            file_path = os.path.join(self.detection_train_path, file)
            sample={}
            sample["file_path"] = file_path  
            self.files.append(sample)

        # add detection test files path 
        detection_test_files = sorted(os.listdir(self.detection_test_path))
        for file in detection_test_files:
            file_path = os.path.join(self.detection_test_path, file)
            sample = {} 
            sample["file_path"] = file_path 

        # add tracking train files path 
        scenes = sorted(os.listdir(self.tracking_train_path))
        for scene in scenes:
            scene_path = os.path.join(self.tracking_train_path, scene)
            tracking_train_files = sorted(os.listdir(scene_path))
            for file in tracking_train_files:
                file_path = os.path.join(scene_path, file)
                sample = {} 
                sample["file_path"] = file_path 
                self.files.append(sample)

        
        # add tracking test files path 
        scenes = sorted(os.listdir(self.tracking_test_path))
        for scene in scenes:
            scene_path = os.path.join(self.tracking_test_path, scene)
            tracking_test_files = sorted(os.listdir(scene_path))
            for file in tracking_test_files:
                file_path = os.path.join(scene_path, file)
                sample = {} 
                sample["file_path"] = file_path 
                self.files.append(sample)

        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]["file_path"] 
        laserscan = LaserScan(project=True) 
        laserscan.open_scan(file_path)
        points = laserscan.proj_xyz 
        

        return {"range_image":points}  
    





class KITTI_NOISE(Dataset):
    def __init__(self, generator_model, detection_train_path, detection_test_path, tracking_train_path, tracking_test_path):
        self.detection_train_path = detection_train_path 
        self.detection_test_path = detection_test_path 
        self.tracking_train_path = tracking_train_path 
        self.tracking_test_path = tracking_test_path 
        self.generator_model = generator_model

        self.files = [] 

        # add detection train files path 
        detection_train_files = sorted(os.listdir(self.detection_train_path)) 
        for file in detection_train_files:
            file_path = os.path.join(self.detection_train_path, file)
            sample={}
            sample["file_path"] = file_path  
            self.files.append(sample)

        # add detection test files path 
        detection_test_files = sorted(os.listdir(self.detection_test_path))
        for file in detection_test_files:
            file_path = os.path.join(self.detection_test_path, file)
            sample = {} 
            sample["file_path"] = file_path 

        # add tracking train files path 
        scenes = sorted(os.listdir(self.tracking_train_path))
        for scene in scenes:
            scene_path = os.path.join(self.tracking_train_path, scene)
            tracking_train_files = sorted(os.listdir(scene_path))
            for file in tracking_train_files:
                file_path = os.path.join(scene_path, file)
                sample = {} 
                sample["file_path"] = file_path 
                self.files.append(sample)

        
        # add tracking test files path 
        scenes = sorted(os.listdir(self.tracking_test_path))
        for scene in scenes:
            scene_path = os.path.join(self.tracking_test_path, scene)
            tracking_test_files = sorted(os.listdir(scene_path))
            for file in tracking_test_files:
                file_path = os.path.join(scene_path, file)
                sample = {} 
                sample["file_path"] = file_path 
                self.files.append(sample)

        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]["file_path"] 
        laserscan = LaserScan(project=True) 
        laserscan.open_scan(file_path)
        points = laserscan.proj_xyz 
        points = torch.from_numpy(points).permute(2, 0, 1) 
        temp_points = points.unsqueeze(dim=0).to(device).float()
        x_return, x_mu, x_log_var = self.generator_model(temp_points) 

        negative_points = x_return[0]
        

        return {"range_image":points, "negative_range_image":negative_points}  
    
    
    




class SemanticSTF(Dataset):
    def __init__(self, velodyne_test_path):
        self.velodyne_test_path = velodyne_test_path 

        self.files = [] 
        velo_files = sorted(os.listdir(self.velodyne_test_path)) 
        for i in range(len(velo_files)):
            velo_file_path = os.path.join(self.velodyne_test_path, velo_files[i])
            sample = {} 
            sample["pcd_path"] = velo_file_path 
            self.files.append(sample)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]["pcd_path"]
        laserscan = LaserScan(project=True)  
        laserscan.open_scan(file_path)
        points = laserscan.proj_xyz 

        return {"range_image":points}
    


        
    


    


class ValidationData(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir 
        self.scenes = os.listdir(self.root)

        self.files = [] 
        for i in range(len(self.scenes)):
            pcd_path = os.path.join(self.root, self.scenes[i], "pcd")
            label_path = os.path.join(self.root, self.scenes[i], "label")

            pcd_files = sorted(os.listdir(pcd_path))
            label_files = sorted(os.listdir(label_path))

            for n in range(len(pcd_files)):
                pcd_file_path = os.path.join(pcd_path, pcd_files[n])
                label_file_path = os.path.join(label_path, label_files[n])
                sample = {} 
                sample["pcd_file_path"] = pcd_file_path 
                sample["label_file_path"] = label_file_path 
                self.files.append(sample) 

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        pcd_file_path = self.files[index]["pcd_file_path"]
        label_file_path = self.files[index]["label_file_path"]

        
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

        points = points.reshape(64, 1024, 3)

        return {"points":points, "labels":labels}
    
    
     



     


class DataScore(Dataset):
    def __init__(self, kitti_path, SementicSTF_path):
        self.kitti_path = kitti_path 
        self.SementicSTF_path = SementicSTF_path 

        self.files = [] 

        kitti_pcds = sorted(os.listdir(self.kitti_path))
        SementicSTF_pcds = sorted(os.listdir(self.SementicSTF_path))
        for i in range(len(SementicSTF_pcds)):
            pcd_path = os.path.join(self.kitti_path, kitti_pcds[i])
            num_feature = 4 
            label = 1
            sample= {} 
            sample["pcd_path"] = pcd_path 
            sample["label"] = label 
            sample["num_feature"] = num_feature 
            self.files.append(sample)

        for i in range(len(SementicSTF_pcds)):
            pcd_path = os.path.join(self.SementicSTF_path, SementicSTF_pcds[i])
            num_feature = 5 
            label = -1 
            sample = {} 
            sample["pcd_path"] = pcd_path 
            sample["label"] = label 
            sample["num_feature"] = num_feature 
            self.files.append(sample)

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        pcd_file_path = self.files[index]["pcd_path"]
        label = self.files[index]["label"]
        num_feature = self.files[index]["num_feature"]  

        laserscan_obj = LaserScan2(num_features=num_feature, project=True)
        
        laserscan_obj.open_scan(pcd_file_path)
        points = laserscan_obj.proj_xyz
        points = np.asarray(points)

        #image_height, image_width, _ = points.shape 
        #patch_per_col = image_height // 4 
        #patch_per_row = image_width // 8 

        #original_image = points 
        

        # convert images into patches 
        #original_image_patches = image_to_patches(original_image, [4, 8]) 

        return {"range_image":points, "label":label}  
    







    







class RB_Data(Dataset):
    def __init__(self, velodyne_test_path):
        self.velodyne_test_path = velodyne_test_path 

        self.files = [] 
        velo_files = sorted(os.listdir(self.velodyne_test_path)) 
        for i in range(len(velo_files)):
            velo_file_path = os.path.join(self.velodyne_test_path, velo_files[i])
            sample = {} 
            sample["pcd_path"] = velo_file_path 
            self.files.append(sample)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]["pcd_path"]
        laserscan = LaserScanRB(project=True)  
        laserscan.open_scan(file_path)
        points = laserscan.proj_xyz 

        return {"range_image":points}
    


        
    

    


       

        


   





    
    






        


    





