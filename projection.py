import numpy as np 



class LaserScan:
    """ class that contains laserscan x y, z, r"""

    EXTENSIONS_SCAN = [".bin"]

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
        scan = np.fromfile(filename, dtype=np.float32) 
        scan = scan.reshape((-1, 5))

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

        










