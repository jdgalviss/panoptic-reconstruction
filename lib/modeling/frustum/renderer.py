import os
import sys
import numpy as np
from lib.config import config
import torch
import pytorch3d
# Data structures and functions for rendering

# from pytorch3d.structures import Meshes
# from pytorch3d.renderer import Textures

from pytorch3d.renderer import look_at_view_transform
# import os
# import sys
# sys.path.append('/usr/src/app/spsg/torch')
from utils.raycast_rgbd.raycast_rgbd import RaycastRGBD
from math import sin, cos,pi
import loss as loss_util

truncation = 1.5
device = torch.device(config.MODEL.DEVICE)
input_dim = (128, 128, 128)
batch_size = 1
style_width = 320
style_height = 240
raycast_depth_max = 6.0
ray_increment = 0.003 * truncation
thresh_sample_dist = 100.5 * ray_increment
max_num_locs_per_sample = 640000
intrinsics = torch.FloatTensor([[277.1281435, 311.76912635, 160.0, 120.0]]).to(device)

def homogeneous_transform(R,t):
    last_row = torch.FloatTensor([[0,0,0,1]]).unsqueeze(0)
    T = torch.cat((R,t),dim=2)
    return torch.cat((T,last_row),dim=1)

def rot_x(t):
    return torch.FloatTensor([[1,0,0],[0,cos(t),-sin(t)],[0,sin(t),cos(t)]])

def rot_y(t):
    return torch.FloatTensor([[cos(t), 0, sin(t)],[0,1,0],[-sin(t),0,cos(t)]])

def rot_z(t):
    return torch.FloatTensor([[cos(t),-sin(t),0],[sin(t),cos(t),0],[0,0,1]])

class Renderer(object):
    def __init__(self, camera_base_transform = None, voxelsize = 0.06):
        R0, t0 = look_at_view_transform(dist=-180, elev=0, azim=90)
        t0 = torch.FloatTensor([[10.0,63.5,63.5]])
        self.T_C1W = homogeneous_transform(R0,t0.transpose(0,1).unsqueeze(0)).to(device)
        if not camera_base_transform is None:
            self.T_GC1 = camera_base_transform.to(device)
            # self.T_GC1[:,:3,-1] = self.T_GC1[:,:3,-1]/voxelsize
        self.raycaster_rgbd = RaycastRGBD(batch_size, input_dim, style_width, style_height, depth_min=0.1/voxelsize, depth_max=raycast_depth_max/voxelsize, 
                             thresh_sample_dist=thresh_sample_dist, ray_increment=ray_increment, max_num_locs_per_sample=max_num_locs_per_sample)
        self.truncation = 1.5
        self.voxelsize = voxelsize

    def set_base_camera_transform(self, T_GC1):
        self.T_GC1 = T_GC1.to(device)
        # self.T_GC1[:,:3,-1] = self.T_GC1[:,:3,-1]/voxelsize

    # def create_pcl(self, points, points_rgb):
    #     point_cloud = Pointclouds(points=[points.type(torch.FloatTensor)], features=[points_rgb.type(torch.FloatTensor)]).to(device)

    #     return point_cloud


    
    def render_image(self, locs, vals, sdf, colors, T_GC2, offset = None, angle=None):

        # Compute transform to auxiliary view
        # print(T_GC2.shape)
        # print(self.T_GC1.shape)
        # print(self.T_C1W.shape)
        T_GC2[:,:3,-1] = T_GC2[:,:3,-1]/self.voxelsize

        _T_GC2 = (torch.inverse(T_GC2[0]) @ self.T_GC1[0] @ self.T_C1W[0]).to(device)

        if not offset is None:
            # print(_T_GC2[:3,-1].shape)
            # print(offset.shape)
            _T_GC2[:3,-1] += offset
        
        if not angle is None:
            _T_GC2[:3,:3] = torch.matmul(rot_y(angle).to(device), _T_GC2[:3,:3])

        _T_GC2 = _T_GC2.unsqueeze(0)
        view_matrix = _T_GC2
        
        # view_matrix = homogeneous_transform(_T_GC2[:3,:3],_T_GC2[:3,-1].transpose(0,1).unsqueeze(0)).to(device)
        target_normals = loss_util.compute_normals_sparse(locs, vals, sdf.shape[2:], transform=torch.inverse(view_matrix))

        raycast_color, _, raycast_normal = self.raycaster_rgbd(locs.to(device), vals.to(device), colors.contiguous().to(device), target_normals.to(device), view_matrix.to(device), intrinsics.to(device))
        raycast_color = torch.fliplr(raycast_color[0]).unsqueeze(0)
        raycast_normal = torch.fliplr(raycast_normal[0]).unsqueeze(0)

        # raycast_normal = torch.fliplr(raycast_normal[0]).unsqueeze(0)

        # print("\nrendered_image shape: {}".format(images.shape))
        # print("rendered_image shape: {}".format(images[0, ..., :3].shape))
        # raycast_color = torch.clamp(raycast_color,-1.0,1.0)
        
        return raycast_color[0], raycast_normal[0]     
        