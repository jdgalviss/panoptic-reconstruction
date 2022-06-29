import os
import sys
from urllib.parse import _NetlocResultMixinStr
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

# TODO: Read parameters from config file
truncation = 1.5
device = torch.device(config.MODEL.DEVICE)
input_dim = (254, 254, 254)
batch_size = 1
style_width = 320
style_height = 240
raycast_depth_max = 6.0
ray_increment = 0.003 * truncation
thresh_sample_dist = 100.5 * ray_increment
max_num_locs_per_sample = 640000
# intrinsics = torch.FloatTensor([[277.1281435, 311.76912635, 160.0, 120.0]]).to(device)
intrinsics = torch.FloatTensor([[277.1281435, 277.1281435, 160.0, 120.0]]).to(device)

num_views = 3

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
    """
    Renderer object used to render views for a given scene given the camera pose, SDF, and color
    - T_GC1 refers to the transform from the ground scene frame (from blender proc) to the base camera
    - T_C1W refers to the transform from the base camera to the renderer world frame

    Parameters
    ----------
    camera_base_transform : torch.Tensor
        4x4 transformation matrix from camera base to ground_scene frame for the original view (From the data)
    voxelsize : float
        voxel size of the SDF
    """
    def __init__(self, camera_base_transform = None, voxelsize = 0.0301):
        R0, t0 = look_at_view_transform(dist=-200, elev=0, azim=90)
        t0 = torch.FloatTensor([[22.5,127.0,127.0]])
        # Base Camera (original view) to Renderer World Transform
        self.T_C1W = homogeneous_transform(R0,t0.transpose(0,1).unsqueeze(0)).to(device)
        if not camera_base_transform is None:
            self.T_GC1 = camera_base_transform.to(device)
        self.raycaster_rgbd = []
        for i in range(num_views):
            self.raycaster_rgbd.append(RaycastRGBD(batch_size, input_dim, style_width, style_height, depth_min=0.1/voxelsize, depth_max=raycast_depth_max/voxelsize, 
                                    thresh_sample_dist=thresh_sample_dist, ray_increment=ray_increment, max_num_locs_per_sample=max_num_locs_per_sample))

        self.truncation = 1.5
        self.voxelsize = voxelsize

    def set_base_camera_transform(self, T_GC1):
        """
        Set the base camera transform
        
        Parameters
        ----------
        T_GC1 : torch.Tensor
            4x4 transformation matrix from camera base to ground_scene frame for the original view (From the data)
        """
        self.T_GC1 = T_GC1.to(device)
    
    def render_image(self, locs, vals, sdf, colors, cam_poses, offsets = None, angle=None):
        """
        Render a set of images from an SDF, Colors and a different camera poses

        Parameters
        ----------
        locs : torch.Tensor
            Locations of non zero values in the SDF (for efficiency)
        vals : torch.Tensor
            Values of non zero values in the SDF
        sdf : torch.Tensor
            Full SDF Tensor (used to compute normals)
        colors : torch.Tensor
            Color Tensor
        cam_poses : torch.Tensor
            4x4 transformation matrices from camera base to ground_scene frame for the original view (From the data)

        Returns
        ----------
        color_images : torch.Tensor
            Color images for each camera pose
        normal_images : torch.Tensor
            Normal images for each camera pose
        """
        # Compute view matrices for all the views from the camera poses
        view_matrices = []
        for T, offset in zip(cam_poses, offsets):
            # Compute the transformation from the camera pose of the aux view. to the Renderer World Frame
            T_C2W = torch.matmul(torch.matmul(torch.inverse(T[0]), self.T_GC1[0]), self.T_C1W[0])
            # Add offsets TODO: This should not be necessary
            if not offset is None:
                T_C2W[:3,-1] += offset
            if not angle is None:
                T_C2W[:3,:3] = torch.matmul(rot_y(angle).to(device), T_C2W[:3,:3])
            T_C2W = T_C2W.unsqueeze(0)
            view_matrices.append(T_C2W)
        # Get all the view matrices (transform from camera pose of the view to the world frame)
        view_matrices = torch.cat(view_matrices).unsqueeze(1)

        # Render color and normal images
        color_imgs = []
        normal_imgs = []
        for i, view_matrix in enumerate(view_matrices):
            target_normals = loss_util.compute_normals_sparse(locs, vals, sdf.shape[2:], transform=torch.inverse(view_matrix))
            raycast_color, _, raycast_normal = self.raycaster_rgbd[i](locs.to(device), vals.to(device), colors.contiguous().to(device), target_normals.to(device), view_matrix.to(device), intrinsics.to(device))
            color_imgs.append(torch.fliplr(raycast_color[0]).unsqueeze(0))
            normal_imgs.append(torch.fliplr(raycast_normal[0]).unsqueeze(0))

        return torch.cat(color_imgs), torch.cat(normal_imgs)