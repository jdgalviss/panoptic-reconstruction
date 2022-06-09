import os
import sys
import numpy as np
from lib.config import config
import torch
import pytorch3d
# Data structures and functions for rendering

# from pytorch3d.structures import Meshes
# from pytorch3d.renderer import Textures
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    OrthographicCameras,
    VolumeRenderer,
    NDCGridRaysampler,
    MonteCarloRaysampler,
    GridRaysampler,
    EmissionAbsorptionRaymarcher,
    AbsorptionOnlyRaymarcher,
    NDCMultinomialRaysampler,
    MultinomialRaysampler,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor
)
from math import sin, cos,pi

device = torch.device(config.MODEL.DEVICE)
FOCAL_LENGTH1 = 277.1281435
FOCAL_LENGTH2 = 311.76912635
color_image_size = (240, 320)

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
    def __init__(self, camera_base_transform = None):
        R0, t0 = look_at_view_transform(-3.1, 0, 0, up=((0, -1, 0), ))
        self.T_C1W = homogeneous_transform(R0,t0.transpose(0,1).unsqueeze(0)).to(device)
        self.T_GC1 = camera_base_transform.to(device)

        self.raster_settings = PointsRasterizationSettings(
            image_size=color_image_size, 
            radius = 0.03,
            points_per_pixel = 20
        )

    def set_base_camera_transform(self, T_GC1):
        self.T_GC1 = T_GC1

    def create_pcl(self, points, points_rgb):
        point_cloud = Pointclouds(points=[points.type(torch.FloatTensor)], features=[points_rgb.type(torch.FloatTensor)]).to(device)

        return point_cloud


    
    def render_image(self, pcl, T_GC2, offset = None, angle=None):

        # Compute transform to auxiliary view
        # print(T_GC2.shape)
        # print(self.T_GC1.shape)
        # print(self.T_C1W.shape)

        _T_GC2 = (torch.inverse(T_GC2)[0] @ self.T_GC1[0] @ self.T_C1W[0]).to(device)
        if not offset is None:
            # print(_T_GC2[:3,-1].shape)
            # print(offset.shape)
            _T_GC2[:3,-1] += offset
        
        if not angle is None:
            _T_GC2[:3,:3] = torch.matmul(rot_y(angle).to(device), _T_GC2[:3,:3])
        # print(_T_GC2.shape)

        cameras = PerspectiveCameras(device=device, R=_T_GC2[:3,:3].unsqueeze(0), T=_T_GC2[:3,-1].unsqueeze(0), focal_length=((FOCAL_LENGTH1,FOCAL_LENGTH1),),
                             principal_point=(([160,120]),), image_size=((240, 320), ),in_ndc=False) #, K=front3d_intrinsic.unsqueeze(0))\


        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        # Render view
        images = renderer(pcl)[0, ..., :3]

        # print("\nrendered_image shape: {}".format(images.shape))
        # print("rendered_image shape: {}".format(images[0, ..., :3].shape))

        return images        
        