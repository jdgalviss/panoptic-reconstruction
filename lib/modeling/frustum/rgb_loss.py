from pty import STDIN_FILENO
from lib.structures import DepthMap
from lib.structures.frustum import compute_camera2frustum_transform
from lib.config import config
from .renderer import Renderer
from torchmcubes import marching_cubes, grid_interp
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.nn import functional as F
import cv2
device = torch.device(config.MODEL.DEVICE)

import matplotlib.pyplot as plt
import numpy as np

def plot_image(img):
    major_ticks = np.arange(0, 320, 40)
    minor_ticks = np.arange(0, 320, 20)

    major_ticksy = np.arange(0, 240, 40)
    minor_ticksy = np.arange(0, 240, 20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True)
    # img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)

    ax.imshow(img)
    circle1 = plt.Circle((160, 120), 5, color='g')
    ax.add_patch(circle1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticksy)
    ax.set_yticks(minor_ticksy, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.show()

class RGBLoss(torch.nn.Module):
    def __init__(self):
        super(RGBLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='sum')
        self.renderer = Renderer(camera_base_transform=None)


    def forward(self, geometry_prediction, rgb_prediction, aux_views, cam_poses, debug=False):
    # def rgb_loss(geometry_prediction, rgb_prediction, aux_views, cam_poses):
        # return 0.0
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION

        # Get Dense Predictions
        sdf, _, _ = geometry_prediction.dense(dense_dimensions, min_coordinates, default_value=truncation)
        rgb, _, _ = rgb_prediction.dense(dense_dimensions, min_coordinates)

        
        rgb = rgb.squeeze()
        sdf = sdf.squeeze()
        # print(sdf.shape)
        rgb = (F.interpolate(rgb.unsqueeze(0), size=(128,128,128), mode="trilinear", align_corners=True))[0]
        sdf = (F.interpolate(sdf.unsqueeze(0).unsqueeze(0), size=(128,128,128), mode="trilinear", align_corners=True))[0,0]

        # print("\ngeometry_dense shape: ", geometry.shape)
        # print("rgb_dense shape: ", rgb.shape)

        truncation = 1.5
        sdf = torch.clamp(sdf,0.0,3.0)
        sdf -= 1.5
        sdf = sdf.unsqueeze(0).unsqueeze(0)
        colors = rgb.permute(1,2,3,0).unsqueeze(0)
        # print("sdf shape: ", sdf.shape)
        # print("colors shape: ", colors.shape)

        locs = torch.nonzero(torch.abs(sdf[:,0]) < truncation)

        locs = torch.cat([locs[:,1:], locs[:,:1]],1).contiguous()

        vals = sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]].contiguous()
        # print("colors shape: ", colors.shape)
        # print("colors range: [{}, {}] ".format(torch.max(colors), torch.min(colors)))
        colors = colors[locs[:,-1],locs[:,0],locs[:,1],locs[:,2],:].float() #/255.0
        # print(surface_mask.shape)
        # print(points_rgb.shape)
        # print(points.shape)





        # Generate mesh from voxel representation
        # print("vertices shape: ", vertices.shape)
        # print("vertices range: [{}, {}]".format(torch.max(vertices), torch.min(vertices)))
        # print("faces shape: ", faces.shape)
        # colors = torch.randn_like(colors)
        # print("colors shape: ", colors.shape)
        # print("colors range: [{}, {}] ".format(torch.max(colors), torch.min(colors)))

        # Render original view
        self.renderer.set_base_camera_transform(T_GC1=cam_poses[0][0])
        # renderer.set_base_camera_transform(cam_poses[0])

        ## TODO: fix offset
        offsets = torch.FloatTensor([[0.0, 0.0, 0.0],[18.0,0.0,22.0],[20.0,0.0,11.5],]).to(device) #[111.0,0.0,165.0]
        angles = torch.FloatTensor([0,0,0.0,0.0]).to(device)
        losses = torch.tensor(0.0).to(device)
        # for T,view, offset, angle in zip(cam_poses[0], aux_views[0], offsets,angles):
        #     img = self.renderer.render_image(locs, vals, sdf, colors, T, offset=offset, angle=None)
        #     view = view.permute(1,2,0).to(device)
        #     mask = torch.where(img == torch.tensor([0.0,0.0,0.0]).to(device), torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        #     mask = img > torch.tensor([0.0,0.0,0.0]).to(device)
        #     mask = (mask[:,:,0] * mask[:,:,1] * mask[:,:,2]).unsqueeze(-1)
        #     N = 3*torch.sum(mask)
        #     loss = self.l1(img*mask,view*mask)/N

        #     if debug:
        #         print('rendered_img range: [{},{}]'.format(torch.min(img),torch.max(img)))
        #         print("mean: ", torch.mean(img))
        #         # plot_image(error.detach().cpu().numpy())
        #         plot_image(img.detach().cpu().numpy())
        #         plot_image(view.cpu().numpy())
        #         print("loss: ", loss)
        #     if not debug:
        #         losses += loss

        T = cam_poses[0,0]
        view = aux_views[0,0]
        offset = offsets[0]
        angle = angles[0]

        img = self.renderer.render_image(locs, vals, sdf, colors, T, offset=offset, angle=None)
        view = view.permute(1,2,0).to(device)
        mask = torch.where(img == torch.tensor([0.0,0.0,0.0]).to(device), torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        mask = img > torch.tensor([0.0,0.0,0.0]).to(device)
        mask = (mask[:,:,0] * mask[:,:,1] * mask[:,:,2]).unsqueeze(-1)
        N = 3*torch.sum(mask)
        loss = self.l1(img*mask,view*mask)/N
        if debug:
            print('rendered_img range: [{},{}]'.format(torch.min(img),torch.max(img)))
            print("mean: ", torch.mean(img))
            # plot_image(error.detach().cpu().numpy())
            plot_image(img.detach().cpu().numpy())
            plot_image(view.cpu().numpy())
            print("loss: ", loss)
        if not debug:
                losses += loss
        return losses


