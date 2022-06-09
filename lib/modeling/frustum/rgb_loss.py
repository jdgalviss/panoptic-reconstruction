from lib.structures import DepthMap
from lib.structures.frustum import compute_camera2frustum_transform
from lib.config import config
from .renderer import Renderer
from torchmcubes import marching_cubes, grid_interp
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

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


    def forward(self, geometry_prediction, rgb_prediction, aux_views, cam_poses, debug=False):
    # def rgb_loss(geometry_prediction, rgb_prediction, aux_views, cam_poses):
        # return 0.0
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION

        # Get Dense Predictions
        geometry, _, _ = geometry_prediction.dense(dense_dimensions, min_coordinates, default_value=truncation)
        rgb, _, _ = rgb_prediction.dense(dense_dimensions, min_coordinates)

        geometry = geometry.unsqueeze(0)
        rgb = rgb.squeeze()

        # print("\ngeometry_dense shape: ", geometry.shape)
        # print("rgb_dense shape: ", rgb.shape)

        surface_mask = geometry.squeeze() < 1.0
        points_rgb = rgb[:,surface_mask].transpose(0,1)
        points = surface_mask.squeeze().nonzero()
        points = points.type(torch.FloatTensor)
        points*=0.03
        center = torch.FloatTensor([3.825+0.00,3.825-0.00,3.825-0.0])
        points-=center
        # print(surface_mask.shape)
        # print(points_rgb.shape)
        # print(points.shape)





        # Generate mesh from voxel representation
        # print("vertices shape: ", vertices.shape)
        # print("vertices range: [{}, {}]".format(torch.max(vertices), torch.min(vertices)))
        # print("faces shape: ", faces.shape)
        # print("colors shape: ", rgb.shape)
        # print("colors range: [{}, {}] ".format(torch.max(rgb), torch.min(rgb)))

        # Render original view
        renderer = Renderer(camera_base_transform=cam_poses[0][0])
        # renderer.set_base_camera_transform(cam_poses[0])

        pcl = renderer.create_pcl(points, points_rgb)

        ## TODO: fix offset
        offsets = torch.FloatTensor([[0, 0, 0],[0.8, 0, 0.15],[0.75, 0, 0.1],[-0.85, 0, 0.1]]).to(device)
        angles = torch.FloatTensor([0,0,0.3,2.15]).to(device)
        losses = torch.tensor(0.0).to(device)
        for T,view, offset, angle in zip(cam_poses[0], aux_views[0], offsets,angles):
            img = renderer.render_image(pcl, T, offset=offset, angle=angle)
            # print('gt_view: {}'.format(view.shape))
            # print('gt_view range: [{},{}]'.format(torch.min(view),torch.max(view)))
            view = view.permute(1,2,0).to(device)
            # mask for pixels outside view
            # print(img[120,180,:])
            mask = torch.where(img == torch.tensor([0.0,0.0,0.0]).to(device), torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
            N = 3*torch.sum(mask)
            loss = self.l1(img*mask,view*mask)/N



            # error = img-view
            # error = torch.abs(error*mask)
            # loss = torch.sum(error)/N
            # error /= torch.max(error)
            # print(error.shape)
            # print("mean: ", torch.mean(img))
            if debug:
                print('rendered_img range: [{},{}]'.format(torch.min(img),torch.max(img)))
                print("mean: ", torch.mean(img))
                # plot_image(error.detach().cpu().numpy())
                plot_image(img.detach().cpu().numpy())
                plot_image(view.cpu().numpy())
                print("loss: ", loss)
            if not debug:
                losses += loss

        # print("losses: ", losses)

        return losses


