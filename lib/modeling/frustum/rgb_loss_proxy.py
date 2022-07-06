from lib.config import config
from .renderer_proxy import Renderer
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

device = torch.device(config.MODEL.DEVICE)
_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def plot_image(img, size=(240,320)):
    major_ticks = np.arange(0, size[1], 40)
    minor_ticks = np.arange(0, size[1], 20)

    major_ticksy = np.arange(0, size[0], 40)
    minor_ticksy = np.arange(0, size[0], 20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True)
    # img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)

    ax.imshow(img)
    circle1 = plt.Circle((size[1]/2, size[0]/2), 5, color='g')
    ax.add_patch(circle1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticksy)
    ax.set_yticks(minor_ticksy, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.show()

class RGBProxyLoss(torch.nn.Module):
    """
    RGB Loss class used to compute reconstruction loss, style loss and GAN loss.
    """
    def __init__(self, volume_size = 64):
        super(RGBProxyLoss, self).__init__()
        # Renderer
        self.volume_size = volume_size
        self.image_size = (int(240*volume_size/256.0), int(320*volume_size/256.0))
        self.renderer = Renderer(camera_base_transform=None, voxelsize=0.03*256./volume_size, volume_size=volume_size)

    def forward(self, rgb, sdf, aux_views, cam_poses, debug = False):
        imgs, normals = self.renderer.render_image(sdf,rgb,cam_poses)
        masks = (torch.logical_or(torch.isinf(imgs),torch.isnan(imgs))).detach()
        views = aux_views[0].to(device)
        views = F.interpolate(views, size=self.image_size, mode="bilinear", align_corners=True)

        views = views.permute(0,2,3,1)

        views[masks] = 0.0
        imgs[masks]  = 0.0
        valids = torch.logical_not(masks)

        if debug:
            for img, view, mask, normal in zip(imgs,views, masks, normals):
                plot_image(img.detach().cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'], size=self.image_size)
                plot_image(normal.detach().cpu().numpy(), size=self.image_size)
                # plot_image(np.float32(mask.detach().cpu().numpy()))
                plot_image(view.detach().cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'], size=self.image_size)

        # Compute Losses
        # L1-loss
        num_valid = torch.sum(valids).item()
        loss = torch.abs(imgs-views)
        loss = torch.sum(loss)/num_valid

        if debug:
            print("L1 loss: ", loss)
            return 0.0

        return loss
