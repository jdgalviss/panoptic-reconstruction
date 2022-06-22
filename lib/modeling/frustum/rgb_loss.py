from lib.structures import DepthMap
from lib.structures.frustum import compute_camera2frustum_transform
from lib.config import config
from .renderer import Renderer
from .style import Model as StyleModel
from .style import gram_matrix
from .discriminator import Discriminator2D, GANLoss
from .utils import weight_codes
import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import cv2
device = torch.device(config.MODEL.DEVICE)
from lib.data import transforms2d as t2d
import MinkowskiEngine as Me


import matplotlib.pyplot as plt
import numpy as np

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
VALID_THRESH = 0.1

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
        self.normalize_transform = t2d.Normalize(_imagenet_stats["mean"], _imagenet_stats["std"])

        # Style Loss
        model_style = torchvision.models.vgg19(pretrained=True).cuda().eval()
        self.model_style = StyleModel(model_style, _imagenet_stats["mean"], _imagenet_stats["std"])

        # GAN Loss
        self.discriminator = Discriminator2D(nf_in=3, nf=8, patch_size=96, image_dims=(320, 240), patch=True, use_bias=True, disc_loss_type='vanilla').to(device)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=4*0.001, weight_decay=0.0)
        self.gan_loss = GANLoss(loss_type='vanilla')

    def compute_style_loss(self, pred_color, target_color, compute_style=True, compute_content=True, mask=None):
        # output_color, target_color = preprocess_rendered_target_images(output_color, target_color, mask)
        target_features = self.model_style(target_color)
        output_features = self.model_style(pred_color)
        loss = 0.0
        loss_content = 0.0
        for k in range(len(output_features)):
            if compute_content:
                loss_content += F.mse_loss(output_features[k], target_features[k])
            if compute_style:
                tgt = gram_matrix(target_features[k])
                pred = gram_matrix(output_features[k])
                loss += F.mse_loss(pred*10, tgt*10)
        return loss, loss_content

    def forward(self, geometry_prediction, rgb_prediction, semantic_prediction, aux_views, cam_poses, debug=False):
    # def rgb_loss(geometry_prediction, rgb_prediction, aux_views, cam_poses):
        # return 0.0
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION

        # Get Dense Predictions
        sdf, _, _ = geometry_prediction.dense(dense_dimensions, min_coordinates, default_value=truncation)
        rgb, _, _ = rgb_prediction.dense(dense_dimensions, min_coordinates)

        # reverse one hot for semantic prediction
        semantic_idx = semantic_prediction.F.argmax(dim=1)
        # print("semantic_prediction range: [{}, {}]".format(torch.min(semantic_prediction), torch.max(semantic_prediction)))
        print("semantic_weights values: ", np.unique(semantic_idx.detach().cpu().numpy()))





        semantic_weights_sparse = weight_codes()[semantic_idx].to(device)
        
        # print("semantic_weights_sparse: ", semantic_weights_sparse.shape)

        semantic_weights_sparse = Me.SparseTensor(semantic_weights_sparse, rgb_prediction.C, coordinate_manager=rgb_prediction.coordinate_manager)
        # print("semantic_weights_sparse_sparse: ", semantic_weights_sparse.shape)

        semantic_weights, _, _ = semantic_weights_sparse.dense(dense_dimensions, min_coordinates)
        # print("semantic_weights values: ", np.unique(semantic_weights.detach().cpu().numpy()))

        # print("semantic_weights: ", semantic_weights.shape)

        
        rgb = rgb.squeeze()
        sdf = sdf.squeeze()
        semantic_weights = semantic_weights.squeeze()
        # print("semantic values: ", np.unique(semantic_weights.detach().cpu().numpy()))

        # print(sdf.shape)

        rgb = (F.interpolate(rgb.unsqueeze(0), size=(254,254,254), mode="trilinear", align_corners=True))[0]
        sdf = (F.interpolate(sdf.unsqueeze(0).unsqueeze(0), size=(254,254,254), mode="trilinear", align_corners=True))[0,0]
        semantic_weights = (F.interpolate(semantic_weights.unsqueeze(0), size=(254,254,254), mode="nearest"))[0]
        # print("\ngeometry_dense shape: ", geometry.shape)
        


        truncation = 1.5
        # sdf = torch.clamp(sdf,0.0,3.0)
        sdf -= 1.5
        sdf = sdf.unsqueeze(0).unsqueeze(0)
        colors = rgb.permute(1,2,3,0).unsqueeze(0)
        semantic_weights = semantic_weights.permute(1,2,3,0).unsqueeze(0)
        # print("sdf shape: ", sdf.shape)
        # print("colors shape: ", colors.shape)

        locs = torch.nonzero(torch.abs(sdf[:,0]) < truncation)

        locs = torch.cat([locs[:,1:], locs[:,:1]],1).contiguous()

        vals = sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]].contiguous()
        # print("colors shape: ", colors.shape)
        # print("colors range: [{}, {}] ".format(torch.max(colors), torch.min(colors)))
        colors = colors[locs[:,-1],locs[:,0],locs[:,1],locs[:,2],:].float() #/255.0
        semantic_weights = semantic_weights[locs[:,-1],locs[:,0],locs[:,1],locs[:,2],:].float() #/255.0


        # Render original view
        cam_poses[:,:,:,:3,-1] /= 0.0301
        self.renderer.set_base_camera_transform(T_GC1=cam_poses[0][0])
        # renderer.set_base_camera_transform(cam_poses[0])

        ## TODO: fix offset
        offsets = torch.FloatTensor([[0.0, 0.0, 0.0],[18.0,0.0,22.0],[20.0,0.0,11.5],]).to(device)*2.0 #[111.0,0.0,165.0]
        angles = torch.FloatTensor([0,0,0.0,0.0]).to(device)
        losses = torch.tensor(0.0).to(device)

        # For use with GAN loss
        imgs = []
        views = []
        valids = []
        weights = []
        # for T,view, offset, angle in zip(cam_poses[0,1:], aux_views[0,1:], offsets[1:],angles[1:]):
        # print("semantic shape: ", semantic_weights.shape)
        # return 0.0
        imgs, _ = self.renderer.render_image(locs, vals, sdf, colors, cam_poses[0], offsets=offsets, angle=None)
        weights, _ = self.renderer.render_image(locs, vals, sdf, semantic_weights, cam_poses[0], offsets=offsets, angle=None)
        

        masks = (torch.logical_or(torch.isinf(imgs),torch.isnan(imgs))).detach()
        masks_semantic = (torch.logical_or(torch.isinf(weights),torch.isnan(weights))).detach()

        views = aux_views[0].to(device)
        views = views.permute(0,2,3,1)
        views[masks] = 0.0
        imgs[masks]  = 0.0
        weights[masks_semantic] = 0.0
        valids = torch.logical_not(masks)
        if debug:
            for img, weight, view, mask in zip(imgs,weights,views, masks):
                plot_image(img.detach().cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'])
                plot_image(weight.detach().cpu().numpy())
                # plot_image(np.float32(mask.detach().cpu().numpy()))
                plot_image(view.detach().cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'])

        
        

        
        # if debug:
        #     plot_image(imgs[0].detach().cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'])
        #     plot_image(weights[0].detach().cpu().numpy())
        #     plot_image(views[0].cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'])
        # return 0.0
        # print("imgs shape: ", imgs.shape)
        # print("views shape: ", views.shape)
        # print("valids shape: ", valids.shape)
        # print("weights shape: ", weights.shape)

        # L1-loss
        weights = weights.detach()
        num_valid = torch.sum(valids).item()
        loss = torch.abs(imgs-views)*weights
        # print("loss shape: ", loss.shape)
        loss = torch.sum(loss)/num_valid

        ## Style Loss
        # style_loss, loss_content = self.compute_style_loss(imgs.permute(0,3,1,2),views.permute(0,3,1,2))

        style_loss = torch.tensor(0.0).to(device)
        loss_content = torch.tensor(0.0).to(device)

        for i in range(3):
            _style_loss, _loss_content = self.compute_style_loss(imgs.permute(0,3,1,2)[i].unsqueeze(0),views.permute(0,3,1,2)[i].unsqueeze(0))
            style_loss += _style_loss
            loss_content += _loss_content

        # GAN Loss
        self.optimizer_disc.zero_grad()
        valid = valids.permute(0,3,1,2)
        valid = (self.discriminator.compute_valids(valid[:,-1,:,:].float().unsqueeze(1)) > VALID_THRESH).squeeze(1)

        # print("si")
        # print(img.permute(2,0,1).unsqueeze(0).shape)
        # print(imgs.permute(0,3,1,2).shape)
        # img_fake = imgs.permute(0,3,1,2).detach()
        with torch.no_grad():
            img_fake = imgs.permute(0,3,1,2).clone()
           
        real_loss, fake_loss, penalty = self.gan_loss.compute_discriminator_loss(self.discriminator, views.permute(0,3,1,2), 
                                                                           img_fake, valid, None )

        real_loss = torch.mean(real_loss)
        fake_loss = torch.mean(fake_loss)
        disc_loss = (real_loss + fake_loss)

        if not debug:
            disc_loss.backward()
            self.optimizer_disc.step()
            
        gen_loss = self.gan_loss.compute_generator_loss(self.discriminator, imgs.permute(0,3,1,2))

        total_loss = (8.0*loss+(0.01*style_loss+loss_content*0.001+0.2*gen_loss))

        if debug:
            print("Total loss: ", total_loss)
            print("L1 loss: ", loss)
            print("style loss: ", style_loss)
            print("loss_content: ", loss_content)
            print("disc_loss: ", disc_loss)
            print("gen_loss: ", gen_loss)

        return total_loss
                

        if debug:
            print('rendered_img range: [{},{}]'.format(torch.min(img),torch.max(img)))
            print("mean: ", torch.mean(img))
            plot_image(img.detach().cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'])
            plot_image(semantic_weights_img.detach().cpu().numpy())
            plot_image(view.cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'])
            print("L1 loss: ", loss)
            print("style loss: ", style_loss)
            print("loss_content: ", loss_content)


        if not debug:
            # losses += (loss+0.1*style_loss+loss_content*0.1)
            losses += (4.0*loss+0.01*style_loss+loss_content*0.001)
            # losses = torch.clamp(losses, min=0.0)


        # views = torch.cat(views)
        # imgs = torch.cat(imgs)
        # valids = torch.cat(valids)
        # self.optimizer_disc.zero_grad()
        # real_loss, fake_loss, penalty = self.gan_loss.compute_discriminator_loss(self.discriminator, views, 
        #                                                                         imgs, valids, None )
        # real_loss = torch.mean(real_loss)
        # fake_loss = torch.mean(fake_loss)
        # disc_losses += (real_loss + fake_loss)
        # gen_loss = self.gan_loss.compute_generator_loss(self.discriminator, imgs)
        # # gen_loss = torch.clamp(gen_loss, min=0.0)
        
        # if not debug:
        #     disc_losses.backward()
        #     self.optimizer_disc.step()
        # else:
        #         print("real_loss: ", real_loss)
        #         print("fake_loss: ", fake_loss)
        #         print("penalty: ", penalty)
        #         print("disc_loss: ", disc_losses)
        #         print("gen_loss: ", gen_loss)

        return losses


