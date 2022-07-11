from lib.config import config
from .renderer import Renderer
from .style import Model as StyleModel
from .style import gram_matrix
from .discriminator import Discriminator2D, GANLoss
from .utils import weight_codes, create_color_palette
import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
device = torch.device(config.MODEL.DEVICE)
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
    """
    RGB Loss class used to compute reconstruction loss, style loss and GAN loss.
    """
    def __init__(self):
        super(RGBLoss, self).__init__()
        # L1 Reconstruction Loss
        self.l1 = nn.L1Loss(reduction='sum')
        # Renderer
        self.renderer = Renderer(camera_base_transform=None)

        # Style Loss
        model_style = torchvision.models.vgg19(pretrained=True).to(device).eval()
        self.model_style = StyleModel(model_style, _imagenet_stats["mean"], _imagenet_stats["std"])

        # GAN Loss
        self.discriminator = Discriminator2D(nf_in=6, nf=8, patch_size=96, image_dims=(320, 240), patch=True, use_bias=True, disc_loss_type='vanilla').to(device)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=4*0.001, weight_decay=0.0)
        self.gan_loss = GANLoss(loss_type='vanilla')

    def compute_style_loss(self, pred_color, target_color, compute_style=True, compute_content=True, mask=None):
        """
        Compute style loss using VGG backbone - Loss is computed by comparing the features generated by the backbone
        for the predicted and target images.

        Parameters
        ----------
        pred_color : torch.Tensor
            Predicted image.
        target_color : torch.Tensor
            Target image.
        
        Returns
        -------
        loss : torch.Tensor
            style loss
        loss_content : torch.Tensor
            content loss
        """
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

    def forward(self, geometry_prediction, rgb_prediction, semantic_prediction, aux_views, cam_poses, target_sdf, debug=False):
        """
        Compute the loss for the RGB prediction (colored geometry).

        Parameters
        ----------
        geometry_prediction : torch.Tensor
            Predicted sparse geometry.
        rgb_prediction : torch.Tensor
            Predicted sparse RGB volume.
        semantic_prediction : torch.Tensor
            Predicted semantic volume labels.
        aux_views : torch.Tensor
            Images of the auxiliary views used to compare with predicted views from diff rendering
        cam_poses : torch.Tensor
            Camera poses of the auxiliary views
        
        Returns
        -------
        losses: dict
            Dictionary containing the losses (TotalRGBLoss, L1-reconstruction, style, content, generator, discriminator).
        """
        
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION

        # Get Dense Predictions
        sdf, _, _ = geometry_prediction.dense(dense_dimensions, min_coordinates, default_value=truncation)
        rgb, _, _ = rgb_prediction.dense(dense_dimensions, min_coordinates)
        # reverse one hot for semantic prediction (semantic mask is used to add weights to color prediction loss for foreground objects)
        # print("semantic_prediction shape: ",semantic_prediction.shape)
        # semantic_idx = semantic_prediction.F.argmax(dim=1)
        # print("semantic_prediction range: [{}, {}]".format(torch.min(semantic_prediction), torch.max(semantic_prediction)))
        # print("semantic_idx shape: ",semantic_idx.shape)
        
        # print("semantic_weights values: ", np.unique(semantic_idx.detach().cpu().numpy()))

        # If debug create colored semantic labels, else, create the weights mask
        
        
        # Get semantic weights Sparse Tensor and then transform to dense tensor (redundant?)
        # semantic_weights_sparse = Me.SparseTensor(semantic_weights_sparse, rgb_prediction.C, coordinate_manager=rgb_prediction.coordinate_manager)
        # semantic_weights, _, _ = semantic_weights_sparse.dense(dense_dimensions, min_coordinates)

        # print("semantic_weights shape: ",semantic_weights.shape)
        
        rgb = rgb.squeeze()
        sdf = sdf.squeeze()
        # semantic_weights = semantic_weights.squeeze()
        target_sdf = target_sdf.squeeze()
        print("semantic_prediction: ",semantic_prediction.shape)

        # Interpolate to size (254,254,254) to stay inside CUDA's limits
        rgb = (F.interpolate(rgb.unsqueeze(0), size=(254,254,254), mode="trilinear", align_corners=True))[0]
        sdf = (F.interpolate(sdf.unsqueeze(0).unsqueeze(0), size=(254,254,254), mode="trilinear", align_corners=True))[0,0]
        semantic_prediction = (F.interpolate(semantic_prediction.float(), size=(254,254,254), mode="trilinear",align_corners=True))[0]
        target_sdf = (F.interpolate(target_sdf.unsqueeze(0).unsqueeze(0), size=(254,254,254), mode="trilinear", align_corners=True))[0,0]
        print("semantic_prediction: ",semantic_prediction.shape)

        # return 0.0
        # TODO: Substraction of -1.5 only to have negative values in SDF, but distorts geometry.
        truncation = 3.0
        if not config.MODEL.FRUSTUM3D.IS_SDF:
            sdf -= 1.5
            target_sdf -= 1.5
            target_sdf *= 2.0
            sdf *= 2.0
            truncation = 3.0

        sdf = sdf.unsqueeze(0).unsqueeze(0)
        target_sdf = target_sdf.unsqueeze(0).unsqueeze(0)
        colors = rgb.permute(1,2,3,0).unsqueeze(0)
        semantic_prediction = semantic_prediction.permute(1,2,3,0).unsqueeze(0)

        # Obtain sparse tensor of sdf, colors and semantic weights
        locs = torch.nonzero(torch.abs(sdf[:,0]) < truncation)
        locs = torch.cat([locs[:,1:], locs[:,:1]],1).contiguous()
        vals = sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]].contiguous()
        colors = colors[locs[:,-1],locs[:,0],locs[:,1],locs[:,2],:].float() #/255.0
        semantic_prediction = semantic_prediction[locs[:,-1],locs[:,0],locs[:,1],locs[:,2],:].float() #/255.0

        print("semantic_prediction: ",semantic_prediction.shape)
        print("semantic_prediction: ",semantic_prediction[:,0].long().shape)
        print("semantic_weights values: ", np.unique(semantic_prediction[:,0].long().detach().cpu().numpy()))

        semantic_weights = weight_codes()[semantic_prediction[:,0].long()].to(device)

        if debug:
            # semantic_weights_sparse = create_color_palette()[semantic_idx].to(device)
            semantic_weights = create_color_palette()[semantic_prediction[:,0].long()].to(device)
        else:
            semantic_weights = weight_codes()[semantic_prediction[:,0].long()].to(device)

        print("semantic_weights: ",semantic_weights.shape)
        print("semantic_weights values: ", np.unique(semantic_weights.detach().cpu().numpy()))

        

        # Divide translation vector in cam_poses by the voxel size
        cam_poses[:,:,:,:3,-1] /= 0.03*256./254.

        # Set the base camera pose of the renderer (For the original view)
        self.renderer.set_base_camera_transform(T_pose1=cam_poses[0][0])

        ## TODO: fix offset - These offsets were calculated manually and should not be necessary
        offsets = torch.FloatTensor([[0.0, -0.7, 0.0],[17.0,-0.7,21.0],[19.5,-0.7,11.7],]).to(device)*2.0 #[111.0,0.0,165.0]
        angles = torch.FloatTensor([0,0,0.0,0.0]).to(device)
        losses = torch.tensor(0.0).to(device)

        # Render views and semantic weights masks for all the given camera poses
        print("colors shape: ",colors.shape)
        # return 0.0
        imgs, normals, target_normals = self.renderer.render_image(locs, vals, sdf, colors, cam_poses[0], rgb=rgb, target_sdf=target_sdf, offsets=offsets, angle=None)
        weights, _, _ = self.renderer.render_image(locs, vals, sdf, semantic_weights, cam_poses[0], offsets=offsets, angle=None)
        
        masks = (torch.logical_or(torch.isinf(imgs),torch.isnan(imgs))).detach()
        masks_semantic = (torch.logical_or(torch.isinf(weights),torch.isnan(weights))).detach()

        views = aux_views[0].to(device)
        views = views.permute(0,2,3,1)
        views[masks] = 0.0
        imgs[masks]  = 0.0
        weights[masks_semantic] = 0.0
        valids = torch.logical_not(masks)
        if debug:
            for img, weight, view, mask, normal, target_normal in zip(imgs,weights,views, masks, normals, target_normals):
                plot_image(img.detach().cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'])
                plot_image(weight.detach().cpu().numpy())
                plot_image(normal.detach().cpu().numpy())
                plot_image(target_normal.detach().cpu().numpy())
                # plot_image(np.float32(mask.detach().cpu().numpy()))
                plot_image(view.detach().cpu().numpy()*_imagenet_stats['std']+_imagenet_stats['mean'])

        # Compute Losses
        # L1-loss
        weights = weights.detach()
        num_valid = torch.sum(valids).item()
        if debug:
            loss = torch.abs(imgs-views)
        else:
            loss = torch.abs(imgs-views)*weights
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
        # Render ground truth normals
        self.optimizer_disc.zero_grad()
        target_normals[target_normals == -float('inf')] = 0.0
        normals[normals == -float('inf')] = 0.0

        # Targets
        target2d = torch.cat([views, target_normals], 3)
        target2d = target2d.permute(0,3,1,2)
        # predictions
        pred2d = torch.cat([imgs, normals], 3)
        pred2d = pred2d.permute(0,3,1,2)

        #valid
        valid = pred2d.detach() != -float('inf')
        # valid = valids.permute(0,3,1,2)
        valid = (self.discriminator.compute_valids(valid[:,-1,:,:].float().unsqueeze(1)) > VALID_THRESH).squeeze(1)
        # with torch.no_grad():
        #     img_fake = imgs.permute(0,3,1,2).clone()
        real_loss, fake_loss, penalty = self.gan_loss.compute_discriminator_loss(self.discriminator, target2d, 
                                                                           pred2d.contiguous().detach(), valid, None )

        real_loss = torch.mean(real_loss)
        fake_loss = torch.mean(fake_loss)
        disc_loss = (real_loss + fake_loss)
        # Training step for the Discriminator
        if not debug:
            disc_loss.backward()
            self.optimizer_disc.step()

        # Generator Loss 
        gen_loss = self.gan_loss.compute_generator_loss(self.discriminator, pred2d) # TODDO: Move up?

        # Total Loss TODO: Define weights for each loss in config file
        total_loss = (8.0*loss+(0.01*style_loss+loss_content*0.001+0.75*gen_loss))
        losses = {"rgb_total_loss":total_loss, "rgb_reconstruction_loss_dbg":loss, "rgb_style_loss_dbg":style_loss, 
                  "rgb_content_loss_dbg":loss_content, "rgb_gen_loss_dbg":gen_loss, "rgb_disc_loss_dbg":disc_loss, "rgb_disc_real_loss_dbg":real_loss, "rgb_disc_fake_loss_dbg":fake_loss}
        
        if debug:
            print("Total loss: ", total_loss)
            print("L1 loss: ", loss)
            print("style loss: ", style_loss)
            print("loss_content: ", loss_content)
            print("disc_loss: ", disc_loss)
            print("gen_loss: ", gen_loss)
            return 0.0


        return losses