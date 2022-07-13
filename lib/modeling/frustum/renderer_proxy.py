from utils.raycast_rgbd.raycast_rgbd import RaycastRGBD
from pytorch3d.renderer import look_at_view_transform
import torch
from lib.config import config
import loss as loss_util

device = torch.device(config.MODEL.DEVICE)

def homogeneous_transform(R,t):
    last_row = torch.FloatTensor([[0,0,0,1]]).unsqueeze(0)
    T = torch.cat((R,t),dim=2)
    return torch.cat((T,last_row),dim=1)

class Renderer(object):
    def __init__(self, camera_base_transform = None, voxelsize = 0.12, truncation=1.5, volume_size=64):
        R0, t0 = look_at_view_transform(dist=-200, elev=0, azim=90)
        t0 = torch.FloatTensor([[0.0, volume_size/2.0,volume_size/2.0]])
        self.T_view1 = homogeneous_transform(R0,t0.transpose(0,1).unsqueeze(0)).to(device)
        self.T_off = torch.FloatTensor([[1.0,0.0,0.0,1.0*volume_size/254.0],[0.0,1.0,0.0,-1.0*volume_size/254.0],[0.0,0.0,1.0,21.5*volume_size/254.0],[0.0,0.0,0.0,1.0]]).to(device)
        self.T_view1[0] = self.T_view1[0] @ self.T_off

        if not camera_base_transform is None:
            self.T_pose1 = camera_base_transform.to(device)
        self.raycaster_rgbd = []

        # TODO: Read parameters from config file
        input_dim = (volume_size, volume_size, volume_size)
        batch_size = 1
        if volume_size != 254:
            style_width = int(320*volume_size/256.)
            style_height = int(240*volume_size/256.)
        else:
            style_width = 320
            style_height = 240
        self.intrinsics = torch.FloatTensor([[277.1281435, 277.1281435, 159.0, 119.0]]).to(device)

        self.intrinsics *= volume_size/256
        raycast_depth_max = 6.0
        self.truncation = truncation
        ray_increment = 0.03 * self.truncation
        thresh_sample_dist = 150.5 * ray_increment
        max_num_locs_per_sample = 640000
        num_views = config.MODEL.FRUSTUM3D.NUM_VIEWS

        self.volume_size = volume_size

        for i in range(num_views):
            self.raycaster_rgbd.append(RaycastRGBD(batch_size, input_dim, style_width, style_height, depth_min=0.1/voxelsize, depth_max=raycast_depth_max/voxelsize, 
                                    thresh_sample_dist=thresh_sample_dist, ray_increment=ray_increment, max_num_locs_per_sample=max_num_locs_per_sample))
        self.voxelsize = voxelsize
    
    def set_base_camera_transform(self, T_pose1):
        self.T_pose1 = T_pose1.to(device)
    
    def render_image(self, sdf, rgb, cam_poses):

        # TODO: Substraction of -1.5 only to have negative values in SDF, but distorts geometry.
        truncation = 3.0
        if not config.MODEL.FRUSTUM3D.IS_SDF:
            sdf -= 1.5
            truncation = 1.5
        rgb = rgb.squeeze()
        colors = rgb.permute(1,2,3,0).unsqueeze(0)


        locs = torch.nonzero(torch.abs(sdf[:,0]) < truncation)
        locs = torch.cat([locs[:,1:], locs[:,:1]],1).contiguous()
        vals = sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]].contiguous()
        colors = colors[locs[:,-1],locs[:,0],locs[:,1],locs[:,2],:].float() #/255.0
        cam_poses[:,:,:,:3,-1] /= 0.03*256./self.volume_size

        self.T_pose1 = cam_poses[0,0]

        view_matrices = []
        for T_pose2 in cam_poses[0]:
            # Compute the that takes from base camera pose to aux camera pose in the renderer world frame
            T_diff2 = torch.inverse(self.T_pose1[0]) @ T_pose2[0]
            # Offset fix, needed due to inversion in the sdf in the dataloader
            T_diff2[:3,-1] *= torch.FloatTensor([-1,0,-1]).to(device)
            view_matrix2 = (self.T_view1[0]@T_diff2).unsqueeze(0)
            view_matrices.append(view_matrix2)
        # Get all the view matrices (transform from camera pose of the view to the world frame)
        view_matrices = torch.cat(view_matrices).unsqueeze(1)
        # Render color and normal images
        color_imgs = []
        normal_imgs = []
        for i, view_matrix in enumerate(view_matrices):
            target_normals = loss_util.compute_normals_sparse(locs, vals, sdf.shape[2:], transform=torch.inverse(view_matrix))
            raycast_color, _, raycast_normal = self.raycaster_rgbd[i](locs.to(device), vals.to(device), colors.contiguous().to(device), target_normals.to(device), view_matrix.to(device), self.intrinsics.to(device))
            color_imgs.append(torch.fliplr(raycast_color[0]).unsqueeze(0))
            normal_imgs.append(torch.fliplr(raycast_normal[0]).unsqueeze(0))
    
        return torch.cat(color_imgs), torch.cat(normal_imgs)