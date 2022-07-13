import time

from collections import OrderedDict
from pathlib import Path

import torch
from lib.structures.field_list import collect

from lib import utils, logger, config, modeling, solver, data

import os
import sys
import lib.data.transforms2d as t2d
from PIL import Image
import numpy as np
from lib.utils.intrinsics import adjust_intrinsic


def configure_inference(opts):
    # load config
    config.OUTPUT_DIR = opts.output
    config.merge_from_file(opts.config_file)
    config.merge_from_list(opts.opts)
    # inference settings
    config.MODEL.FRUSTUM3D.IS_LEVEL_64 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_128 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_256 = False
    config.MODEL.FRUSTUM3D.FIX = True


config.merge_from_file('configs/front3d_train_3d_test.yaml')
config.MODEL.PRETRAIN = "/usr/src/app/panoptic-reconstruction/output/00_pretrained/model_overfit_6classes.pth"
# inference settings
config.MODEL.FRUSTUM3D.IS_LEVEL_64 = False
config.MODEL.FRUSTUM3D.IS_LEVEL_128 = False
config.MODEL.FRUSTUM3D.IS_LEVEL_256 = False
config.MODEL.FRUSTUM3D.FIX = True

device = torch.device("cuda:0")
print("Load model...")
model = modeling.PanopticReconstruction()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of Trainable Parameters: {}".format(pytorch_total_params))

model_dict = model.state_dict()
checkpoint = torch.load(config.MODEL.PRETRAIN, map_location=device)
pretrained_dict = checkpoint["model"]
model_dict.update(pretrained_dict)
# model.load_state_dict(checkpoint["model"])  # load model checkpoint
model.load_state_dict(model_dict) # Load pretrained parameters
model = model.to(device)  # move to gpu
model.switch_test()


from pathlib import Path
import lib.visualize as vis
from lib.structures.frustum import compute_camera2frustum_transform
from lib.structures import DepthMap

img_names = ["b4dd7f88-1d68-47ff-b99a-0a96d8b8b116/0009"]

for img_name in img_names:
    img_folder = img_name.split('/')[0]
    img_id = img_name.split('/')[1]
    
    input_path = "data/front3d/{}/rgb_{}.png".format(img_folder, img_id)
    print("\nprocessing image: ", input_path)
    
    config.OUTPUT_DIR='output/00/{}/{}_{}'.format("model_color",img_folder,img_id)

    # Define image transformation.
    color_image_size = (320, 240)
    depth_image_size = (160, 120)

    imagenet_stats = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_transforms = t2d.Compose([
        t2d.Resize(color_image_size),
        t2d.ToTensor(),
        t2d.Normalize(imagenet_stats[0], imagenet_stats[1]),  # use imagenet stats to normalize image
    ])
    #Open and prepare input image.
    input_image = Image.open(input_path)
    input_image = image_transforms(input_image)
    input_image = input_image.unsqueeze(0).to(device)

    # Prepare intrinsic matrix.
    front3d_intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC)
    front3d_intrinsic = adjust_intrinsic(front3d_intrinsic, color_image_size, depth_image_size)
    front3d_intrinsic = torch.from_numpy(front3d_intrinsic).to(device).float()

    # Prepare frustum mask.
    front3d_frustum_mask = np.load(str("data/frustum_mask.npz"))["mask"]
    front3d_frustum_mask = torch.from_numpy(front3d_frustum_mask).bool().to(device).unsqueeze(0).unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        results = model.inference(input_image, front3d_intrinsic, front3d_frustum_mask)
    end = time.time()
    print("Inference time: {}".format(end - start))
    print(f"Visualize results, save them at {config.OUTPUT_DIR}")
    
    
    ## Save imgs
    depth_map: DepthMap = results["depth"]

    # visualize results
    output_path = config.OUTPUT_DIR
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Save Meshes
    dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
    min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
    truncation = config.MODEL.FRUSTUM3D.TRUNCATION
    iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

    geometry = results["frustum"]["geometry"]
    surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)
    instances = results["panoptic"]["panoptic_instances"]
    semantics = results["panoptic"]["panoptic_semantics"]
    color = results["panoptic"]["rgb"]
    color = color.permute(1,2,3,0)

    camera2frustum = compute_camera2frustum_transform(depth_map.intrinsic_matrix.cpu(), torch.tensor(results["input"].size()) / 2.0,
                                                          config.MODEL.PROJECTION.DEPTH_MIN,
                                                          config.MODEL.PROJECTION.DEPTH_MAX,
                                                          config.MODEL.PROJECTION.VOXEL_SIZE)

    # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
    camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
    frustum2camera = torch.inverse(camera2frustum)
    vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply", transform=frustum2camera)
    vis.write_distance_field(surface.squeeze(), color.squeeze(), output_path / "mesh_color.ply", transform=frustum2camera, is_color=True)
    vis.write_distance_field(surface.squeeze(), instances.squeeze(), output_path / "mesh_instances.ply", transform=frustum2camera)
    vis.write_distance_field(surface.squeeze(), semantics.squeeze(), output_path / "mesh_semantics.ply", transform=frustum2camera)


