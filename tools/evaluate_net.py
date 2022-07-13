import argparse
import os
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple, Dict

from lib import modeling, metrics, visualize, utils

from lib.data import setup_dataloader

from lib.modeling.utils import thicken_grid, UnNormalize
from lib.visualize.mesh import get_mesh

from lib.config import config
from lib.structures.field_list import collect


from tools.test_net_single_image import configure_inference
from lib.utils import re_seed
import glob
from datetime import datetime
from lib.utils.intrinsics import adjust_intrinsic
from torch.utils.tensorboard import SummaryWriter
from lib.modeling.frustum.renderer_proxy import Renderer
from lib.modeling.frustum.utils import convert_lab01_to_rgb_pt

def main(opts, start=0, end=None):
    configure_inference(opts)

    re_seed(0)
    device = torch.device("cuda:0")

    # basic paths
    files = glob.glob(config.OUTPUT_DIR + "*")
    max_file = max(files)
    now = str(datetime.now()) # current date and time
    out_dir = config.OUTPUT_DIR + "{:02d}_eval_{}".format(int(max_file.split('/')[-1].split('_')[0])+1, now.replace(' ','').replace(':','_'))
    # out_dir = 'output/01' #TODO: Temporary
    # os.mkdir(out_dir)
    print('Saving Evaluation results to:', out_dir)

    output_path = Path(out_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    output_config_path = output_path / "config.yaml"
    utils.save_config(config, output_config_path)
    utils.setup_logger(output_path, "log.txt")
    writer = SummaryWriter(log_dir=str(output_path / "tensorboard"))


    # Define model and load checkpoint.
    print("Load model...")
    model = modeling.PanopticReconstruction()
    checkpoint = torch.load(config.MODEL.PRETRAIN)
    model.load_state_dict(checkpoint["model"])  # load model checkpoint
    model = model.to(device)  # move to gpu
    model.switch_test()

    # Define dataset
    # config.DATALOADER.NUM_WORKERS=0
    dataloader = setup_dataloader(config.DATASETS.TRAINVAL, is_train=False, shuffle=False)
    # dataloader.dataset.samples = dataloader.dataset.samples[start:end]
    print("Evaluating on {} samples".format(len(dataloader)))


    # Prepare metric
    metric = metrics.PanopticReconstructionQuality()

    # Prepare intrinsic matrix for eval
    color_image_size = (320, 240)
    depth_image_size = (160, 120)
    front3d_intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC)
    front3d_intrinsic = adjust_intrinsic(front3d_intrinsic, color_image_size, depth_image_size)
    front3d_intrinsic = torch.from_numpy(front3d_intrinsic).to(device).float()

    # Prepare frustum mask for eval
    front3d_frustum_mask = np.load(str("data/frustum_mask.npz"))["mask"]
    front3d_frustum_mask = torch.from_numpy(front3d_frustum_mask).bool().to(device).unsqueeze(0).unsqueeze(0)

    # For Color Image Rendering
    renderer_256 = Renderer(camera_base_transform=None, voxelsize=0.03*256./254, volume_size=254)
    # Unnormalize transform
    unnormalize = UnNormalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))

    # Color Metrics
    l1_metric = metrics.L1ReconstructionLoss()
    ssim_metric = metrics.SSIM()
    feat_l1_metric = metrics.Feature_L1()
    color_metrics = {"L1": 0.0, "SSIM": 0.0, "FeatureL1": 0.0}

    num_images = 0
    for idx, (image_ids, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if idx % 2 == 0:
            continue
        if targets is None:
            print(f"Error, {image_ids[0]}")
            continue

        # Get input images
        images = collect(targets, "color")

        # Pass through model
        with torch.no_grad():
            try:
                results = model.inference(images, front3d_intrinsic, front3d_frustum_mask)
            except Exception as e:
                print(e)
                del targets, images
                continue

        if config.DATASETS.NAME == "front3d":
            frustum_mask = dataloader.dataset.frustum_mask
        else:
            frustum_mask = targets[0].get_field("frustum_mask").squeeze()

        # Prepare ground truth
        instances_gt, instance_semantic_classes_gt = _prepare_semantic_mapping(targets[0].get_field("instance3d").squeeze(),
                                                                               targets[0].get_field("semantic3d").squeeze())
        distance_field_gt = targets[0].get_field("geometry").squeeze()
        instance_information_gt = _prepare_instance_masks_thicken(instances_gt, instance_semantic_classes_gt,
                                                                  distance_field_gt, frustum_mask)

        # Prepare prediction
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        distance_field_pred = results["frustum"]["geometry"].dense(dense_dimensions, min_coordinates, default_value=truncation)[0].to("cpu", non_blocking=False)
        instances_pred = results["panoptic"]["panoptic_instances"].to("cpu", non_blocking=False)

        instance_semantic_classes_pred = results["panoptic"]["panoptic_semantic_mapping"]
        instance_information_pred = _prepare_instance_masks_thicken(instances_pred, instance_semantic_classes_pred,
                                                                    distance_field_pred, frustum_mask)

        # Add to metric
        # Format: Dict[instance_id: instance_mask, semantic_label]
        per_sample_result = metric.add(instance_information_pred, instance_information_gt)

        file_name = image_ids[0].replace("/", "_")
        with open(output_path / f"{file_name}.json", "w") as f:
            json.dump({k: cat.as_metric for k, cat in per_sample_result.items()}, f, indent=4)

        # Color Metrics
        cam_poses = collect(targets, "cam_poses")
        views = collect(targets, "aux_views")
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to("cuda")
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        sdf, _, _ = results['frustum']['geometry'].dense(dense_dimensions, min_coordinates, default_value=truncation)
        rgb, _, _ = results['frustum']['rgb'].dense(dense_dimensions, min_coordinates, default_value=truncation)
        imgs, _ = renderer_256.render_image(sdf,rgb,cam_poses)

        views = views.permute(0,1,3,4,2)
        views = views[0]
        masks = (torch.logical_or(torch.isinf(imgs),torch.isnan(imgs)))
        imgs[masks] = 0.0
        views[masks] = 0.0
        N = (torch.sum(torch.logical_not(masks)).item())
        imgs = imgs.permute(0,3,1,2)
        views = views.permute(0,3,1,2)

        imgs = unnormalize(imgs)
        views = unnormalize(views)

        imgs = torch.clamp(imgs, 0.0, 1.0)
        with torch.no_grad():
            l1_metric.add(imgs, views,N)
            ssim_metric.add(imgs, views)
            feat_l1_metric.add(imgs, views)
            num_images +=1
    
    # Reduce metric
    quantitative = metric.reduce()
    writer.add_scalar('eval/panoptic_quality', quantitative["pq"])
    writer.add_scalar('eval/segmentation_quality', quantitative["sq"])
    writer.add_scalar('eval/recognition_quality', quantitative["rq"])

    # Color Metrics
    color_metrics['L1'] = l1_metric.reduce()
    color_metrics['SSIM'] = ssim_metric.reduce()
    color_metrics['FeatureL1'] = feat_l1_metric.reduce()

    print("\nColor Metrics: ")
    for k, v in color_metrics.items():
        writer.add_scalar(f'eval/{k}', v)
        print(f"{k}: {v}")

    print("\nPanoptic Reconstruction Metrics:")
    # Print results
    for k, v in quantitative.items():
        print(f"{k:>5}", f"{v:.3f}")


def _prepare_instance_masks_thicken(instances, semantic_mapping, distance_field, frustum_mask) -> Dict[int, Tuple[torch.Tensor, int]]:
    instance_information = {}

    for instance_id, semantic_class in semantic_mapping.items():
        instance_mask: torch.Tensor = (instances == instance_id)
        instance_distance_field = torch.full_like(instance_mask, dtype=torch.float, fill_value=3.0)
        instance_distance_field[instance_mask] = distance_field.squeeze()[instance_mask]
        instance_distance_field_masked = instance_distance_field.abs() < 1.0

        # instance_grid = instance_grid & frustum_mask
        instance_grid = thicken_grid(instance_distance_field_masked, [256, 256, 256], frustum_mask)
        instance_information[instance_id] = instance_grid, semantic_class

    return instance_information


def _prepare_semantic_mapping(instances, semantics, offset=2):
    semantic_mapping = {}
    panoptic_instances = torch.zeros_like(instances).int()

    things_start_index = offset  # map wall and floor to id 1 and 2

    unique_instances = instances.unique()
    for index, instance_id in enumerate(unique_instances):
        # Ignore freespace
        if instance_id != 0:
            # Compute 3d instance surface mask
            instance_mask: torch.Tensor = (instances == instance_id)
            # instance_surface_mask = instance_mask & surface_mask
            panoptic_instance_id = index + things_start_index
            panoptic_instances[instance_mask] = panoptic_instance_id

            # get semantic prediction
            semantic_region = torch.masked_select(semantics, instance_mask)
            semantic_things = semantic_region[
                (semantic_region != 0) & (semantic_region != 10) & (semantic_region != 11)]

            unique_labels, semantic_counts = torch.unique(semantic_things, return_counts=True)
            max_count, max_count_index = torch.max(semantic_counts, dim=0)
            selected_label = unique_labels[max_count_index]

            semantic_mapping[panoptic_instance_id] = selected_label.int().item()

    # Merge stuff classes
    # Merge floor class
    wall_class = 10
    wall_id = 1
    wall_mask = semantics == wall_class
    panoptic_instances[wall_mask] = wall_id
    semantic_mapping[wall_id] = wall_class

    # Merge floor class
    floor_class = 11
    floor_id = 2
    floor_mask = semantics == floor_class
    panoptic_instances[floor_mask] = floor_id
    semantic_mapping[floor_id] = floor_class

    return panoptic_instances, semantic_mapping


def evaluate_jsons(opts):
    result_path = Path(opts.output)
    samples = [file for file in result_path.iterdir() if file.suffix == ".json"]

    print(f"Found {len(samples)} samples")
    metric = metrics.PanopticReconstructionQuality()
    for sample in tqdm(samples):
        try:
            content = json.load(open(sample))
            data = {}
            for k, cat in content.items():
                panoptic_sample = metrics.PQStatCategory()
                panoptic_sample.iou = cat["iou"]
                panoptic_sample.tp = cat["tp"]
                panoptic_sample.fp = cat["fp"]
                panoptic_sample.fn = cat["fn"]
                panoptic_sample.n = cat["n"]
                data[int(k)] = panoptic_sample

            metric.add_sample(data)
        except Exception as e:
            print(f"Error with {sample}")
            continue

    summary = metric.reduce()

    for name, value in summary.items():
        if name[0] == "n":
            print(f"{name:>10}\t\t{value:>5d}")
        else:
            print(f"{name:>10}\t\t{value:>5.3f}")

    with open(result_path / "panoptic_result.json", "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="output/")
    parser.add_argument("--config-file", "-c", type=str, default="configs/front3d_evaluate.yaml")
    parser.add_argument("--eval-only", type=bool, default=False)
    # parser.add_argument("--model", "-m", type=str, default="resources/panoptic-front3d.pth")
    parser.add_argument("-s", type=int, default=0)
    parser.add_argument("-e", type=int, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    config.OUTPUT_DIR = args.output
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)

    if args.eval_only:
        evaluate_jsons(args)
    else:
        main(args, args.s, args.e)
