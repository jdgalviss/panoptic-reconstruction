import time
from collections import OrderedDict
from pathlib import Path

import torch
from lib.structures.field_list import collect

from lib import utils, logger, config, modeling, solver, data, metrics

import glob
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict

from lib.utils import re_seed
from tqdm import tqdm
from lib.modeling.utils import thicken_grid

from lib.modeling.frustum.renderer_proxy import Renderer
import torchvision
import numpy as np
from lib.utils.intrinsics import adjust_intrinsic
import pprint

stages = ["64","128","256"]
_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for im in tensor:
            for ch, m, s in zip(im, self.mean, self.std):
                ch.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return tensor

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


class Trainer:
    def __init__(self, output_path: Path) -> None:
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.checkpointer = None
        self.dataloader = None
        self.val_dataloader = None
        self.logger = logger
        self.meters = utils.MetricLogger(delimiter="  ")
        self.checkpoint_arguments = {}
        self.output_path = output_path
        self.val_metric = None
        self.unnormalize = None
        self.setup()

    def setup(self) -> None:
        re_seed(0)

        # Setup model
        self.model = modeling.PanopticReconstruction()

        device = torch.device(config.MODEL.DEVICE)
        self.model.to(device, non_blocking=True)

        self.model.log_model_info()
        self.model.fix_weights()

        # Setup optimizer, scheduler, checkpointer
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.SOLVER.BASE_LR,
                                          betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                          weight_decay=config.SOLVER.WEIGHT_DECAY)
        self.scheduler = solver.WarmupMultiStepLR(self.optimizer, config.SOLVER.STEPS, config.SOLVER.GAMMA,
                                                  warmup_factor=1,
                                                  warmup_iters=0,
                                                  warmup_method="linear")

        
        self.checkpointer = utils.DetectronCheckpointer(self.model, self.optimizer, self.scheduler, self.output_path)

        # Load the checkpoint
        checkpoint_data = self.checkpointer.load()

        # Additionally load a 2D model which overwrites the previously loaded weights
        # TODO: move to checkpointer?
        if config.MODEL.PRETRAIN2D:
            pretrain_2d = torch.load(config.MODEL.PRETRAIN2D)
            self.model.load_state_dict(pretrain_2d["model"])

        self.checkpoint_arguments["iteration"] = 0

        if config.SOLVER.LOAD_SCHEDULER:
            self.checkpoint_arguments.update(checkpoint_data)

        # Dataloader
        self.dataloader = data.setup_dataloader(config.DATASETS.TRAIN)
        print("VAL DATASET FOLDER: ", config.DATASETS.VAL)
        self.val_dataloader = data.setup_dataloader(config.DATASETS.VAL, is_train=False, shuffle=False)

        # Prepare evaluation metric
        self.val_metric = metrics.PanopticReconstructionQuality()

        # Renderers
        self.renderer_64 = Renderer(camera_base_transform=None, voxelsize=0.03*256./64, volume_size=64)
        self.renderer_128 = Renderer(camera_base_transform=None, voxelsize=0.03*256./128, volume_size=128)
        self.renderer_256 = Renderer(camera_base_transform=None, voxelsize=0.03*256./254, volume_size=254)

        # Unnormalize transform
        self.unnormalize = UnNormalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))

    def do_train(self) -> None:
        device = torch.device(config.MODEL.DEVICE)

        # Log start logging
        self.logger.info(f"Start training {self.checkpointer.output_path.name}")

        # Switch training mode
        self.model.switch_training()

        # Main loop
        iteration = config.MODEL.ITERATION
        iteration_end = time.time()

        # Summary Writter
        writer = SummaryWriter(log_dir=str(self.output_path / "tensorboard"))

        # Prepare intrinsic matrix for eval
        color_image_size = (320, 240)
        depth_image_size = (160, 120)
        front3d_intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC)
        front3d_intrinsic = adjust_intrinsic(front3d_intrinsic, color_image_size, depth_image_size)
        front3d_intrinsic = torch.from_numpy(front3d_intrinsic).to(device).float()

        # Prepare frustum mask for eval
        front3d_frustum_mask = np.load(str("data/frustum_mask.npz"))["mask"]
        front3d_frustum_mask = torch.from_numpy(front3d_frustum_mask).bool().to(device).unsqueeze(0).unsqueeze(0)


        for idx, (image_ids, targets) in enumerate(self.dataloader):
            assert targets is not None, "error during data loading"
            data_time = time.time() - iteration_end

            # Get input images
            images = collect(targets, "color")

            # Pass through model
            try:
                losses, results = self.model(images, targets)
            except Exception as e:
                print(e, "skipping", image_ids[0])
                del targets, images
                continue

            # Accumulate total loss
            total_loss: torch.Tensor = 0.0
            log_meters = OrderedDict()

            for loss_group in losses.values():
                for loss_name, loss in loss_group.items():

                    if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
                        if "256" in loss_name:
                            writer.add_scalar('train_256/'+loss_name, loss.detach().cpu(), iteration)
                        elif "128" in loss_name:
                            writer.add_scalar('train_128/'+loss_name, loss.detach().cpu(), iteration)
                        elif "64" in loss_name:
                            writer.add_scalar('train_64/'+loss_name, loss.detach().cpu(), iteration)
                        else:
                            writer.add_scalar('train_full/'+loss_name, loss.detach().cpu(), iteration)
                        if not ("dbg" in loss_name):
                            total_loss += loss
                            log_meters[loss_name] = loss.item()

            writer.add_scalar('train/total_loss', total_loss.detach().cpu(), iteration)

            # Loss backpropagation, optimizer & scheduler step
            self.optimizer.zero_grad()

            if torch.is_tensor(total_loss):
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                log_meters["total"] = total_loss.item()
            else:
                log_meters["total"] = total_loss

            # Minkowski Engine recommendation
            torch.cuda.empty_cache()

            # Save checkpoint
            if iteration % config.SOLVER.CHECKPOINT_PERIOD == 0:
                self.checkpointer.save(f"model_{iteration:07d}", **self.checkpoint_arguments)

            last_training_stage = self.model.set_current_training_stage(iteration)

            # Save additional checkpoint after hierarchy level
            if last_training_stage is not None:
                self.checkpointer.save(f"model_{last_training_stage}_{iteration:07d}", **self.checkpoint_arguments)
                self.logger.info(f"Finish {last_training_stage} hierarchy level")

            # Gather logging information
            self.meters.update(**log_meters)
            batch_time = time.time() - iteration_end
            self.meters.update(time=batch_time, data=data_time)
            current_learning_rate = self.scheduler.get_lr()[0]
            current_training_stage = self.model.get_current_training_stage()

            self.logger.info(self.meters.delimiter.join([f"IT: {iteration:06d}", current_training_stage,
                                                         f"{str(self.meters)}", f"LR: {current_learning_rate}"]))
            writer.add_scalar('train/lr', current_learning_rate, iteration)


            # Evaluation
            if iteration % config.SOLVER.EVALUATION_PERIOD == 0 and (current_training_stage == "FULL" or current_training_stage == "LEVEL-256"):
                self.model.switch_test()
                print("Evaluating on {} samples".format(len(self.val_dataloader)))
                metrics = {}
                if True:
                    rendered_imgs = []
                    aux_views = []
                    for idx, (image_ids, targets) in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
                        if targets is None:
                            print(f"Error, {image_ids[0]}")
                            continue

                        # Get input images
                        images = collect(targets, "color")

                        # Pass through model
                        with torch.no_grad():
                            try:
                                results = self.model.inference(images, front3d_intrinsic, front3d_frustum_mask)
                            except Exception as e:
                                print(e)
                                del targets, images
                                continue

                        if config.DATASETS.NAME == "front3d":
                            frustum_mask = self.val_dataloader.dataset.frustum_mask
                        else:
                            frustum_mask = targets[0].get_field("frustum_mask").squeeze()

                        # Panoptic Reconstruction:
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
                        per_sample_result = self.val_metric.add(instance_information_pred, instance_information_gt)
                        # metrics.update(per_sample_result)
                        # pprint(per_sample_result)

                        # Render images to compute rgb metrics
                        cam_poses = collect(targets, "cam_poses")
                        views = collect(targets, "aux_views")
                        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
                        min_coordinates = torch.IntTensor([0, 0, 0]).to("cuda")
                        truncation = config.MODEL.FRUSTUM3D.TRUNCATION
                        sdf, _, _ = results['frustum']['geometry'].dense(dense_dimensions, min_coordinates, default_value=truncation)
                        rgb, _, _ = results['frustum']['rgb'].dense(dense_dimensions, min_coordinates, default_value=truncation)
                        imgs, _ = self.renderer_256.render_image(sdf,rgb,cam_poses)
                        # imgs = imgs.detach().cpu()
                        rendered_imgs.append(imgs)
                        aux_views.append(views[0])

                    # Compute L1 reconstruction loss
                    rendered_imgs = torch.cat(rendered_imgs)
                    aux_views = torch.cat(aux_views)
                    rendered_imgs = rendered_imgs.permute(0,3,1,2)
                    masks = (torch.logical_or(torch.isinf(rendered_imgs),torch.isnan(rendered_imgs)))
                    rendered_imgs[masks] = 0.0
                    aux_views[masks] = 0.0
                    l1_reconstruction_loss = torch.abs(rendered_imgs-aux_views)
                    N = (torch.sum(torch.logical_not(masks)).item())
                    if N != 0:
                        l1_reconstruction_loss  = torch.sum(l1_reconstruction_loss) / N
                        writer.add_scalar('eval/l1_reconstruction', l1_reconstruction_loss, iteration)
                    else:
                        print("No valid samples")
                    rendered_imgs = self.unnormalize(rendered_imgs)
                    rendered_imgs = torch.clamp(rendered_imgs, 0.0, 1.0)
                    grid_rendered_imgs = torchvision.utils.make_grid(rendered_imgs, nrow=4)
                    writer.add_image('eval/rendered_256', grid_rendered_imgs, iteration)
                    quantitative = self.val_metric.reduce()
                    # Print results
                    writer.add_scalar('eval/panoptic_quality', quantitative["pq"], iteration)
                    writer.add_scalar('eval/segmentation_quality', quantitative["sq"], iteration)
                    writer.add_scalar('eval/recognition_quality', quantitative["rq"], iteration)
                self.model.switch_training()
                
            iteration += 1
            iteration_end = time.time()

        self.checkpointer.save("model_final", **self.checkpoint_arguments)
