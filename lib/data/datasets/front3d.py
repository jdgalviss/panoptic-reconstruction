import os
import random
import zipfile
from pathlib import Path
from typing import Dict, Union, List, Tuple

import numpy as np
import torch.utils.data
from PIL import Image, ImageEnhance
import pyexr

from lib import data
from lib.data import transforms2d as t2d
from lib.data import transforms3d as t3d
from lib.structures import FieldList
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
import torch
from skimage import io, color

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

class ToLABTensor(object):
    """Convert RGB image to LAB."""
    def __int__(self):
        print("ToLab")

    def __call__(self, rgb):
        """
        Args:
            rgb (np.array): RGB image array.
        Returns:
            np.array: LAB image array.
        """

        rgb = np.asarray(rgb, dtype=np.float32)/255.0
        # print("\nRGB shape:", rgb.shape)
        # print("RGB range: [{}, {}]".format(np.amin(rgb), np.amax(rgb)))
        lab = color.rgb2lab(rgb)
        lab[:,:,0] = (lab[:,:,0] / 100.0)
        lab[:,:,1:] = (lab[:,:,1:] + 100.0)/200.0
        
        #Normalize
        lab = 2.0 * (lab - 0.5)

        lab = torch.from_numpy(lab.astype(np.float32))
        return lab.permute(2,0,1)

class Front3D(torch.utils.data.Dataset):
    def __init__(self, file_list_path: os.PathLike, dataset_root_path: os.PathLike, fields: List[str],
                 num_samples: int = None, shuffle: bool = False) -> None:
        super().__init__()

        self.dataset_root_path = Path(dataset_root_path)

        self.samples: List = self.load_and_filter_file_list(file_list_path)

        if shuffle:
            random.shuffle(self.samples)

        self.samples = self.samples[:num_samples]

        # Fields defines which data should be loaded
        if fields is None:
            fields = []

        self.fields = fields

        self.image_size = (320, 240)
        self.depth_image_size = (160, 120)
        self.intrinsic = self.prepare_intrinsic()
        self.voxel_size = config.MODEL.PROJECTION.VOXEL_SIZE
        self.depth_min = config.MODEL.PROJECTION.DEPTH_MIN
        self.depth_max = config.MODEL.PROJECTION.DEPTH_MAX
        self.grid_dimensions = config.MODEL.FRUSTUM3D.GRID_DIMENSIONS
        self.truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        self.max_instances = config.MODEL.INSTANCE2D.MAX
        self.num_min_instance_pixels = config.MODEL.INSTANCE2D.MIN_PIXELS
        self.stuff_classes = [0, 10, 11, 12]

        self.frustum_mask: torch.Tensor = self.load_frustum_mask()

        self.transforms: Dict = self.define_transformations()

    def __getitem__(self, index) -> Tuple[str, FieldList]:
        sample_path = self.samples[index]
        scene_id = sample_path.split("/")[0]
        image_id = sample_path.split("/")[1]

        sample = FieldList(self.image_size, mode="xyxy")
        sample.add_field("index", index)
        sample.add_field("name", sample_path)

        try:

            # 2D data
            if "color" in self.fields:
                color = Image.open(self.dataset_root_path / scene_id / f"rgb_{image_id}.png", formats=["PNG"])
                color = self.transforms["color"](color)
                sample.add_field("color", color)

            if "depth" in self.fields:
                depth = pyexr.read(str(self.dataset_root_path / scene_id / f"depth_{image_id}.exr")).squeeze()[::-1, ::-1].copy()
                depth = self.transforms["depth"](depth)
                sample.add_field("depth", depth)

            if "instance2d" in self.fields:
                segmentation2d = np.load(self.dataset_root_path / scene_id / f"segmap_{image_id}.mapped.npz")["data"]
                instance2d = self.transforms["instance2d"](segmentation2d)
                sample.add_field("instance2d", instance2d)

            # 3D data
            needs_weighting = False

            if "geometry" in self.fields:
                
                if config.MODEL.FRUSTUM3D.IS_SDF:
                    geometry_path = self.dataset_root_path / scene_id / f"geometry_sdf_{image_id}.npz"
                    geometry = np.load(geometry_path)["arr_0"]
                else:
                    geometry_path = self.dataset_root_path / scene_id / f"geometry_{image_id}.npz"
                    geometry = np.load(geometry_path)["data"]
                geometry = np.ascontiguousarray(np.flip(geometry, axis=[0, 1]))  # Flip order, thanks for pointing that out.
                geometry = self.transforms["geometry"](geometry)
                # process hierarchy
                sample.add_field("occupancy_256", self.transforms["occupancy_256"](geometry))
                sample.add_field("occupancy_128", self.transforms["occupancy_128"](geometry))
                sample.add_field("occupancy_64", self.transforms["occupancy_64"](geometry))

                geometry = self.transforms["geometry_truncate"](geometry)
                sample.add_field("geometry", geometry)

                # add frustum mask
                sample.add_field("frustum_mask", self.frustum_mask.clone())

                needs_weighting = True

            if "semantic3d" or "instance3d" in self.fields:
                segmentation3d_path = self.dataset_root_path / scene_id / f"segmentation_{image_id}.mapped.npz"
                segmentation3d_data = np.load(segmentation3d_path)["data"]
                segmentation3d_data = np.copy(np.flip(segmentation3d_data, axis=[1, 2]))  # Flip order, thanks for pointing that out.
                semantic3d, instance3d = segmentation3d_data
                needs_weighting = True

                if "semantic3d" in self.fields:
                    semantic3d = self.transforms["semantic3d"](semantic3d)
                    sample.add_field("semantic3d", semantic3d)

                    # process semantic3d hierarchy
                    sample.add_field("semantic3d_64", self.transforms["segmentation3d_64"](semantic3d))
                    sample.add_field("semantic3d_128", self.transforms["segmentation3d_128"](semantic3d))

                if "instance3d" in self.fields:
                    # Ensure consistent instance id shuffle between 2D and 3D instances
                    instance_mapping = sample.get_field("instance2d").get_field("instance_mapping")
                    instance3d = self.transforms["instance3d"](instance3d, mapping=instance_mapping)
                    sample.add_field("instance3d", instance3d)

                    # process instance3d hierarchy
                    sample.add_field("instance3d_64", self.transforms["segmentation3d_64"](instance3d))
                    sample.add_field("instance3d_128", self.transforms["segmentation3d_128"](instance3d))
            
            if "aux_views" in self.fields:
                aux_views = []
                cam_poses = []
                
                # views_names = [image_id]
                views_names = []

                if config.MODEL.FRUSTUM3D.NUM_VIEWS > 0:
                    try:
                        views_list = open(self.dataset_root_path / scene_id / f"viewlist_{image_id}.txt", 'r')
                        for view_name in views_list.readlines():
                            views_names.append(view_name.replace('\n',''))
                    except:
                        print("\nCould not find viewlist.txt")
                        print(self.dataset_root_path / scene_id / f"viewslist_{image_id}.txt")

                for i, aux_img_id in enumerate(views_names):
                    if i < config.MODEL.FRUSTUM3D.NUM_VIEWS:
                        aux_img = Image.open(self.dataset_root_path / scene_id / f"rgb_{aux_img_id}.png", formats=["PNG"])
                        if config.MODEL.FRUSTUM3D.ENHANCE_CONTRAST:
                            enhancer = ImageEnhance.Contrast(aux_img)
                            aux_img = enhancer.enhance(1.2)
                        # aux_img = aux_img.astype(np.float32)/255.0
                        aux_img = self.transforms["aux_views"](aux_img)
                        # aux_image = t2d.ToTensor(aux_img)
                        # print("aux_img.shape", aux_img.shape)
                        # print("aux_image range: [{},{}]".format(torch.min(aux_img), torch.max(aux_img)))
                        aux_views.append(aux_img)

                        campose_path = self.dataset_root_path / scene_id / f"campose_{aux_img_id}.npz"
                        cam2world = np.load(campose_path)["blender_matrix"]
                        cam_poses.append(torch.from_numpy(cam2world).type(torch.FloatTensor).unsqueeze(0))
                sample.add_field("aux_views", torch.stack(aux_views))
                sample.add_field("cam_poses", torch.stack(cam_poses))


            if needs_weighting:
                weighting_path = self.dataset_root_path / scene_id / f"weighting_{image_id}.npz"
                weighting = np.load(weighting_path)["data"]
                weighting = np.copy(np.flip(weighting, axis=[0, 1]))  # Flip order, thanks for pointing that out.
                weighting = self.transforms["weighting3d"](weighting)
                sample.add_field("weighting3d", weighting)

                # Process weighting mask hierarchy
                sample.add_field("weighting3d_64", self.transforms["weighting3d_64"](weighting))
                sample.add_field("weighting3d_128", self.transforms["weighting3d_128"](weighting))

            return sample_path, sample
        except Exception as e:
            print(sample_path)
            print(e)
            return None, sample

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_and_filter_file_list(file_list_path: os.PathLike) -> List[str]:
        with open(file_list_path) as f:
            content = f.readlines()

        images = [line.strip() for line in content]

        return images

    def load_frustum_mask(self) -> torch.Tensor:
        mask_path = self.dataset_root_path / "frustum_mask.npz"
        mask = np.load(str(mask_path))["mask"]
        mask = torch.from_numpy(mask).bool()

        return mask

    def define_transformations(self) -> Dict:
        transforms = dict()

        # 2D transforms
        
        if config.MODEL.COLOR_SPACE == "LAB":
            transforms["aux_views"] = t2d.Compose([
                ToLABTensor(),
            ])
        else:
            transforms["aux_views"] = t2d.Compose([
                t2d.ToTensor(),
                t2d.Normalize(_imagenet_stats["mean"], _imagenet_stats["std"])
            ])

        transforms["color"] = t2d.Compose([
            t2d.ToTensor(),
            t2d.Normalize(_imagenet_stats["mean"], _imagenet_stats["std"])
        ])

        transforms["depth"] = t2d.Compose([
            t2d.ToImage(),
            t2d.Resize(self.depth_image_size, Image.NEAREST),
            t2d.ToNumpyArray(),
            t2d.ToTensorFromNumpy(),
            t2d.ToDepthMap(self.intrinsic)  # 3D-Front has single intrinsic matrix
        ])

        transforms["instance2d"] = t2d.Compose([
            t2d.SegmentationToMasks(self.image_size, self.num_min_instance_pixels, self.max_instances, True, self.stuff_classes)
        ])

        # 3D transforms
        transforms["geometry"] = t3d.Compose([
            t3d.ToTensor(dtype=torch.float),
            t3d.Unsqueeze(0),
            t3d.ToTSDF(truncation=12)
        ])

        transforms["geometry_truncate"] = t3d.ToTDF(truncation=self.truncation)

        transforms["occupancy_64"] = t3d.Compose([t3d.ResizeTrilinear(0.25), t3d.ToBinaryMask(8), t3d.ToTensor(dtype=torch.float)])
        transforms["occupancy_128"] = t3d.Compose([t3d.ResizeTrilinear(0.5), t3d.ToBinaryMask(6), t3d.ToTensor(dtype=torch.float)])
        transforms["occupancy_256"] = t3d.Compose([t3d.ToBinaryMask(self.truncation), t3d.ToTensor(dtype=torch.float)])

        transforms["weighting3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.float), t3d.Unsqueeze(0)])
        transforms["weighting3d_64"] = t3d.ResizeTrilinear(0.25)
        transforms["weighting3d_128"] = t3d.ResizeTrilinear(0.5)

        transforms["semantic3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.long)])

        transforms["instance3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.long), t3d.Mapping(mapping={}, ignore_values=[0])])

        transforms["segmentation3d_64"] = t3d.Compose([t3d.ResizeMax(8, 4, 2)])
        transforms["segmentation3d_128"] = t3d.Compose([t3d.ResizeMax(4, 2, 1)])

        return transforms

    def prepare_intrinsic(self) -> torch.Tensor:
        intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC).reshape((4, 4))
        intrinsic_adjusted = adjust_intrinsic(intrinsic, self.image_size, self.depth_image_size)
        intrinsic_adjusted = torch.from_numpy(intrinsic_adjusted).float()

        return intrinsic_adjusted
