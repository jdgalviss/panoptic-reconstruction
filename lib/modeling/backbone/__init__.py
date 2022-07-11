from .resnet_encoder import ResNetEncoder
from .resnet import ResNet
from .utils import build_backbone
from lib.config import config

if config.MODEL.FRUSTUM3D.SEPARATE_ENCODER:
    from .unet_sparse_panoptic import UNetSparse
else:
    from .unet_sparse import UNetSparse

from .unet_sparse_rgb import UNetSparseRGB
from .multitask_heads_sparse import GeometryHeadSparse, ClassificationHeadSparse, ColorHeadSparse
