import torch
from torchmetrics import StructuralSimilarityIndexMeasure
from lib.metrics import Metric
from torchvision import transforms
import torch.nn as nn

class L1ReconstructionLoss(Metric):
    """
    Compute L1 reconstruction loss
    
    """
    def __init__(self):
        super().__init__()
        self.loss = 0.0
        self.num_imgs = 0

    def add(self, imgs, targets, num_valid):
        """
        Add L1 reconstruction loss
        Args:
            imgs: images to be reconstructed
            targets: ground truth images
            num_valid: number of valid pixels
        """
        if num_valid != 0:
            loss = torch.sum(torch.abs(imgs - targets)) / num_valid
        else:
            loss = 0.0
        self.loss += loss
        self.num_imgs += 1

    def reduce(self):
        return self.loss / self.num_imgs

class SSIM(Metric):
    """
    Compute STRUCTURAL SIMILARITY INDEX MEASURE (SSIM) Metric
    
    """
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.value = 0.0
        self.num_imgs = 0

    def add(self, imgs, targets):
        """
        Add SSIM value
        
        Args:
            img: images to be reconstructed
            targets: ground truth images
        s"""
        self.value += self.ssim(imgs, targets)
        self.num_imgs += 1
        self.ssim.reset()

    def reduce(self):
        return self.value / self.num_imgs

class Feature_L1(Metric):
    """
    Compute Feature L1
    
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='mean')
        self.value = 0.0
        self.num_imgs = 0
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).cuda().eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model.fc = nn.Identity()


    def add(self, imgs, targets):
        """
        Add L1 reconstruction loss
        Args:
            imgs: images to be reconstructed
            targets: ground truth images
            num_valid: number of valid pixels
        """
        preprocessed_imgs = self.preprocess(imgs)
        preprocessed_targets = self.preprocess(targets)
        features_imgs = self.model(preprocessed_imgs)
        features_targets = self.model(preprocessed_targets)
        l1_dist = self.l1(features_imgs, features_targets)
        self.value += l1_dist
        self.num_imgs += 1
        
    def reduce(self):
        return self.value / self.num_imgs