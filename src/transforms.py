# transforms.py
from albumentations import (Compose, HorizontalFlip, Normalize,
                            RandomResizedCrop, Resize, ShiftScaleRotate,
                            Transpose, VerticalFlip)
from albumentations.pytorch import ToTensorV2

from config import CFG


def get_transforms(*, data):
    if data == 'train':
        return Compose([
            # Resize(CFG.size, CFG.size),
            RandomResizedCrop(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
