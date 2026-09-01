"""
Follica Dataset & Calibration Loader Module.

Handles loading base public datasets (Kaggle/Roboflow) and merging the
volunteer-collected calibration set (100-200 consented photos across Fitzpatrick skin types).
"""

import os
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ScalpDataset(Dataset):
    """
    Standard PyTorch Dataset for Scalp Condition Classification.
    """
    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms():
    """Returns training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def load_dataset_with_calibration(
    base_data_dir: str,
    calibration_data_dir: Optional[str] = None
) -> Tuple[List[str], List[int], dict]:
    """
    Loads base Kaggle/Roboflow images and seamlessly merges the volunteer-collected calibration set.

    ===================================================================================
    [PLACEHOLDER STEP: VOLUNTEER CALIBRATION SET MERGE]
    Consented calibration photos (100-200 samples) should be organized into:
      calibration_data_dir/
         0_normal/
         1_dry_scalp_dandruff/
         2_seborrheic_dermatitis/
         3_alopecia_areata/
         4_folliculitis/
    ===================================================================================
    """
    classes = [
        "0_normal",
        "1_dry_scalp_dandruff",
        "2_seborrheic_dermatitis",
        "3_alopecia_areata",
        "4_folliculitis"
    ]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    file_paths = []
    labels = []

    # 1. Load Base Dataset
    if os.path.exists(base_data_dir):
        for cls_name in classes:
            cls_dir = os.path.join(base_data_dir, cls_name)
            if os.path.isdir(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        file_paths.append(os.path.join(cls_dir, fname))
                        labels.append(class_to_idx[cls_name])

    # 2. Merge Volunteer Calibration Set
    calibration_count = 0
    if calibration_data_dir and os.path.exists(calibration_data_dir):
        print(f"\n[Follica Pipeline] Merging Volunteer Calibration Set from {calibration_data_dir}...")
        for cls_name in classes:
            cls_dir = os.path.join(calibration_data_dir, cls_name)
            if os.path.isdir(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        file_paths.append(os.path.join(cls_dir, fname))
                        labels.append(class_to_idx[cls_name])
                        calibration_count += 1
        print(f"[Follica Pipeline] Successfully incorporated {calibration_count} consented volunteer calibration images.")
    else:
        print("[Follica Pipeline] Notice: No external calibration directory specified. Using base public dataset only.")

    return file_paths, labels, class_to_idx
