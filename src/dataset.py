import torch
import albumentations as A
from torch.utils.data import Dataset

# Data augmentation operations to be used only in the Train set
train_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5, border_mode=0),
])

class FastBratsDataset(Dataset):
    """
    PyTorch Dataset class for loading preprocessed BraTS 2D slices.
    """
    def __init__(self, files_list_path, augment=False):
        # Since the files are already filtered, we load them directly here.
        self.files = torch.load(files_list_path)
        # Flag indicating whether data augmentation will be applied (True for Train, False for Val/Test)
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])
        img = sample['image'].float()   # Shape: (4, 240, 240)
        mask = sample['mask'].float()   # Shape: (3, 240, 240)

        # If augment=True (meaning we are in the Train set), apply Albumentations
        if self.augment:
            # Albumentations expects (H, W, C) format, converting tensors to this format
            img_np = img.permute(1, 2, 0).numpy()
            mask_np = mask.permute(1, 2, 0).numpy()
            
            # Apply the SAME random transformations to both image and mask synchronously
            augmented = train_augmentation(image=img_np, mask=mask_np)
            
            # Convert back to PyTorch tensor format (C, H, W)
            img = torch.tensor(augmented['image'], dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(augmented['mask'], dtype=torch.float32).permute(2, 0, 1)

        return img, mask