import os
import glob
import torch
import shutil
import numpy as np
import nibabel as nib
import random  
from scipy import ndimage
from tqdm import tqdm

DATA_DIR = './data/BraTS2021_Training_Data'
OUT_DIR = './data/preprocessed_data'

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

def _normalize_modality(img_slice):
    mean = np.mean(img_slice)
    std = np.std(img_slice)
    
    if std > 0:
        # Standard Z-score formula: z = (x - µ) / σ
        # Added epsilon (1e-8) to denominator to prevent ZeroDivisionError.
        normalized = (img_slice - mean) / (std + 1e-8)
    else:
        # If the entire matrix has the same value (completely empty/black slice etc.), 
        # standard deviation becomes 0. In this case, fill the matrix with 0s.
        normalized = np.zeros_like(img_slice)
        
    return normalized.astype(np.float32)

def _find_file(patient_path, keyword):
    patterns = [os.path.join(patient_path, f"*_{keyword}.nii.gz"), os.path.join(patient_path, f"*_{keyword}.nii")]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches: return matches[0]
    return None

# PATIENT-LEVEL SPLITTING
print("Patient list -> Splitting into Train/Val/Test")
all_entries = glob.glob(os.path.join(DATA_DIR, '*'))

# Only the first 300 patients (to save resources)
patient_paths = sorted([p for p in all_entries if os.path.isdir(p)])[:300]

# Shuffle for random but reproducible slicing
random.seed(42)
random.shuffle(patient_paths)

# Split patients into three: 70% Train, 15% Validation, 15% Test
n_total = len(patient_paths)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)

train_paths = patient_paths[:n_train]                    # First 70% portion
val_paths   = patient_paths[n_train:n_train + n_val]     # Middle 15% portion
test_paths  = patient_paths[n_train + n_val :]           # Remaining 15% portion

splits = {'train': train_paths, 'val': val_paths, 'test': test_paths}
saved_lists = {'train': [], 'val': [], 'test': []}

# Create folder for each set (train, val, test) and loop
for split_name, p_list in splits.items():
    out_split = os.path.join(OUT_DIR, split_name)
    os.makedirs(out_split, exist_ok=True)
    
    # Progress bar (tqdm) over patients in that group
    for patient_path in tqdm(p_list, desc=f"Processing {split_name} Set"):
        pid = os.path.basename(patient_path) # Get Patient ID from folder name
        
        # Find 4 different MRI sequences and 1 mask file for the patient
        flair_path = _find_file(patient_path, "flair")
        t1_path    = _find_file(patient_path, "t1")
        t1ce_path  = _find_file(patient_path, "t1ce")
        t2_path    = _find_file(patient_path, "t2")
        seg_path   = _find_file(patient_path, "seg")
        
        # If even one file is missing, skip this patient (to avoid errors)
        if not all([flair_path, t1_path, t1ce_path, t2_path, seg_path]): continue
        
        # Load 3D files in NIfTI (.nii.gz) format as arrays
        try:
            flair_img = nib.load(flair_path).get_fdata()
            t1_img    = nib.load(t1_path).get_fdata()
            t1ce_img  = nib.load(t1ce_path).get_fdata()
            t2_img    = nib.load(t2_path).get_fdata()
            mask_img  = nib.load(seg_path).get_fdata()
        except Exception:
            # Skip patient if there is a corrupted file error during reading
            continue
            
        # Find total number of slices in Z-axis (top to bottom)
        num_slices = flair_img.shape[2]
        
        # Iterate through each slice one by one (going from 3D to 2D)
        for z in range(num_slices):
            flair_slice = flair_img[:, :, z].astype(np.float32)
            
            # If slice is completely black (no brain tissue), skip it
            if np.max(flair_slice) == 0: continue
            
            # Extract mask to find coordinates where brain starts and ends
            non_zero_mask = flair_slice > 0
            coords = np.argwhere(non_zero_mask)
            if len(coords) == 0: continue
            
            # Find extreme x and y coordinates of the brain to crop black spaces
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            padding = 5 # Leave a 5-pixel padding around the brain so it doesn't get cut
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(flair_slice.shape[0], x_max + padding)
            y_max = min(flair_slice.shape[1], y_max + padding)

            # Crop and resize to 240x240 (Zoom) function
            def _crop_and_zoom(img_2d, is_mask=False):
                cropped = img_2d[x_min:x_max, y_min:y_max] # Discard black spaces
                if cropped.size == 0: return np.zeros((240, 240), dtype=np.float32)
                
                # Find the zoom factor needed to make the size exactly 240x240
                zoom_factors = (240 / cropped.shape[0], 240 / cropped.shape[1])
                # If mask, do not interpolate (order=0); if MRI image, smooth (order=1)
                zoom_order = 0 if is_mask else 1
                return ndimage.zoom(cropped, zoom_factors, order=zoom_order)

            # Take mask slice and extract Whole Tumor region (labels 1, 2, and 4)
            mask_slice = mask_img[:, :, z].astype(np.float32)
            mask_wt = np.isin(mask_slice, [1, 2, 4]).astype(np.float32)
            
            # Take 2D slices of other MRI sequences as well
            t1_slice  = t1_img[:, :, z].astype(np.float32)
            t1ce_slice= t1ce_img[:, :, z].astype(np.float32)
            t2_slice  = t2_img[:, :, z].astype(np.float32)

            # Crop 4 MRI sequences, resize to 240x240, normalize, and create a single 4-channel image
            modalities = []
            for m_slice in [flair_slice, t1_slice, t1ce_slice, t2_slice]:
                zoomed_m = _crop_and_zoom(m_slice, is_mask=False)
                normalized = _normalize_modality(zoomed_m) # Z-score normalization
                modalities.append(normalized)
            img_4channel = np.stack(modalities, axis=0) # Shape: (4, 240, 240)

            # Create 3 different mask regions suitable for BraTS dataset
            mask_tc = np.isin(mask_slice, [1, 4]).astype(np.float32) # Tumor Core
            mask_et = (mask_slice == 4).astype(np.float32)           # Enhancing Tumor
            
            # Crop masks, resize to 240x240, and combine as a 3-channel target mask
            mask_3channel = []
            for m in [mask_wt, mask_tc, mask_et]:
                zoomed_mask = _crop_and_zoom(m, is_mask=True)
                mask_3channel.append(zoomed_mask.astype(np.uint8))
            mask_3channel = np.stack(mask_3channel, axis=0) # Shape: (3, 240, 240)

            # Determine filename to save the slice (e.g., Patient1_slice_45.pt)
            out_file = os.path.join(out_split, f"{pid}_slice_{z}.pt")
            
            # Save to disk in Tensor format.
            # float16 and uint8 are chosen to save RAM and disk space.
            torch.save({
                'image': torch.tensor(img_4channel, dtype=torch.float16),
                'mask' : torch.tensor(mask_3channel, dtype=torch.uint8)
            }, out_file)
            
            saved_lists[split_name].append(out_file)

torch.save(saved_lists['train'], os.path.join(OUT_DIR, 'train_files.pt'))
torch.save(saved_lists['val'],   os.path.join(OUT_DIR, 'val_files.pt'))
torch.save(saved_lists['test'],  os.path.join(OUT_DIR, 'test_files.pt'))
print("\nPreprocessing completed, Train/Val/Test lists created.")