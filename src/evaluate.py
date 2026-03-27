import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FastBratsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation Hyperparameters
BATCH_SIZE = 32
DATA_DIR = './data/preprocessed_data'

test_dataset = FastBratsDataset(os.path.join(DATA_DIR, 'test_files.pt'), augment=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Test Slice Count: {len(test_dataset)}")

# Load the trained model architecture and weights
model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=4, classes=3)
model.load_state_dict(torch.load("brats_unet_model_4channel.pth", map_location=device))
model.to(device)
model.eval()

print("\n" + "=" * 60)
print("INFERENCE & EVALUATION")
print("=" * 60)

def compute_clinical_metrics(predictions, ground_truths):
    """
    Computes Dice Score and IoU metrics for specific clinical regions.
    WT: Whole Tumor, TC: Tumor Core, ET: Enhancing Tumor.
    """
    region_names = ['WT', 'TC', 'ET'] 
    metrics = {
        'WT': {'dice': [], 'iou': []},
        'TC': {'dice': [], 'iou': []},
        'ET': {'dice': [], 'iou': []}
    }
    
    for pred, target in zip(predictions, ground_truths):
        for ch_idx, region_key in enumerate(region_names):
            pred_channel = pred[ch_idx].flatten().astype(np.float32)
            target_channel = target[ch_idx].flatten().astype(np.float32)
            
            # MATHEMATICAL CORRECTION (TRAP FIX)
            sum_pred = np.sum(pred_channel)
            sum_target = np.sum(target_channel)
            
            # True Negative: If there is no tumor in reality AND model predicted no tumor
            if sum_target == 0 and sum_pred == 0:
                dice = 1.0
                iou = 1.0
            # False Positive: If there is no tumor in reality BUT model predicted one
            elif sum_target == 0 and sum_pred > 0:
                dice = 0.0
                iou = 0.0
            # Standard Dice calculation (if tumor exists)
            else:
                intersection = np.sum(pred_channel * target_channel)
                dice = 2.0 * intersection / (sum_pred + sum_target + 1e-8)
                iou = intersection / (np.sum(np.logical_or(pred_channel, target_channel)) + 1e-8)
            
            metrics[region_key]['dice'].append(dice)
            metrics[region_key]['iou'].append(iou)
            
    return metrics

all_preds  = []
all_labels = []

print("Running inference on Test Set...")
with torch.no_grad():
    for imgs, masks in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device)
        outputs = model(imgs)

        preds_batch  = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(np.float32)
        labels_batch = masks.numpy().astype(np.float32)

        all_preds.extend(preds_batch)
        all_labels.extend(labels_batch)

clinical_metrics = compute_clinical_metrics(all_preds, all_labels)

print("\n" + "=" * 60)
print("CLINICAL MULTI-CLASS METRICS (3-Channel Multilabel)")
print("=" * 60)

region_names_full = {'WT': 'Whole Tumor (Labels 1,2,4)', 'TC': 'Tumor Core (Labels 1,4)', 'ET': 'Enhancing Tumor (Label 4)'}

for region in ['WT', 'TC', 'ET']:
    dice_scores = clinical_metrics[region]['dice']
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f"\n{region} - {region_names_full[region]}:")
    print(f"  - Mean Dice Score (Average Success) : {mean_dice:.4f}")
    print(f"  - Standard Deviation (Variance)     : {std_dice:.4f}")

print("\n" + "-" * 60)
print("PER-CHANNEL PIXEL-LEVEL METRICS")
print("-" * 60)

for ch_idx, region in enumerate(['WT', 'TC', 'ET']):
    all_preds_channel = np.array([p[ch_idx].flatten() for p in all_preds]).flatten()
    all_labels_channel = np.array([l[ch_idx].flatten() for l in all_labels]).flatten()
    
    # Random sampling if array is too large to prevent memory overflow during metrics calc
    if len(all_labels_channel) > 500000:
        indices = np.random.choice(len(all_labels_channel), 500000, replace=False)
        target_true = all_labels_channel[indices].astype(np.int32)
        target_pred = all_preds_channel[indices].astype(np.int32)
    else:
        target_true = all_labels_channel.astype(np.int32)
        target_pred = all_preds_channel.astype(np.int32)
    
    print(f"\n{region}:")
    print(classification_report(target_true, target_pred, target_names=['Background', 'Positive'], digits=4))

# Detailed Visualization
def overlay_mask_on_mri(mri_img, mask, alpha=0.5):
    """Overlays the predicted/ground-truth mask transparently onto the MRI slice."""
    # Normalize MRI image between 0-1 (For visualization)
    mri_norm = (mri_img - mri_img.min()) / (mri_img.max() - mri_img.min() + 1e-8)
    mri_rgb = np.stack([mri_norm]*3, axis=-1) # Convert Grayscale to RGB
    
    overlay = mri_rgb.copy()
    # WT: Red, TC: Green, ET: Blue
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] 
    
    for ch in range(3):
        mask_channel = mask[ch] == 1
        for i in range(3): # R, G, B channels
            overlay[:, :, i][mask_channel] = overlay[:, :, i][mask_channel] * (1 - alpha) + colors[ch][i] * alpha
            
    return overlay

print("\nGenerating Random Test Sample Overlays...")
n_show = 10
indices = random.sample(range(len(test_dataset)), n_show)

fig, axes = plt.subplots(n_show, 3, figsize=(15, 4 * n_show))

for row, idx in enumerate(indices):
    img, mask = test_dataset[idx]
    
    # Model Prediction
    with torch.no_grad():
        pred_tensor = model(img.unsqueeze(0).to(device))
        pred = (torch.sigmoid(pred_tensor) > 0.5).cpu().numpy()[0]
    
    flair_channel = img[0].numpy() # Use FLAIR sequence as background
    gt_mask = mask.numpy()
    
    # Overlay Operations
    gt_overlay = overlay_mask_on_mri(flair_channel, gt_mask)
    pred_overlay = overlay_mask_on_mri(flair_channel, pred)
    
    # Plotting
    axes[row, 0].imshow(flair_channel, cmap='gray')
    axes[row, 0].set_title(f'Sample {idx} - Original FLAIR', fontsize=12)
    
    axes[row, 1].imshow(gt_overlay)
    axes[row, 1].set_title('Ground Truth\n(WT:Red, TC:Green, ET:Blue)', fontsize=12)
    
    axes[row, 2].imshow(pred_overlay)
    axes[row, 2].set_title('Model Prediction', fontsize=12)
    
    for ax in axes[row]:
        ax.axis('off')

plt.tight_layout()
plt.savefig('mri_predictions_overlay.png', dpi=200, bbox_inches='tight')
print("\nOverlay images saved successfully as 'mri_predictions_overlay.png'.")
# plt.show() is omitted to run cleanly from terminal/scripts without blocking