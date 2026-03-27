import os
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FastBratsDataset

# Device configuration (GPU check)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
DATA_DIR = './data/preprocessed_data'

# Augmentation is only applied to the Train set (must be False for Validation/Test)
train_dataset = FastBratsDataset(os.path.join(DATA_DIR, 'train_files.pt'), augment=True)
val_dataset   = FastBratsDataset(os.path.join(DATA_DIR, 'val_files.pt'),   augment=False)

print(f"Train Slice Count: {len(train_dataset)}")
print(f"Val   Slice Count: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("\n" + "=" * 60)
print("TRAINING STARTED (Estimated time: ~45-60 mins)")
print("=" * 60)

# Model Architecture: U-Net with EfficientNet-B0 backbone
model = smp.Unet(
    encoder_name="efficientnet-b0", 
    encoder_weights="imagenet", 
    in_channels=4,  # 4-channel input: FLAIR, T1, T1ce, T2
    classes=3       # 3 output channels: WT, TC, ET
)
model.to(device)

# Combining two loss functions: Dice Loss and BCE
criterion_dice = smp.losses.DiceLoss(mode='multilabel')
criterion_bce = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = None
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            return True
        return False
    
    def restore_model(self, model):
        """Restores the best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

# If validation loss doesn't improve for 3 epochs ("patience=3"), reduce learning rate by half ("factor=0.5")
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

loss_history = {'train': [], 'val': []}
lr_history = []

# Main Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", unit="batch")

    for imgs, masks in progress_bar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss_dice = criterion_dice(outputs, masks)
        loss_bce  = criterion_bce(outputs, masks)
        loss = loss_dice + loss_bce

        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_epoch_loss / len(train_loader)
    loss_history['train'].append(avg_train_loss)

    # --- Validation Phase ---
    val_epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            loss_dice = criterion_dice(outputs, masks)
            loss_bce  = criterion_bce(outputs, masks)
            loss = loss_dice + loss_bce

            val_epoch_loss += loss.item()

    avg_val_loss = val_epoch_loss / len(val_loader)
    loss_history['val'].append(avg_val_loss)

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)

    print(f" Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}")

    if early_stopping(avg_val_loss, model):
        print(f"\n Early stopping triggered at epoch {epoch+1}")
        print(f"   Best validation loss: {early_stopping.best_val_loss:.4f}")
        early_stopping.restore_model(model)
        break

    if device.type == 'cuda':
        torch.cuda.empty_cache()

# Save the final trained model
torch.save(model.state_dict(), "brats_unet_model_4channel.pth")
print("\n Model weights saved: brats_unet_model_4channel.pth")