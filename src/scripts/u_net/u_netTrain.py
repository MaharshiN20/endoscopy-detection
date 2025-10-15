import osimport os

import sysimport sys

from pathlib import Pathfrom pathlib import Path

from tqdm import tqdmfrom tqdm import tqdm

import numpy as npimport numpy as np

import cv2import cv2

import torchimport torch

import torch.nn as nnimport torch.nn as nn

import torch.nn.functional as Fimport torch.nn.functional as F

from torch.utils.data import Dataset, DataLoaderfrom torch.utils.data import Dataset, DataLoader



# Setup paths# ----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent# Setup paths

REPO_ROOT = SCRIPT_DIR.parents[2]# ----------------------------

sys.path.insert(0, str(REPO_ROOT))SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = SCRIPT_DIR.parents[2]

from src.models.unet.unet_model import UNetsys.path.insert(0, str(REPO_ROOT))



class DiceLoss(nn.Module):# Import your U-Net model

    def __init__(self, smooth=1e-6):from src.models.unet.unet_model import UNet

        super().__init__()

        self.smooth = smooth# ----------------------------

        # Dataset class

    def forward(self, pred, target):# ----------------------------

        pred = torch.sigmoid(pred)class EsophagusMaskDataset(Dataset):

        # Flatten the tensors    def __init__(self, images_dir, masks_dir, img_size=256):

        pred_flat = pred.view(pred.size(0), -1)        self.images_dir = images_dir

        target_flat = target.view(target.size(0), -1)        self.masks_dir = masks_dir

                self.img_size = img_size

        intersection = (pred_flat * target_flat).sum(1)        self.ids = sorted([

        union = pred_flat.sum(1) + target_flat.sum(1)            f for f in os.listdir(images_dir)

                    if f.lower().endswith((".png", ".jpg", ".jpeg"))

        dice = (2. * intersection + self.smooth)/(union + self.smooth)        ])

        return 1 - dice.mean()

    def __len__(self):

class EsophagusMaskDataset(Dataset):        return len(self.ids)

    def __init__(self, images_dir, masks_dir, img_size=256):

        self.images_dir = images_dir    def __getitem__(self, idx):

        self.masks_dir = masks_dir        img_name = self.ids[idx]

        self.img_size = img_size        img_path = os.path.join(self.images_dir, img_name)

        self.ids = sorted([        mask_path = os.path.join(self.masks_dir, img_name)

            f for f in os.listdir(images_dir)

            if f.lower().endswith((".png", ".jpg", ".jpeg"))        # load image & mask

        ])        image = cv2.imread(img_path)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    def __len__(self):

        return len(self.ids)        if image is None or mask is None:

            raise ValueError(f"Missing image or mask: {img_name}")

    def __getitem__(self, idx):

        img_name = self.ids[idx]        # resize to consistent size

        img_path = os.path.join(self.images_dir, img_name)        image = cv2.resize(image, (self.img_size, self.img_size))

        mask_path = os.path.join(self.masks_dir, img_name)        mask = cv2.resize(mask, (self.img_size, self.img_size))



        # Load image & mask        # normalize

        image = cv2.imread(str(img_path))        image = image.astype(np.float32) / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)        mask = (mask > 127).astype(np.float32)

        

        if image is None or mask is None:        # convert to tensors

            raise ValueError(f"Could not load image or mask: {img_path}")        image = np.transpose(image, (2, 0, 1))  # CHW

                image = torch.from_numpy(image).float()

        # Resize        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)

        image = cv2.resize(image, (self.img_size, self.img_size))

        mask = cv2.resize(mask, (self.img_size, self.img_size))        return image, mask

        

        # Normalize image and mask

        image = image.astype(np.float32) / 255.0# ----------------------------

        mask = (mask > 127).astype(np.float32)# Dice Loss (common for segmentation)

        # ----------------------------

        # Convert to tensorsclass DiceLoss(nn.Module):

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()  # CHW    def __init__(self, smooth=1e-6):

        mask = torch.from_numpy(mask).unsqueeze(0).float()  # Add channel dim        super().__init__()

        self.smooth = smooth

        return image, mask

    def forward(self, preds, targets):

def main():        preds = torch.sigmoid(preds)

    # Setup device        num = 2 * (preds * targets).sum(dim=(2, 3)) + self.smooth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        den = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth

    print(f"Training on: {device}")        dice = num / den

        return 1 - dice.mean()

    # Paths

    train_img_dir = REPO_ROOT / "data" / "dataset" / "data" / "train" / "images"

    train_mask_dir = REPO_ROOT / "data" / "dataset" / "data" / "train" / "masks"# ----------------------------

    val_img_dir = REPO_ROOT / "data" / "dataset" / "data" / "val" / "images"# Training Loop

    val_mask_dir = REPO_ROOT / "data" / "dataset" / "data" / "val" / "masks"# ----------------------------

    output_dir = REPO_ROOT / "src" / "outputs" / "u_net" / "esophagus_opening"def main():

    os.makedirs(output_dir, exist_ok=True)    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on: {device}")

    # Datasets

    train_ds = EsophagusMaskDataset(train_img_dir, train_mask_dir, img_size=256)    # paths

    val_ds = EsophagusMaskDataset(val_img_dir, val_mask_dir, img_size=256)    train_img_dir = REPO_ROOT / "data" / "dataset" / "data" / "train" / "images"

    train_mask_dir = REPO_ROOT / "data" / "dataset" / "data" / "train" / "masks"

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)    val_img_dir = REPO_ROOT / "data" / "dataset" / "data" / "val" / "images"

    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)    val_mask_dir = REPO_ROOT / "data" / "dataset" / "data" / "val" / "masks"

    output_dir = REPO_ROOT / "src" / "outputs" / "u_net" / "esophagus_opening"

    # Model    os.makedirs(output_dir, exist_ok=True)

    model = UNet(n_channels=3, n_classes=1).to(device)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")    # datasets

        train_ds = EsophagusMaskDataset(train_img_dir, train_mask_dir, img_size=256)

    # Loss and optimizer    val_ds = EsophagusMaskDataset(val_img_dir, val_mask_dir, img_size=256)

    criterion = DiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

        optimizer, mode='min', factor=0.5, patience=5

    )    # model

    model = UNet(n_channels=3, n_classes=1).to(device)

    # Training loop    bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")    dice = DiceLoss()

    num_epochs = 100    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    early_stopping_patience = 15

    no_improve_count = 0    best_val_loss = float("inf")



    for epoch in range(1, num_epochs + 1):    for epoch in range(1, 31):

        # Training        model.train()

        model.train()        train_loss = 0

        train_loss = 0        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):

        train_dice = 0            imgs, masks = imgs.to(device), masks.to(device)

        

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):            preds = model(imgs)

            imgs, masks = imgs.to(device), masks.to(device)            loss = 0.5 * bce(preds, masks) + 0.5 * dice(preds, masks)

            

            optimizer.zero_grad()            optimizer.zero_grad()

            preds = model(imgs)            loss.backward()

            loss = criterion(preds, masks)            optimizer.step()

                        train_loss += loss.item()

            loss.backward()

            optimizer.step()        train_loss /= len(train_loader)

            

            train_loss += loss.item()        # validation

                    model.eval()

            # Calculate Dice score for monitoring        val_loss = 0

            with torch.no_grad():        with torch.no_grad():

                pred_masks = (torch.sigmoid(preds) > 0.5).float()            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):

                dice = (2 * (pred_masks * masks).sum()) / (pred_masks.sum() + masks.sum() + 1e-6)                imgs, masks = imgs.to(device), masks.to(device)

                train_dice += dice.item()                preds = model(imgs)

                loss = 0.5 * bce(preds, masks) + 0.5 * dice(preds, masks)

        train_loss /= len(train_loader)                val_loss += loss.item()

        train_dice /= len(train_loader)        val_loss /= len(val_loader)

        

        # Validation        print(f"Epoch {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")

        model.eval()

        val_loss = 0        # save checkpoints

        val_dice = 0        torch.save(model.state_dict(), output_dir / "last.pt")

                if val_loss < best_val_loss:

        with torch.no_grad():            best_val_loss = val_loss

            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):            torch.save(model.state_dict(), output_dir / "best.pt")

                imgs, masks = imgs.to(device), masks.to(device)            print(f"✅ Saved best model (Val Loss: {val_loss:.4f})")

                preds = model(imgs)

                loss = criterion(preds, masks)    print("Training complete. Models saved to:", output_dir)

                

                val_loss += loss.item()

                if __name__ == "__main__":

                # Calculate Dice score    main()

                pred_masks = (torch.sigmoid(preds) > 0.5).float()
                dice = (2 * (pred_masks * masks).sum()) / (pred_masks.sum() + masks.sum() + 1e-6)
                val_dice += dice.item()
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 50)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice
            }
            torch.save(checkpoint, output_dir / "best.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            
            print(f"Saved new best model (val_loss: {val_loss:.4f})")
        else:
            no_improve_count += 1
            
            # Save latest model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice
            }
            torch.save(checkpoint, output_dir / "last.pt")
        
        # Early stopping
        if no_improve_count >= early_stopping_patience:
            print(f"No improvement for {early_stopping_patience} epochs. Stopping training.")
            break

if __name__ == "__main__":
    main()