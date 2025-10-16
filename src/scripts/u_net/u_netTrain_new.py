import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Setup paths
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import your U-Net model
from src.models.unet.unet_model import UNet

# ----------------------------
# Dataset class
# ----------------------------
class EsophagusMaskDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.ids = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        # load image & mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Missing image or mask: {img_name}")

        # resize to consistent size
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # normalize
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # convert to tensors
        image = np.transpose(image, (2, 0, 1))  # CHW
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)

        return image, mask


# ----------------------------
# Dice Loss (common for segmentation)
# ----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        num = 2 * (preds * targets).sum(dim=(2, 3)) + self.smooth
        den = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        dice = num / den
        return 1 - dice.mean()


# ----------------------------
# Training Loop
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # paths
    train_img_dir = REPO_ROOT / "data" / "dataset" / "data" / "train" / "images"
    train_mask_dir = REPO_ROOT / "data" / "dataset" / "data" / "train" / "masks"
    val_img_dir = REPO_ROOT / "data" / "dataset" / "data" / "val" / "images"
    val_mask_dir = REPO_ROOT / "data" / "dataset" / "data" / "val" / "masks"
    output_dir = REPO_ROOT / "src" / "outputs" / "u_net" / "esophagus_opening"
    os.makedirs(output_dir, exist_ok=True)

    # datasets
    train_ds = EsophagusMaskDataset(train_img_dir, train_mask_dir, img_size=256)
    val_ds = EsophagusMaskDataset(val_img_dir, val_mask_dir, img_size=256)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

    # model
    model = UNet(n_channels=3, n_classes=1).to(device)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, 31):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = 0.5 * bce(preds, masks) + 0.5 * dice(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = 0.5 * bce(preds, masks) + 0.5 * dice(preds, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")

        # save checkpoints
        torch.save(model.state_dict(), output_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"✅ Saved best model (Val Loss: {val_loss:.4f})")

    print("Training complete. Models saved to:", output_dir)


if __name__ == "__main__":
    main()