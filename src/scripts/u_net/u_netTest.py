import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.unet.unet_model import UNet

def load_image(image_path, size=256):
    """Load and preprocess an image for prediction"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    print(f"Original image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
    
    image = cv2.resize(image, (size, size))
    image = image.astype(np.float32) / 255.0
    print(f"After normalization range: [{image.min():.4f}, {image.max():.4f}]")
    
    image = np.transpose(image, (2, 0, 1))  # CHW
    return torch.from_numpy(image).unsqueeze(0)  # Add batch dimension

def predict_mask(model, image):
    """Generate prediction mask for an image with both thresholds"""
    model.eval()
    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred)
        mask_03 = (pred > 0.3).float()
        mask_05 = (pred > 0.5).float()
    return pred, mask_03, mask_05

def save_visualization(image, pred, mask_03, mask_05, save_path):
    """Save original image and predicted masks with different thresholds using heatmap"""
    # Convert tensors to numpy arrays
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    pred = pred.squeeze().cpu().numpy()
    mask_03 = mask_03.squeeze().cpu().numpy()
    mask_05 = mask_05.squeeze().cpu().numpy()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot raw prediction heatmap
    im2 = ax2.imshow(pred, cmap='jet')
    ax2.set_title('Raw Prediction (Heatmap)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    # Plot threshold 0.3 mask
    ax3.imshow(mask_03, cmap='jet')
    ax3.set_title('Threshold 0.3')
    ax3.axis('off')
    
    # Plot threshold 0.5 mask
    ax4.imshow(mask_05, cmap='jet')
    ax4.set_title('Threshold 0.5')
    ax4.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = UNet(n_channels=3, n_classes=1)
    model_path = REPO_ROOT / "src" / "outputs" / "u_net" / "esophagus_opening" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the model weights directly
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Setup paths
    test_img_dir = REPO_ROOT / "data" / "dataset" / "data" / "test" / "images"
    if not test_img_dir.exists():
        raise FileNotFoundError(f"Test images directory not found at {test_img_dir}")
        
    # For prediction only mode (no ground truth masks available)
    test_mask_dir = None
    output_dir = REPO_ROOT / "src" / "outputs" / "u_net" / "val_prediction"
    os.makedirs(output_dir, exist_ok=True)

    # Process test images
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(test_images)} test images...")
    
    for img_name in tqdm(test_images, desc="Generating predictions"):
        # Load image
        img_path = test_img_dir / img_name
        image = load_image(img_path)
        
        # Generate predictions with different thresholds
        image = image.to(device)
        pred, mask_03, mask_05 = predict_mask(model, image)
        
        # Save visualization
        save_path = output_dir / f"{img_name.split('.')[0]}_result.png"
        save_visualization(image, pred, mask_03, mask_05, save_path)

    print(f"\nPredictions completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
     #~0.6 val loss range gives good results