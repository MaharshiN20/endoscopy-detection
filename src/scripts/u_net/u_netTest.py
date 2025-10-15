import osimport os

import sysimport sys

from pathlib import Pathfrom pathlib import Path

import numpy as npimport numpy as np

import cv2import cv2

import torchimport torch

from tqdm import tqdmfrom tqdm import tqdm

import matplotlib.pyplot as pltimport matplotlib.pyplot as plt



# Setup paths# Setup paths

SCRIPT_DIR = Path(__file__).resolve().parentSCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = SCRIPT_DIR.parents[2]REPO_ROOT = SCRIPT_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT))sys.path.insert(0, str(REPO_ROOT))



from src.models.unet.unet_model import UNetfrom src.models.unet.unet_model import UNet



def load_image(image_path, size=256):def load_image(image_path, size=256):

    """Load and preprocess an image for prediction"""    """Load and preprocess an image for prediction"""

    image = cv2.imread(str(image_path))    image = cv2.imread(str(image_path))

    if image is None:    if image is None:

        raise ValueError(f"Could not load image: {image_path}")        raise ValueError(f"Could not load image: {image_path}")

        

    image = cv2.resize(image, (size, size))    image = cv2.resize(image, (size, size))

    image = image.astype(np.float32) / 255.0    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))  # CHW    image = np.transpose(image, (2, 0, 1))  # CHW

    return torch.from_numpy(image).unsqueeze(0)  # Add batch dimension    return torch.from_numpy(image).unsqueeze(0)  # Add batch dimension



def predict_mask(model, image):def predict_mask(model, image):

    """Generate prediction mask for an image"""    """Generate prediction mask for an image"""

    model.eval()    model.eval()

    with torch.no_grad():    with torch.no_grad():

        pred = model(image)        pred = model(image)

        pred = torch.sigmoid(pred)  # Convert to probability (0-1)        pred = torch.sigmoid(pred)  # Convert to probability (0-1)

        return pred  # Return raw predictions        return pred  # Return raw predictions



def save_visualization(image, pred_mask, save_path):def save_visualization(image, pred_mask, save_path):

    """Save original image and predicted masks side by side"""    """Save original image and predicted masks side by side"""

    # Convert tensors to numpy arrays    # Convert tensors to numpy arrays

    image = image.squeeze().permute(1, 2, 0).cpu().numpy()    image = image.squeeze().permute(1, 2, 0).cpu().numpy()

    pred_mask = pred_mask.squeeze().cpu().numpy()    pred_mask = pred_mask.squeeze().cpu().numpy()

        

    # Create thresholded versions at different levels    # Create thresholded versions at different levels

    thresh_low = (pred_mask > 0.3).astype(np.float32)    thresh_low = (pred_mask > 0.3).astype(np.float32)

    thresh_med = (pred_mask > 0.5).astype(np.float32)    thresh_med = (pred_mask > 0.5).astype(np.float32)

        

    # Create figure with subplots    # Create figure with subplots

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

        

    # Plot original image    # Plot original image

    ax1.imshow(image)    ax1.imshow(image)

    ax1.set_title('Original Image')    ax1.set_title('Original Image')

    ax1.axis('off')    ax1.axis('off')

        

    # Plot raw predictions    # Plot raw predictions

    im2 = ax2.imshow(pred_mask, cmap='jet')    im2 = ax2.imshow(pred_mask, cmap='jet')

    ax2.set_title('Raw Predictions')    ax2.set_title('Raw Predictions')

    ax2.axis('off')    ax2.axis('off')

    plt.colorbar(im2, ax=ax2)    plt.colorbar(im2, ax=ax2)

        

    # Plot thresholded masks    # Plot thresholded masks

    ax3.imshow(thresh_low, cmap='gray')    ax3.imshow(thresh_low, cmap='gray')

    ax3.set_title('Threshold > 0.3')    ax3.set_title('Threshold > 0.3')

    ax3.axis('off')    ax3.axis('off')

        

    ax4.imshow(thresh_med, cmap='gray')    ax4.imshow(thresh_med, cmap='gray')

    ax4.set_title('Threshold > 0.5')    ax4.set_title('Threshold > 0.5')

    ax4.axis('off')    ax4.axis('off')

        

    # Save the figure    # Save the figure

    plt.savefig(str(save_path), bbox_inches='tight', dpi=150)    plt.savefig(str(save_path), bbox_inches='tight', dpi=150)

    plt.close()    plt.close()



def main():def main():

    # Setup device    # Setup device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")    print(f"Using device: {device}")



    # Load model    # Load model

    model = UNet(n_channels=3, n_classes=1)    model = UNet(n_channels=3, n_classes=1)

    model_path = REPO_ROOT / "src" / "outputs" / "u_net" / "esophagus_opening" / "best.pt"    model_path = REPO_ROOT / "src" / "outputs" / "u_net" / "esophagus_opening" / "best.pt"

    if not model_path.exists():    if not model_path.exists():

        raise FileNotFoundError(f"Model file not found at {model_path}")        raise FileNotFoundError(f"Model file not found at {model_path}")

        

    # Load model weights    # Load model weights

    checkpoint = torch.load(model_path, map_location=device)    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:

        state_dict = checkpoint['model_state_dict']        state_dict = checkpoint['model_state_dict']

    else:    else:

        state_dict = checkpoint        state_dict = checkpoint

        

    try:    try:

        model.load_state_dict(state_dict)        model.load_state_dict(state_dict)

        print(f"Loaded model from: {model_path}")        print(f"Loaded model from: {model_path}")

    except RuntimeError as e:    except RuntimeError as e:

        print(f"Error loading model state dict. This might be a v2 model. Please check if you're using the correct model file.")        print(f"Error loading model state dict. This might be a v2 model. Please check if you're using the correct model file.")

        raise e        raise e

        

    # Move model to device and set to eval mode    # Move model to device and set to eval mode

    model = model.to(device)    model = model.to(device)

    model.eval()    model.eval()

        

    # Print model summary    # Print model summary

    total_params = sum(p.numel() for p in model.parameters())    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")    print(f"Total parameters: {total_params:,}")

    print(f"Trainable parameters: {trainable_params:,}")    print(f"Trainable parameters: {trainable_params:,}")

        

    # Setup paths    # Setup paths

    test_img_dir = REPO_ROOT / "data" / "dataset" / "data" / "test" / "images"    test_img_dir = REPO_ROOT / "data" / "dataset" / "data" / "test" / "images"

    if not test_img_dir.exists():    if not test_img_dir.exists():

        raise FileNotFoundError(f"Test images directory not found at {test_img_dir}")        raise FileNotFoundError(f"Test images directory not found at {test_img_dir}")

                

    # For prediction only mode (no ground truth masks available)    # For prediction only mode (no ground truth masks available)

    test_mask_dir = None    test_mask_dir = None

    output_dir = REPO_ROOT / "src" / "outputs" / "u_net" / "val_prediction"    output_dir = REPO_ROOT / "src" / "outputs" / "u_net" / "val_prediction"

    os.makedirs(output_dir, exist_ok=True)    os.makedirs(output_dir, exist_ok=True)



    # Process test images    # Process test images

    test_images = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])    test_images = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        

    # Create output directory    # Create output directory

    output_dir.mkdir(parents=True, exist_ok=True)    output_dir.mkdir(parents=True, exist_ok=True)

        

    print(f"\nProcessing {len(test_images)} test images...")    print(f"\nProcessing {len(test_images)} test images...")

        

    for img_name in tqdm(test_images, desc="Generating predictions"):    for img_name in tqdm(test_images, desc="Generating predictions"):

        # Load image        # Load image

        img_path = test_img_dir / img_name        img_path = test_img_dir / img_name

        image = load_image(img_path)        image = load_image(img_path)

                

        # Generate prediction        # Generate prediction

        image = image.to(device)        image = image.to(device)

        pred_mask = predict_mask(model, image)        pred_mask = predict_mask(model, image)

                

        # Save visualization        # Save visualization

        save_path = output_dir / f"{img_name.split('.')[0]}_result.png"        save_path = output_dir / f"{img_name.split('.')[0]}_result.png"

        save_visualization(image, pred_mask, save_path)        save_visualization(image, pred_mask, save_path)



    print(f"\nPredictions completed!")    print(f"\nPredictions completed!")

    print(f"Results saved to: {output_dir}")    print(f"Results saved to: {output_dir}")



if __name__ == "__main__":if __name__ == "__main__":

    main()    main()