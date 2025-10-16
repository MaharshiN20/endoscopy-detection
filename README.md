# Endoscopy Detection

Implementations of various segmentation & object detection algorithms for endoscopy images.

## UNet Model Configuration

### Optimal Training Parameters
- Model: UNet (3 input channels, 1 output channel)
- Loss Function: Combined BCE and Dice Loss (0.5 * BCE + 0.5 * Dice)
- Batch Size: 8
- Learning Rate: 1e-4
- Image Size: 256x256
- Optimizer: Adam

### Target Metrics
- Target Validation Loss: ~0.6
  - This validation loss range produces optimal results for:
    - Clear detection of large, obvious openings
    - Good sensitivity to subtle, smaller openings
    - Balanced heatmap visualization

### Visualization Settings
- Raw prediction heatmap (best for subtle features)
- Multiple threshold outputs:
  - 0.3 threshold for higher sensitivity
  - 0.5 threshold for higher specificity

### Model Output Characteristics
- Heatmap visualization:
  - Excellent for detecting smaller, subtle openings
  - Provides detailed confidence gradients
  - Best viewed with 'jet' colormap
- Threshold outputs:
  - Better for larger, more obvious openings
  - Provides clear binary segmentation

### Training Tips
1. Monitor validation loss during training
2. Save model when validation loss approaches 0.6
3. Use both heatmap and threshold visualizations for evaluation
4. Balance between sensitivity (lower threshold) and specificity (higher threshold)

## Directory Structure
- `src/models/unet/`: UNet model implementation
- `src/scripts/u_net/`: Training and testing scripts
- `data/dataset/`: Dataset organization
- `src/outputs/u_net/`: Model outputs and results (not tracked in git due to file size)

## Note
Model checkpoint files (*.pt) are not included in the repository due to size limitations. Train the model using the provided configuration to achieve similar results.