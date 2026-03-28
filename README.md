# Concrete Crack Segmentation with UNet

## 📋 Project Overview

A deep learning project for detecting and segmenting concrete cracks in images using an improved UNet architecture. The model achieves exceptional performance with Dice coefficient of **99.77%** and IoU of **99.54%**.

## 🎯 Key Features

- **High-Performance UNet**: 4-level encoder-decoder with BatchNorm and Dropout
- **Advanced Data Augmentation**: 8 different augmentation techniques
- **Automatic Mask Inversion**: Detects and handles inverted masks automatically
- **Combined Loss Function**: Weighted combination of Dice (60%) + BCE (40%) loss
- **Robust Preprocessing**: Automatic dataset path detection (Local/Colab)
- **Comprehensive Validation**: Multiple metrics (Dice, IoU, F1, Precision, Recall)
- **Production Ready**: Easy inference function for new predictions

## 📊 Results

| Metric | Value |
|--------|-------|
| **Dice Score** | 99.77% |
| **IoU Score** | 99.54% |
| **F1 Score** | 99.77% |
| **Precision** | 99.54% |
| **Recall** | 100.00% |
| **Training Loss** | 0.0162 |
| **Validation Loss** | 0.0139 |

## 🏗️ Project Structure

```
concreate-crack-segmentation/
├── Concreate_Crack_Segmentation.ipynb    # Main notebook
├── Dataset/
│   ├── images/                           # 800 concrete images (256x256)
│   └── masks/                            # Corresponding crack masks
├── unet_model_weights.pth                # Trained model weights
├── model_config.json                     # Model configuration
├── training_history.json                 # Training metrics history
├── training_history.png                  # Training plots
├── predictions_visualization.png         # Sample predictions
├── requirements.txt                      # Dependencies
├── README.md                             # This file
├── model_card.md                        # Model card for HF
└── inference.py                         # Inference script
```

## 🔧 Requirements

```
torch==2.0.0
torchvision==0.15.0
Pillow==9.5.0
numpy==1.24.3
opencv-python==4.8.0.74
matplotlib==3.7.2
tqdm==4.66.1
```

## 🚀 Quick Start

### 🌐 Try the Demo (No Installation Needed!)

**Easiest Way**: Use the interactive Gradio Space directly in your browser:
👉 **[Launch Demo on Hugging Face](https://huggingface.co/spaces/samir-mohamed/concrete-crack-segmentation)**

- Upload any concrete image
- Get instant segmentation results
- No installation or coding required!

### Local Usage
```python
# Option 1: Using inference.py
from inference import CrackSegmentationModel

model = CrackSegmentationModel('unet_model_weights.pth')
prediction = model.predict('image.jpg', threshold=0.5)

# Option 2: Direct from Hugging Face (Recommended)
from huggingface_hub import hf_hub_download
import torch

model_path = hf_hub_download(
    repo_id="samir-mohamed/concrete-crack-segmentation",
    filename="unet_model_weights.pth"
)
model = torch.load(model_path)
```

### Colab Usage
```python
# Mount Google Drive and run notebook cells in order
# Dataset path: /MyDrive/computer vision course/projects/Concreate Crack Segmentation/Dataset/
```

## 📈 Model Architecture

### ImprovedUNet
- **Input**: RGB images (3 channels, 256×256)
- **Encoder**: 4 levels with convolution + BatchNorm + ReLU + Dropout
- **Bottleneck**: Deep feature extraction
- **Decoder**: 4 levels with transposed convolution + skip connections
- **Output**: Binary segmentation mask (1 channel)
- **Total Parameters**: ~7.8M

### Core Components
- **Encoder Filters**: [64, 128, 256, 512]
- **Activation**: ReLU
- **Normalization**: BatchNorm2d
- **Regularization**: Dropout2d (rate: 0.2-0.3)

## 🎓 Training Configuration

- **Optimizer**: AdamW (lr=2e-3, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR (T_max=50, eta_min=1e-6)
- **Loss Function**: CombinedLoss(0.4×BCE + 0.6×Dice)
- **Batch Size**: 16
- **Train/Val Split**: 80/20 (640 train, 160 val)
- **Epochs**: 25 (with early stopping)
- **Data Augmentation**:
  - Random flip (H/V)
  - Random rotation (±15°)
  - Random affine (translate, scale)
  - Color jitter (brightness, contrast, saturation)
  - Gaussian blur

## 🔍 Dataset Details

### Concrete Crack Dataset
- **Total Images**: 800
- **Image Size**: 256×256 pixels RGB
- **Mask Format**: Binary (black=background, gray/white=cracks)
- **Automatic Features**:
  - Mask inversion detection (100% of masks inverted automatically)
  - Image normalization using ImageNet statistics
  - Stratified train/val split

### Key Observations
- Masks use quantized values: {0.6549 (cracks), 1.0 (background)}
- Average crack area per image: 0-3% of image
- Binary nature enables excellent performance

## 📚 File Descriptions

### Main Files
- **Concreate_Crack_Segmentation.ipynb**: Complete training notebook
- **unet_model_weights.pth**: PyTorch model state dict (recommended for loading)
- **model_config.json**: Hyperparameters and training configuration
- **training_history.json**: Epoch-by-epoch training metrics

### Generated Outputs
- **training_history.png**: 4-panel plot showing:
  - Training/Validation loss
  - Validation Dice score
  - Validation IoU
  - Validation F1 score
- **predictions_visualization.png**: 6 random predictions with:
  - Original images
  - Ground truth masks
  - Model predictions
  - Binary predictions
  - Error maps
  - Metrics per sample

## 🔄 Data Processing Pipeline

1. **Loading Phase**
   - Load RGB image + grayscale mask
   - Automatic mask inversion check

2. **Preprocessing**
   - Resize to 256×256
   - Convert to tensor

3. **Normalization**
   - Images: ImageNet normalization
   - Masks: No normalization (raw [0, 1])

4. **Augmentation** (train only)
   - Horizontal/vertical flip
   - Rotation ±15°
   - Affine transforms
   - Color jitter
   - Gaussian blur

## 💡 Usage Examples

### Load Trained Model
```python
import torch
from model import ImprovedUNet

model = ImprovedUNet(in_channels=3, out_channels=1, depth=4, start_filters=64)
model.load_state_dict(torch.load('unet_model_weights.pth'))
model.eval()
```

### Make Predictions
```python
from PIL import Image
import torchvision.transforms as transforms

image = Image.open('concrete.jpg').convert('RGB')
image = transforms.Resize((256, 256))(image)
image_tensor = transforms.ToTensor()(image).unsqueeze(0)

with torch.no_grad():
    prediction = model(image_tensor)
    
segmentation_mask = prediction.squeeze().numpy()
```

### Batch Processing
```python
from torch.utils.data import DataLoader
from dataset_class import CrackDataset

dataset = CrackDataset(images_path, masks_path, transform=val_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

predictions = []
for images, masks in loader:
    with torch.no_grad():
        preds = model(images)
    predictions.extend(preds.cpu().numpy())
```

## ⚙️ Advanced Configuration

### Custom Training
```python
# Modify these in the notebook:
BATCH_SIZE = 16              # Change batch size
NUM_EPOCHS = 25              # Change max epochs
LEARNING_RATE = 2e-3         # Change learning rate
IMG_WIDTH = IMG_HEIGHT = 256 # Change image size
```

### Loss Function Weights
```python
# In CombinedLoss:
bce_weight = 0.4     # Weight for BCE loss
dice_weight = 0.6    # Weight for Dice loss
# Adjust for different focus (higher dice_weight → more focus on segmentation)
```

## 🐛 Troubleshooting

### Issue: DataLoader Error on Windows
**Solution**: Set `num_workers=0` in DataLoader (already implemented)

### Issue: Out of Memory (OOM)
**Solution**: 
- Reduce BATCH_SIZE (16 → 8)
- Reduce image size (256 → 128)
- Enable gradient checkpointing

### Issue: Model Convergence Issue (loss stays high)
**Solution**:
- Check if masks are inverted (auto-detection should handle)
- Verify dataset loading with diagnostic cells
- Increase learning rate or change optimizer

## 📖 Dependencies & Versions

Tested with:
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (optional, CPU supported)
- CUDA-enabled GPU (RTX 3060, A100, etc.)

## 📄 Model Card

See [model_card.md](model_card.md) for detailed model information.

## 👤 Author

Created as educational project for concrete crack detection using deep learning.

## 📝 License

MIT License - Feel free to use for educational and commercial purposes.

## 🙏 Acknowledgments

- UNet Architecture: [Ronneberger et al., 2015]
- Dice Loss: [Milletari et al., 2016]
- Dataset: Concrete Crack Dataset (800 annotated images)

## 📧 Support & Links

- **Hugging Face Model**: https://huggingface.co/samir-mohamed/concrete-crack-segmentation
- **GitHub Repository**: https://github.com/samir-m0hamed/concrete-crack-segmentation
- For issues or questions, please create an issue in the repository.

---

**Last Updated**: March 28, 2026  
**Model Status**: Production Ready ✓
