# 🎯 Stanford40-DeepClassifier: Multi-Architecture Action Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.18+](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive deep learning framework for human action recognition using the Stanford 40 Actions dataset. This project implements and compares 5 state-of-the-art CNN architectures with end-to-end training, evaluation, and inference pipelines.

## 🌟 Key Features

- **5 Deep Learning Models**: Custom CNN, ResNet50, VGG16, EfficientNetB0, MobileNetV2
- **Complete ML Pipeline**: Data loading, preprocessing, augmentation, training, evaluation, and inference
- **Production-Ready**: Model checkpointing, early stopping, learning rate scheduling
- **Real-time Inference**: Support for webcam predictions and batch processing
- **Comprehensive Visualization**: Training curves, confusion matrices, performance metrics
- **Transfer Learning**: Leverages ImageNet pre-trained weights with fine-tuning

## 📊 Model Performance (Expected)

| Model | Accuracy | Parameters | Training Time | Best For |
|-------|----------|------------|---------------|----------|
| **ResNet50** | ~88.7% | 25M | ~50 min | Highest accuracy |
| **EfficientNetB0** | ~87.1% | 5M | ~40 min | Best efficiency |
| **MobileNetV2** | ~84.5% | 3.5M | ~30 min | Mobile deployment |
| **VGG16** | ~82.6% | 15M | ~38 min | Feature extraction |
| **Custom CNN** | ~78.3% | 5.2M | ~45 min | Learning baseline |

*Training times are approximate and depend on hardware (CPU vs GPU)*

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.18+
8GB+ RAM (16GB recommended)
CUDA-compatible GPU (optional, recommended for faster training)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/bharat3645/stanford40_action_classification-ML-Sem7-.git
cd stanford40_action_classification-ML-Sem7-

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

**Option 1: Download and Setup Automatically**
```bash
python download_dataset.py
```

**Option 2: Manual Setup**
1. Download Stanford 40 Actions dataset from [Stanford Vision Lab](http://vision.stanford.edu/Datasets/40actions.html)
2. Extract images to `data/Stanford40/JPEGImages/`
3. Generate annotations:
```bash
python generate_annotations.py
```

### Training Models

**Train a single model (recommended for first run):**
```bash
python main.py --mode train --models resnet50 --epochs 20
```

**Train all models:**
```bash
python main.py --mode full --epochs 20
```

**Quick test on subset (for development):**
```bash
python main.py --mode train --models custom_cnn --epochs 5 --limit 1000
```

### Evaluation

```bash
# Evaluate specific model
python main.py --mode evaluate --models resnet50

# Evaluate all trained models
python main.py --mode evaluate
```

### Inference

**Single image prediction:**
```bash
python main.py --mode inference --models resnet50 --image path/to/image.jpg
```

**Programmatic inference:**
```python
from inference import ActionClassifier

# Load trained model
classifier = ActionClassifier('models/saved_models/resnet50.keras', 'ResNet50')
classifier.load_model()

# Predict
result = classifier.predict_single('test_image.jpg', top_k=5)
print(f"Predicted Action: {result['top_class']}")
print(f"Confidence: {result['top_probability']:.2%}")

# Visualize prediction
classifier.visualize_prediction('test_image.jpg', save_path='results/predictions/result.png')
```

## 📁 Project Structure

```
stanford40_action_classification/
│
├── config.py                    # Configuration and hyperparameters
├── data_loader.py               # Dataset loading and preprocessing
├── models.py                    # Model architectures
├── train.py                     # Training pipeline
├── evaluate.py                  # Evaluation and metrics
├── inference.py                 # Inference and prediction
├── main.py                      # Main entry point
├── utils.py                     # Utility functions
├── download_dataset.py          # Dataset downloader
├── generate_annotations.py      # Generate XML annotations
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── QUICK_START.md              # Quick start guide
├── PROJECT_SUMMARY.md          # Detailed project summary
├── EXAMPLE_USAGE.md            # Usage examples
│
├── data/
│   └── Stanford40/
│       ├── JPEGImages/         # Dataset images (9,532 images)
│       └── XMLAnnotations/      # Annotation files
│
├── models/
│   ├── saved_models/           # Trained models (.keras format)
│   └── checkpoints/            # Training checkpoints
│
└── results/
    ├── plots/                  # Training curves, confusion matrices
    ├── metrics/                # Performance metrics (CSV)
    └── predictions/            # Inference results
```

## 🎯 Stanford 40 Actions Dataset

The dataset contains **9,532 images** across **40 action classes**:

```
applauding, blowing_bubbles, brushing_teeth, cleaning_the_floor, climbing,
cooking, cutting_trees, cutting_vegetables, drinking, feeding_a_horse,
fishing, fixing_a_bike, fixing_a_car, gardening, holding_an_umbrella,
jumping, looking_through_a_microscope, looking_through_a_telescope,
playing_guitar, playing_violin, pouring_liquid, pushing_a_cart, reading,
phoning, riding_a_bike, riding_a_horse, rowing_a_boat, running,
shooting_an_arrow, smoking, taking_photos, texting_message, throwing_frisby,
using_a_computer, walking_the_dog, washing_dishes, watching_TV,
waving_hands, writing_on_a_board, writing_on_a_book
```

**Dataset Split:**
- Training: 70% (~6,672 images)
- Validation: 20% (~1,906 images)
- Test: 10% (~954 images)

## 🔧 Configuration

Edit `config.py` to customize training parameters:

```python
# Training Configuration
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Data Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True
}

# Early Stopping
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
```

## 📈 Model Architectures

### 1. Custom CNN
- 4 Convolutional blocks with batch normalization
- Dropout layers for regularization
- Dense layers: 512 → 256 → 40 (classes)
- ~5.2M parameters

### 2. ResNet50 (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuning last 30 layers
- Custom classification head
- ~25M parameters (8M trainable)

### 3. VGG16 (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuning last 5 layers
- Custom classification head
- ~15M parameters (3M trainable)

### 4. EfficientNetB0 (Transfer Learning)
- Pre-trained on ImageNet
- Efficient compound scaling
- Fine-tuning last 30 layers
- ~5M parameters (2M trainable)

### 5. MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet
- Optimized for mobile deployment
- Depthwise separable convolutions
- ~3.5M parameters (1.5M trainable)

## 🎨 Visualizations

The project generates comprehensive visualizations:

1. **Dataset Samples**: Grid of sample images from each class
2. **Class Distribution**: Bar chart showing dataset balance
3. **Training Curves**: Loss and accuracy over epochs
4. **Confusion Matrix**: Per-class prediction analysis
5. **Per-Class Metrics**: Precision, recall, F1-score for each action
6. **Top-K Accuracy**: Accuracy at K=1, 3, 5 predictions
7. **Model Comparison**: Side-by-side performance comparison

## 🔬 Advanced Features

### Data Augmentation
- Rotation: ±20 degrees
- Width/Height shift: ±20%
- Zoom: ±20%
- Horizontal flip
- Shear transformations

### Training Optimizations
- **Early Stopping**: Stops training when validation loss plateaus
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Learning Rate Scheduling**: Reduces LR on plateau
- **TensorBoard Logging**: Real-time training monitoring
- **CSV Logging**: Training history saved to CSV

### Callbacks
```python
- ModelCheckpoint: Save best model
- EarlyStopping: Prevent overfitting
- ReduceLROnPlateau: Adaptive learning rate
- TensorBoard: Visualization
- CSVLogger: Training history
```

## 📊 Evaluation Metrics

Each model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **Top-5 Accuracy**: Target class in top 5 predictions
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class prediction analysis
- **Per-Class Metrics**: Individual performance for each action

## 🐛 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16  # or even 8

# Or use a smaller model
python main.py --models mobilenet
```

### Slow Training
```bash
# Enable GPU (if available)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Or reduce epochs
python main.py --epochs 10

# Or use smaller dataset
python main.py --limit 2000
```

### Dataset Not Found
```bash
# Run dataset downloader
python download_dataset.py

# Or manually place images in data/Stanford40/JPEGImages/
# Then run: python generate_annotations.py
```

### Module Not Found
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 🔄 Training Pipeline

```
1. Data Loading
   └── Load images from JPEGImages/
   └── Parse XML annotations
   └── Preprocess and normalize

2. Data Augmentation
   └── Apply transformations
   └── Generate augmented samples

3. Model Building
   └── Load pre-trained weights
   └── Add custom classification head
   └── Compile with optimizer

4. Training
   └── Fit model with callbacks
   └── Monitor validation metrics
   └── Save best model

5. Evaluation
   └── Load best model
   └── Predict on test set
   └── Generate metrics and plots

6. Inference
   └── Load trained model
   └── Predict on new images
   └── Visualize results
```

## 📚 Usage Examples

See [EXAMPLE_USAGE.md](EXAMPLE_USAGE.md) for detailed examples including:
- Custom training loops
- Ensemble predictions
- Batch processing
- Real-time webcam inference
- Custom data augmentation
- Model fine-tuning strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{stanford40-deepclassifier-2025,
  author = {Bharat Singh Parihar},
  title = {Stanford40-DeepClassifier: Multi-Architecture Action Recognition System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/bharat3645/stanford40_action_classification-ML-Sem7-}
}
```

### Original Dataset Citation

```bibtex
@inproceedings{yao2011human,
  title={Human action recognition by learning bases of action attributes and parts},
  author={Yao, Bangpeng and Jiang, Xiaoye and Khosla, Aditya and Lin, Andy Lai and Guibas, Leonidas and Fei-Fei, Li},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2011}
}
```

## 🙏 Acknowledgments

- **Stanford Vision Lab** for the Stanford 40 Actions dataset
- **TensorFlow/Keras teams** for the excellent deep learning framework
- **ImageNet** for pre-trained model weights
- **Open-source community** for various tools and libraries

## 📞 Contact

- **GitHub**: [@bharat3645](https://github.com/bharat3645)
- **Project Link**: [https://github.com/bharat3645/stanford40_action_classification-ML-Sem7-](https://github.com/bharat3645/stanford40_action_classification-ML-Sem7-)

## 🎓 Academic Context

This project was developed as part of the Machine Learning (Semester 7) coursework, demonstrating:
- Deep learning model implementation
- Transfer learning techniques
- Computer vision applications
- Production-ready ML systems
- Model evaluation and comparison

---

**Made with ❤️ for Computer Vision and Deep Learning**

⭐ **Star this repository if you find it helpful!**
