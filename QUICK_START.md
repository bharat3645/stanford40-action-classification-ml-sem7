# Quick Start Guide

This guide will help you get started with the Stanford 40 Actions Classification system in minutes.

## Prerequisites

- Python 3.8+
- 8GB+ RAM
- 10GB+ free disk space
- (Optional) NVIDIA GPU with CUDA support

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python download_dataset.py
```

This will download and extract the Stanford 40 Actions dataset to `data/Stanford40/`.

## Training

### Quick Training (Single Model)

Train ResNet50 (recommended):

```bash
python main.py --mode train --models resnet50 --epochs 30
```

### Train All Models

```bash
python main.py --mode full --epochs 50
```

### Train Specific Models

```bash
python main.py --mode train --models resnet50 efficientnet mobilenet --epochs 30
```

### Quick Test (Limited Data)

```bash
python main.py --mode train --models custom_cnn --epochs 10 --limit 1000
```

## Evaluation

### Evaluate Trained Models

```bash
python main.py --mode evaluate --models resnet50
```

### Evaluate All Models

```bash
python main.py --mode evaluate
```

## Inference

### Single Image Prediction

```bash
python main.py --mode inference --models resnet50 --image path/to/your/image.jpg
```

### Programmatic Inference

```python
from inference import ActionClassifier

# Load classifier
classifier = ActionClassifier(
    model_path='models/saved_models/resnet50.h5',
    model_name='ResNet50'
)
classifier.load_model()

# Predict
result = classifier.predict_single(image_path='test.jpg', top_k=5)
print(f"Action: {result['top_class']}")
print(f"Confidence: {result['top_probability']:.2%}")

# Visualize
classifier.visualize_prediction(image_path='test.jpg')
```

### Real-time Webcam

```python
from inference import ActionClassifier

classifier = ActionClassifier('models/saved_models/resnet50.h5')
classifier.load_model()
classifier.predict_from_webcam(duration=30)
```

## Project Structure

```
stanford40_action_classification/
├── config.py              # Configuration
├── data_loader.py         # Data loading
├── models.py              # Model architectures
├── train.py               # Training
├── evaluate.py            # Evaluation
├── inference.py           # Inference
├── main.py                # Main pipeline
├── utils.py               # Utilities
├── requirements.txt       # Dependencies
└── README.md              # Full documentation
```

## Common Commands

### View Model Summary

```python
from models import ActionClassificationModels

builder = ActionClassificationModels()
model = builder.build_resnet50()
builder.get_model_summary(model)
```

### Load and Explore Data

```python
from data_loader import Stanford40DataLoader

loader = Stanford40DataLoader()
X, y, metadata = loader.load_dataset()

# Visualize samples
loader.visualize_samples(X, y, num_samples=16)

# Check class distribution
loader.get_class_distribution(y)
```

### Compare Model Performance

```python
from evaluate import compare_models

model_paths = [
    'models/saved_models/resnet50.h5',
    'models/saved_models/efficientnet.h5'
]
model_names = ['ResNet50', 'EfficientNet']

comparison = compare_models(model_paths, model_names, X_test, y_test)
print(comparison)
```

## Tips

1. **Start Small**: Use `--limit 1000` flag to test on subset
2. **GPU Recommended**: Training is much faster with GPU
3. **Data Augmentation**: Enabled by default, disable with `--no-augmentation`
4. **Early Stopping**: Training stops automatically if no improvement
5. **Best Model**: ResNet50 typically achieves 85-90% accuracy

## Troubleshooting

### Out of Memory

- Reduce batch size in `config.py`
- Use MobileNetV2 (smaller model)
- Use CPU instead of GPU

### Dataset Not Found

```bash
python download_dataset.py
```

### Slow Training

- Enable GPU support
- Reduce number of epochs
- Use smaller model (MobileNetV2)

### Import Errors

```bash
pip install -r requirements.txt --upgrade
```

## Expected Results

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Custom CNN | 78-80% | ~45 min |
| ResNet50 | **88-90%** | ~50 min |
| EfficientNet | 86-88% | ~40 min |
| MobileNetV2 | 84-86% | ~30 min |

## Next Steps

1. Read full documentation in `README.md`
2. Explore `examples/` directory for notebooks
3. Customize models in `models.py`
4. Tune hyperparameters in `config.py`

## Support

For issues and questions:
- Open GitHub issue
- Check documentation
- Review code comments

---

Happy Coding! 🚀
