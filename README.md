# Stanford 40 Actions Classification System

A comprehensive deep learning system for human action recognition in still images using the Stanford 40 Actions dataset. This project implements and compares multiple state-of-the-art deep learning architectures for action classification.

> ## ⚠️ Status: Pipeline implemented, training not yet run
>
> This repository contains a complete, runnable data-loading / training / evaluation / inference pipeline for the Stanford 40 Actions dataset. **No training run has actually been executed in this repository.** `data/Stanford40/JPEGImages/` and `XMLAnnotations/` contain only `.gitkeep` placeholders (the dataset was never downloaded here), and `models/saved_models/` and `models/checkpoints/` are likewise empty — there are no trained weights, no `results/` directory, and no logs.
>
> Every accuracy/F1/training-time figure in the **[Experiments and Results](#experiments-and-results)**, **[Model Comparison](#model-comparison)**, and **[Conclusion](#conclusion)** sections below is a **projected/target metric**, not a measurement produced by running this code — they are estimates based on how these architectures typically perform on comparable image-classification benchmarks in the published literature, written down as the goals this pipeline was designed to hit. They are explicitly labeled "(projected)" throughout. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the full, honest completion status, and the [Known Limitations](#known-limitations) section below.

## Table of Contents
- [Problem Description](#problem-description)
- [Dataset Information](#dataset-information)
- [System Architecture](#system-architecture)
- [Methods and Approaches](#methods-and-approaches)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments and Results](#experiments-and-results)
- [Model Comparison](#model-comparison)
- [Conclusion](#conclusion)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Problem Description

### Overview
Human action recognition in still images is a fundamental computer vision task with applications in surveillance, human-computer interaction, content-based image retrieval, and assistive technologies. Unlike video-based action recognition that leverages temporal information, recognizing actions from single images requires understanding subtle visual cues, poses, object interactions, and contextual information.

### Problem Statement
Given a still image containing a human subject, the objective is to accurately classify the action being performed from 40 predefined action categories. This is challenging because:
- **Visual Ambiguity**: Similar poses can represent different actions depending on context
- **Intra-class Variation**: Same action can be performed in different ways, angles, and environments
- **Inter-class Similarity**: Different actions may have similar visual appearances
- **Scale and Viewpoint Variation**: Images captured from different distances and angles
- **Occlusion and Clutter**: Real-world images often contain partial occlusions and complex backgrounds

### Importance
Accurate action recognition enables:
- **Surveillance Systems**: Automated monitoring and anomaly detection
- **Assistive Technologies**: Support for visually impaired individuals through scene understanding
- **Content Organization**: Automatic tagging and retrieval in large image databases
- **Human-Robot Interaction**: Understanding human intentions and behaviors
- **Healthcare**: Monitoring patient activities and rehabilitation progress

### Project Goals
1. Implement multiple deep learning architectures (Custom CNN, Transfer Learning models)
2. Compare performance across different model architectures
3. Achieve high accuracy (>85%) on the Stanford 40 Actions dataset
4. Develop a production-ready inference system for real-time classification
5. Provide comprehensive analysis and visualizations of model performance

---

## Dataset Information

### Stanford 40 Actions Dataset

**Source**: Stanford Vision Lab  
**Paper**: "Human Action Recognition by Learning Bases of Action Attributes and Parts"  
**Authors**: B. Yao, X. Jiang, A. Khosla, A.L. Lin, L.J. Guibas, and L. Fei-Fei  
**Conference**: ICCV 2011

### Dataset Statistics
- **Total Images**: 9,532 images
- **Number of Classes**: 40 action categories
- **Images per Class**: 180-300 images
- **Image Format**: JPEG
- **Annotations**: XML format with bounding boxes and action labels
- **Image Resolution**: Variable (resized to 224×224 for model input)

### Action Classes (40 Categories)
The dataset contains the following action categories:

1. applauding
2. blowing_bubbles
3. brushing_teeth
4. cleaning_the_floor
5. climbing
6. cooking
7. cutting_trees
8. cutting_vegetables
9. drinking
10. feeding_a_horse
11. fishing
12. fixing_a_bike
13. fixing_a_car
14. gardening
15. holding_an_umbrella
16. jumping
17. looking_through_a_microscope
18. looking_through_a_telescope
19. playing_guitar
20. playing_violin
21. pouring_liquid
22. pushing_a_cart
23. reading
24. phoning
25. riding_a_bike
26. riding_a_horse
27. rowing_a_boat
28. running
29. shooting_an_arrow
30. smoking
31. taking_photos
32. texting_message
33. throwing_frisby
34. using_a_computer
35. walking_the_dog
36. washing_dishes
37. watching_TV
38. waving_hands
39. writing_on_a_board
40. writing_on_a_book

### Dataset Structure
```
Stanford40/
├── JPEGImages/          # All image files
├── XMLAnnotations/      # XML annotation files with labels and bounding boxes
└── ImageSplits/         # Train/test split information
```

### Data Preprocessing
1. **Image Loading**: Images loaded from JPEG format
2. **Annotation Parsing**: XML files parsed to extract action labels and bounding boxes
3. **Resizing**: All images resized to 224×224 pixels
4. **Normalization**: Pixel values normalized to [0, 1] range
5. **Data Augmentation** (Training only):
   - Random rotation (±20 degrees)
   - Width/height shifts (±20%)
   - Horizontal flipping
   - Zoom (±20%)
   - Shear transformations (±15%)

### Data Split
- **Training Set**: 70% (~6,672 images)
- **Validation Set**: 20% (~1,906 images)
- **Test Set**: 10% (~954 images)

The split is stratified to maintain class distribution across all sets.

---

## System Architecture

### Overall Pipeline

```
┌─────────────────┐
│   Input Image   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - Resize       │
│  - Normalize    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │
│  Extraction     │
│  (CNN/Transfer) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Classification │
│  Head           │
│  (Dense Layers) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Action         │
│  Prediction     │
└─────────────────┘
```

### Technology Stack
- **Framework**: TensorFlow 2.15.0 / Keras
- **Programming Language**: Python 3.8+
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Model Evaluation**: scikit-learn

---

## Methods and Approaches

### 1. Custom CNN Architecture

**Design Philosophy**: Built from scratch to learn hierarchical features specific to action recognition.

**Architecture**:
- **4 Convolutional Blocks**: Each containing:
  - 2 Conv2D layers (filters: 32→64→128→256)
  - Batch Normalization
  - ReLU activation
  - MaxPooling2D (2×2)
  - Dropout (0.25)
- **Dense Layers**:
  - Flatten layer
  - Dense(512) + BatchNorm + ReLU + Dropout(0.5)
  - Dense(256) + BatchNorm + ReLU + Dropout(0.5)
  - Dense(40, softmax) - Output layer

**Total Parameters**: ~5.2M trainable parameters

**Advantages**:
- Designed specifically for action recognition task
- Efficient with moderate computational requirements
- Good baseline for comparison

**Training Strategy**:
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Cross-Entropy
- Metrics: Accuracy, Top-5 Accuracy, Precision, Recall
- Data Augmentation: Applied during training

---

### 2. Transfer Learning Models

Transfer learning leverages pre-trained models on ImageNet (1.2M images, 1000 classes) and fine-tunes them for action classification.

#### a) ResNet50

**Base Architecture**: Residual Network with 50 layers

**Key Features**:
- Residual connections to address vanishing gradient problem
- Deep architecture (50 layers) for complex feature learning
- Pre-trained on ImageNet

**Modifications**:
- Freeze early layers, fine-tune last 30 layers
- Remove original classification head
- Add custom classification layers:
  - GlobalAveragePooling2D
  - Dense(512) + BatchNorm + Dropout(0.5)
  - Dense(256) + BatchNorm + Dropout(0.3)
  - Dense(40, softmax)

**Parameters**: ~25M total, ~8M trainable

**Expected Performance**: High accuracy due to powerful feature extraction

---

#### b) VGG16

**Base Architecture**: Visual Geometry Group 16-layer network

**Key Features**:
- Simple architecture with small (3×3) filters
- Deep network (16 layers)
- Strong performance on various vision tasks

**Modifications**:
- Fine-tune last 5 layers
- Custom classification head similar to ResNet50

**Parameters**: ~15M total, ~3M trainable

**Expected Performance**: Good accuracy, slightly lower than ResNet50

---

#### c) EfficientNetB0

**Base Architecture**: Efficient Neural Network with compound scaling

**Key Features**:
- Balanced scaling of depth, width, and resolution
- Mobile-optimized architecture
- State-of-the-art efficiency

**Modifications**:
- Fine-tune last 30 layers
- Custom classification head

**Parameters**: ~5M total, ~2M trainable

**Expected Performance**: High accuracy with lower computational cost

---

#### d) MobileNetV2

**Base Architecture**: Mobile-optimized convolutional neural network

**Key Features**:
- Depthwise separable convolutions
- Inverted residual blocks
- Lightweight and fast

**Modifications**:
- Fine-tune last 20 layers
- Custom classification head

**Parameters**: ~3.5M total, ~1.5M trainable

**Expected Performance**: Good accuracy with fastest inference speed

---

#### e) Vision Transformer (ViT)

**Architecture**: Transformer-based model for image classification

**Key Features**:
- Patch-based image processing
- Multi-head self-attention mechanism
- Global context understanding

**Implementation**:
- Patch size: 16×16
- Projection dimension: 256
- 4 Transformer blocks
- 8 attention heads
- MLP dimension: 512

**Parameters**: ~8M trainable

**Expected Performance**: Competitive with CNNs, especially on larger datasets

---

### Training Configuration

**Common Settings**:
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam
- **Initial Learning Rate**: 0.001
- **Learning Rate Schedule**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping**: Patience=10 epochs

**Callbacks**:
1. ModelCheckpoint - Save best model based on validation accuracy
2. EarlyStopping - Stop training if no improvement
3. ReduceLROnPlateau - Reduce learning rate when validation loss plateaus
4. TensorBoard - Log training metrics
5. CSVLogger - Save training history to CSV

---

### Evaluation Metrics

1. **Accuracy**: Overall classification accuracy
2. **Top-5 Accuracy**: Target class in top 5 predictions
3. **Precision**: Ratio of correct positive predictions
4. **Recall**: Ratio of actual positives correctly identified
5. **F1-Score**: Harmonic mean of precision and recall
6. **Confusion Matrix**: Per-class prediction analysis
7. **Per-Class Metrics**: Individual performance for each action

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.x (for GPU support)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone Repository
```bash
git clone https://github.com/bharat3645/stanford40_action_classification-ML-Sem7.git
cd stanford40_action_classification-ML-Sem7
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n action_classification python=3.8
conda activate action_classification

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Download Stanford 40 Actions dataset from:
   - Images: [Stanford Vision Lab](http://vision.stanford.edu/Datasets/40actions.html)
   - Or use the provided download script:
   ```bash
   python download_dataset.py
   ```

2. Extract dataset to `data/Stanford40/` directory:
   ```
   data/
   └── Stanford40/
       ├── JPEGImages/
       ├── XMLAnnotations/
       └── ImageSplits/
   ```

### Step 5: Verify Installation
```bash
python -c "import tensorflow as tf; print(tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### Troubleshooting

**Out of Memory Error**
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16  # or even 8

# Or use a smaller model
python main.py --models mobilenet
```

**Slow Training**
```bash
# Enable GPU (if available)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Or reduce epochs
python main.py --epochs 10

# Or use smaller dataset
python main.py --limit 2000
```

**Dataset Not Found**
```bash
# Run dataset downloader
python download_dataset.py

# Or manually place images in data/Stanford40/JPEGImages/
# Then run: python generate_annotations.py
```

**Module Not Found**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## Usage

### 1. Data Loading and Preprocessing

```python
from data_loader import Stanford40DataLoader

# Initialize loader
loader = Stanford40DataLoader()

# Load dataset
X, y, metadata = loader.load_dataset(use_bbox=False)

# Split dataset
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(X, y)

# Visualize samples
loader.visualize_samples(X_train, y_train, num_samples=16)

# Get class distribution
loader.get_class_distribution(y)
```

### 2. Model Building

```python
from models import ActionClassificationModels

# Initialize model builder
model_builder = ActionClassificationModels()

# Build custom CNN
custom_cnn = model_builder.build_custom_cnn()
custom_cnn = model_builder.compile_model(custom_cnn)

# Build ResNet50 with transfer learning
resnet = model_builder.build_resnet50(trainable_layers=30)
resnet = model_builder.compile_model(resnet)

# Get model summary
model_builder.get_model_summary(custom_cnn)
```

### 3. Training

```python
from train import ModelTrainer

# Create trainer
trainer = ModelTrainer(model_name='resnet50')

# Train model
history = trainer.train(
    model=resnet,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=50,
    batch_size=32
)

# Plot training history
trainer.plot_training_history(
    save_path='results/plots/resnet50_training.png'
)
```

**Train All Models**:
```bash
python train.py
```

This will:
- Train all implemented models
- Save best models to `models/saved_models/`
- Generate training plots in `results/plots/`
- Save metrics to `results/metrics/`

### 4. Evaluation

```python
from evaluate import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(
    model_path='models/saved_models/resnet50.h5',
    model_name='ResNet50'
)

# Load model and evaluate
evaluator.load_model()
results, y_pred, y_pred_probs = evaluator.evaluate_model(X_test, y_test)

# Generate confusion matrix
evaluator.plot_confusion_matrix(y_test, y_pred)

# Generate classification report
report = evaluator.generate_classification_report(y_test, y_pred)

# Plot per-class metrics
evaluator.plot_per_class_metrics(y_test, y_pred)

# Visualize predictions
evaluator.visualize_predictions(X_test, y_test, y_pred, show_correct=True)
```

**Evaluate All Models**:
```bash
python evaluate.py
```

### 5. Inference (Real-time Classification)

```python
from inference import ActionClassifier

# Initialize classifier
classifier = ActionClassifier(
    model_path='models/saved_models/resnet50.h5',
    model_name='ResNet50'
)

# Load model
classifier.load_model()

# Predict single image
result = classifier.predict_single(
    image_path='path/to/image.jpg',
    top_k=5
)

print(f"Predicted Action: {result['top_class']}")
print(f"Confidence: {result['top_probability']:.2%}")

# Visualize prediction
classifier.visualize_prediction(
    image_path='path/to/image.jpg',
    save_path='results/predictions/prediction.png'
)

# Batch prediction
results = classifier.predict_batch(
    image_paths=['image1.jpg', 'image2.jpg', 'image3.jpg']
)

# Real-time webcam classification
classifier.predict_from_webcam(duration=30)
```

**Quick Inference**:
```bash
python inference.py --model resnet50 --image path/to/image.jpg
```

---

## Experiments and Results

> **⚠️ Projected, not measured.** No training run has been executed in this repository — see the [status note](#stanford-40-actions-classification-system) at the top of this README and [PROJECT_STATUS.md](PROJECT_STATUS.md). Everything in this section (hardware/software "setup", the performance table, key findings, per-class analysis, training curves, and ablations) describes the target configuration and the results this pipeline is designed to produce, not numbers that were actually observed. Numbers are marked **(projected)** throughout.

### Experimental Setup (Target Configuration — Not Yet Run)

**Hardware this pipeline targets** (recommended for a real run; the environment this repo was authored in has no GPU, per [PROJECT_STATUS.md](PROJECT_STATUS.md)):
- CPU: Intel i7/AMD Ryzen 7 or better
- GPU: NVIDIA RTX 3060 or better (recommended)
- RAM: 16GB
- Storage: 20GB SSD

**Software**:
- OS: Ubuntu 20.04 / Windows 10/11
- Python: 3.8.10
- TensorFlow: 2.15.0
- CUDA: 11.8 (for GPU training)

**Training Configuration** (as implemented in `config.py` / `train.py`; not yet executed):
- Epochs: 50 (with early stopping)
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Data Augmentation: Enabled for training
- Stratified split: 70% train, 20% validation, 10% test

### Model Performance Comparison (Projected — Not Measured)

All figures below are **projected targets**, not results from an actual run — no `results/metrics/*.csv` or trained checkpoint exists in this repository to back them. They are estimates based on how these architectures typically perform on comparable image-classification transfer-learning tasks in the published literature, offered as the goals this pipeline was designed to hit.

| Model | Accuracy (projected) | Top-5 Acc (projected) | Precision (projected) | Recall (projected) | F1-Score (projected) | Parameters | Training Time (projected) |
|-------|----------|-----------|-----------|--------|----------|------------|---------------|
| **Custom CNN** | 78.3% | 94.2% | 0.7756 | 0.7830 | 0.7791 | 5.2M | 45 min |
| **VGG16** | 82.6% | 96.1% | 0.8189 | 0.8260 | 0.8223 | 15M | 38 min |
| **ResNet50** | **88.7%** | **97.8%** | **0.8832** | **0.8870** | **0.8850** | 25M | 52 min |
| **EfficientNetB0** | 87.1% | 97.3% | 0.8656 | 0.8710 | 0.8682 | 5M | 42 min |
| **MobileNetV2** | 84.5% | 96.7% | 0.8398 | 0.8450 | 0.8423 | 3.5M | 28 min |
| **Vision Transformer** | 85.9% | 96.9% | 0.8542 | 0.8590 | 0.8565 | 8M | 58 min |

### Key Findings (Projected — Not Measured)

These are the outcomes the design anticipates, not observations from a completed run:

1. **Best Overall Performance (projected)**: ResNet50 is expected to achieve the highest accuracy (~88.7%) with strong performance across all metrics.

2. **Efficiency vs. Performance (projected)**: EfficientNetB0 is expected to offer an excellent balance with ~87.1% accuracy and only 5M parameters.

3. **Speed (projected)**: MobileNetV2 should be the fastest model (~28 min training) with a respectable ~84.5% accuracy, ideal for mobile deployment.

4. **Custom CNN (projected)**: Expected to provide a solid baseline (~78.3%), consistent with task-specific architectures competing reasonably with limited resources.

5. **Vision Transformer (projected)**: Expected to be competitive (~85.9%), consistent with published results for attention-based approaches on similarly sized datasets.

### Per-Class Analysis (Illustrative Example — Not Measured)

**No per-class evaluation has been run.** The numbers below are an illustrative example of the kind of breakdown `evaluate.py` would produce (per-class accuracy, best/worst performing classes, commonly confused pairs) — they are not measurements from this repository and should not be cited as such. They are included only to show what the evaluation pipeline reports.

Illustrative best-performing actions (>90% accuracy, hypothetical):
- riding_a_bike, playing_guitar, reading, using_a_computer, watching_TV

Illustrative challenging actions (<75% accuracy, hypothetical), based on classes with visually similar poses/objects where confusion is plausible:
- waving_hands (plausible confusion with applauding)
- throwing_frisby (plausible confusion with shooting_an_arrow)
- fixing_a_bike (plausible confusion with fixing_a_car)
- brushing_teeth (plausible confusion with drinking)

### Training Curves (Expected Behavior — Not Measured)

No training has been run, so no curves exist yet. Once trained, `train.py`'s `plot_training_history` and TensorBoard/CSV logging are expected to show the typical, well-behaved pattern for this setup: steady improvement in training accuracy, a small train-val gap thanks to augmentation and regularization, and convergence within the configured epoch budget — but this has not been verified.

### Ablation Studies (Illustrative — Not Measured)

**No ablation experiments have been run.** The figures below illustrate the kind of comparison the pipeline supports (e.g., toggling `--no-augmentation`, varying `trainable_layers`, changing `config.py`'s input resolution) and a plausible direction/magnitude of effect based on general transfer-learning literature — they are not results measured on this dataset.

Illustrative data augmentation impact (hypothetical):
- Without augmentation: ~82% accuracy
- With augmentation: ~88.7% accuracy (projected target above)

Illustrative fine-tuning strategy impact (hypothetical):
- Freeze all layers: lower accuracy than fine-tuning
- Fine-tune last 30 layers: closer to the ~88.7% projected target

Illustrative input resolution impact (hypothetical):
- 128×128: lower accuracy than 224×224
- 224×224: the configuration used for the ~88.7% projected target
- 299×299: marginal further gain possible, at higher compute cost

---

## Model Comparison

> **Note**: Accuracy/training-time figures quoted in this section (e.g. "88.7%", "52 min") are the same **projected, not measured** numbers from [Model Performance Comparison](#model-performance-comparison-projected--not-measured) above. Architectural strengths/weaknesses (parameter counts, relative speed) are structural facts about the models; the accuracy figures are not.

### Strengths and Weaknesses

#### Custom CNN
**Strengths**:
- Task-specific design
- Moderate computational requirements
- Good learning capability
- Full control over architecture

**Weaknesses**:
- Lower accuracy compared to transfer learning
- Requires more training data for optimal performance
- Limited pre-learned features

**Best For**: Resource-constrained environments, educational purposes, baseline comparison

---

#### ResNet50
**Strengths**:
- Highest projected accuracy (88.7%, not measured)
- Strong feature extraction
- Handles complex patterns well
- Pre-trained on large dataset

**Weaknesses**:
- Largest model size (25M parameters)
- Longer training time (52 min)
- Higher memory requirements

**Best For**: Production systems prioritizing accuracy, applications with sufficient computational resources

---

#### EfficientNetB0
**Strengths**:
- Excellent projected accuracy-efficiency balance (87.1%, not measured)
- Compact model size (5M parameters)
- Reasonable training time (42 min)
- State-of-the-art architecture

**Weaknesses**:
- Slightly lower accuracy than ResNet50
- Complex architecture

**Best For**: Balanced deployment scenarios, cloud-based applications, cost-effective solutions

---

#### MobileNetV2
**Strengths**:
- Fastest training (28 min)
- Smallest model size (3.5M)
- Mobile-optimized
- Good real-time performance

**Weaknesses**:
- Lower projected accuracy (84.5%, not measured)
- Limited capacity for complex patterns

**Best For**: Mobile applications, edge devices, real-time systems, IoT deployment

---

### Deployment Recommendations

| Use Case | Recommended Model | Justification |
|----------|-------------------|---------------|
| **Production API** | ResNet50 | Highest accuracy, acceptable latency |
| **Mobile App** | MobileNetV2 | Lightweight, fast inference on devices |
| **Edge Computing** | EfficientNetB0 | Best balance of accuracy and efficiency |
| **Real-time Video** | MobileNetV2 | Fast enough for 30+ FPS processing |
| **Cloud Service** | ResNet50 | Leverage cloud GPUs for best accuracy |
| **Embedded Systems** | MobileNetV2 | Fits memory constraints of embedded hardware |

---

## Conclusion

> **⚠️ Projected, not measured.** As throughout this README, the specific figures in this section (88.7%, 78.3%, 6.6 percentage points, etc.) are projected targets carried over from the [Model Performance Comparison](#model-performance-comparison-projected--not-measured) table, not results from a completed run. See [Known Limitations](#known-limitations) below.

### Summary of What This Project Delivers

This project developed a comprehensive, runnable action classification pipeline for the Stanford 40 Actions dataset. Concretely, as of this writing:

1. **Accuracy (projected, not measured)**: The pipeline targets ~88.7% test accuracy with ResNet50 based on how this architecture performs on comparable transfer-learning benchmarks; this repository has not yet produced a trained ResNet50 checkpoint or a measured test accuracy.

2. **Comprehensive Comparison (implemented, not yet run)**: The code implements and is designed to compare 6 different architectures (Custom CNN, ResNet50, VGG16, EfficientNetB0, MobileNetV2, Vision Transformer) — see `models.py`. No comparison run has actually been executed.

3. **Pipeline (implemented, not "production-ready")**: A complete pipeline exists for data loading, preprocessing, training, evaluation, and inference (`data_loader.py`, `train.py`, `evaluate.py`, `inference.py`). It compiles and is structurally complete, but has not been exercised end-to-end against the real dataset in this repository, so it should not be considered validated or production-ready.

4. **Detailed Analysis (capability exists, no output generated)**: `evaluate.py` is written to generate confusion matrices, per-class performance, and training curves, but no such artifacts currently exist in `results/` because no training/evaluation run has happened.

5. **Flexible Deployment (architectural claim only)**: Models ranging from lightweight (MobileNetV2, 3.5M params) to larger (ResNet50, 25M params) are implemented in code; this is a statement about the architectures offered, not about deployment-tested, trained weights.

### Key Learnings (Anticipated — Not Empirically Validated)

These are expectations drawn from the general transfer-learning literature that motivated this project's design, not conclusions drawn from an experiment run on this dataset:

1. **Transfer Learning Power (expected)**: Pre-trained models are expected to outperform the custom CNN (~88.7% vs ~78.3%, projected), consistent with the general effectiveness of transfer learning — not yet confirmed here.

2. **Data Augmentation (expected)**: Augmentation is expected to meaningfully improve accuracy on a moderate-sized dataset like this one (9,532 images) — the specific "+6.6 percentage points" figure is a hypothetical illustration (see [Ablation Studies](#ablation-studies-illustrative--not-measured)), not a measured result.

3. **Fine-tuning Strategy (expected)**: Fine-tuning more layers of ResNet50 is expected to outperform freezing all layers — not yet confirmed here.

4. **Model Selection Trade-offs**: No single model is universally best; choice depends on accuracy requirements, computational constraints, and deployment environment. (This is a general, architecture-level observation, independent of the unmeasured accuracy figures.)

5. **Action Recognition Challenges (expected)**: Similar poses and contexts (e.g., waving vs. applauding, fixing bike vs. fixing car) are plausible sources of confusion based on the class list, not confirmed by an actual confusion matrix.

### Future Work

1. **Ensemble Methods**: Combine multiple models (ResNet50 + EfficientNet) to boost accuracy beyond 90%.

2. **Attention Mechanisms**: Integrate spatial attention to focus on relevant image regions (hands, objects, poses).

3. **Multi-Modal Learning**: Incorporate object detection and pose estimation as auxiliary tasks.

4. **Temporal Extension**: Extend to video action recognition using 3D CNNs or two-stream networks.

5. **Few-Shot Learning**: Enable recognition of new actions with limited training examples.

6. **Explainability**: Implement Grad-CAM and attention visualization to understand model decisions.

7. **Real-World Deployment**: Optimize models for specific hardware (TensorRT for NVIDIA, CoreML for iOS).

8. **Dataset Expansion**: Collect additional data for challenging actions to balance class performance.

9. **Cross-Dataset Evaluation**: Test generalization on other action recognition datasets (PASCAL VOC Actions, MPII Human Pose).

10. **Active Learning**: Implement active learning pipeline to iteratively improve model with human feedback.

### Broader Impact

This action classification system has potential applications in:

- **Assistive Technology**: Help visually impaired individuals understand activities in their environment
- **Smart Surveillance**: Automated monitoring in public spaces, elderly care facilities
- **Content Moderation**: Automatic flagging of inappropriate activities in user-generated content
- **Sports Analytics**: Automated annotation of player actions in sports footage
- **Healthcare**: Monitoring patient activities for rehabilitation and safety
- **Robotics**: Enabling robots to understand and respond to human actions
- **E-commerce**: Action-based product recommendations and search

### Conclusion Statement

This project implements a deep learning pipeline for action recognition in still images, designed to compare multiple architectures and provide practical tools for training, evaluation, and inference. The comprehensive comparison of architectures is intended to offer insights for practitioners choosing models for specific use cases once it has actually been run. The modular codebase is intended as a foundation for further work, but it should be treated as unvalidated until a real training run is completed.

**Target Performance**: 88.7% accuracy with ResNet50 (projected, not measured — see [Known Limitations](#known-limitations)). This target is based on how ResNet50 typically performs on comparable transfer-learning image-classification tasks; it has not been reproduced by an actual training run in this repository, so it should not be read as a validated result or a claim of practical deployment readiness.

## Known Limitations

1. **No training has been run in this repository.** `data/Stanford40/JPEGImages/` and `XMLAnnotations/` contain only `.gitkeep` placeholders, and `models/saved_models/`, `models/checkpoints/`, and `results/` are empty or absent. Every accuracy, F1, per-class, ablation, and training-curve figure in this README is a **projected/target metric**, not a measured one. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the itemized completion status.

2. **No GPU was available in the environment this repo was authored in** (per PROJECT_STATUS.md), and the full 6-model × up-to-50-epoch training plan is estimated at several hours even on the recommended GPU hardware — impractical to run as a quick coursework sanity check on CPU only.

3. **Dataset Size**: 9,532 images is moderate; larger datasets could improve performance further, whenever a real run is done.

4. **Single Person Focus**: The dataset/models are designed around single-person actions; multi-person scenarios are not addressed.

5. **Temporal Information**: Still images lack the temporal context that videos provide, limiting understanding of dynamic actions.

6. **Background Bias**: Models may end up relying on background context rather than pure action features — a risk to check for once real training happens, not something that has been tested.

7. **Computational Requirements**: The best-performing models (ResNet50, EfficientNetB0) are expected to require a GPU for practical training and inference.

---

## References

### Primary Dataset Paper
1. B. Yao, X. Jiang, A. Khosla, A.L. Lin, L.J. Guibas, and L. Fei-Fei. **"Human Action Recognition by Learning Bases of Action Attributes and Parts."** *International Conference on Computer Vision (ICCV)*, Barcelona, Spain, November 6-13, 2011.  
   - Paper: [PDF](http://vision.stanford.edu/documents/YaoJiangKhoslaLinGuibasFei-Fei_ICCV2011.pdf)  
   - Dataset: [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html)

### Deep Learning Architectures

2. K. He, X. Zhang, S. Ren, and J. Sun. **"Deep Residual Learning for Image Recognition."** *CVPR 2016*.  
   - Paper: https://arxiv.org/abs/1512.03385  
   - ResNet architecture foundational paper

3. K. Simonyan and A. Zisserman. **"Very Deep Convolutional Networks for Large-Scale Image Recognition."** *ICLR 2015*.  
   - Paper: https://arxiv.org/abs/1409.1556  
   - VGG architecture paper

4. M. Tan and Q. Le. **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."** *ICML 2019*.  
   - Paper: https://arxiv.org/abs/1905.11946  
   - EfficientNet architecture

5. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. Chen. **"MobileNetV2: Inverted Residuals and Linear Bottlenecks."** *CVPR 2018*.  
   - Paper: https://arxiv.org/abs/1801.04381  
   - MobileNetV2 architecture

6. A. Dosovitskiy et al. **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."** *ICLR 2021*.  
   - Paper: https://arxiv.org/abs/2010.11929  
   - Vision Transformer (ViT)

### Action Recognition Research

7. K. Simonyan and A. Zisserman. **"Two-Stream Convolutional Networks for Action Recognition in Videos."** *NIPS 2014*.  
   - Paper: https://arxiv.org/abs/1406.2199

8. J. Carreira and A. Zisserman. **"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset."** *CVPR 2017*.  
   - Paper: https://arxiv.org/abs/1705.07750

### Transfer Learning

9. J. Yosinski, J. Clune, Y. Bengio, and H. Lipson. **"How transferable are features in deep neural networks?"** *NIPS 2014*.  
   - Paper: https://arxiv.org/abs/1411.1792

### Frameworks and Tools

10. TensorFlow Documentation. https://www.tensorflow.org/  
11. Keras Documentation. https://keras.io/  
12. scikit-learn Documentation. https://scikit-learn.org/

### Related Datasets

13. M. Everingham et al. **"The PASCAL Visual Object Classes Challenge."** *IJCV 2010*.  
    - PASCAL VOC Action Classification dataset

14. M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele. **"2D Human Pose Estimation: New Benchmark and State of the Art Analysis."** *CVPR 2014*.  
    - MPII Human Pose dataset

### Online Resources

15. Stanford Vision Lab. http://vision.stanford.edu/  
16. Papers With Code - Action Recognition. https://paperswithcode.com/task/action-recognition-in-videos  
17. TensorFlow Hub - Pre-trained Models. https://tfhub.dev/

---

## Project Structure

```
stanford40_action_classification/
│
├── data/
│   └── Stanford40/              # Dataset directory (download separately)
│       ├── JPEGImages/
│       ├── XMLAnnotations/
│       └── ImageSplits/
│
├── models/
│   ├── saved_models/            # Trained model files (.h5)
│   └── checkpoints/             # Training checkpoints
│
├── results/
│   ├── plots/                   # Training curves, confusion matrices
│   ├── metrics/                 # CSV files with evaluation metrics
│   └── predictions/             # Sample prediction visualizations
│
├── config.py                    # Configuration and hyperparameters
├── data_loader.py               # Data loading and preprocessing
├── models.py                    # Model architectures
├── train.py                     # Training pipeline
├── evaluate.py                  # Evaluation and analysis
├── inference.py                 # Real-time inference
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## License

This project is licensed under the MIT License. The Stanford 40 Actions dataset has its own license - please refer to the [dataset website](http://vision.stanford.edu/Datasets/40actions.html) for terms of use.

## Acknowledgments

- Stanford Vision Lab for creating and maintaining the Stanford 40 Actions dataset
- TensorFlow and Keras teams for excellent deep learning frameworks
- Research community for open-source pre-trained models and architectures

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{stanford40_action_classification,
  author ={Bharat Singh Parihar},
  title = {Stanford40 Actions Classification System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/bharat3645/stanford40_action_classification-ML-Sem7}
}
```

And please cite the original dataset paper:

```bibtex
@inproceedings{yao2011human,
  title={Human action recognition by learning bases of action attributes and parts},
  author={Yao, Bangpeng and Jiang, Xiaoye and Khosla, Aditya and Lin, Andy Lai and Guibas, Leonidas and Fei-Fei, Li},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2011}
}
```

---

## Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub: [Project Issues](https://github.com/bharat3645/stanford40_action_classification-ML-Sem7/issues)
- GitHub: [@bharat3645](https://github.com/bharat3645)

---

**Last Updated**: August 2026  
**Version**: 1.0.0  
**Status**: Pipeline implemented; no training run has been executed yet — see [Known Limitations](#known-limitations) and [PROJECT_STATUS.md](PROJECT_STATUS.md) 🚧
