# Example Usage Guide

This document provides practical examples for using the Stanford 40 Actions Classification system.

## Table of Contents
1. [Data Loading Examples](#data-loading-examples)
2. [Model Training Examples](#model-training-examples)
3. [Evaluation Examples](#evaluation-examples)
4. [Inference Examples](#inference-examples)
5. [Advanced Usage](#advanced-usage)

---

## Data Loading Examples

### Example 1: Load and Visualize Dataset

```python
from data_loader import Stanford40DataLoader
import config

# Initialize loader
loader = Stanford40DataLoader()

# Load full dataset
X, y, metadata = loader.load_dataset(use_bbox=False)

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of images: {len(X)}")

# Visualize random samples
loader.visualize_samples(X, y, num_samples=16, 
                        save_path='results/plots/samples.png')
```

### Example 2: Load with Bounding Box Cropping

```python
# Load dataset with bounding box cropping
X_cropped, y_cropped, metadata_cropped = loader.load_dataset(use_bbox=True)

print(f"Cropped dataset shape: {X_cropped.shape}")
```

### Example 3: Analyze Class Distribution

```python
# Get class distribution
dist_df = loader.get_class_distribution(y, 
                                        save_path='results/plots/distribution.png')

print("\nClass Distribution:")
print(dist_df.head(10))

# Check for imbalance
max_samples = dist_df['Count'].max()
min_samples = dist_df['Count'].min()
print(f"\nImbalance ratio: {max_samples/min_samples:.2f}")
```

---

## Model Training Examples

### Example 1: Train Custom CNN

```python
from models import ActionClassificationModels
from train import ModelTrainer

# Build model
builder = ActionClassificationModels()
model = builder.build_custom_cnn()
model = builder.compile_model(model)

# Train
trainer = ModelTrainer(model_name='custom_cnn')
history = trainer.train(model, X_train, y_train, X_val, y_val, epochs=50)

# Plot training history
trainer.plot_training_history(save_path='results/plots/custom_cnn_history.png')
```

### Example 2: Transfer Learning with ResNet50

```python
# Build ResNet50 with transfer learning
resnet = builder.build_resnet50(trainable_layers=30)
resnet = builder.compile_model(resnet, learning_rate=0.0001)

# Train with data augmentation
trainer = ModelTrainer(model_name='resnet50')

# Create data generators
train_gen, val_gen = loader.create_data_generators(X_train, y_train, X_val, y_val)

# Train
history = trainer.train_with_generator(
    resnet, train_gen, val_gen,
    steps_per_epoch=len(X_train) // config.BATCH_SIZE,
    validation_steps=len(X_val) // config.BATCH_SIZE,
    epochs=50
)
```

### Example 3: Fine-tune Pre-trained Model

```python
# Load pre-trained model
from tensorflow import keras
model = keras.models.load_model('models/saved_models/resnet50.h5')

# Unfreeze more layers for fine-tuning
for layer in model.layers[-50:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val)
)
```

---

## Evaluation Examples

### Example 1: Basic Model Evaluation

```python
from evaluate import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(
    model_path='models/saved_models/resnet50.h5',
    model_name='ResNet50'
)

# Load and evaluate
evaluator.load_model()
results, y_pred, y_pred_probs = evaluator.evaluate_model(X_test, y_test)

print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
```

### Example 2: Generate Comprehensive Reports

```python
# Confusion matrix
cm = evaluator.plot_confusion_matrix(
    y_test, y_pred,
    save_path='results/plots/confusion_matrix.png'
)

# Classification report
report = evaluator.generate_classification_report(
    y_test, y_pred,
    save_path='results/metrics/classification_report.csv'
)

# Per-class metrics
metrics_df = evaluator.plot_per_class_metrics(
    y_test, y_pred,
    save_path='results/plots/per_class_metrics.png'
)

# Top-K accuracy
topk = evaluator.plot_top_k_accuracy(
    y_test, y_pred_probs,
    k_values=[1, 3, 5, 10],
    save_path='results/plots/topk_accuracy.png'
)
```

### Example 3: Compare Multiple Models

```python
from evaluate import compare_models

# Define models to compare
model_paths = [
    'models/saved_models/custom_cnn.h5',
    'models/saved_models/resnet50.h5',
    'models/saved_models/efficientnet.h5'
]
model_names = ['Custom CNN', 'ResNet50', 'EfficientNet']

# Compare
comparison_df = compare_models(model_paths, model_names, X_test, y_test)

print("\nModel Comparison:")
print(comparison_df)

# Find best model
best_model = comparison_df['accuracy'].idxmax()
print(f"\nBest Model: {best_model}")
print(f"Accuracy: {comparison_df.loc[best_model, 'accuracy']:.4f}")
```

---

## Inference Examples

### Example 1: Single Image Prediction

```python
from inference import ActionClassifier

# Initialize classifier
classifier = ActionClassifier(
    model_path='models/saved_models/resnet50.h5',
    model_name='ResNet50'
)
classifier.load_model()

# Predict
image_path = 'test_images/sample.jpg'
result = classifier.predict_single(image_path=image_path, top_k=5)

print(f"\nPredicted Action: {result['top_class']}")
print(f"Confidence: {result['top_probability']:.2%}")

print("\nTop-5 Predictions:")
for action, prob in zip(result['top_k_classes'], result['top_k_probabilities']):
    print(f"  {action}: {prob:.2%}")

# Visualize prediction
classifier.visualize_prediction(
    image_path=image_path,
    save_path='results/predictions/sample_prediction.png'
)
```

### Example 2: Batch Prediction

```python
import os

# Get list of test images
test_images_dir = 'test_images/'
image_paths = [os.path.join(test_images_dir, f) 
               for f in os.listdir(test_images_dir) 
               if f.endswith('.jpg')]

# Batch prediction
results = classifier.predict_batch(image_paths=image_paths)

# Print results
for img_path, result in zip(image_paths, results):
    print(f"\n{os.path.basename(img_path)}:")
    print(f"  Action: {result['top_class']}")
    print(f"  Confidence: {result['top_probability']:.2%}")
```

### Example 3: Real-time Webcam Inference

```python
# Start webcam inference
classifier.predict_from_webcam(duration=60, show_visualization=True)
# Press 'q' to quit
```

### Example 4: Create Inference Report

```python
import pandas as pd

# Create comprehensive report for directory
report_df = classifier.create_inference_report(
    test_images_dir='test_images/',
    output_path='results/predictions/inference_report.csv'
)

print("\nInference Report:")
print(report_df)

# Analyze results
confidence_threshold = 0.8
high_confidence = report_df[report_df['confidence'] >= confidence_threshold]
print(f"\nHigh confidence predictions (>={confidence_threshold}): {len(high_confidence)}/{len(report_df)}")
```

---

## Advanced Usage

### Example 1: Custom Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define custom augmentation
custom_augmentation = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Create generator
train_generator = custom_augmentation.flow(
    X_train, y_train,
    batch_size=32
)

# Train with custom augmentation
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=50,
    validation_data=(X_val, y_val)
)
```

### Example 2: Learning Rate Scheduling

```python
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch, lr):
    """Learning rate schedule"""
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    elif epoch < 30:
        return lr * 0.25
    else:
        return lr * 0.1

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

# Train with custom LR schedule
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler]
)
```

### Example 3: Ensemble Predictions

```python
# Load multiple models
models = {
    'resnet50': keras.models.load_model('models/saved_models/resnet50.h5'),
    'efficientnet': keras.models.load_model('models/saved_models/efficientnet.h5'),
    'vgg16': keras.models.load_model('models/saved_models/vgg16.h5')
}

# Get predictions from all models
ensemble_preds = []
for model_name, model in models.items():
    preds = model.predict(X_test, verbose=0)
    ensemble_preds.append(preds)

# Average predictions
ensemble_pred = np.mean(ensemble_preds, axis=0)
ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)

# Evaluate ensemble
from sklearn.metrics import accuracy_score
y_test_classes = np.argmax(y_test, axis=1)
ensemble_accuracy = accuracy_score(y_test_classes, ensemble_pred_classes)
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
```

### Example 4: Error Analysis

```python
# Find misclassified samples
y_test_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)
misclassified_idx = np.where(y_test_classes != y_pred_classes)[0]

print(f"Number of misclassifications: {len(misclassified_idx)}")

# Analyze confusion pairs
from collections import Counter
confusion_pairs = []
for idx in misclassified_idx:
    true_class = config.ACTION_CLASSES[y_test_classes[idx]]
    pred_class = config.ACTION_CLASSES[y_pred_classes[idx]]
    confusion_pairs.append((true_class, pred_class))

most_common = Counter(confusion_pairs).most_common(10)
print("\nMost common confusions:")
for (true, pred), count in most_common:
    print(f"  {true} → {pred}: {count} times")
```

### Example 5: Model Compression

```python
import tensorflow as tf

# Load model
model = keras.models.load_model('models/saved_models/resnet50.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save compressed model
with open('models/saved_models/resnet50.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model compressed and saved!")
```

---

## Tips and Best Practices

1. **Data Preparation**:
   - Always visualize data before training
   - Check for class imbalance
   - Use stratified splitting

2. **Training**:
   - Start with smaller learning rates for transfer learning
   - Use data augmentation to prevent overfitting
   - Monitor validation metrics closely

3. **Evaluation**:
   - Look beyond overall accuracy
   - Analyze per-class performance
   - Identify confusion patterns

4. **Inference**:
   - Batch predictions for efficiency
   - Consider confidence thresholds
   - Implement fallback mechanisms

5. **Production**:
   - Optimize models for deployment
   - Implement proper error handling
   - Monitor inference times

---

For more examples and detailed documentation, see `README.md`.
