"""
Configuration file for Stanford 40 Actions Classification
All hyperparameters, paths, and model settings
"""

import os

# ============================================================================
# PATHS
# ============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Dataset path
DATASET_PATH = os.path.join(DATA_DIR, 'Stanford40')
IMAGES_DIR = os.path.join(DATASET_PATH, 'JPEGImages')
ANNOTATIONS_DIR = os.path.join(DATASET_PATH, 'XMLAnnotations')
SPLITS_DIR = os.path.join(DATASET_PATH, 'ImageSplits')

# Model paths
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')
CHECKPOINT_DIR = os.path.join(MODELS_DIR, 'checkpoints')

# Results paths
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, SAVED_MODELS_DIR, 
                  CHECKPOINT_DIR, PLOTS_DIR, METRICS_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Number of action classes
NUM_CLASSES = 40

# Action class names (in alphabetical order)
ACTION_CLASSES = [
    'applauding', 'blowing_bubbles', 'brushing_teeth', 'cleaning_the_floor',
    'climbing', 'cooking', 'cutting_trees', 'cutting_vegetables',
    'drinking', 'feeding_a_horse', 'fishing', 'fixing_a_bike',
    'fixing_a_car', 'gardening', 'holding_an_umbrella', 'jumping',
    'looking_through_a_microscope', 'looking_through_a_telescope',
    'playing_guitar', 'playing_violin', 'pouring_liquid', 'pushing_a_cart',
    'reading', 'phoning', 'riding_a_bike', 'riding_a_horse',
    'rowing_a_boat', 'running', 'shooting_an_arrow', 'smoking',
    'taking_photos', 'texting_message', 'throwing_frisby', 'using_a_computer',
    'walking_the_dog', 'washing_dishes', 'watching_TV', 'waving_hands',
    'writing_on_a_board', 'writing_on_a_book'
]

# Image preprocessing
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Training settings
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Model checkpoint settings
SAVE_BEST_ONLY = True
MONITOR_METRIC = 'val_accuracy'
MODE = 'max'

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

# Augmentation parameters for training
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Model names mapping
MODEL_NAMES = {
    'custom_cnn': 'Custom_CNN',
    'resnet50': 'ResNet50',
    'vgg16': 'VGG16',
    'efficientnet': 'EfficientNetB0',
    'mobilenet': 'MobileNetV2',
    'vit': 'VisionTransformer'
}

# Custom CNN architecture
CUSTOM_CNN_CONFIG = {
    'conv_blocks': [
        {'filters': 32, 'dropout': 0.25},
        {'filters': 64, 'dropout': 0.25},
        {'filters': 128, 'dropout': 0.25},
        {'filters': 256, 'dropout': 0.25}
    ],
    'dense_layers': [512, 256],
    'dense_dropout': [0.5, 0.5]
}

# Transfer learning fine-tuning layers
FINE_TUNE_LAYERS = {
    'resnet50': 30,
    'vgg16': 5,
    'efficientnet': 30,
    'mobilenet': 20
}

# Classification head configuration
CLASSIFICATION_HEAD = {
    'global_pooling': 'avg',  # 'avg' or 'max'
    'dense_units': [512, 256],
    'dropout_rates': [0.5, 0.3],
    'use_batch_norm': True
}

# Vision Transformer configuration
VIT_CONFIG = {
    'patch_size': 16,
    'projection_dim': 256,
    'num_heads': 8,
    'transformer_layers': 4,
    'mlp_head_units': [512, 256]
}

# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================

OPTIMIZER_CONFIG = {
    'type': 'adam',  # 'adam', 'sgd', 'rmsprop'
    'learning_rate': LEARNING_RATE,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-07,
    'clipnorm': 1.0
}

# Loss function
LOSS_FUNCTION = 'categorical_crossentropy'

# Metrics to track
METRICS = ['accuracy']

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Top-K accuracy
TOP_K = [1, 3, 5]

# Confusion matrix settings
CM_FIGSIZE = (20, 18)
CM_CMAP = 'Blues'

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 100
SAVE_DPI = 300

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

# Prediction settings
PREDICTION_BATCH_SIZE = 16
TOP_PREDICTIONS = 5

# Visualization settings for predictions
PREDICTION_FIGSIZE = (12, 8)
SHOW_CONFIDENCE = True

# Webcam settings
WEBCAM_FPS = 30
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# ============================================================================
# LOGGING AND MONITORING
# ============================================================================

# Logging level
LOG_LEVEL = 'INFO'

# TensorBoard settings
USE_TENSORBOARD = True
TENSORBOARD_UPDATE_FREQ = 'epoch'

# Experiment tracking
TRACK_EXPERIMENTS = True
EXPERIMENT_LOG_FILE = os.path.join(RESULTS_DIR, 'experiments.log')

# ============================================================================
# DATASET DOWNLOAD CONFIGURATION
# ============================================================================

# Dataset URL
DATASET_URL = 'http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip'
ANNOTATIONS_URL = 'http://vision.stanford.edu/Datasets/Stanford40_XMLAnnotations.zip'

# Download settings
DOWNLOAD_TIMEOUT = 600  # seconds
EXTRACT_PATH = DATA_DIR

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# GPU settings
USE_GPU = True
GPU_MEMORY_GROWTH = True
MIXED_PRECISION = False  # Use mixed precision training (requires GPU)

# Multi-GPU settings
MULTI_GPU = False
NUM_GPUS = 1

# Number of workers for data loading
NUM_WORKERS = 4
USE_MULTIPROCESSING = False

# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

# Validation settings
VALIDATION_FREQ = 1  # Validate every N epochs
SHUFFLE_VAL = False

# Test settings
TEST_TIME_AUGMENTATION = False  # Use TTA for evaluation
TTA_STEPS = 5

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable features
USE_BOUNDING_BOXES = False  # Use bounding box cropping
USE_DATA_AUGMENTATION = True
USE_MIXED_PRECISION = False
USE_EARLY_STOPPING = True
USE_REDUCE_LR = True
USE_MODEL_CHECKPOINT = True
SAVE_TRAINING_HISTORY = True

# Visualization flags
PLOT_TRAINING_CURVES = True
PLOT_CONFUSION_MATRIX = True
PLOT_PER_CLASS_METRICS = True
SAVE_SAMPLE_PREDICTIONS = True

# ============================================================================
# VERSION INFO
# ============================================================================

CONFIG_VERSION = '1.0.0'
PROJECT_NAME = 'Stanford 40 Actions Classification'
PROJECT_VERSION = '1.0.0'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path(model_name):
    """Get the full path for a saved model"""
    return os.path.join(SAVED_MODELS_DIR, f"{model_name}.keras")

def get_checkpoint_path(model_name):
    """Get the checkpoint path for a model"""
    return os.path.join(CHECKPOINT_DIR, model_name)

def get_plot_path(plot_name):
    """Get the full path for a plot"""
    return os.path.join(PLOTS_DIR, plot_name)

def get_metrics_path(metrics_name):
    """Get the full path for metrics file"""
    return os.path.join(METRICS_DIR, metrics_name)

def print_config():
    """Print current configuration"""
    print("="*80)
    print(f"{PROJECT_NAME} - Configuration")
    print("="*80)
    print(f"\nVersion: {PROJECT_VERSION}")
    print(f"\nDataset Path: {DATASET_PATH}")
    print(f"Models Directory: {SAVED_MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nNumber of Classes: {NUM_CLASSES}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"\nData Split - Train: {TRAIN_RATIO}, Val: {VAL_RATIO}, Test: {TEST_RATIO}")
    print(f"Data Augmentation: {USE_DATA_AUGMENTATION}")
    print(f"Early Stopping: {USE_EARLY_STOPPING}")
    print("="*80)

if __name__ == "__main__":
    print_config()
