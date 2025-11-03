"""
Main Pipeline Script for Stanford 40 Actions Classification
Complete end-to-end training and evaluation
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

import config
from data_loader import Stanford40DataLoader
from models import ActionClassificationModels
from train import ModelTrainer, train_all_models
from evaluate import ModelEvaluator, compare_models
from inference import ActionClassifier


def setup_environment():
    """Setup environment and check requirements"""
    print("="*80)
    print("Stanford 40 Actions Classification System")
    print("="*80)
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")
        print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        if tf.config.list_physical_devices('GPU'):
            for gpu in tf.config.list_physical_devices('GPU'):
                print(f"  - {gpu}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Check dataset
    if not os.path.exists(config.DATASET_PATH):
        print(f"\nWarning: Dataset not found at {config.DATASET_PATH}")
        print("Please download the dataset using: python download_dataset.py")
        return False
    
    print(f"\nDataset location: {config.DATASET_PATH}")
    print(f"Models will be saved to: {config.SAVED_MODELS_DIR}")
    print(f"Results will be saved to: {config.RESULTS_DIR}")
    
    return True


def load_data(use_bbox=False, limit=None):
    """Load and split dataset"""
    print("\n" + "="*80)
    print("Step 1: Loading Dataset")
    print("="*80)
    
    loader = Stanford40DataLoader()
    
    # Load dataset
    X, y, metadata = loader.load_dataset(use_bbox=use_bbox, limit=limit)
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(X, y)
    
    # Visualize samples
    print("\nGenerating sample visualizations...")
    loader.visualize_samples(
        X_train, y_train, num_samples=16,
        save_path=os.path.join(config.PLOTS_DIR, 'dataset_samples.png')
    )
    
    # Get class distribution
    print("\nAnalyzing class distribution...")
    dist_df = loader.get_class_distribution(
        y,
        save_path=os.path.join(config.PLOTS_DIR, 'class_distribution.png')
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, loader


def train_models(X_train, X_val, X_test, y_train, y_val, y_test, 
                model_list=None, epochs=50, use_augmentation=True):
    """Train specified models"""
    print("\n" + "="*80)
    print("Step 2: Training Models")
    print("="*80)
    
    if model_list is None:
        # Train all models
        print("\nTraining all models...")
        results_df = train_all_models(
            X_train, X_val, X_test, y_train, y_val, y_test,
            epochs=epochs,
            use_augmentation=use_augmentation
        )
    else:
        # Train specific models
        model_builder = ActionClassificationModels()
        results = {}
        
        for model_name in model_list:
            print(f"\n{'#'*80}")
            print(f"# Training {model_name.upper()}")
            print(f"{'#'*80}\n")
            
            try:
                # Build model
                if model_name == 'custom_cnn':
                    model = model_builder.build_custom_cnn()
                elif model_name == 'resnet50':
                    model = model_builder.build_resnet50(trainable_layers=30)
                elif model_name == 'vgg16':
                    model = model_builder.build_vgg16(trainable_layers=5)
                elif model_name == 'efficientnet':
                    model = model_builder.build_efficientnet(trainable_layers=30)
                elif model_name == 'mobilenet':
                    model = model_builder.build_mobilenet(trainable_layers=20)
                else:
                    print(f"Unknown model: {model_name}")
                    continue
                
                # Compile model
                model = model_builder.compile_model(model)
                
                # Train model
                trainer = ModelTrainer(model_name=model_name)
                history = trainer.train(
                    model, X_train, y_train, X_val, y_val,
                    epochs=epochs
                )
                
                # Plot history
                trainer.plot_training_history(
                    save_path=os.path.join(config.PLOTS_DIR, f'{model_name}_training.png')
                )
                
                # Evaluate on test set
                test_results = model.evaluate(X_test, y_test, verbose=0)
                results[model_name] = {
                    'test_accuracy': test_results[1],
                    'test_loss': test_results[0]
                }
                
                print(f"\n{model_name} Test Accuracy: {test_results[1]:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
    
    print("\nTraining completed!")


def evaluate_models(X_test, y_test, model_list=None):
    """Evaluate trained models"""
    print("\n" + "="*80)
    print("Step 3: Evaluating Models")
    print("="*80)
    
    if model_list is None:
        model_list = ['custom_cnn', 'resnet50', 'vgg16', 'efficientnet', 'mobilenet']
    
    # Compare all models
    model_paths = [os.path.join(config.SAVED_MODELS_DIR, f"{name}.keras") for name in model_list]
    model_names = [config.MODEL_NAMES.get(name, name) for name in model_list]
    
    # Filter existing models
    existing_paths = []
    existing_names = []
    for path, name in zip(model_paths, model_names):
        if os.path.exists(path):
            existing_paths.append(path)
            existing_names.append(name)
    
    if not existing_paths:
        print("No trained models found. Please train models first.")
        return
    
    print(f"\nFound {len(existing_paths)} trained models.")
    
    # Compare models
    comparison_df = compare_models(existing_paths, existing_names, X_test, y_test)
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Detailed evaluation for best model
    best_model_idx = comparison_df['accuracy'].idxmax()
    best_model_name = model_list[model_names.index(best_model_idx)]
    
    print(f"\nPerforming detailed evaluation on best model: {best_model_name}")
    
    evaluator = ModelEvaluator(
        model_path=os.path.join(config.SAVED_MODELS_DIR, f"{best_model_name}.keras"),
        model_name=best_model_name
    )
    
    if evaluator.load_model():
        results, y_pred, y_pred_probs = evaluator.evaluate_model(X_test, y_test)
        
        # Generate visualizations
        print("\nGenerating evaluation visualizations...")
        
        # Confusion matrix
        evaluator.plot_confusion_matrix(
            y_test, y_pred,
            save_path=os.path.join(config.PLOTS_DIR, f'{best_model_name}_confusion_matrix.png')
        )
        
        # Normalized confusion matrix
        evaluator.plot_normalized_confusion_matrix(
            y_test, y_pred,
            save_path=os.path.join(config.PLOTS_DIR, f'{best_model_name}_normalized_cm.png')
        )
        
        # Classification report
        evaluator.generate_classification_report(
            y_test, y_pred,
            save_path=os.path.join(config.METRICS_DIR, f'{best_model_name}_classification_report.csv')
        )
        
        # Per-class metrics
        evaluator.plot_per_class_metrics(
            y_test, y_pred,
            save_path=os.path.join(config.PLOTS_DIR, f'{best_model_name}_per_class_metrics.png')
        )
        
        # Top-K accuracy
        evaluator.plot_top_k_accuracy(
            y_test, y_pred_probs,
            save_path=os.path.join(config.PLOTS_DIR, f'{best_model_name}_topk_accuracy.png')
        )
        
        print("\nEvaluation completed!")


def run_inference(model_name='resnet50', image_path=None):
    """Run inference on test images"""
    print("\n" + "="*80)
    print("Step 4: Running Inference")
    print("="*80)
    
    model_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}.keras")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first.")
        return
    
    classifier = ActionClassifier(model_path, model_name)
    
    if not classifier.load_model():
        return
    
    if image_path and os.path.exists(image_path):
        print(f"\nPredicting action for: {image_path}")
        result = classifier.predict_single(image_path=image_path, top_k=5)
        
        print(f"\nTop Prediction: {result['top_class']}")
        print(f"Confidence: {result['top_probability']:.2%}")
        print(f"\nTop-5 Predictions:")
        for action, prob in zip(result['top_k_classes'], result['top_k_probabilities']):
            print(f"  {action}: {prob:.2%}")
        
        # Visualize
        classifier.visualize_prediction(
            image_path=image_path,
            save_path=os.path.join(config.PREDICTIONS_DIR, 'inference_result.png')
        )
    else:
        print("No test image provided. Skipping inference demo.")
    
    print("\nInference system ready!")


def main():
    """Main pipeline"""
    parser = argparse.ArgumentParser(description='Stanford 40 Actions Classification Pipeline')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train', 'evaluate', 'inference'],
                       help='Pipeline mode: full (all steps), train, evaluate, or inference')
    parser.add_argument('--models', type=str, nargs='+',
                       default=None,
                       help='Specific models to train (e.g., resnet50 vgg16)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples (for testing)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--image', type=str, default=None,
                       help='Image path for inference')
    
    args = parser.parse_args()
    
    # Setup
    if not setup_environment():
        return
    
    start_time = datetime.now()
    
    # Full pipeline
    if args.mode in ['full', 'train', 'evaluate']:
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test, loader = load_data(
            use_bbox=False,
            limit=args.limit
        )
        
        # Train models
        if args.mode in ['full', 'train']:
            train_models(
                X_train, X_val, X_test, y_train, y_val, y_test,
                model_list=args.models,
                epochs=args.epochs,
                use_augmentation=not args.no_augmentation
            )
        
        # Evaluate models
        if args.mode in ['full', 'evaluate']:
            evaluate_models(X_test, y_test, model_list=args.models)
    
    # Inference
    if args.mode in ['full', 'inference']:
        model_name = args.models[0] if args.models else 'resnet50'
        run_inference(model_name=model_name, image_path=args.image)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print(f"Pipeline completed in {total_time:.2f} seconds")
    print("="*80)


if __name__ == "__main__":
    main()
