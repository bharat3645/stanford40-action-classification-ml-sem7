"""
Utility Functions for Stanford 40 Actions Classification
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def save_results(results, filepath):
    """Save results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def save_history(history, filepath):
    """Save training history"""
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    print(f"History saved to {filepath}")


def load_history(filepath):
    """Load training history"""
    with open(filepath, 'rb') as f:
        history = pickle.load(f)
    return history


def plot_comparison(metrics_dict, metric_name, save_path=None):
    """
    Plot comparison of a specific metric across models
    
    Args:
        metrics_dict: Dictionary with model names as keys and metric values as values
        metric_name: Name of the metric to plot
        save_path: Path to save the plot
    """
    models = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(models)), values, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def create_summary_table(models_results):
    """
    Create a formatted summary table of model results
    
    Args:
        models_results: Dictionary with model names and their metrics
    
    Returns:
        Formatted string table
    """
    # Header
    table = "\n" + "="*100 + "\n"
    table += "MODEL PERFORMANCE SUMMARY\n"
    table += "="*100 + "\n"
    table += f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Loss':<12}\n"
    table += "-"*100 + "\n"
    
    # Rows
    for model_name, metrics in models_results.items():
        table += f"{model_name:<20} "
        table += f"{metrics.get('accuracy', 0):<12.4f} "
        table += f"{metrics.get('precision', 0):<12.4f} "
        table += f"{metrics.get('recall', 0):<12.4f} "
        table += f"{metrics.get('f1_score', 0):<12.4f} "
        table += f"{metrics.get('test_loss', 0):<12.4f}\n"
    
    table += "="*100 + "\n"
    
    return table


def log_experiment(model_name, config_params, results, log_file='experiments.log'):
    """
    Log experiment details
    
    Args:
        model_name: Name of the model
        config_params: Dictionary of configuration parameters
        results: Dictionary of results
        log_file: Path to log file
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log_entry = f"\n{'='*80}\n"
    log_entry += f"Experiment: {model_name}\n"
    log_entry += f"Timestamp: {timestamp}\n"
    log_entry += f"{'-'*80}\n"
    log_entry += "Configuration:\n"
    for key, value in config_params.items():
        log_entry += f"  {key}: {value}\n"
    log_entry += f"{'-'*80}\n"
    log_entry += "Results:\n"
    for key, value in results.items():
        log_entry += f"  {key}: {value}\n"
    log_entry += f"{'='*80}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)
    
    print(f"Experiment logged to {log_file}")


def get_model_size(model_path):
    """
    Get model file size in MB
    
    Args:
        model_path: Path to model file
    
    Returns:
        Size in MB
    """
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0


def calculate_inference_time(model, sample_input, num_iterations=100):
    """
    Calculate average inference time
    
    Args:
        model: Trained model
        sample_input: Sample input for inference
        num_iterations: Number of iterations to average
    
    Returns:
        Average inference time in milliseconds
    """
    import time
    
    times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = model.predict(sample_input, verbose=0)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


def visualize_model_architecture(model, save_path):
    """
    Visualize model architecture
    
    Args:
        model: Keras model
        save_path: Path to save visualization
    """
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file=save_path, show_shapes=True, show_layer_names=True,
                  rankdir='TB', expand_nested=True, dpi=96)
        print(f"Model architecture saved to {save_path}")
    except Exception as e:
        print(f"Could not visualize model: {e}")


def create_latex_table(results_df, save_path=None):
    """
    Create LaTeX table from results DataFrame
    
    Args:
        results_df: Pandas DataFrame with results
        save_path: Path to save LaTeX code
    
    Returns:
        LaTeX table string
    """
    latex_str = results_df.to_latex(float_format="%.4f", index=True)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table saved to {save_path}")
    
    return latex_str


def check_gpu_availability():
    """Check GPU availability and memory"""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n{len(gpus)} GPU(s) available:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"    Memory growth enabled")
            except:
                pass
        return True
    else:
        print("\nNo GPU available. Using CPU.")
        return False


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print(f"Random seeds set to {seed}")


def estimate_training_time(num_samples, batch_size, epochs, time_per_batch=1.0):
    """
    Estimate training time
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        epochs: Number of epochs
        time_per_batch: Estimated time per batch in seconds
    
    Returns:
        Estimated time in hours
    """
    num_batches = num_samples / batch_size
    total_batches = num_batches * epochs
    total_seconds = total_batches * time_per_batch
    total_hours = total_seconds / 3600
    
    return total_hours


def print_system_info():
    """Print system information"""
    import platform
    import tensorflow as tf
    
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    print(f"Python Version: {platform.python_version()}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {len(gpus)} device(s) available")
        for i, gpu in enumerate(gpus):
            print(f"  - {gpu.name}")
    else:
        print("GPU: Not available")
    
    print("="*80 + "\n")


def main():
    """Test utility functions"""
    print_system_info()
    check_gpu_availability()
    set_random_seeds(42)


if __name__ == "__main__":
    main()
