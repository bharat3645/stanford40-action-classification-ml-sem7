"""
Model Evaluation and Analysis Module
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
import config
from data_loader import Stanford40DataLoader


class ModelEvaluator:
    """
    Class for comprehensive model evaluation
    """
    
    def __init__(self, model_path, model_name='model'):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.loader = Stanford40DataLoader()
        
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded: {self.model_name}")
            return True
        else:
            print(f"Model not found: {self.model_path}")
            return False
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            print("Model not loaded. Call load_model() first.")
            return None
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        print(f"\n{'='*80}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*80}\n")
        
        # Get predictions
        y_pred_probs = self.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Model evaluation
        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        
        results = {
            'model_name': self.model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_loss': test_results[0] if isinstance(test_results, list) else test_results
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Test Loss: {results['test_loss']:.4f}")
        
        return results, y_pred, y_pred_probs
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix
        """
        # Convert to class indices
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=config.ACTION_CLASSES,
                   yticklabels=config.ACTION_CLASSES,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_normalized_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot normalized confusion matrix (percentages)
        """
        # Convert to class indices
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=config.ACTION_CLASSES,
                   yticklabels=config.ACTION_CLASSES,
                   cbar_kws={'label': 'Percentage'})
        plt.title(f'Normalized Confusion Matrix - {self.model_name}', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Normalized confusion matrix saved to {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """
        Generate detailed classification report
        """
        # Convert to class indices
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Generate report
        report = classification_report(
            y_true, y_pred,
            target_names=config.ACTION_CLASSES,
            digits=4,
            output_dict=True
        )
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        print(f"\nClassification Report - {self.model_name}")
        print("="*80)
        print(report_df)
        
        if save_path:
            report_df.to_csv(save_path)
            print(f"\nClassification report saved to {save_path}")
        
        return report_df
    
    def plot_per_class_metrics(self, y_true, y_pred, save_path=None):
        """
        Plot per-class precision, recall, and F1-score
        """
        # Convert to class indices
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(config.NUM_CLASSES)
        )
        
        # Create DataFrame
        metrics_df = pd.DataFrame({
            'Action': config.ACTION_CLASSES,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Sort by F1-score
        metrics_df = metrics_df.sort_values('F1-Score', ascending=True)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(20, 12))
        
        # Precision
        axes[0].barh(range(len(metrics_df)), metrics_df['Precision'])
        axes[0].set_yticks(range(len(metrics_df)))
        axes[0].set_yticklabels(metrics_df['Action'], fontsize=8)
        axes[0].set_xlabel('Precision', fontsize=12)
        axes[0].set_title('Per-Class Precision', fontsize=14)
        axes[0].grid(axis='x', alpha=0.3)
        
        # Recall
        axes[1].barh(range(len(metrics_df)), metrics_df['Recall'], color='orange')
        axes[1].set_yticks(range(len(metrics_df)))
        axes[1].set_yticklabels(metrics_df['Action'], fontsize=8)
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_title('Per-Class Recall', fontsize=14)
        axes[1].grid(axis='x', alpha=0.3)
        
        # F1-Score
        axes[2].barh(range(len(metrics_df)), metrics_df['F1-Score'], color='green')
        axes[2].set_yticks(range(len(metrics_df)))
        axes[2].set_yticklabels(metrics_df['Action'], fontsize=8)
        axes[2].set_xlabel('F1-Score', fontsize=12)
        axes[2].set_title('Per-Class F1-Score', fontsize=14)
        axes[2].grid(axis='x', alpha=0.3)
        
        plt.suptitle(f'Per-Class Metrics - {self.model_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to {save_path}")
        
        plt.show()
        
        return metrics_df
    
    def plot_top_k_accuracy(self, y_true, y_pred_probs, k_values=[1, 3, 5, 10], save_path=None):
        """
        Plot Top-K accuracy
        """
        # Convert to class indices
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        
        top_k_accuracies = []
        
        for k in k_values:
            top_k_pred = np.argsort(y_pred_probs, axis=1)[:, -k:]
            correct = sum([y_true[i] in top_k_pred[i] for i in range(len(y_true))])
            accuracy = correct / len(y_true)
            top_k_accuracies.append(accuracy)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, top_k_accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('K', fontsize=12)
        plt.ylabel('Top-K Accuracy', fontsize=12)
        plt.title(f'Top-K Accuracy - {self.model_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        
        # Add value labels
        for k, acc in zip(k_values, top_k_accuracies):
            plt.text(k, acc, f'{acc:.3f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Top-K accuracy plot saved to {save_path}")
        
        plt.show()
        
        return dict(zip([f'top_{k}' for k in k_values], top_k_accuracies))
    
    def visualize_predictions(self, X_test, y_true, y_pred, num_samples=16, 
                            show_correct=True, save_path=None):
        """
        Visualize sample predictions
        """
        # Convert to class indices
        if len(y_true.shape) > 1:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true
            
        if len(y_pred.shape) > 1:
            y_pred_idx = np.argmax(y_pred, axis=1)
        else:
            y_pred_idx = y_pred
        
        # Select samples
        if show_correct:
            indices = np.where(y_true_idx == y_pred_idx)[0]
            title = "Correct Predictions"
        else:
            indices = np.where(y_true_idx != y_pred_idx)[0]
            title = "Incorrect Predictions"
        
        if len(indices) < num_samples:
            num_samples = len(indices)
        
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        
        # Plot
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, idx in enumerate(selected_indices):
            axes[i].imshow(X_test[idx])
            true_label = config.ACTION_CLASSES[y_true_idx[idx]]
            pred_label = config.ACTION_CLASSES[y_pred_idx[idx]]
            
            if show_correct:
                axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", 
                                fontsize=9, color='green')
            else:
                axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", 
                                fontsize=9, color='red')
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{title} - {self.model_name}', fontsize=16, y=1.0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions visualization saved to {save_path}")
        
        plt.show()


def compare_models(model_paths, model_names, X_test, y_test):
    """
    Compare multiple models
    """
    results = []
    
    for model_path, model_name in zip(model_paths, model_names):
        evaluator = ModelEvaluator(model_path, model_name)
        if evaluator.load_model():
            result, _, _ = evaluator.evaluate_model(X_test, y_test)
            results.append(result)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model_name')
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    comparison_df['accuracy'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Model Accuracy Comparison', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xticklabels(comparison_df.index, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # F1-Score comparison
    comparison_df['f1_score'].plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Model F1-Score Comparison', fontsize=14)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_xticklabels(comparison_df.index, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'models_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comparison
    comparison_path = os.path.join(config.METRICS_DIR, 'models_comparison.csv')
    comparison_df.to_csv(comparison_path)
    print(f"\nComparison saved to {comparison_path}")
    
    return comparison_df


def main():
    """Main evaluation pipeline"""
    # Load test data
    loader = Stanford40DataLoader()
    X, y, _ = loader.load_dataset(use_bbox=False, limit=None)
    _, _, X_test, _, _, y_test = loader.split_dataset(X, y)
    
    # Evaluate single model
    model_name = 'resnet50'
    model_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}.keras")
    
    evaluator = ModelEvaluator(model_path, model_name)
    
    if evaluator.load_model():
        # Evaluate
        results, y_pred, y_pred_probs = evaluator.evaluate_model(X_test, y_test)
        
        # Confusion matrix
        evaluator.plot_confusion_matrix(
            y_test, y_pred,
            save_path=os.path.join(config.PLOTS_DIR, f'{model_name}_confusion_matrix.png')
        )
        
        # Normalized confusion matrix
        evaluator.plot_normalized_confusion_matrix(
            y_test, y_pred,
            save_path=os.path.join(config.PLOTS_DIR, f'{model_name}_normalized_cm.png')
        )
        
        # Classification report
        evaluator.generate_classification_report(
            y_test, y_pred,
            save_path=os.path.join(config.METRICS_DIR, f'{model_name}_classification_report.csv')
        )
        
        # Per-class metrics
        evaluator.plot_per_class_metrics(
            y_test, y_pred,
            save_path=os.path.join(config.PLOTS_DIR, f'{model_name}_per_class_metrics.png')
        )
        
        # Top-K accuracy
        evaluator.plot_top_k_accuracy(
            y_test, y_pred_probs,
            save_path=os.path.join(config.PLOTS_DIR, f'{model_name}_topk_accuracy.png')
        )
        
        # Visualize correct predictions
        evaluator.visualize_predictions(
            X_test, y_test, y_pred, num_samples=16, show_correct=True,
            save_path=os.path.join(config.PLOTS_DIR, f'{model_name}_correct_predictions.png')
        )
        
        # Visualize incorrect predictions
        evaluator.visualize_predictions(
            X_test, y_test, y_pred, num_samples=16, show_correct=False,
            save_path=os.path.join(config.PLOTS_DIR, f'{model_name}_incorrect_predictions.png')
        )


if __name__ == "__main__":
    main()
