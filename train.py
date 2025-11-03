"""
Training Pipeline for Action Classification Models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)

import config
from data_loader import Stanford40DataLoader
from models import ActionClassificationModels


class ModelTrainer:
    """
    Class to handle training of multiple models
    """
    
    def __init__(self, model_name='custom_cnn'):
        self.model_name = model_name
        self.model = None
        self.history = None
        self.model_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}.keras")
        self.history_path = os.path.join(config.METRICS_DIR, f"{model_name}_history.pkl")
        
    def create_callbacks(self):
        """
        Create training callbacks
        """
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint = ModelCheckpoint(
            filepath=self.model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard_dir = os.path.join(config.CHECKPOINT_DIR, 'tensorboard', self.model_name)
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_logger = CSVLogger(
            os.path.join(config.METRICS_DIR, f"{self.model_name}_training.csv")
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train(self, model, X_train, y_train, X_val, y_val, 
             epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
        """
        Train the model
        """
        self.model = model
        
        print(f"\n{'='*80}")
        print(f"Training {self.model_name}")
        print(f"{'='*80}\n")
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        start_time = datetime.now()
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Save history
        with open(self.history_path, 'wb') as f:
            pickle.dump(self.history.history, f)
        
        return self.history
    
    def train_with_generator(self, model, train_generator, val_generator,
                            steps_per_epoch, validation_steps,
                            epochs=config.EPOCHS):
        """
        Train model using data generators
        """
        self.model = model
        
        print(f"\n{'='*80}")
        print(f"Training {self.model_name} with Data Augmentation")
        print(f"{'='*80}\n")
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        start_time = datetime.now()
        
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Save history
        with open(self.history_path, 'wb') as f:
            pickle.dump(self.history.history, f)
        
        return self.history
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
        
        history_dict = self.history.history if hasattr(self.history, 'history') else self.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot accuracy
        axes[0, 0].plot(history_dict['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(history_dict['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title(f'{self.model_name} - Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(history_dict['loss'], label='Train Loss')
        axes[0, 1].plot(history_dict['val_loss'], label='Val Loss')
        axes[0, 1].set_title(f'{self.model_name} - Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot top-5 accuracy
        if 'top_5_accuracy' in history_dict:
            axes[1, 0].plot(history_dict['top_5_accuracy'], label='Train Top-5 Acc')
            axes[1, 0].plot(history_dict['val_top_5_accuracy'], label='Val Top-5 Acc')
            axes[1, 0].set_title(f'{self.model_name} - Top-5 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-5 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot precision and recall
        if 'precision' in history_dict and 'recall' in history_dict:
            axes[1, 1].plot(history_dict['precision'], label='Train Precision')
            axes[1, 1].plot(history_dict['val_precision'], label='Val Precision')
            axes[1, 1].plot(history_dict['recall'], label='Train Recall')
            axes[1, 1].plot(history_dict['val_recall'], label='Val Recall')
            axes[1, 1].set_title(f'{self.model_name} - Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def load_model(self):
        """
        Load trained model
        """
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return self.model
        else:
            print(f"Model file not found: {self.model_path}")
            return None
    
    def load_history(self):
        """
        Load training history
        """
        if os.path.exists(self.history_path):
            with open(self.history_path, 'rb') as f:
                self.history = pickle.load(f)
            print(f"History loaded from {self.history_path}")
            return self.history
        else:
            print(f"History file not found: {self.history_path}")
            return None


def train_all_models(X_train, X_val, X_test, y_train, y_val, y_test, 
                     epochs=config.EPOCHS, use_augmentation=False):
    """
    Train all models and compare results
    """
    model_builder = ActionClassificationModels()
    results = {}
    
    # Models to train
    models_to_train = [
        ('custom_cnn', lambda: model_builder.build_custom_cnn()),
        ('resnet50', lambda: model_builder.build_resnet50(trainable_layers=30)),
        ('vgg16', lambda: model_builder.build_vgg16(trainable_layers=5)),
        ('efficientnet', lambda: model_builder.build_efficientnet(trainable_layers=30)),
        ('mobilenet', lambda: model_builder.build_mobilenet(trainable_layers=20))
    ]
    
    for model_name, model_func in models_to_train:
        print(f"\n{'#'*80}")
        print(f"# Training {model_name.upper()}")
        print(f"{'#'*80}\n")
        
        try:
            # Build and compile model
            model = model_func()
            model = model_builder.compile_model(model)
            
            # Create trainer
            trainer = ModelTrainer(model_name=model_name)
            
            # Train model
            if use_augmentation:
                # Create data generators
                loader = Stanford40DataLoader()
                train_gen, val_gen = loader.create_data_generators(X_train, y_train, X_val, y_val)
                steps_per_epoch = len(X_train) // config.BATCH_SIZE
                validation_steps = len(X_val) // config.BATCH_SIZE
                
                history = trainer.train_with_generator(
                    model, train_gen, val_gen,
                    steps_per_epoch, validation_steps,
                    epochs=epochs
                )
            else:
                history = trainer.train(
                    model, X_train, y_train, X_val, y_val,
                    epochs=epochs
                )
            
            # Plot training history
            plot_path = os.path.join(config.PLOTS_DIR, f"{model_name}_training_history.png")
            trainer.plot_training_history(save_path=plot_path)
            
            # Evaluate on test set
            test_loss, test_acc, test_top5, test_prec, test_rec = model.evaluate(
                X_test, y_test, verbose=0
            )
            
            # Store results
            results[model_name] = {
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'test_top5_accuracy': test_top5,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'best_val_accuracy': max(history.history['val_accuracy']),
                'best_val_loss': min(history.history['val_loss'])
            }
            
            print(f"\n{model_name.upper()} Results:")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test Top-5 Accuracy: {test_top5:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(config.METRICS_DIR, 'all_models_results.csv')
    results_df.to_csv(results_path)
    print(f"\nAll results saved to {results_path}")
    
    return results_df


def main():
    """Main training pipeline"""
    print("="*80)
    print("STANFORD 40 ACTIONS CLASSIFICATION - TRAINING PIPELINE")
    print("="*80)
    
    # Load data
    loader = Stanford40DataLoader()
    X, y, metadata = loader.load_dataset(use_bbox=False, limit=None)  # Remove limit for full dataset
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(X, y)
    
    # Train all models
    results_df = train_all_models(
        X_train, X_val, X_test, y_train, y_val, y_test,
        epochs=30,  # Adjust as needed
        use_augmentation=True
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(results_df)


if __name__ == "__main__":
    main()
