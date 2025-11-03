"""
Data Loading and Preprocessing Module for Stanford 40 Actions Dataset
"""

import os
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import config


class Stanford40DataLoader:
    """
    Class to load and preprocess Stanford 40 Actions dataset
    """
    
    def __init__(self, dataset_path=config.DATASET_PATH):
        self.dataset_path = dataset_path
        self.images_path = config.IMAGES_DIR
        self.annotations_path = config.ANNOTATIONS_DIR
        self.image_sets_path = config.SPLITS_DIR
        self.action_classes = config.ACTION_CLASSES
        self.img_size = config.IMG_SIZE
        self.label_encoder = LabelEncoder()
        
    def parse_xml_annotation(self, xml_file):
        """Parse XML annotation file to extract action label and bounding box"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract filename
        filename = root.find('filename').text
        
        # Extract action label - try both root level and object level
        action_elem = root.find('action')
        if action_elem is None:
            action_elem = root.find('.//object/action')
        action = action_elem.text if action_elem is not None else None
        
        # Extract bounding box
        bbox = None
        bndbox = root.find('.//object/bndbox')
        if bndbox is not None:
            bbox = {
                'xmin': int(bndbox.find('xmin').text),
                'ymin': int(bndbox.find('ymin').text),
                'xmax': int(bndbox.find('xmax').text),
                'ymax': int(bndbox.find('ymax').text)
            }
        
        return filename, action, bbox
    
    def load_and_preprocess_image(self, image_path, bbox=None, use_bbox=False):
        """Load and preprocess image with optional bounding box cropping"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop using bounding box if available and requested
        if use_bbox and bbox is not None:
            image = image[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
        
        # Resize to target size
        image = cv2.resize(image, self.img_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype('float32') / 255.0
        
        return image
    
    def load_dataset(self, use_bbox=False, limit=None):
        """
        Load entire dataset with labels
        
        Args:
            use_bbox: Whether to crop images using bounding boxes
            limit: Maximum number of images to load (for testing purposes)
        
        Returns:
            X: Array of images
            y: Array of labels
            metadata: DataFrame with image information
        """
        print("Loading Stanford 40 Actions dataset...")
        
        images = []
        labels = []
        metadata = []
        
        # Get all XML annotation files
        xml_files = [f for f in os.listdir(self.annotations_path) if f.endswith('.xml')]
        
        if limit:
            xml_files = xml_files[:limit]
        
        # Process each annotation file
        for xml_file in tqdm(xml_files, desc="Processing images"):
            xml_path = os.path.join(self.annotations_path, xml_file)
            
            # Parse annotation
            filename, action, bbox = self.parse_xml_annotation(xml_path)
            
            if action is None:
                continue
            
            # Construct image path
            image_path = os.path.join(self.images_path, filename)
            
            if not os.path.exists(image_path):
                continue
            
            # Load and preprocess image
            image = self.load_and_preprocess_image(image_path, bbox, use_bbox)
            
            if image is None:
                continue
            
            images.append(image)
            labels.append(action)
            metadata.append({
                'filename': filename,
                'action': action,
                'bbox': bbox
            })
        
        # Convert to numpy arrays
        X = np.array(images)
        
        # Encode labels
        self.label_encoder.fit(self.action_classes)
        y_encoded = self.label_encoder.transform(labels)
        y = to_categorical(y_encoded, num_classes=config.NUM_CLASSES)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata)
        
        print(f"\nDataset loaded successfully!")
        print(f"Total images: {len(X)}")
        print(f"Image shape: {X[0].shape}")
        print(f"Number of classes: {config.NUM_CLASSES}")
        
        return X, y, metadata_df
    
    def split_dataset(self, X, y, test_size=config.TEST_RATIO, val_size=config.VAL_RATIO, 
                     random_state=config.RANDOM_SEED):
        """
        Split dataset into train, validation, and test sets
        """
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=np.argmax(y, axis=1)
        )
        
        # Second split: separate validation set from training set
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, stratify=np.argmax(y_train_val, axis=1)
        )
        
        print(f"\nDataset split:")
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """
        Create data generators with augmentation for training
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=config.AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=config.AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=config.AUGMENTATION_CONFIG['height_shift_range'],
            horizontal_flip=config.AUGMENTATION_CONFIG['horizontal_flip'],
            zoom_range=config.AUGMENTATION_CONFIG['zoom_range'],
            shear_range=config.AUGMENTATION_CONFIG['shear_range'],
            fill_mode=config.AUGMENTATION_CONFIG['fill_mode']
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Fit generators
        train_generator = train_datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=config.BATCH_SIZE)
        
        return train_generator, val_generator
    
    def visualize_samples(self, X, y, num_samples=16, save_path=None):
        """
        Visualize random samples from the dataset
        """
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.ravel()
        
        # Get random indices
        indices = np.random.choice(len(X), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            axes[i].imshow(X[idx])
            label_idx = np.argmax(y[idx])
            label_name = self.label_encoder.inverse_transform([label_idx])[0]
            axes[i].set_title(f"Action: {label_name}", fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample visualization saved to {save_path}")
        
        plt.show()
    
    def get_class_distribution(self, y, save_path=None):
        """
        Get and visualize class distribution
        """
        # Decode labels
        y_decoded = np.argmax(y, axis=1)
        class_names = self.label_encoder.inverse_transform(y_decoded)
        
        # Count occurrences
        unique, counts = np.unique(class_names, return_counts=True)
        
        # Create DataFrame
        dist_df = pd.DataFrame({'Action': unique, 'Count': counts})
        dist_df = dist_df.sort_values('Count', ascending=False)
        
        # Plot
        plt.figure(figsize=(16, 8))
        plt.bar(range(len(dist_df)), dist_df['Count'])
        plt.xticks(range(len(dist_df)), dist_df['Action'], rotation=90)
        plt.xlabel('Action Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution in Stanford 40 Actions Dataset')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()
        
        return dist_df


def main():
    """Test data loader"""
    loader = Stanford40DataLoader()
    
    # Load dataset (limit for testing)
    X, y, metadata = loader.load_dataset(use_bbox=False, limit=100)
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(X, y)
    
    # Visualize samples
    loader.visualize_samples(X_train, y_train, num_samples=16, 
                            save_path=os.path.join(config.PLOTS_DIR, 'sample_images.png'))
    
    # Get class distribution
    dist_df = loader.get_class_distribution(y, 
                                           save_path=os.path.join(config.PLOTS_DIR, 'class_distribution.png'))
    print("\nClass Distribution:")
    print(dist_df)


if __name__ == "__main__":
    main()
