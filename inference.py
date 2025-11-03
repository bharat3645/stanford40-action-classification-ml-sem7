"""
Inference Module for Real-time Action Classification
"""

import os
import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import config
from data_loader import Stanford40DataLoader


class ActionClassifier:
    """
    Class for real-time action classification inference
    """
    
    def __init__(self, model_path, model_name='action_classifier'):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.loader = Stanford40DataLoader()
        self.img_size = config.IMG_SIZE
        
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded successfully: {self.model_name}")
            return True
        else:
            print(f"Model not found: {self.model_path}")
            return False
    
    def preprocess_image(self, image_path=None, image_array=None):
        """
        Preprocess single image for inference
        
        Args:
            image_path: Path to image file
            image_array: Numpy array of image (alternative to path)
        
        Returns:
            Preprocessed image ready for model input
        """
        if image_path is not None:
            # Load from file
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_array is not None:
            # Use provided array
            image = image_array.copy()
        else:
            print("Either image_path or image_array must be provided")
            return None
        
        # Resize
        image = cv2.resize(image, self.img_size)
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_single(self, image_path=None, image_array=None, top_k=5):
        """
        Predict action for a single image
        
        Args:
            image_path: Path to image file
            image_array: Numpy array of image
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        # Preprocess image
        image = self.preprocess_image(image_path, image_array)
        if image is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_probabilities = predictions[top_indices]
        top_classes = [config.ACTION_CLASSES[i] for i in top_indices]
        
        results = {
            'top_class': top_classes[0],
            'top_probability': float(top_probabilities[0]),
            'top_k_classes': top_classes,
            'top_k_probabilities': [float(p) for p in top_probabilities],
            'all_probabilities': predictions.tolist()
        }
        
        return results
    
    def predict_batch(self, image_paths=None, image_arrays=None):
        """
        Predict actions for multiple images
        
        Args:
            image_paths: List of image paths
            image_arrays: List of image arrays
        
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        results = []
        
        if image_paths is not None:
            for img_path in image_paths:
                result = self.predict_single(image_path=img_path)
                results.append(result)
        
        elif image_arrays is not None:
            for img_array in image_arrays:
                result = self.predict_single(image_array=img_array)
                results.append(result)
        
        return results
    
    def visualize_prediction(self, image_path=None, image_array=None, 
                           save_path=None, top_k=5):
        """
        Visualize prediction with image and top-k results
        """
        # Load original image
        if image_path is not None:
            orig_image = cv2.imread(image_path)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        elif image_array is not None:
            orig_image = image_array.copy()
        else:
            print("Either image_path or image_array must be provided")
            return
        
        # Get prediction
        result = self.predict_single(image_path, image_array, top_k)
        
        if result is None:
            return
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display image
        axes[0].imshow(orig_image)
        axes[0].set_title(f"Predicted: {result['top_class']}\n"
                         f"Confidence: {result['top_probability']:.2%}",
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Display top-k predictions
        y_pos = np.arange(len(result['top_k_classes']))
        axes[1].barh(y_pos, result['top_k_probabilities'], color='skyblue')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(result['top_k_classes'])
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_title(f'Top-{top_k} Predictions', fontsize=14)
        axes[1].set_xlim(0, 1)
        
        # Add value labels
        for i, v in enumerate(result['top_k_probabilities']):
            axes[1].text(v + 0.01, i, f'{v:.2%}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def predict_from_webcam(self, duration=30, show_visualization=True):
        """
        Real-time prediction from webcam
        
        Args:
            duration: Duration in seconds to run
            show_visualization: Whether to show live visualization
        """
        if self.model is None:
            if not self.load_model():
                return
        
        print("Starting webcam... Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        frame_count = 0
        prediction_interval = 10  # Predict every N frames
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Make prediction periodically
                if frame_count % prediction_interval == 0:
                    result = self.predict_single(image_array=frame_rgb)
                    
                    if result is not None:
                        action = result['top_class']
                        confidence = result['top_probability']
                        
                        # Display on frame
                        text = f"{action}: {confidence:.2%}"
                        cv2.putText(frame, text, (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                  (0, 255, 0), 3)
                
                if show_visualization:
                    cv2.imshow('Action Classification', frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam closed")
    
    def create_inference_report(self, test_images_dir, output_path):
        """
        Create inference report for a directory of test images
        """
        if self.model is None:
            if not self.load_model():
                return
        
        print(f"Processing images from {test_images_dir}")
        
        # Get all image files
        image_files = [f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        results = []
        
        for img_file in image_files:
            img_path = os.path.join(test_images_dir, img_file)
            result = self.predict_single(image_path=img_path)
            
            if result is not None:
                results.append({
                    'filename': img_file,
                    'predicted_action': result['top_class'],
                    'confidence': result['top_probability']
                })
        
        # Save report
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Inference report saved to {output_path}")
        
        return df


def main():
    """Test inference module"""
    
    # Load model
    model_name = 'resnet50'
    model_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}.keras")
    
    classifier = ActionClassifier(model_path, model_name)
    
    if not classifier.load_model():
        print("Failed to load model. Please train the model first.")
        return
    
    # Test with sample image
    print("\nTesting with sample image...")
    
    # You can replace this with your own test image
    test_image_path = "path/to/your/test/image.jpg"
    
    if os.path.exists(test_image_path):
        # Single prediction
        result = classifier.predict_single(image_path=test_image_path, top_k=5)
        
        print(f"\nPrediction Results:")
        print(f"Top Action: {result['top_class']}")
        print(f"Confidence: {result['top_probability']:.2%}")
        print(f"\nTop-5 Predictions:")
        for action, prob in zip(result['top_k_classes'], result['top_k_probabilities']):
            print(f"  {action}: {prob:.2%}")
        
        # Visualize
        classifier.visualize_prediction(
            image_path=test_image_path,
            save_path=os.path.join(config.PREDICTIONS_DIR, 'sample_prediction.png')
        )
    else:
        print(f"Test image not found: {test_image_path}")
    
    # Uncomment to test webcam inference
    # print("\nStarting webcam inference...")
    # classifier.predict_from_webcam(duration=30)


if __name__ == "__main__":
    main()
