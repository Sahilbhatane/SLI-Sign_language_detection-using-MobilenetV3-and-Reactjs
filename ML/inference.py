"""
Model Inference Script
Test the trained ONNX model on individual images
"""

import os
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort
from pathlib import Path

class SignLanguagePredictor:
    """Sign language prediction using ONNX model"""
    
    def __init__(self, model_path="./backend/model_v2.onnx", labels_path="./backend/class_labels.txt"):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to ONNX model
            labels_path: Path to class labels file
        """
        self.model_path = model_path
        self.labels_path = labels_path
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found: {labels_path}")
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Load class labels
        with open(labels_path, 'r') as f:
            self.class_labels = [line.strip() for line in f.readlines()]
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Classes loaded: {len(self.class_labels)} classes")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image file
            target_size: Target image size (width, height)
        
        Returns:
            Preprocessed image array
        """
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        
        # Convert to array and standardize like backend
        img = np.array(img, dtype=np.float32)
        try:
            import tensorflow as tf
            tensor = tf.convert_to_tensor(img)
            tensor = tf.image.per_image_standardization(tensor)
            img_array = tensor.numpy()
        except Exception:
            m = np.mean(img, dtype=np.float32)
            s = np.std(img, dtype=np.float32)
            s = float(max(s, 1.0/np.sqrt(img.size)))
            img_array = (img - m) / s
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path, top_k=3):
        """
        Predict sign language phrase from image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            List of (class_name, confidence) tuples
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: img_array})
        predictions = outputs[0][0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.class_labels[idx]
            confidence = predictions[idx]
            results.append((class_name, confidence))
        
        return results
    
    def predict_and_display(self, image_path, top_k=3):
        """
        Predict and display results
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to display
        """
        print(f"\n{'='*70}")
        print(f"Analyzing: {image_path}")
        print(f"{'='*70}")
        
        try:
            results = self.predict(image_path, top_k)
            
            print(f"\nTop {top_k} Predictions:")
            print(f"{'-'*70}")
            
            for i, (class_name, confidence) in enumerate(results, 1):
                confidence_pct = confidence * 100
                bar_length = int(confidence_pct / 2)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                
                print(f"{i}. {class_name:<25} {confidence_pct:6.2f}% {bar}")
            
            print(f"{'-'*70}")
            print(f"\n✓ Predicted Sign: {results[0][0].upper()}")
            print(f"  Confidence: {results[0][1]*100:.2f}%")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

def test_sample_images(predictor, data_dir="./data", samples_per_class=2):
    """
    Test model on random samples from each class
    
    Args:
        predictor: SignLanguagePredictor instance
        data_dir: Path to data directory
        samples_per_class: Number of samples to test per class
    """
    print(f"\n{'='*70}")
    print("TESTING MODEL ON SAMPLE IMAGES")
    print(f"{'='*70}\n")
    
    data_path = Path(data_dir)
    class_folders = sorted([f for f in data_path.iterdir() if f.is_dir()])
    
    correct = 0
    total = 0
    
    for class_folder in class_folders[:5]:  # Test first 5 classes
        class_name = class_folder.name
        image_files = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg"))
        
        if not image_files:
            continue
        
        # Sample random images
        samples = np.random.choice(image_files, min(samples_per_class, len(image_files)), replace=False)
        
        for img_path in samples:
            results = predictor.predict(str(img_path), top_k=1)
            predicted = results[0][0]
            confidence = results[0][1]
            
            is_correct = predicted == class_name
            if is_correct:
                correct += 1
            total += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} True: {class_name:<20} | Predicted: {predicted:<20} | Conf: {confidence*100:5.1f}%")
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n{'='*70}")
        print(f"Sample Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"{'='*70}\n")

def main():
    """Main inference function"""
    
    print("="*70)
    print("SIGN LANGUAGE RECOGNITION - MODEL INFERENCE")
    print("="*70)
    
    try:
        # Initialize predictor
        predictor = SignLanguagePredictor()
        
        # Check if command line argument provided
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            if os.path.exists(image_path):
                predictor.predict_and_display(image_path, top_k=5)
            else:
                print(f"❌ Image not found: {image_path}")
        else:
            # Run tests on sample images
            print("\nNo image provided. Running sample tests...")
            test_sample_images(predictor)
            
            print("\nUsage for single image prediction:")
            print("  python inference.py <path_to_image>")
            print("\nExample:")
            print("  python inference.py ./data/stop/1.png")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have trained the model first:")
        print("  python ML/train_v2.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
