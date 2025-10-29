"""
ONNX Model Utilities for FastAPI Backend
Helper functions for loading and using the ONNX model
"""

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from typing import List, Tuple, Dict
import io

class ONNXSignLanguageModel:
    """
    Wrapper class for ONNX sign language recognition model
    Optimized for FastAPI backend integration
    """
    
    def __init__(self, model_path: str = "./model_v2.onnx", labels_path: str = "./class_labels.txt"):
        """
        Initialize ONNX model
        
        Args:
            model_path: Path to ONNX model file
            labels_path: Path to class labels file
        """
        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        # Load ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # Try GPU first
        )
        
        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Load class labels
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.class_labels = [line.strip() for line in f.readlines()]
        
        self.num_classes = len(self.class_labels)
        
        print(f"✓ ONNX Model loaded successfully")
        print(f"  - Model: {model_path}")
        print(f"  - Input shape: {self.input_shape}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Device: {self.session.get_providers()[0]}")
    
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess PIL Image for model input
        
        Args:
            image: PIL Image object
            target_size: Target size (width, height)
        
        Returns:
            Preprocessed numpy array
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and standardize like backend
        img = np.array(image, dtype=np.float32)
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
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def preprocess_image_bytes(self, image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image from bytes (useful for FastAPI file uploads)
        
        Args:
            image_bytes: Image data as bytes
            target_size: Target size (width, height)
        
        Returns:
            Preprocessed numpy array
        """
        image = Image.open(io.BytesIO(image_bytes))
        return self.preprocess_image(image, target_size)
    
    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """
        Run model inference
        
        Args:
            image_array: Preprocessed image array
        
        Returns:
            Prediction probabilities (1D array)
        """
        # Run inference
        outputs = self.session.run(None, {self.input_name: image_array})
        
        # Return probabilities (remove batch dimension)
        return outputs[0][0]
    
    def predict_top_k(self, image_array: np.ndarray, k: int = 3) -> List[Dict[str, float]]:
        """
        Get top-k predictions with class names and confidences
        
        Args:
            image_array: Preprocessed image array
            k: Number of top predictions to return
        
        Returns:
            List of dicts with 'class', 'confidence', 'rank'
        """
        # Get predictions
        predictions = self.predict(image_array)
        
        # Get top-k indices
        top_indices = np.argsort(predictions)[-k:][::-1]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'rank': rank,
                'class': self.class_labels[idx],
                'confidence': float(predictions[idx]),
                'confidence_percent': float(predictions[idx] * 100)
            })
        
        return results
    
    def predict_from_pil(self, image: Image.Image, top_k: int = 3) -> List[Dict[str, float]]:
        """
        Full pipeline: PIL Image -> Predictions
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions
        
        Returns:
            List of prediction dicts
        """
        img_array = self.preprocess_image(image)
        return self.predict_top_k(img_array, k=top_k)
    
    def predict_from_bytes(self, image_bytes: bytes, top_k: int = 3) -> List[Dict[str, float]]:
        """
        Full pipeline: Image bytes -> Predictions
        
        Args:
            image_bytes: Image data as bytes
            top_k: Number of top predictions
        
        Returns:
            List of prediction dicts
        """
        img_array = self.preprocess_image_bytes(image_bytes)
        return self.predict_top_k(img_array, k=top_k)
    
    def predict_from_file(self, file_path: str, top_k: int = 3) -> List[Dict[str, float]]:
        """
        Full pipeline: File path -> Predictions
        
        Args:
            file_path: Path to image file
            top_k: Number of top predictions
        
        Returns:
            List of prediction dicts
        """
        image = Image.open(file_path)
        return self.predict_from_pil(image, top_k)
    
    def get_all_classes(self) -> List[str]:
        """Get list of all class labels"""
        return self.class_labels.copy()
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'input_name': self.input_name,
            'classes': self.class_labels,
            'providers': self.session.get_providers()
        }


# Example FastAPI integration
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# Initialize model at startup
model = ONNXSignLanguageModel(
    model_path="./backend/model.onnx",
    labels_path="./backend/class_labels.txt"
)

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    '''Predict sign language phrase from uploaded image'''
    
    # Read image bytes
    image_bytes = await file.read()
    
    # Get predictions
    predictions = model.predict_from_bytes(image_bytes, top_k=5)
    
    return JSONResponse(content={
        'success': True,
        'predictions': predictions,
        'top_prediction': predictions[0]['class'],
        'confidence': predictions[0]['confidence_percent']
    })

@app.get("/classes")
async def get_classes():
    '''Get all available sign language classes'''
    return {
        'classes': model.get_all_classes(),
        'count': model.num_classes
    }

@app.get("/model-info")
async def get_model_info():
    '''Get model information'''
    return model.get_model_info()
"""

if __name__ == "__main__":
    # Test the model
    print("Testing ONNX Model Utilities\n")
    
    try:
        # Initialize model
        model = ONNXSignLanguageModel(
            model_path="./backend/model.onnx",
            labels_path="./backend/class_labels.txt"
        )
        
        # Get model info
        info = model.get_model_info()
        print(f"\nModel Information:")
        print(f"  - Classes: {info['num_classes']}")
        print(f"  - Input Shape: {info['input_shape']}")
        print(f"  - Execution Provider: {info['providers'][0]}")
        
        # Test with a sample image
        import sys
        if len(sys.argv) > 1:
            test_image = sys.argv[1]
            print(f"\nTesting with: {test_image}")
            
            predictions = model.predict_from_file(test_image, top_k=5)
            
            print("\nTop 5 Predictions:")
            for pred in predictions:
                print(f"  {pred['rank']}. {pred['class']:<25} {pred['confidence_percent']:6.2f}%")
        else:
            print("\nModel loaded successfully!")
            print("Run with an image path to test: python onnx_utils.py <image_path>")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train the model first: python train.py")
    except Exception as e:
        print(f"Error: {e}")
