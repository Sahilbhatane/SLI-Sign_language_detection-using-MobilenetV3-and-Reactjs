"""
FastAPI Backend Test Script
Tests all endpoints with sample data
"""

import base64
import requests
import json
from pathlib import Path

# API Base URL
BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_root():
    """Test root endpoint"""
    print_section("Testing Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        
        data = response.json()
        print("✓ Root endpoint accessible")
        print(f"  Message: {data.get('message')}")
        print(f"  Version: {data.get('version')}")
        print(f"  Endpoints: {list(data.get('endpoints', {}).keys())}")
        return True
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False


def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print("✓ Health check passed")
        print(f"  Status: {data.get('status')}")
        print(f"  Model Loaded: {data.get('model_loaded')}")
        print(f"  Number of Classes: {data.get('num_classes')}")
        print(f"  Input Shape: {data.get('input_shape')}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_classes():
    """Test get classes endpoint"""
    print_section("Testing Get Classes Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/classes")
        response.raise_for_status()
        
        data = response.json()
        print("✓ Classes retrieved successfully")
        print(f"  Total Classes: {data.get('count')}")
        print(f"  Sample Classes: {data.get('classes', [])[:5]}...")
        return True
    except Exception as e:
        print(f"✗ Get classes failed: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print_section("Testing Model Info Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        response.raise_for_status()
        
        data = response.json()
        model_info = data.get('model_info', {})
        print("✓ Model info retrieved successfully")
        print(f"  Classes: {model_info.get('num_classes')}")
        print(f"  Input Shape: {model_info.get('input_shape')}")
        print(f"  Provider: {model_info.get('providers', ['N/A'])[0]}")
        return True
    except Exception as e:
        print(f"✗ Model info failed: {e}")
        return False


def test_predict_with_image(image_path):
    """Test prediction endpoint with actual image"""
    print_section(f"Testing Prediction with Image: {image_path}")
    
    try:
        # Check if image exists
        if not Path(image_path).exists():
            print(f"✗ Image not found: {image_path}")
            print("  Skipping prediction test with real image")
            return False
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare request
        payload = {
            "image": base64_image,
            "top_k": 5
        }
        
        # Send request
        print("  Sending prediction request...")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        print("✓ Prediction successful")
        print(f"  Top Prediction: {data.get('prediction')}")
        print(f"  Confidence: {data.get('confidence'):.2f}%")
        print(f"  Processing Time: {data.get('processing_time_ms'):.2f}ms")
        
        print("\n  Top 5 Predictions:")
        for pred in data.get('predictions', [])[:5]:
            class_name = pred.get('class')
            confidence = pred.get('confidence_percent', 0)
            rank = pred.get('rank')
            bar_length = int(confidence / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"    {rank}. {class_name:<25} {confidence:6.2f}% {bar}")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        print(f"  Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def test_predict_with_dummy_image():
    """Test prediction endpoint with dummy base64 image"""
    print_section("Testing Prediction with Dummy Image")
    
    try:
        from PIL import Image
        import io
        
        # Create a dummy 224x224 RGB image
        img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Prepare request
        payload = {
            "image": base64_image,
            "top_k": 3
        }
        
        # Send request
        print("  Sending prediction request with dummy image...")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        print("✓ Prediction successful (dummy image)")
        print(f"  Top Prediction: {data.get('prediction')}")
        print(f"  Confidence: {data.get('confidence'):.2f}%")
        print(f"  Processing Time: {data.get('processing_time_ms'):.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction with dummy image failed: {e}")
        return False


def test_invalid_base64():
    """Test prediction endpoint with invalid base64"""
    print_section("Testing Invalid Base64 Handling")
    
    try:
        payload = {
            "image": "not-valid-base64!!!",
            "top_k": 5
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            print("✓ Invalid base64 correctly rejected (400 Bad Request)")
            return True
        else:
            print(f"✗ Expected 400, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("  FASTAPI BACKEND TEST SUITE")
    print("  Sign Language Recognition API")
    print("=" * 70)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the server is running: python backend/main.py")
    
    # Wait for user confirmation
    input("\nPress Enter to start tests...")
    
    results = {}
    
    # Run tests
    results['root'] = test_root()
    results['health'] = test_health()
    results['classes'] = test_classes()
    results['model_info'] = test_model_info()
    results['predict_dummy'] = test_predict_with_dummy_image()
    results['invalid_base64'] = test_invalid_base64()
    
    # Try to test with a real image
    sample_images = [
        "../data/stop/1.png",
        "./data/stop/1.png",
        "data/stop/1.png"
    ]
    
    for img_path in sample_images:
        if Path(img_path).exists():
            results['predict_real'] = test_predict_with_image(img_path)
            break
    else:
        print_section("Real Image Test")
        print("⚠ No sample images found, skipping real image test")
        results['predict_real'] = None
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        print(f"  {test_name:<20} {status}")
    
    print("\n" + "-" * 70)
    print(f"  Total Tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print("-" * 70)
    
    if failed == 0:
        print("\n✓ All tests passed! API is working correctly.")
    else:
        print(f"\n✗ {failed} test(s) failed. Please check the errors above.")
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nTest suite error: {e}")
