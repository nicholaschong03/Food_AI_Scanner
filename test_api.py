import requests
import json
import base64
from PIL import Image
import io
import numpy as np

# API base URL (change this to your deployed URL)
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint to verify models are loaded"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"Health check passed: {data}")
            return data.get('models_loaded', False)
        else:
            print(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {str(e)}")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a simple colored image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return img_bytes

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_food_detection():
    """Test food detection endpoint"""
    try:
        img_bytes = create_test_image()
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}

        response = requests.post(f"{BASE_URL}/detect-food", files=files)
        print(f"Food detection: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Food detection failed: {e}")
        return False

def test_food_classification():
    """Test food classification endpoint"""
    try:
        img_bytes = create_test_image()
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}

        response = requests.post(f"{BASE_URL}/classify-food", files=files)
        print(f"Food classification: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Food classification failed: {e}")
        return False

def test_combined_analysis():
    """Test combined analysis endpoint"""
    try:
        img_bytes = create_test_image()
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}

        response = requests.post(f"{BASE_URL}/analyze-food", files=files)
        print(f"Combined analysis: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Combined analysis failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Food AI Scanner API...")
    print("=" * 50)

    # Test health endpoint
    print("\n1. Testing health endpoint...")
    health_ok = test_health()

    if health_ok:
        # Test individual endpoints
        print("\n2. Testing food detection...")
        test_food_detection()

        print("\n3. Testing food classification...")
        test_food_classification()

        print("\n4. Testing combined analysis...")
        test_combined_analysis()

        print("\n" + "=" * 50)
        print("API testing completed!")
    else:
        print("Health check failed. Make sure the API is running.")