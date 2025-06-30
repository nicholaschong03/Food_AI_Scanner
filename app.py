import os
import io
import json
import base64
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import gc
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Constants
FOOD_CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5

# Memory optimization for free tier
def optimize_memory():
    """Optimize memory usage for free tier"""
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    # Force garbage collection
    gc.collect()

# Get configuration from environment variables
def get_config():
    """Get configuration from environment variables with defaults"""
    return {
        'port': int(os.getenv('PORT', 8000)),
        'host': os.getenv('HOST', '0.0.0.0'),
        'classification_threshold': float(os.getenv('CLASSIFICATION_THRESHOLD', 0.5)),
        'segmentation_threshold': float(os.getenv('SEGMENTATION_THRESHOLD', 0.5)),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'max_image_size': int(os.getenv('MAX_IMAGE_SIZE', 10485760)),  # 10MB default
        'image_timeout': int(os.getenv('IMAGE_TIMEOUT', 30)),
        'model_device': os.getenv('MODEL_DEVICE', 'cpu'),
        'plan': os.getenv('PLAN', 'free_tier'),
        'region': os.getenv('REGION', 'singapore')
    }

# Configure logging
config = get_config()
logging.basicConfig(level=getattr(logging, config['log_level']))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Food AI Scanner API",
    description="ML inference service for food detection, classification, and ingredient segmentation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),  # Use env var or default to '*'
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
food_nonfood_model = None
food_classifier_model = None
segmentation_model = None
food_labels = None
gemini_model = None

# Lazy loading functions
def load_food_nonfood_model():
    """Lazy load food/non-food classifier"""
    global food_nonfood_model
    if food_nonfood_model is None:
        logger.info("Loading food/non-food classifier...")
        optimize_memory()
        food_nonfood_model = keras.models.load_model('models/food_nonfood_mnv2.h5', compile=False)
        optimize_memory()
    return food_nonfood_model

def load_food_classifier_model():
    """Lazy load food dish classifier"""
    global food_classifier_model
    if food_classifier_model is None:
        logger.info("Loading food dish classifier...")
        optimize_memory()
        food_classifier_model = keras.models.load_model('models/food_dish_classifier_efficientnetb2_combined_latest.h5', compile=False)
        optimize_memory()
    return food_classifier_model

def load_segmentation_model():
    """Lazy load YOLO segmentation model"""
    global segmentation_model
    if segmentation_model is None:
        logger.info("Loading YOLO segmentation model...")
        optimize_memory()
        segmentation_model = YOLO('models/best_ingredient_seg.pt')
        optimize_memory()
    return segmentation_model

def load_food_labels():
    """Lazy load food labels"""
    global food_labels
    if food_labels is None:
        logger.info("Loading food labels...")
        with open('models/labels_combined.json', 'r') as f:
            food_labels = json.load(f)
    return food_labels

def load_gemini_model():
    """Lazy load Gemini model"""
    global gemini_model
    if gemini_model is None:
        logger.info("Setting up Gemini API...")
        setup_gemini()
    return gemini_model

# Nutritionix API configuration
def setup_nutritionix():
    """Setup Nutritionix API if credentials are available"""
    app_id = os.getenv('NUTRITIONIX_APP_ID')
    app_key = os.getenv('NUTRITIONIX_APP_KEY')

    if app_id and app_key:
        logger.info("Nutritionix API configured successfully")
        return True
    else:
        logger.warning("NUTRITIONIX_APP_ID or NUTRITIONIX_APP_KEY not found. Nutrition features will be disabled.")
        return False

def get_nutrition_data(ingredient_name: str) -> Dict[str, Any]:
    """Get nutrition data for an ingredient using Nutritionix API"""
    app_id = os.getenv('NUTRITIONIX_APP_ID')
    app_key = os.getenv('NUTRITIONIX_APP_KEY')

    if not app_id or not app_key:
        return {"error": "Nutritionix API not configured"}

    try:
        url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
        headers = {
            'Content-Type': 'application/json',
            'x-app-id': app_id,
            'x-app-key': app_key
        }
        data = {
            "query": ingredient_name
        }

        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()

        result = response.json()

        if result.get('foods') and len(result['foods']) > 0:
            food = result['foods'][0]

            # Extract main nutrition values
            nutrition_data = {
                "food_name": food.get('food_name', ingredient_name),
                "serving_qty": food.get('serving_qty', 1),
                "serving_unit": food.get('serving_unit', 'serving'),
                "serving_weight_grams": round(food.get('serving_weight_grams', 0), 2),

                # Main macros
                "calories": round(food.get('nf_calories', 0), 2),
                "protein": round(food.get('nf_protein', 0), 2),
                "total_fat": round(food.get('nf_total_fat', 0), 2),
                "saturated_fat": round(food.get('nf_saturated_fat', 0), 2),
                "total_carbohydrate": round(food.get('nf_total_carbohydrate', 0), 2),
                "dietary_fiber": round(food.get('nf_dietary_fiber', 0), 2),
                "sugars": round(food.get('nf_sugars', 0), 2),
                "cholesterol": round(food.get('nf_cholesterol', 0), 2),
                "sodium": round(food.get('nf_sodium', 0), 2),
                "potassium": round(food.get('nf_potassium', 0), 2),

                # Additional nutrients
                "vitamin_c": None,
                "vitamin_a": None,
                "vitamin_d": None,
                "vitamin_e": None,
                "vitamin_k": None,
                "calcium": None,
                "iron": None,
                "magnesium": None,
                "zinc": None,
                "phosphorus": None,

                "source": "nutritionix",
                "success": True
            }

            # Extract additional nutrients from full_nutrients array
            if food.get('full_nutrients'):
                for nutrient in food['full_nutrients']:
                    attr_id = nutrient.get('attr_id')
                    value = nutrient.get('value', 0)

                    # Map Nutritionix attribute IDs to nutrient names
                    if attr_id == 401:  # Vitamin C
                        nutrition_data["vitamin_c"] = round(value, 2)
                    elif attr_id == 320:  # Vitamin A
                        nutrition_data["vitamin_a"] = round(value, 2)
                    elif attr_id == 328:  # Vitamin D
                        nutrition_data["vitamin_d"] = round(value, 2)
                    elif attr_id == 323:  # Vitamin E
                        nutrition_data["vitamin_e"] = round(value, 2)
                    elif attr_id == 430:  # Vitamin K
                        nutrition_data["vitamin_k"] = round(value, 2)
                    elif attr_id == 301:  # Calcium
                        nutrition_data["calcium"] = round(value, 2)
                    elif attr_id == 303:  # Iron
                        nutrition_data["iron"] = round(value, 2)
                    elif attr_id == 304:  # Magnesium
                        nutrition_data["magnesium"] = round(value, 2)
                    elif attr_id == 309:  # Zinc
                        nutrition_data["zinc"] = round(value, 2)
                    elif attr_id == 305:  # Phosphorus
                        nutrition_data["phosphorus"] = round(value, 2)

            return nutrition_data
        else:
            return {
                "food_name": ingredient_name,
                "error": "No nutrition data found",
                "success": False
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Nutritionix API: {str(e)}")
        return {
            "food_name": ingredient_name,
            "error": f"API request failed: {str(e)}",
            "success": False
        }
    except Exception as e:
        logger.error(f"Error processing Nutritionix response: {str(e)}")
        return {
            "food_name": ingredient_name,
            "error": f"Processing error: {str(e)}",
            "success": False
        }

def calculate_total_nutrition(ingredients_nutrition: List[Dict]) -> Dict[str, Any]:
    """Calculate total nutrition from a list of ingredient nutrition data"""
    try:
        total_nutrition = {
            "total_calories": 0.0,
            "total_protein": 0.0,
            "total_fat": 0.0,
            "total_saturated_fat": 0.0,
            "total_carbohydrate": 0.0,
            "total_fiber": 0.0,
            "total_sugars": 0.0,
            "total_cholesterol": 0.0,
            "total_sodium": 0.0,
            "total_potassium": 0.0,
            "total_vitamin_c": 0.0,
            "total_vitamin_a": 0.0,
            "total_calcium": 0.0,
            "total_iron": 0.0,
            "ingredients_count": len(ingredients_nutrition),
            "successful_lookups": 0
        }

        for nutrition in ingredients_nutrition:
            if nutrition.get("success", False):
                total_nutrition["successful_lookups"] += 1

                # Add all nutrition values, handling None values
                for key in total_nutrition:
                    if key in ["ingredients_count", "successful_lookups"]:
                        continue

                    value = nutrition.get(key.replace("total_", ""), 0)
                    if value is not None:
                        total_nutrition[key] += value

        # Round all numeric values to 2 decimal places
        for key in total_nutrition:
            if key not in ["ingredients_count", "successful_lookups"]:
                total_nutrition[key] = round(total_nutrition[key], 2)

        return total_nutrition

    except Exception as e:
        logger.error(f"Error calculating total nutrition: {str(e)}")
        return {
            "total_calories": 0.0,
            "total_protein": 0.0,
            "total_fat": 0.0,
            "total_saturated_fat": 0.0,
            "total_carbohydrate": 0.0,
            "total_fiber": 0.0,
            "total_sugars": 0.0,
            "total_cholesterol": 0.0,
            "total_sodium": 0.0,
            "total_potassium": 0.0,
            "total_vitamin_c": 0.0,
            "total_vitamin_a": 0.0,
            "total_calcium": 0.0,
            "total_iron": 0.0,
            "ingredients_count": 0,
            "successful_lookups": 0
        }

# YOLO segmentation class names (from your Colab)
segmentation_class_names = [
    "background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding",
    "ice cream", "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond",
    "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot",
    "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon",
    "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck",
    "sausage", "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn",
    "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu",
    "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger",
    "okra", "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli",
    "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans",
    "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom",
    "salad", "other ingredients"
]

# Gemini API configuration
def setup_gemini():
    """Setup Gemini API if API key is available"""
    global gemini_model
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Gemini API configured successfully")
        else:
            logger.warning("GEMINI_API_KEY not found. Gemini features will be disabled.")
    except Exception as e:
        logger.error(f"Error setting up Gemini: {str(e)}")

def analyze_ingredients_with_gemini(image: Image.Image) -> Dict[str, Any]:
    if not gemini_model:
        return {"ingredients": [], "confidence": "low", "source": "gemini_disabled"}
    try:
        prompt = """
        Analyze this food image and identify all visible ingredients.
        Please provide ONLY a JSON response with the following format (no markdown, no explanation, no extra text):
        {
            "ingredients": [
                {"name": "ingredient_name", "confidence": "high/medium/low", "description": "brief description"}
            ],
            "dish_type": "type of dish",
            "cooking_method": "how it's prepared if visible"
        }
        Do not include any explanation, markdown, or extra text. Only output the JSON block.
        If you're unsure about an ingredient, mark confidence as 'low'.
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        response = gemini_model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
        response_text = response.text
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text
        result = json.loads(json_str)
        result["source"] = "gemini"
        return result
    except json.JSONDecodeError:
        # Fallback: extract ingredients from text response
        ingredients = []
        lines = response.text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['ingredient', 'contains', 'made with']):
                ingredients.append(line.strip())
        return {
            "ingredients": ingredients,
            "confidence": "medium",
            "source": "gemini_text_fallback"
        }
    except Exception as e:
        logger.error(f"Error with Gemini API: {str(e)}")
        return {"ingredients": [], "confidence": "low", "source": "gemini_error"}

def combine_ingredient_results(yolo_ingredients: List[str], yolo_confidences: List[Dict], gemini_result: Dict) -> Dict[str, Any]:
    """Combine YOLO and Gemini ingredient detection results"""
    combined_ingredients = []

    # Add YOLO detected ingredients
    for i, ingredient in enumerate(yolo_ingredients):
        if ingredient.lower() != "background":
            confidence_value = list(yolo_confidences[i].values())[0] if i < len(yolo_confidences) else 0.5
            combined_ingredients.append({
                "name": ingredient,
                "confidence": "high" if confidence_value > 0.7 else "medium" if confidence_value > 0.5 else "low",
                "description": f"Detected by YOLO model with {confidence_value:.2f} confidence",
                "source": "yolo"
            })

    # Add Gemini detected ingredients (avoid duplicates)
    if gemini_result and "ingredients" in gemini_result:
        existing_names = {ing["name"].lower() for ing in combined_ingredients}
        for ingredient in gemini_result["ingredients"]:
            if isinstance(ingredient, dict) and "name" in ingredient:
                name = ingredient["name"]
                if name.lower() not in existing_names and name.lower() != "background":
                    combined_ingredients.append({
                        "name": name,
                        "confidence": ingredient.get("confidence", "medium"),
                        "description": ingredient.get("description", "Detected by Gemini Vision"),
                        "source": "gemini"
                    })
                    existing_names.add(name.lower())

    # Create combined analysis
    combined_analysis = {
        "ingredients": combined_ingredients,
        "dish_type": gemini_result.get("dish_type", "Unknown") if gemini_result else "Unknown",
        "cooking_method": gemini_result.get("cooking_method", "Not visible") if gemini_result else "Not visible",
        "source": "combined"
    }

    return combined_analysis

# Image preprocessing functions
def preprocess_image_for_classification(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess image for classification models"""
    # Resize image
    image = image.resize(target_size)
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_for_food_classification(image: Image.Image, img_size: int = 224) -> np.ndarray:
    """Preprocess image for food classification (matches Google Colab implementation)"""
    # Resize image to IMG_SIZE
    img = image.convert('RGB').resize((img_size, img_size))
    # Convert to numpy array with float32
    arr = np.array(img, dtype=np.float32)
    # Apply EfficientNet preprocessing (same as Colab)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    # Add batch dimension
    return arr[None, ...]

def preprocess_image_for_detection(image: Image.Image, img_size: int = 224) -> np.ndarray:
    """Preprocess image for food detection (matches Google Colab implementation)"""
    # Use ImageOps.fit for consistent cropping like in Colab
    img_cropped = ImageOps.fit(image, (img_size, img_size))
    # Convert to numpy array and normalize to float32(0-1)
    arr = np.asarray(img_cropped, dtype=np.float32) / 255.0
    # Add batch dimension
    return arr[None, ...]  # shape (1,H,W,3)

def preprocess_image_for_segmentation(image: Image.Image, target_size: tuple = (512, 512)) -> torch.Tensor:
    """Preprocess image for segmentation model"""
    # Resize image
    image = image.resize(target_size)
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def process_yolo_segmentation(image: Image.Image, results, confidence_threshold: float = None):
    """Process YOLO segmentation results (matches Google Colab implementation)"""
    # Use environment variable if not provided
    if confidence_threshold is None:
        confidence_threshold = config['segmentation_threshold']

    try:
        # Get masks, boxes, class predictions, and confidence scores
        masks = results[0].masks.data.cpu().numpy()  # [num_masks, H, W]
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [num_masks, 4]
        cls = results[0].boxes.cls.cpu().numpy().astype(int)  # [num_masks]
        conf = results[0].boxes.conf.cpu().numpy()  # [num_masks] - confidence scores

        # Get detected ingredients with confidence filtering
        ingredients_detected = []
        ingredient_confidences = []
        high_confidence_masks = []
        high_confidence_boxes = []
        high_confidence_cls = []

        for i, (class_id, confidence) in enumerate(zip(cls, conf)):
            if confidence >= confidence_threshold and class_id < len(segmentation_class_names):
                ingredient_name = segmentation_class_names[class_id]
                if ingredient_name not in ingredients_detected and ingredient_name != "background":
                    ingredients_detected.append(ingredient_name)
                    ingredient_confidences.append({ingredient_name: float(confidence)})
                    # Keep only high confidence detections for visualization
                    high_confidence_masks.append(masks[i])
                    high_confidence_boxes.append(boxes[i])
                    high_confidence_cls.append(class_id)

        # Create segmentation visualization with only high confidence detections
        mask_h, mask_w = masks.shape[1:] if len(masks) > 0 else (image.size[1], image.size[0])
        img_np = np.array(image.resize((mask_w, mask_h)))

        # Create colored mask overlay
        import random
        colored_mask = np.zeros_like(img_np, dtype=np.float32)

        for i, mask in enumerate(high_confidence_masks):
            class_id = int(high_confidence_cls[i])
            if class_id < len(segmentation_class_names):
                # Generate consistent color for each class
                random.seed(class_id)
                color = [random.random(), random.random(), random.random()]

                mask_bool = mask > 0.5
                for c in range(3):
                    colored_mask[..., c] += mask_bool * color[c]

        # Convert to PIL image
        colored_mask = np.clip(colored_mask, 0, 1)
        mask_image = Image.fromarray((colored_mask * 255).astype(np.uint8))
        mask_image = mask_image.resize(image.size)

        return mask_image, ingredients_detected, ingredient_confidences

    except Exception as e:
        logger.error(f"Error processing YOLO segmentation: {str(e)}")
        # Return empty results if processing fails
        return image, ["segmentation_failed"], []

def classify_food_dish(image: Image.Image):
    """Classify the food dish using the loaded model and return a FoodClassificationResponse."""
    processed_image = preprocess_image_for_food_classification(image)
    model = load_food_classifier_model()
    labels = load_food_labels()
    predictions = model.predict(processed_image, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [
        {"dish_name": labels[idx], "confidence": float(predictions[idx])}
        for idx in top_indices
    ]
    best_idx = top_indices[0]
    dish_name = labels[best_idx]
    confidence = float(predictions[best_idx])
    return FoodClassificationResponse(
        dish_name=dish_name,
        confidence=confidence,
        top_predictions=top_predictions
    )

def gemini_confidence_to_float(conf):
    mapping = {"high": 0.95, "medium": 0.7, "low": 0.3}
    return mapping.get(str(conf).lower(), 0.0)

def classify_dish_with_gemini(image: Image.Image) -> dict:
    model = load_gemini_model()
    if not model:
        return {"dish_name": "unknown", "confidence": "low", "source": "gemini_disabled"}
    try:
        prompt = """
        You are a food recognition expert. Analyze this food image and identify the most likely dish name (e.g., 'chicken chop', 'laksa', 'nasi lemak', etc).
        Please provide ONLY a JSON response in the following format (no markdown, no explanation, no extra text):
        {
            "dish_name": "the most likely dish name",
            "confidence": "high/medium/low",
            "description": "brief description of the dish"
        }
        Do not include any explanation, markdown, or extra text. Only output the JSON block.
        If you are unsure, set confidence to 'low'.
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
        response_text = response.text
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            json_str = response_text[json_start:json_end].strip()
        elif '```' in response_text:
            json_start = response_text.find('```') + 3
            json_end = response_text.find('```', json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)
        result["source"] = "gemini"
        return result
    except Exception as e:
        logger.error(f"Error in Gemini classification: {str(e)}")
        return {"dish_name": "unknown", "confidence": "low", "source": "gemini_error", "error": str(e)}

@app.post("/classify-food", response_model=FoodClassificationResponse)
async def classify_food(file: UploadFile = File(...)):
    """Classify the food dish in the uploaded image"""
    try:
        # Read and validate image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Check file size
        image_data = await file.read()
        if len(image_data) > config['max_image_size']:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {config['max_image_size']} bytes"
            )

        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Try both classification methods
        try:
            # First try with the trained model
            result = classify_food_dish(image)
            return result
        except Exception as model_error:
            logger.warning(f"Model classification failed: {str(model_error)}")
            # Fallback to Gemini
            gemini_result = classify_dish_with_gemini(image)
            return FoodClassificationResponse(
                dish_name=gemini_result.get('dish_name', 'unknown'),
                confidence=gemini_confidence_to_float(gemini_result.get('confidence', 'low')),
                top_predictions=[{
                    "dish_name": gemini_result.get('dish_name', 'unknown'),
                    "confidence": gemini_confidence_to_float(gemini_result.get('confidence', 'low')),
                    "source": gemini_result.get('source', 'unknown')
                }]
            )

    except Exception as e:
        logger.error(f"Error in food classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/segment", response_model=SegmentationResponse)
async def segment_ingredients(file: UploadFile = File(...)):
    """Segment ingredients from the uploaded food image"""
    try:
        # Read and validate image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Check file size
        image_data = await file.read()
        if len(image_data) > config['max_image_size']:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {config['max_image_size']} bytes"
            )

        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Process segmentation with lazy loading
        try:
            # Run YOLO segmentation
            results = load_segmentation_model()(np.array(image), verbose=False)

            # Process results
            ingredients_detected = []
            ingredient_confidences = []

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        if box.conf is not None and box.cls is not None:
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])

                            # Get class name from the model
                            class_name = result.names[class_id] if class_id in result.names else f"class_{class_id}"

                            # Filter out background class
                            if class_name.lower() != "background":
                                ingredients_detected.append(class_name)
                                ingredient_confidences.append({class_name: confidence})

            # If YOLO didn't detect anything, try Gemini
            if not ingredients_detected:
                logger.info("No ingredients detected by YOLO, trying Gemini...")
                gemini_result = analyze_ingredients_with_gemini(image)
                if gemini_result.get('ingredients'):
                    ingredients_detected = gemini_result['ingredients']
                    ingredient_confidences = [{"gemini_detected": 0.7} for _ in ingredients_detected]

            # Get nutrition data
            nutrition_data = None
            health_analysis = None
            if ingredients_detected:
                nutrition_data = analyze_nutrition(ingredients_detected)
                if nutrition_data.get('total_nutrition'):
                    health_analysis = calculate_health_metrics(nutrition_data['total_nutrition'])

            # Combine YOLO and Gemini results
            combined_analysis = combine_ingredient_results(
                ingredients_detected,
                ingredient_confidences,
                analyze_ingredients_with_gemini(image)
            )

            return SegmentationResponse(
                ingredients_detected=ingredients_detected,
                ingredient_confidences=ingredient_confidences,
                combined_analysis=combined_analysis,
                nutrition_data=nutrition_data,
                health_analysis=health_analysis
            )

        except Exception as seg_error:
            logger.error(f"Segmentation error: {str(seg_error)}")
            # Fallback to Gemini only
            gemini_result = analyze_ingredients_with_gemini(image)
            ingredients_detected = gemini_result.get('ingredients', [])
            ingredient_confidences = [{"gemini_detected": 0.7} for _ in ingredients_detected]

            nutrition_data = None
            health_analysis = None
            if ingredients_detected:
                nutrition_data = analyze_nutrition(ingredients_detected)
                if nutrition_data.get('total_nutrition'):
                    health_analysis = calculate_health_metrics(nutrition_data['total_nutrition'])

            return SegmentationResponse(
                ingredients_detected=ingredients_detected,
                ingredient_confidences=ingredient_confidences,
                combined_analysis=gemini_result,
                nutrition_data=nutrition_data,
                health_analysis=health_analysis
            )

    except Exception as e:
        logger.error(f"Error in ingredient segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/nutrition")
async def analyze_nutrition_endpoint(ingredients: List[str]):
    """Analyze nutrition for a list of ingredients"""
    try:
        nutrition_data = analyze_nutrition(ingredients)
        return nutrition_data
    except Exception as e:
        logger.error(f"Error in nutrition analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing nutrition: {str(e)}")

@app.post("/analyze", response_model=CombinedResponse)
async def analyze_food(file: UploadFile = File(...)):
    """Complete food analysis: detection, classification, and segmentation"""
    try:
        # Read and validate image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Check file size
        image_data = await file.read()
        if len(image_data) > config['max_image_size']:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {config['max_image_size']} bytes"
            )

        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # 1. Food detection
        detection_result = detect_food_nonfood(image)
        food_detection = FoodDetectionResponse(
            is_food=detection_result['is_food'],
            confidence=detection_result['confidence']
        )

        # 2. Food classification (only if food is detected)
        food_classification = None
        if detection_result['is_food']:
            try:
                food_classification = classify_food_dish(image)
            except Exception as e:
                logger.warning(f"Food classification failed: {str(e)}")
                # Fallback to Gemini
                gemini_result = classify_dish_with_gemini(image)
                food_classification = FoodClassificationResponse(
                    dish_name=gemini_result.get('dish_name', 'unknown'),
                    confidence=gemini_confidence_to_float(gemini_result.get('confidence', 'low')),
                    top_predictions=[{
                        "dish_name": gemini_result.get('dish_name', 'unknown'),
                        "confidence": gemini_confidence_to_float(gemini_result.get('confidence', 'low')),
                        "source": gemini_result.get('source', 'unknown')
                    }]
                )

        # 3. Ingredient segmentation (only if food is detected)
        segmentation = None
        if detection_result['is_food']:
            try:
                # Run YOLO segmentation
                results = load_segmentation_model()(np.array(image), verbose=False)

                ingredients_detected = []
                ingredient_confidences = []

                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        for box in result.boxes:
                            if box.conf is not None and box.cls is not None:
                                confidence = float(box.conf[0])
                                class_id = int(box.cls[0])

                                # Get class name from the model
                                class_name = result.names[class_id] if class_id in result.names else f"class_{class_id}"

                                # Filter out background class
                                if class_name.lower() != "background":
                                    ingredients_detected.append(class_name)
                                    ingredient_confidences.append({class_name: confidence})

                # If YOLO didn't detect anything, try Gemini
                if not ingredients_detected:
                    logger.info("No ingredients detected by YOLO, trying Gemini...")
                    gemini_result = analyze_ingredients_with_gemini(image)
                    if gemini_result.get('ingredients'):
                        ingredients_detected = gemini_result['ingredients']
                        ingredient_confidences = [{"gemini_detected": 0.7} for _ in ingredients_detected]

                # Get nutrition data
                nutrition_data = None
                health_analysis = None
                if ingredients_detected:
                    nutrition_data = analyze_nutrition(ingredients_detected)
                    if nutrition_data.get('total_nutrition'):
                        health_analysis = calculate_health_metrics(nutrition_data['total_nutrition'])

                # Combine YOLO and Gemini results
                combined_analysis = combine_ingredient_results(
                    ingredients_detected,
                    ingredient_confidences,
                    analyze_ingredients_with_gemini(image)
                )

                segmentation = SegmentationResponse(
                    ingredients_detected=ingredients_detected,
                    ingredient_confidences=ingredient_confidences,
                    combined_analysis=combined_analysis,
                    nutrition_data=nutrition_data,
                    health_analysis=health_analysis
                )

            except Exception as seg_error:
                logger.error(f"Segmentation error: {str(seg_error)}")
                # Fallback to Gemini only
                gemini_result = analyze_ingredients_with_gemini(image)
                ingredients_detected = gemini_result.get('ingredients', [])
                ingredient_confidences = [{"gemini_detected": 0.7} for _ in ingredients_detected]

                nutrition_data = None
                health_analysis = None
                if ingredients_detected:
                    nutrition_data = analyze_nutrition(ingredients_detected)
                    if nutrition_data.get('total_nutrition'):
                        health_analysis = calculate_health_metrics(nutrition_data['total_nutrition'])

                segmentation = SegmentationResponse(
                    ingredients_detected=ingredients_detected,
                    ingredient_confidences=ingredient_confidences,
                    combined_analysis=gemini_result,
                    nutrition_data=nutrition_data,
                    health_analysis=health_analysis
                )

        return CombinedResponse(
            food_detection=food_detection,
            food_classification=food_classification,
            segmentation=segmentation
        )

    except Exception as e:
        logger.error(f"Error in complete food analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_food_ai(request: ChatRequest = Body(...)):
    """Chat with the food AI about nutrition and health"""
    try:
        model = load_gemini_model()
        if not model:
            raise HTTPException(status_code=503, detail="Gemini API not available")

        # Build context from request
        context = "You are a helpful nutrition and food expert. "

        if request.dish_name:
            context += f"The user is asking about: {request.dish_name}. "

        if request.ingredients:
            context += f"Ingredients: {', '.join(request.ingredients)}. "

        if request.nutrition:
            context += f"Nutrition data: {json.dumps(request.nutrition)}. "

        if request.image:
            # Decode base64 image
            try:
                image_data = base64.b64decode(request.image)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')

                # Convert image to bytes for Gemini
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                # Generate response with image
                response = model.generate_content([
                    context + request.question,
                    {"mime_type": "image/png", "data": img_byte_arr}
                ])
            except Exception as img_error:
                logger.error(f"Error processing image in chat: {str(img_error)}")
                # Fallback to text-only
                response = model.generate_content(context + request.question)
        else:
            # Text-only response
            response = model.generate_content(context + request.question)

        return ChatResponse(
            answer=response.text,
            gemini_raw={"response": response.text}
        )

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

def analyze_nutrition(ingredients: List[str]) -> Dict[str, Any]:
    """Analyze nutrition for a list of ingredients"""
    try:
        ingredients_nutrition = []
        for ingredient in ingredients:
            nutrition = get_nutrition_data(ingredient)
            ingredients_nutrition.append(nutrition)

        # Calculate total nutrition
        total_nutrition = calculate_total_nutrition(ingredients_nutrition)

        return {
            "ingredients_nutrition": ingredients_nutrition,
            "total_nutrition": total_nutrition,
            "summary": {
                "total_ingredients": len(ingredients),
                "successful_lookups": total_nutrition["successful_lookups"],
                "total_calories": round(total_nutrition["total_calories"], 2),
                "main_macros": {
                    "protein_g": round(total_nutrition["total_protein"], 2),
                    "carbs_g": round(total_nutrition["total_carbohydrate"], 2),
                    "fat_g": round(total_nutrition["total_fat"], 2),
                    "fiber_g": round(total_nutrition["total_fiber"], 2),
                    "sugars_g": round(total_nutrition["total_sugars"], 2)
                }
            }
        }

    except Exception as e:
        logger.error(f"Error in nutrition analysis: {str(e)}")
        return {
            "error": f"Nutrition analysis failed: {str(e)}",
            "ingredients_nutrition": [],
            "total_nutrition": {},
            "summary": {}
        }

def detect_food_nonfood(image: Image.Image):
    """Detect if the image contains food using the loaded model and return a dictionary."""
    processed_image = preprocess_image_for_detection(image)
    model = load_food_nonfood_model()
    prob = model.predict(processed_image, verbose=0)[0][0]
    threshold = config['classification_threshold']
    is_food = prob < threshold  # If prob < threshold, it's FOOD
    confidence = prob if not is_food else (1 - prob)
    return {
        "is_food": is_food,
        "confidence": confidence
    }

# Pydantic models for responses
class FoodDetectionResponse(BaseModel):
    is_food: bool
    confidence: float

class FoodClassificationResponse(BaseModel):
    dish_name: str
    confidence: Optional[float]
    top_predictions: List[Dict[str, Any]]

class SegmentationResponse(BaseModel):
    ingredients_detected: List[str]
    ingredient_confidences: List[Dict[str, float]]  # List of {ingredient: confidence} pairs
    combined_analysis: Optional[Dict[str, Any]] = None  # Combined YOLO + Gemini results
    nutrition_data: Optional[Dict[str, Any]] = None  # Nutrition information for ingredients
    health_analysis: Optional[Dict[str, Any]] = None  # Health metrics and recommendations

class CombinedResponse(BaseModel):
    food_detection: FoodDetectionResponse
    food_classification: Optional[FoodClassificationResponse]
    segmentation: Optional[SegmentationResponse]

class ChatRequest(BaseModel):
    question: str
    dish_name: Optional[str] = None
    ingredients: Optional[List[str]] = None
    nutrition: Optional[Dict[str, Any]] = None
    image: Optional[str] = None  # base64-encoded image

class ChatResponse(BaseModel):
    answer: str
    gemini_raw: Optional[dict] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint - doesn't require model loading"""
    try:
        # Check Nutritionix configuration
        nutritionix_configured = setup_nutritionix()

        return {
            "status": "healthy",
            "models_loaded": False,  # Models are loaded on-demand now
            "nutritionix_configured": nutritionix_configured,
            "gemini_configured": False,  # Will be loaded when needed
            "lazy_loading": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Food detection endpoint
@app.post("/detect-food", response_model=FoodDetectionResponse)
async def detect_food(file: UploadFile = File(...)):
    """Detect if the uploaded image contains food"""
    try:
        # Read and validate image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Check file size
        image_data = await file.read()
        if len(image_data) > config['max_image_size']:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {config['max_image_size']} bytes"
            )

        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Use the lazy loading function
        result = detect_food_nonfood(image)

        return FoodDetectionResponse(
            is_food=result['is_food'],
            confidence=result['confidence']
        )

    except Exception as e:
        logger.error(f"Error in food detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Simple startup event - no model loading
@app.on_event("startup")
async def startup_event():
    """Startup event - only setup basic configuration"""
    setup_nutritionix()
    logger.info("FastAPI app started with lazy loading enabled")

@app.get("/model-status")
async def model_status():
    """Check which models are currently loaded in memory"""
    return {
        "food_nonfood_model": food_nonfood_model is not None,
        "food_classifier_model": food_classifier_model is not None,
        "segmentation_model": segmentation_model is not None,
        "food_labels": food_labels is not None,
        "gemini_model": gemini_model is not None,
        "lazy_loading": True,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['host'],
        port=config['port'],
        timeout_keep_alive=config['image_timeout']
    )