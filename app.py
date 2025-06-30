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

def get_food_nonfood_model():
    global food_nonfood_model
    if food_nonfood_model is None:
        logger.info("Loading food/non-food classifier...")
        food_nonfood_model = keras.models.load_model('models/food_nonfood_mnv2.h5', compile=False)
    return food_nonfood_model

def get_food_classifier_model():
    global food_classifier_model
    if food_classifier_model is None:
        logger.info("Loading food dish classifier...")
        food_classifier_model = keras.models.load_model('models/food_dish_classifier_efficientnetb2_combined_latest.h5', compile=False)
    return food_classifier_model

def get_segmentation_model():
    global segmentation_model
    if segmentation_model is None:
        logger.info("Loading YOLO segmentation model...")
        segmentation_model = YOLO('models/best_ingredient_seg.pt')
    return segmentation_model

def get_food_labels():
    global food_labels
    if food_labels is None:
        logger.info("Loading food labels...")
        with open('models/labels_combined.json', 'r') as f:
            food_labels = json.load(f)
    return food_labels

def get_gemini_model():
    global gemini_model
    if gemini_model is None:
        logger.info("Setting up Gemini API...")
        setup_gemini()
    return gemini_model

# Remove model loading from startup event
@app.on_event("startup")
async def startup_event():
    setup_nutritionix()
    logger.info("FastAPI app started with lazy loading enabled")

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
    """Health check endpoint"""
    try:
        # Check if models are loaded
        models_loaded = all([
            get_food_nonfood_model() is not None,
            get_food_classifier_model() is not None,
            get_segmentation_model() is not None,
            get_food_labels() is not None
        ])

        # Check Nutritionix configuration
        nutritionix_configured = setup_nutritionix()

        # Check Gemini configuration
        gemini_configured = get_gemini_model() is not None

        return {
            "status": "healthy" if models_loaded else "unhealthy",
            "models_loaded": models_loaded,
            "nutritionix_configured": nutritionix_configured,
            "gemini_configured": gemini_configured,
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

        # Preprocess image using the same logic as Google Colab
        processed_image = preprocess_image_for_detection(image)

        # Make prediction (same as Colab: model.predict(x, verbose=0)[0][0])
        model = get_food_nonfood_model()
        prob = model.predict(processed_image, verbose=0)[0][0]

        # Apply threshold logic (same as Colab: prob < threshold for FOOD)
        threshold = config['classification_threshold']
        is_food = prob < threshold  # If prob < threshold, it's FOOD

        # Calculate confidence (same as Colab logic)
        confidence = prob if not is_food else (1 - prob)

        # Optimize memory after prediction
        optimize_memory()

        return FoodDetectionResponse(
            is_food=is_food,
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Error in food detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def classify_food_dish(image: Image.Image):
    """Classify the food dish using the loaded model and return a FoodClassificationResponse."""
    processed_image = preprocess_image_for_food_classification(image)
    model = get_food_classifier_model()
    labels = get_food_labels()
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
    if not get_gemini_model():
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
        response = get_gemini_model().generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
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
    except Exception as e:
        logger.error(f"Error with Gemini API: {str(e)}")
        return {"dish_name": "unknown", "confidence": "low", "source": "gemini_error"}

@app.post("/classify-food", response_model=FoodClassificationResponse)
async def classify_food(file: UploadFile = File(...)):
    """Classify the food dish, fallback to Gemini Vision Pro if confidence is low"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        result = classify_food_dish(image)
        used_model = "local"
        gemini_result = None
        # If confidence is below threshold, use Gemini for dish name
        if result.confidence < FOOD_CLASSIFICATION_CONFIDENCE_THRESHOLD and get_gemini_model():
            try:
                gemini_result = classify_dish_with_gemini(image)
                used_model = "gemini"
                return {
                    "dish_name": gemini_result.get("dish_name", "unknown"),
                    "confidence": gemini_confidence_to_float(gemini_result.get("confidence")),
                    "top_predictions": [],
                    "used_model": used_model,
                    "gemini_result": gemini_result
                }
            except Exception as e:
                logger.warning(f"Gemini dish classification failed: {str(e)}")
        # Return local model result
        return {
            **result.dict(),
            "used_model": used_model,
            "gemini_result": gemini_result
        }
    except Exception as e:
        logger.error(f"Error in food classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Segmentation endpoint
@app.post("/segment", response_model=SegmentationResponse)
async def segment_ingredients(file: UploadFile = File(...)):
    """
    Segment ingredients in food image using YOLO model and Gemini (if available)
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        # Run YOLO segmentation
        results = get_segmentation_model()(image_np, verbose=False)

        # Process results
        ingredients_detected = []
        ingredient_confidences = []
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.data.cpu().numpy()
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                class_id = int(box[5])
                confidence = float(box[4])
                if class_id < len(segmentation_class_names):
                    ingredient_name = segmentation_class_names[class_id]
                    if ingredient_name == "background":
                        continue  # Skip background class
                    if confidence >= config['segmentation_threshold']:
                        ingredients_detected.append(ingredient_name)
                        ingredient_confidences.append({ingredient_name: confidence})

        # Always call Gemini for ingredient analysis if available
        gemini_result = None
        if get_gemini_model():
            try:
                gemini_result = analyze_ingredients_with_gemini(image)
            except Exception as e:
                logger.warning(f"Gemini analysis failed: {str(e)}")

        # --- Combine YOLO & Gemini detected ingredients for nutrition lookup ---
        yolo_ingredients = set([i for i in ingredients_detected if i.lower() != "background"])
        gemini_ingredients = set()
        if gemini_result and "ingredients" in gemini_result:
            for i in gemini_result["ingredients"]:
                if isinstance(i, dict) and "name" in i:
                    name = i["name"]
                else:
                    name = str(i)
                if name.lower() != "background":
                    gemini_ingredients.add(name)

        all_ingredients = sorted(yolo_ingredients.union(gemini_ingredients))

        # Lookup nutrtion for All unique ingredients
        nutrition_data = None
        if all_ingredients:
            ingredients_nutrition = []
            for ingredient in all_ingredients:
                nutrition = get_nutrition_data(ingredient)
                ingredients_nutrition.append(nutrition)
            total_nutrition = calculate_total_nutrition(ingredients_nutrition)
            nutrition_data = {
                "ingredients_nutrition": ingredients_nutrition,
                "total_nutrition": total_nutrition,
                "used_ingredients": all_ingredients,
            }

        # Create combined analysis with both YOLO and Gemini ingredients
        combined_analysis = combine_ingredient_results(ingredients_detected, ingredient_confidences, gemini_result)

        # Calculate health metrics
        health_analysis = calculate_health_metrics(nutrition_data) if nutrition_data else None

        return SegmentationResponse(
            ingredients_detected=ingredients_detected,
            ingredient_confidences=ingredient_confidences,
            combined_analysis=combined_analysis,
            nutrition_data=nutrition_data,
            health_analysis=health_analysis
        )

    except Exception as e:
        logger.error(f"Error in segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

# Nutrition analysis endpoint
@app.post("/nutrition")
async def analyze_nutrition_endpoint(ingredients: List[str]):
    """
    Analyze nutrition for a list of ingredients
    """
    try:
        if not ingredients:
            raise HTTPException(status_code=400, detail="No ingredients provided")

        nutrition_result = analyze_nutrition(ingredients)

        return {
            "ingredients": ingredients,
            "nutrition_analysis": nutrition_result
        }

    except Exception as e:
        logger.error(f"Error in nutrition analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Nutrition analysis failed: {str(e)}")

# Combined endpoint for all predictions
@app.post("/analyze", response_model=CombinedResponse)
async def analyze_food(file: UploadFile = File(...)):
    """
    Complete food analysis: detection, classification, segmentation, and nutrition
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image once
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        # 1. Food/Non-food detection
        food_detection_result = detect_food_nonfood(image)

        # Initialize responses
        food_classification_result = None
        segmentation_result = None

        # Only proceed with detailed analysis if food is detected
        if food_detection_result.is_food:
            # 2. Food classification with Gemini fallback
            try:
                food_classification_result = classify_food_dish(image)
                # If confidence is below threshold, use Gemini for dish name
                if food_classification_result.confidence < FOOD_CLASSIFICATION_CONFIDENCE_THRESHOLD and get_gemini_model():
                    try:
                        gemini_result = classify_dish_with_gemini(image)
                        food_classification_result = FoodClassificationResponse(
                            dish_name=gemini_result.get("dish_name", "unknown"),
                            confidence=gemini_confidence_to_float(gemini_result.get("confidence")),
                            top_predictions=[]
                        )
                    except Exception as e:
                        logger.warning(f"Gemini dish classification failed: {str(e)}")
            except Exception as e:
                logger.warning(f"Food classification failed: {str(e)}")

            # 3. Ingredient segmentation (using same logic as /segment endpoint)
            try:
                # Run YOLO segmentation
                results = get_segmentation_model()(image_np, verbose=False)
                ingredients_detected = []
                ingredient_confidences = []

                if len(results) > 0 and results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    boxes = results[0].boxes.data.cpu().numpy()
                    for i, (mask, box) in enumerate(zip(masks, boxes)):
                        class_id = int(box[5])
                        confidence = float(box[4])
                        if class_id < len(segmentation_class_names):
                            ingredient_name = segmentation_class_names[class_id]
                            if ingredient_name == "background":
                                continue  # Skip background class
                            if confidence >= config['segmentation_threshold']:
                                ingredients_detected.append(ingredient_name)
                                ingredient_confidences.append({ingredient_name: confidence})

                # Always call Gemini for ingredient analysis if available
                gemini_result = None
                if get_gemini_model():
                    try:
                        gemini_result = analyze_ingredients_with_gemini(image)
                    except Exception as e:
                        logger.warning(f"Gemini analysis failed: {str(e)}")

                # --- Combine YOLO & Gemini detected ingredients for nutrition lookup ---
                yolo_ingredients = set([i for i in ingredients_detected if i.lower() != "background"])
                gemini_ingredients = set()
                if gemini_result and "ingredients" in gemini_result:
                    for i in gemini_result["ingredients"]:
                        if isinstance(i, dict) and "name" in i:
                            name = i["name"]
                        else:
                            name = str(i)
                        if name.lower() != "background":
                            gemini_ingredients.add(name)

                all_ingredients = sorted(yolo_ingredients.union(gemini_ingredients))

                # Lookup nutrition for all unique ingredients
                nutrition_data = None
                if all_ingredients:
                    ingredients_nutrition = []
                    for ingredient in all_ingredients:
                        nutrition = get_nutrition_data(ingredient)
                        ingredients_nutrition.append(nutrition)
                    total_nutrition = calculate_total_nutrition(ingredients_nutrition)
                    nutrition_data = {
                        "ingredients_nutrition": ingredients_nutrition,
                        "total_nutrition": total_nutrition,
                        "used_ingredients": all_ingredients,
                    }

                # Create combined analysis with both YOLO and Gemini ingredients
                combined_analysis = combine_ingredient_results(ingredients_detected, ingredient_confidences, gemini_result)

                # Calculate health metrics
                health_analysis = calculate_health_metrics(nutrition_data) if nutrition_data else None

                # Create segmentation result
                segmentation_result = SegmentationResponse(
                    ingredients_detected=ingredients_detected,
                    ingredient_confidences=ingredient_confidences,
                    combined_analysis=combined_analysis,
                    nutrition_data=nutrition_data,
                    health_analysis=health_analysis
                )

            except Exception as e:
                logger.warning(f"Segmentation failed: {str(e)}")

        return CombinedResponse(
            food_detection=food_detection_result,
            food_classification=food_classification_result,
            segmentation=segmentation_result
        )

    except Exception as e:
        logger.error(f"Error in combined analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

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
    """Detect if the image contains food using the loaded model and return a FoodDetectionResponse."""
    processed_image = preprocess_image_for_detection(image)
    model = get_food_nonfood_model()
    prob = model.predict(processed_image, verbose=0)[0][0]
    threshold = config['classification_threshold']
    is_food = prob < threshold  # If prob < threshold, it's FOOD
    confidence = prob if not is_food else (1 - prob)
    return FoodDetectionResponse(
        is_food=is_food,
        confidence=confidence
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_food_ai(request: ChatRequest = Body(...)):
    """
    Conversational Q&A about the captured food using Gemini Vision Pro.
    Accepts question, dish_name, ingredients, nutrition, and optionally image (base64).
    Returns the AI's answer.
    """
    if not get_gemini_model():
        raise HTTPException(status_code=503, detail="Gemini Vision Pro is not configured.")
    try:
        # Compose context for Gemini
        context = ""
        if request.dish_name:
            context += f"Dish name: {request.dish_name}\n"
        if request.ingredients:
            context += f"Ingredients: {', '.join(request.ingredients)}\n"
        if request.nutrition:
            context += f"Nutrition: {json.dumps(request.nutrition)}\n"
        prompt = f"You are a food and nutrition expert AI. Answer the user's question about the food below.\n{context}\nQuestion: {request.question}"
        # Prepare image if provided
        gemini_inputs = [prompt]
        if request.image:
            try:
                image_bytes = base64.b64decode(request.image.split(',')[-1])
                gemini_inputs.append({"mime_type": "image/jpeg", "data": image_bytes})
            except Exception as e:
                logger.warning(f"Failed to decode image for chat: {str(e)}")
        # Get answer from Gemini
        response = get_gemini_model().generate_content(gemini_inputs)
        answer = response.text.strip()
        return ChatResponse(answer=answer, gemini_raw=getattr(response, 'result', None))
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

def calculate_health_metrics(nutrition_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate health metrics and recommendations based on nutrition data"""
    try:
        total_nutrition = nutrition_data.get("total_nutrition", {})

        # Debug logging to see what we're working with
        logger.info(f"Total nutrition keys: {list(total_nutrition.keys())}")

        # Daily Value (DV) percentages based on 2000 calorie diet
        daily_values = {
            "calories": 2000,
            "protein": 50,  # grams
            "total_fat": 65,  # grams
            "saturated_fat": 20,  # grams
            "total_carbohydrate": 300,  # grams
            "dietary_fiber": 25,  # grams
            "sugars": 50,  # grams
            "cholesterol": 300,  # mg
            "sodium": 2300,  # mg
            "potassium": 3500,  # mg
            "vitamin_c": 90,  # mg
            "vitamin_a": 900,  # mcg RAE
            "calcium": 1300,  # mg
            "iron": 18  # mg
        }

        # Calculate DV percentages
        dv_percentages = {}
        for nutrient, dv in daily_values.items():
            # Map nutrient names to actual keys in total_nutrition
            if nutrient == "calories":
                key = "total_calories"
            elif nutrient == "protein":
                key = "total_protein"
            elif nutrient == "total_fat":
                key = "total_fat"
            elif nutrient == "saturated_fat":
                key = "total_saturated_fat"
            elif nutrient == "total_carbohydrate":
                key = "total_carbohydrate"
            elif nutrient == "dietary_fiber":
                key = "total_fiber"
            elif nutrient == "sugars":
                key = "total_sugars"
            elif nutrient == "cholesterol":
                key = "total_cholesterol"
            elif nutrient == "sodium":
                key = "total_sodium"
            elif nutrient == "potassium":
                key = "total_potassium"
            elif nutrient == "vitamin_c":
                key = "total_vitamin_c"
            elif nutrient == "vitamin_a":
                key = "total_vitamin_a"
            elif nutrient == "calcium":
                key = "total_calcium"
            elif nutrient == "iron":
                key = "total_iron"
            else:
                key = f"total_{nutrient}"

            if key in total_nutrition:
                value = total_nutrition[key]
                percentage = round((value / dv) * 100, 1)
                dv_percentages[nutrient] = {
                    "value": value,
                    "daily_value": dv,
                    "percentage": percentage,
                    "unit": "g" if nutrient in ["protein", "total_fat", "saturated_fat", "total_carbohydrate", "dietary_fiber", "sugars"] else "mg" if nutrient in ["cholesterol", "sodium", "potassium", "vitamin_c", "calcium", "iron"] else "mcg" if nutrient == "vitamin_a" else "kcal"
                }

        # Calculate macronutrient ratios
        total_calories = total_nutrition.get("total_calories", 0)
        if total_calories > 0:
            protein_cals = total_nutrition.get("total_protein", 0) * 4
            fat_cals = total_nutrition.get("total_fat", 0) * 9
            carb_cals = total_nutrition.get("total_carbohydrate", 0) * 4

            macro_ratios = {
                "protein": round((protein_cals / total_calories) * 100, 1),
                "fat": round((fat_cals / total_calories) * 100, 1),
                "carbohydrate": round((carb_cals / total_calories) * 100, 1)
            }
        else:
            macro_ratios = {"protein": 0, "fat": 0, "carbohydrate": 0}

        # Health score calculation (1-10 scale)
        health_score = 10

        # Deduct points for excessive values
        if dv_percentages.get("saturated_fat", {}).get("percentage", 0) > 100:
            health_score -= 2.0
        if dv_percentages.get("sodium", {}).get("percentage", 0) > 100:
            health_score -= 2.0
        if dv_percentages.get("sugars", {}).get("percentage", 0) > 100:
            health_score -= 2.0
        if dv_percentages.get("cholesterol", {}).get("percentage", 0) > 100:
            health_score -= 2.0

        # Add points for good values
        if dv_percentages.get("dietary_fiber", {}).get("percentage", 0) > 80:
            health_score += 1.0
        if dv_percentages.get("protein", {}).get("percentage", 0) > 80:
            health_score += 0.5
        if dv_percentages.get("vitamin_c", {}).get("percentage", 0) > 50:
            health_score += 0.5
        if dv_percentages.get("calcium", {}).get("percentage", 0) > 50:
            health_score += 0.5

        health_score = max(1, min(10, round(health_score)))

        # Dietary recommendations
        recommendations = []

        if dv_percentages.get("saturated_fat", {}).get("percentage", 0) > 100:
            recommendations.append("High in saturated fat - consider choosing leaner protein sources")
        if dv_percentages.get("sodium", {}).get("percentage", 0) > 100:
            recommendations.append("High in sodium - consider reducing salt or choosing lower-sodium options")
        if dv_percentages.get("sugars", {}).get("percentage", 0) > 100:
            recommendations.append("High in added sugars - consider natural sweeteners or reducing sugar intake")
        if dv_percentages.get("dietary_fiber", {}).get("percentage", 0) < 50:
            recommendations.append("Low in fiber - consider adding more vegetables, fruits, or whole grains")
        if dv_percentages.get("protein", {}).get("percentage", 0) < 50:
            recommendations.append("Low in protein - consider adding more protein-rich foods")
        if dv_percentages.get("vitamin_c", {}).get("percentage", 0) < 30:
            recommendations.append("Low in vitamin C - consider adding citrus fruits or vegetables")
        if dv_percentages.get("calcium", {}).get("percentage", 0) < 30:
            recommendations.append("Low in calcium - consider adding dairy products or calcium-fortified foods")

        if not recommendations:
            recommendations.append("This meal appears to be well-balanced nutritionally")

        # Meal type classification
        meal_type = "Unknown"
        if total_calories < 300:
            meal_type = "Snack"
        elif total_calories < 600:
            meal_type = "Light Meal"
        elif total_calories < 1000:
            meal_type = "Regular Meal"
        else:
            meal_type = "Heavy Meal"

        # Dietary restrictions check
        dietary_flags = []
        if total_nutrition.get("total_cholesterol", 0) > 200:
            dietary_flags.append("high_cholesterol")
        if total_nutrition.get("total_sodium", 0) > 1000:
            dietary_flags.append("high_sodium")
        if total_nutrition.get("total_saturated_fat", 0) > 10:
            dietary_flags.append("high_saturated_fat")
        if total_nutrition.get("total_sugars", 0) > 25:
            dietary_flags.append("high_sugar")

        return {
            "health_score": health_score,
            "meal_type": meal_type,
            "daily_value_percentages": dv_percentages,
            "macro_ratios": macro_ratios,
            "recommendations": recommendations,
            "dietary_flags": dietary_flags,
            "summary": {
                "total_calories": round(total_calories, 2),
                "calorie_density": "High" if total_calories > 500 else "Medium" if total_calories > 300 else "Low",
                "nutritional_balance": "Good" if health_score > 7 else "Fair" if health_score > 5 else "Poor"
            }
        }

    except Exception as e:
        logger.error(f"Error calculating health metrics: {str(e)}")
        return {
            "health_score": 1,
            "meal_type": "Unknown",
            "daily_value_percentages": {},
            "macro_ratios": {},
            "recommendations": [f"Unable to calculate health metrics: {str(e)}"],
            "dietary_flags": [],
            "summary": {}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['host'],
        port=config['port'],
        timeout_keep_alive=config['image_timeout']
    )