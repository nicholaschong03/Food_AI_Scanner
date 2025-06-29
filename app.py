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
                "serving_weight_grams": food.get('serving_weight_grams', 0),

                # Main macros
                "calories": food.get('nf_calories', 0),
                "protein": food.get('nf_protein', 0),
                "total_fat": food.get('nf_total_fat', 0),
                "saturated_fat": food.get('nf_saturated_fat', 0),
                "total_carbohydrate": food.get('nf_total_carbohydrate', 0),
                "dietary_fiber": food.get('nf_dietary_fiber', 0),
                "sugars": food.get('nf_sugars', 0),
                "cholesterol": food.get('nf_cholesterol', 0),
                "sodium": food.get('nf_sodium', 0),
                "potassium": food.get('nf_potassium', 0),

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
                        nutrition_data["vitamin_c"] = value
                    elif attr_id == 320:  # Vitamin A
                        nutrition_data["vitamin_a"] = value
                    elif attr_id == 328:  # Vitamin D
                        nutrition_data["vitamin_d"] = value
                    elif attr_id == 323:  # Vitamin E
                        nutrition_data["vitamin_e"] = value
                    elif attr_id == 430:  # Vitamin K
                        nutrition_data["vitamin_k"] = value
                    elif attr_id == 301:  # Calcium
                        nutrition_data["calcium"] = value
                    elif attr_id == 303:  # Iron
                        nutrition_data["iron"] = value
                    elif attr_id == 304:  # Magnesium
                        nutrition_data["magnesium"] = value
                    elif attr_id == 309:  # Zinc
                        nutrition_data["zinc"] = value
                    elif attr_id == 305:  # Phosphorus
                        nutrition_data["phosphorus"] = value

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
    """Calculate total nutrition for all ingredients"""
    total_nutrition = {
        "total_calories": 0,
        "total_protein": 0,
        "total_fat": 0,
        "total_saturated_fat": 0,
        "total_carbohydrate": 0,
        "total_fiber": 0,
        "total_sugars": 0,
        "total_cholesterol": 0,
        "total_sodium": 0,
        "total_potassium": 0,
        "total_vitamin_c": 0,
        "total_vitamin_a": 0,
        "total_calcium": 0,
        "total_iron": 0,
        "ingredients_count": len(ingredients_nutrition),
        "successful_lookups": 0
    }

    for nutrition in ingredients_nutrition:
        if nutrition.get('success', False):
            total_nutrition["successful_lookups"] += 1
            total_nutrition["total_calories"] += nutrition.get('calories', 0)
            total_nutrition["total_protein"] += nutrition.get('protein', 0)
            total_nutrition["total_fat"] += nutrition.get('total_fat', 0)
            total_nutrition["total_saturated_fat"] += nutrition.get('saturated_fat', 0)
            total_nutrition["total_carbohydrate"] += nutrition.get('total_carbohydrate', 0)
            total_nutrition["total_fiber"] += nutrition.get('dietary_fiber', 0)
            total_nutrition["total_sugars"] += nutrition.get('sugars', 0)
            total_nutrition["total_cholesterol"] += nutrition.get('cholesterol', 0)
            total_nutrition["total_sodium"] += nutrition.get('sodium', 0)
            total_nutrition["total_potassium"] += nutrition.get('potassium', 0)

            # Add vitamins and minerals if available
            if nutrition.get('vitamin_c') is not None:
                total_nutrition["total_vitamin_c"] += nutrition['vitamin_c']
            if nutrition.get('vitamin_a') is not None:
                total_nutrition["total_vitamin_a"] += nutrition['vitamin_a']
            if nutrition.get('calcium') is not None:
                total_nutrition["total_calcium"] += nutrition['calcium']
            if nutrition.get('iron') is not None:
                total_nutrition["total_iron"] += nutrition['iron']

    return total_nutrition

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
            gemini_model = genai.GenerativeModel('gemini-pro-vision')
            logger.info("Gemini API configured successfully")
        else:
            logger.warning("GEMINI_API_KEY not found. Gemini features will be disabled.")
    except Exception as e:
        logger.error(f"Error setting up Gemini: {str(e)}")

def analyze_ingredients_with_gemini(image: Image.Image) -> Dict[str, Any]:
    """Use Gemini Vision to analyze ingredients in the image"""
    if not gemini_model:
        return {"ingredients": [], "confidence": "low", "source": "gemini_disabled"}

    try:
        # Prepare the prompt for ingredient detection
        prompt = """
        Analyze this food image and identify all visible ingredients.
        Please provide a JSON response with the following format:
        {
            "ingredients": [
                {"name": "ingredient_name", "confidence": "high/medium/low", "description": "brief description"}
            ],
            "dish_type": "type of dish",
            "cooking_method": "how it's prepared if visible"
        }

        Focus on:
        1. Main ingredients (proteins, vegetables, grains)
        2. Spices and seasonings if visible
        3. Sauces and condiments
        4. Garnishes and toppings

        Be specific but don't guess. If you're unsure about an ingredient, mark confidence as "low".
        """

        # Convert PIL image to bytes for Gemini
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Generate response
        response = gemini_model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])

        # Parse response
        try:
            # Try to extract JSON from response
            response_text = response.text
            # Find JSON in the response (Gemini might wrap it in markdown)
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
    """Combine YOLO and Gemini results for comprehensive ingredient detection"""

    # Process YOLO results
    yolo_ingredient_data = []
    for i, ingredient in enumerate(yolo_ingredients):
        confidence = list(yolo_confidences[i].values())[0]
        yolo_ingredient_data.append({
            "name": ingredient,
            "confidence": confidence,
            "source": "yolo",
            "detection_type": "segmented"
        })

    # Process Gemini results
    gemini_ingredient_data = []
    if gemini_result.get("ingredients"):
        for item in gemini_result["ingredients"]:
            if isinstance(item, dict):
                gemini_ingredient_data.append({
                    "name": item.get("name", str(item)),
                    "confidence": item.get("confidence", "medium"),
                    "source": "gemini",
                    "detection_type": "vision_analysis"
                })
            else:
                gemini_ingredient_data.append({
                    "name": str(item),
                    "confidence": "medium",
                    "source": "gemini",
                    "detection_type": "vision_analysis"
                })

    # Combine and deduplicate
    all_ingredients = {}

    # Add YOLO ingredients
    for item in yolo_ingredient_data:
        name_lower = item["name"].lower()
        if name_lower not in all_ingredients:
            all_ingredients[name_lower] = item
        else:
            # If both detected, mark as high confidence
            all_ingredients[name_lower]["source"] = "both"
            all_ingredients[name_lower]["confidence"] = max(
                all_ingredients[name_lower]["confidence"],
                item["confidence"]
            )

    # Add Gemini ingredients
    for item in gemini_ingredient_data:
        name_lower = item["name"].lower()
        if name_lower not in all_ingredients:
            all_ingredients[name_lower] = item
        else:
            # If both detected, mark as high confidence
            all_ingredients[name_lower]["source"] = "both"
            all_ingredients[name_lower]["confidence"] = max(
                all_ingredients[name_lower]["confidence"],
                item["confidence"]
            )

    return {
        "combined_ingredients": list(all_ingredients.values()),
        "yolo_count": len(yolo_ingredients),
        "gemini_count": len(gemini_ingredient_data),
        "total_detected": len(all_ingredients),
        "dish_type": gemini_result.get("dish_type", "unknown"),
        "cooking_method": gemini_result.get("cooking_method", "unknown")
    }

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

def load_models():
    """Load all ML models with memory optimization"""
    global food_nonfood_model, food_classifier_model, segmentation_model, food_labels

    try:
        # Optimize memory before loading
        optimize_memory()

        # Load food/non-food classifier
        logger.info("Loading food/non-food classifier...")
        food_nonfood_model = keras.models.load_model('models/food_nonfood_mnv2.h5', compile=False)

        # Load food dish classifier
        logger.info("Loading food dish classifier...")
        food_classifier_model = keras.models.load_model('models/food_dish_classifier_efficientnetb2_combined_latest.h5', compile=False)

        # Load YOLO segmentation model
        logger.info("Loading YOLO segmentation model...")
        segmentation_model = YOLO('models/best_ingredient_seg.pt')

        # Load food labels
        logger.info("Loading food labels...")
        with open('models/labels_combined.json', 'r') as f:
            food_labels = json.load(f)

        # Setup Gemini API
        logger.info("Setting up Gemini API...")
        setup_gemini()

        logger.info("All models loaded successfully!")

        # Optimize memory after loading
        optimize_memory()

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()
    setup_nutritionix()

# Pydantic models for responses
class FoodDetectionResponse(BaseModel):
    is_food: bool
    confidence: float

class FoodClassificationResponse(BaseModel):
    dish_name: str
    confidence: float
    top_predictions: List[Dict[str, Any]]

class SegmentationResponse(BaseModel):
    segmented_image: str  # Base64 encoded image
    ingredients_detected: List[str]
    ingredient_confidences: List[Dict[str, float]]  # List of {ingredient: confidence} pairs
    combined_analysis: Optional[Dict[str, Any]] = None  # Combined YOLO + Gemini results
    nutrition_data: Optional[Dict[str, Any]] = None  # Nutrition information for ingredients

class CombinedResponse(BaseModel):
    food_detection: FoodDetectionResponse
    food_classification: Optional[FoodClassificationResponse]
    segmentation: Optional[SegmentationResponse]
    gemini_analysis: Optional[Dict[str, Any]] = None  # Additional Gemini insights
    nutrition_analysis: Optional[Dict[str, Any]] = None  # Total nutrition analysis

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
            food_nonfood_model is not None,
            food_classifier_model is not None,
            segmentation_model is not None,
            food_labels is not None
        ])

        # Check Nutritionix configuration
        nutritionix_configured = setup_nutritionix()

        # Check Gemini configuration
        gemini_configured = gemini_model is not None

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
        prob = food_nonfood_model.predict(processed_image, verbose=0)[0][0]

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
    predictions = food_classifier_model.predict(processed_image, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [
        {"dish_name": food_labels[idx], "confidence": float(predictions[idx])}
        for idx in top_indices
    ]
    best_idx = top_indices[0]
    dish_name = food_labels[best_idx]
    confidence = float(predictions[best_idx])
    return FoodClassificationResponse(
        dish_name=dish_name,
        confidence=confidence,
        top_predictions=top_predictions
    )

# Food classification threshold
FOOD_CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.50

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
        # If confidence is below threshold, use Gemini Vision Pro
        if result.confidence < FOOD_CLASSIFICATION_CONFIDENCE_THRESHOLD and gemini_model:
            try:
                gemini_result = analyze_ingredients_with_gemini(image)
                used_model = "gemini"
                # Optionally, you can set the Gemini result as the main result
                # or return both for transparency
                return {
                    "dish_name": gemini_result.get("dish_type", "unknown"),
                    "confidence": None,
                    "top_predictions": [],
                    "used_model": used_model,
                    "gemini_result": gemini_result
                }
            except Exception as e:
                logger.warning(f"Gemini Vision Pro classification failed: {str(e)}")
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
    Segment ingredients in food image using YOLO model
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
        results = segmentation_model(image_np, verbose=False)

        # Process results
        ingredients_detected = []
        ingredient_confidences = []

        if len(results) > 0 and results[0].masks is not None:
            # Get masks and class predictions
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.data.cpu().numpy()

            # Create colored mask overlay
            colored_mask = np.zeros_like(image_np)

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                class_id = int(box[5])
                confidence = float(box[4])

                if class_id < len(segmentation_class_names):
                    ingredient_name = segmentation_class_names[class_id]

                    # Apply confidence threshold
                    if confidence >= config['segmentation_threshold']:
                        ingredients_detected.append(ingredient_name)
                        ingredient_confidences.append({ingredient_name: confidence})

                        # Color the mask (different color for each ingredient)
                        color = np.random.randint(0, 255, 3)
                        colored_mask[mask.astype(bool)] = color

            # Blend mask with original image
            alpha = 0.5
            segmented_image = cv2.addWeighted(image_np, 1-alpha, colored_mask, alpha, 0)
        else:
            segmented_image = image_np

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', segmented_image)
        segmented_image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Get nutrition data for detected ingredients
        nutrition_data = None
        if ingredients_detected:
            ingredients_nutrition = []
            for ingredient in ingredients_detected:
                nutrition = get_nutrition_data(ingredient)
                ingredients_nutrition.append(nutrition)

            # Calculate total nutrition
            total_nutrition = calculate_total_nutrition(ingredients_nutrition)

            nutrition_data = {
                "ingredients_nutrition": ingredients_nutrition,
                "total_nutrition": total_nutrition
            }

        return SegmentationResponse(
            segmented_image=f"data:image/jpeg;base64,{segmented_image_b64}",
            ingredients_detected=ingredients_detected,
            ingredient_confidences=ingredient_confidences,
            nutrition_data=nutrition_data
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
        gemini_analysis = None
        nutrition_analysis = None

        # Only proceed with detailed analysis if food is detected
        if food_detection_result.is_food:
            # 2. Food classification
            try:
                food_classification_result = classify_food_dish(image)
            except Exception as e:
                logger.warning(f"Food classification failed: {str(e)}")

            # 3. Ingredient segmentation
            try:
                segmentation_result = await segment_ingredients_internal(image_np)

                # 4. Gemini analysis for additional insights
                if gemini_model:
                    try:
                        gemini_analysis = analyze_ingredients_with_gemini(image)
                    except Exception as e:
                        logger.warning(f"Gemini analysis failed: {str(e)}")

                # 5. Nutrition analysis
                if segmentation_result and segmentation_result.ingredients_detected:
                    nutrition_analysis = analyze_nutrition(segmentation_result.ingredients_detected)

            except Exception as e:
                logger.warning(f"Segmentation failed: {str(e)}")

        return CombinedResponse(
            food_detection=food_detection_result,
            food_classification=food_classification_result,
            segmentation=segmentation_result,
            gemini_analysis=gemini_analysis,
            nutrition_analysis=nutrition_analysis
        )

    except Exception as e:
        logger.error(f"Error in combined analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def segment_ingredients_internal(image_np: np.ndarray) -> SegmentationResponse:
    """Internal segmentation function for combined analysis"""
    try:
        # Run YOLO segmentation
        results = segmentation_model(image_np, verbose=False)

        # Process results
        ingredients_detected = []
        ingredient_confidences = []

        if len(results) > 0 and results[0].masks is not None:
            # Get masks and class predictions
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.data.cpu().numpy()

            # Create colored mask overlay
            colored_mask = np.zeros_like(image_np)

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                class_id = int(box[5])
                confidence = float(box[4])

                if class_id < len(segmentation_class_names):
                    ingredient_name = segmentation_class_names[class_id]

                    # Apply confidence threshold
                    if confidence >= config['segmentation_threshold']:
                        ingredients_detected.append(ingredient_name)
                        ingredient_confidences.append({ingredient_name: confidence})

                        # Color the mask (different color for each ingredient)
                        color = np.random.randint(0, 255, 3)
                        colored_mask[mask.astype(bool)] = color

            # Blend mask with original image
            alpha = 0.5
            segmented_image = cv2.addWeighted(image_np, 1-alpha, colored_mask, alpha, 0)
        else:
            segmented_image = image_np

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', segmented_image)
        segmented_image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Get nutrition data for detected ingredients
        nutrition_data = None
        if ingredients_detected:
            ingredients_nutrition = []
            for ingredient in ingredients_detected:
                nutrition = get_nutrition_data(ingredient)
                ingredients_nutrition.append(nutrition)

            # Calculate total nutrition
            total_nutrition = calculate_total_nutrition(ingredients_nutrition)

            nutrition_data = {
                "ingredients_nutrition": ingredients_nutrition,
                "total_nutrition": total_nutrition
            }

        return SegmentationResponse(
            segmented_image=f"data:image/jpeg;base64,{segmented_image_b64}",
            ingredients_detected=ingredients_detected,
            ingredient_confidences=ingredient_confidences,
            nutrition_data=nutrition_data
        )

    except Exception as e:
        logger.error(f"Error in internal segmentation: {str(e)}")
        raise e

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
    prob = food_nonfood_model.predict(processed_image, verbose=0)[0][0]
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
    if not gemini_model:
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
        response = gemini_model.generate_content(gemini_inputs)
        answer = response.text.strip()
        return ChatResponse(answer=answer, gemini_raw=getattr(response, 'result', None))
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['host'],
        port=config['port'],
        timeout_keep_alive=config['image_timeout']
    )