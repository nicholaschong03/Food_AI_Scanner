# Food AI Scanner API

A FastAPI-based ML inference service for food detection, classification, ingredient segmentation, and nutrition analysis. This service provides comprehensive food analysis capabilities:

1. **Food Detection**: Determines if an image contains food
2. **Food Classification**: Identifies the specific dish in the image
3. **Ingredient Segmentation**: Segments ingredients from food images
4. **Nutrition Analysis**: Provides detailed nutrition information for detected ingredients

## Features

- 🍽️ **Food Detection**: Binary classification (food vs non-food)
- 🍕 **Food Classification**: Multi-class classification with 101 food categories
- 🥕 **Ingredient Segmentation**: Semantic segmentation of food ingredients
- 🧬 **Nutrition Analysis**: Comprehensive nutrition data via Nutritionix API
- 🤖 **AI-Powered Insights**: Google Gemini Vision API for enhanced ingredient detection
- 🚀 **FastAPI**: High-performance async API framework
- 📊 **Comprehensive Responses**: Detailed predictions with confidence scores
- 🔒 **Error Handling**: Robust error handling and validation
- 📝 **API Documentation**: Auto-generated OpenAPI documentation

## API Endpoints

### Health Check
- `GET /health` - Check service health and model status

### Individual Services
- `POST /detect-food` - Detect if image contains food
- `POST /classify-food` - Classify the food dish
- `POST /segment` - Segment ingredients from food with nutrition data
- `POST /nutrition` - Analyze nutrition for a list of ingredients

### Combined Analysis
- `POST /analyze` - Complete analysis (detection + classification + segmentation + nutrition)

## Model Files

The service uses three pre-trained models:

1. **`food_nonfood_mnv2.h5`** - MobileNetV2-based food/non-food classifier
2. **`food_dish_classifier_efficientnetb2_combined_latest.h5`** - EfficientNetB2-based food dish classifier
3. **`best_ingredient_seg.pt`** - PyTorch-based ingredient segmentation model
4. **`labels_combined.json`** - Food category labels (101 classes)

## Environment Configuration

Create a `.env` file with the following variables:

```bash
# API Configuration
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO

# Model Configuration
SEGMENTATION_CONFIDENCE_THRESHOLD=0.5
MAX_IMAGE_SIZE=10485760  # 10MB

# Google Gemini API (optional)
GEMINI_API_KEY=your_gemini_api_key_here

# Nutritionix API (required for nutrition features)
NUTRITIONIX_APP_ID=your_nutritionix_app_id_here
NUTRITIONIX_APP_KEY=your_nutritionix_app_key_here
```

### Getting API Keys

1. **Nutritionix API**:
   - Sign up at [nutritionix.com](https://www.nutritionix.com/business/api)
   - Get your App ID and App Key from the dashboard

2. **Google Gemini API** (optional):
   - Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Used for enhanced ingredient detection and analysis

## Local Development

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd food_AI_api
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the API**:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Testing

Run the test script to verify all endpoints:
```bash
python test_api.py
```

## Docker Deployment

### Build the Docker image:
```bash
docker build -t food-ai-scanner-api .
```

### Run the container:
```bash
docker run -p 8000:8000 food-ai-scanner-api
```

## Render Deployment

### 1. Prepare for Render

1. **Create a Render account** at [render.com](https://render.com)

2. **Connect your repository** to Render

3. **Create a new Web Service**:
   - **Name**: `food-ai-scanner-api`
   - **Environment**: `Docker`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)

### 2. Configure Environment Variables

In Render dashboard, add these environment variables:
```
PYTHON_VERSION=3.9
PORT=8000
```

### 3. Build Configuration

Render will automatically detect the Dockerfile and build the service.

### 4. Deploy

Click "Create Web Service" and Render will:
1. Build your Docker image
2. Deploy it to their infrastructure
3. Provide you with a public URL

## API Usage Examples

### Using curl

**Health Check**:
```bash
curl http://your-render-url/health
```

**Food Detection**:
```bash
curl -X POST "http://your-render-url/detect-food" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

**Complete Analysis**:
```bash
curl -X POST "http://your-render-url/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

### Using JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');

async function analyzeFood(imagePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));

    const response = await fetch('http://your-render-url/analyze', {
        method: 'POST',
        body: form
    });

    const result = await response.json();
    return result;
}

// Usage
analyzeFood('path/to/food-image.jpg')
    .then(result => {
        console.log('Food detected:', result.food_detection.is_food);
        console.log('Dish name:', result.food_classification?.dish_name);
        console.log('Confidence:', result.food_classification?.confidence);
    })
    .catch(error => console.error('Error:', error));
```

### Using Python

```python
import requests

def analyze_food(image_path, api_url):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f'{api_url}/analyze', files=files)
        return response.json()

# Usage
result = analyze_food('food-image.jpg', 'http://your-render-url')
print(f"Food detected: {result['food_detection']['is_food']}")
print(f"Dish: {result['food_classification']['dish_name']}")
```

## Nutrition Analysis

The API provides comprehensive nutrition analysis for detected ingredients using the Nutritionix API.

### Nutrition Data Included

**Main Macronutrients:**
- Calories
- Protein (g)
- Total Fat (g)
- Saturated Fat (g)
- Total Carbohydrates (g)
- Dietary Fiber (g)
- Sugars (g)
- Cholesterol (mg)
- Sodium (mg)
- Potassium (mg)

**Vitamins & Minerals:**
- Vitamin C (mg)
- Vitamin A (IU)
- Vitamin D (IU)
- Vitamin E (mg)
- Vitamin K (mcg)
- Calcium (mg)
- Iron (mg)
- Magnesium (mg)
- Zinc (mg)
- Phosphorus (mg)

### Nutrition Endpoints

**Individual Ingredient Nutrition:**
```bash
curl -X POST "http://your-render-url/nutrition" \
     -H "Content-Type: application/json" \
     -d '["apple", "banana", "orange"]'
```

**Complete Analysis with Nutrition:**
```bash
curl -X POST "http://your-render-url/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@food_image.jpg"
```

### Nutrition Response Format

```json
{
  "nutrition_analysis": {
    "ingredients_nutrition": [
      {
        "food_name": "apple",
        "serving_qty": 1,
        "serving_unit": "medium",
        "serving_weight_grams": 182,
        "calories": 94.6,
        "protein": 0.47,
        "total_fat": 0.31,
        "saturated_fat": 0.05,
        "total_carbohydrate": 25.13,
        "dietary_fiber": 4.37,
        "sugars": 18.91,
        "cholesterol": 0,
        "sodium": 1.82,
        "potassium": 194.74,
        "vitamin_c": 8.37,
        "vitamin_a": 98.28,
        "calcium": 10.92,
        "iron": 0.22,
        "source": "nutritionix",
        "success": true
      }
    ],
    "total_nutrition": {
      "total_calories": 94.6,
      "total_protein": 0.47,
      "total_fat": 0.31,
      "total_saturated_fat": 0.05,
      "total_carbohydrate": 25.13,
      "total_fiber": 4.37,
      "total_sugars": 18.91,
      "total_cholesterol": 0,
      "total_sodium": 1.82,
      "total_potassium": 194.74,
      "total_vitamin_c": 8.37,
      "total_vitamin_a": 98.28,
      "total_calcium": 10.92,
      "total_iron": 0.22,
      "ingredients_count": 1,
      "successful_lookups": 1
    },
    "summary": {
      "total_ingredients": 1,
      "successful_lookups": 1,
      "total_calories": 94.6,
      "main_macros": {
        "protein_g": 0.47,
        "carbs_g": 25.13,
        "fat_g": 0.31,
        "fiber_g": 4.37,
        "sugars_g": 18.91
      }
    }
  }
}
```

## Response Format

### Food Detection Response
```json
{
    "is_food": true,
    "confidence": 0.95
}
```

### Food Classification Response
```json
{
    "dish_name": "pizza",
    "confidence": 0.87,
    "top_predictions": [
        {"dish_name": "pizza", "confidence": 0.87},
        {"dish_name": "lasagna", "confidence": 0.08},
        {"dish_name": "cheesecake", "confidence": 0.03}
    ]
}
```

### Segmentation Response
```json
{
    "segmented_image": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "ingredients_detected": ["cheese", "tomato", "dough"],
    "ingredient_confidences": [
        {"cheese": 0.95},
        {"tomato": 0.87},
        {"dough": 0.92}
    ],
    "nutrition_data": {
        "ingredients_nutrition": [...],
        "total_nutrition": {...}
    }
}
```

### Combined Analysis Response
```json
{
    "food_detection": {
        "is_food": true,
        "confidence": 0.95
    },
    "food_classification": {
        "dish_name": "pizza",
        "confidence": 0.87,
        "top_predictions": [...]
    },
    "segmentation": {
        "segmented_image": "data:image/jpeg;base64,...",
        "ingredients_detected": ["cheese", "tomato", "dough"],
        "ingredient_confidences": [...],
        "nutrition_data": {...}
    },
    "gemini_analysis": {
        "ingredients": [...],
        "dish_type": "pizza",
        "cooking_method": "baked"
    },
    "nutrition_analysis": {
        "ingredients_nutrition": [...],
        "total_nutrition": {...},
        "summary": {...}
    }
}
```

## Performance Considerations

- **Model Loading**: Models are loaded once at startup for optimal performance
- **Image Processing**: Images are automatically resized to appropriate dimensions
- **Memory Usage**: Models are loaded in CPU mode to reduce memory requirements
- **Response Time**: Typical response time is 1-3 seconds depending on image size
- **API Rate Limits**: Nutritionix API has rate limits; consider caching for production

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure all model files are in the correct directory
   - Check file permissions
   - Verify model file integrity

2. **Memory Issues**:
   - Reduce image size before sending
   - Consider using smaller model variants
   - Monitor memory usage in deployment

3. **Nutritionix API Errors**:
   - Verify API credentials are correct
   - Check API rate limits
   - Ensure ingredient names are in English

4. **Timeout Errors**:
   - Increase timeout settings in your client
   - Optimize image preprocessing
   - Consider async processing for large images

### Logs

Check application logs for detailed error information:
```bash
# Local development
tail -f app.log

# Docker
docker logs <container-id>

# Render
# Check logs in Render dashboard
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the API documentation at `/docs`