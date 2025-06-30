# Railway Deployment Guide

## Prerequisites
- Railway account
- Git repository with your code
- Environment variables configured

## Environment Variables Required

Set these in your Railway project settings:

### Required:
```
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO
MAX_IMAGE_SIZE=10485760
IMAGE_TIMEOUT=30
MODEL_DEVICE=cpu
PLAN=free_tier
REGION=singapore
CLASSIFICATION_THRESHOLD=0.5
SEGMENTATION_THRESHOLD=0.5
ALLOWED_ORIGINS=*
```

### Optional (for enhanced features):
```
# Nutritionix API - for nutrition analysis
NUTRITIONIX_APP_ID=your_nutritionix_app_id_here
NUTRITIONIX_APP_KEY=your_nutritionix_app_key_here

# Google Gemini API - for enhanced analysis
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Deployment Steps

1. **Connect your GitHub repository to Railway**
2. **Set environment variables** in Railway dashboard
3. **Deploy** - Railway will automatically detect the FastAPI app
4. **Monitor** the deployment logs for any issues

## Health Check
Your app includes a health check endpoint at `/health` that Railway will use to verify the service is running.

## API Endpoints
- `GET /health` - Health check
- `POST /detect-food` - Food detection
- `POST /classify-food` - Food classification
- `POST /segment` - Ingredient segmentation
- `POST /analyze` - Complete analysis
- `POST /nutrition` - Nutrition analysis
- `POST /chat` - AI chat about food

## Model Files
Make sure your `models/` directory contains:
- `food_nonfood_mnv2.h5`
- `food_dish_classifier_efficientnetb2_combined_latest.h5`
- `best_ingredient_seg.pt`
- `labels_combined.json`

## Troubleshooting
- Check Railway logs for any model loading errors
- Verify all environment variables are set
- Ensure model files are included in your repository