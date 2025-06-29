#!/usr/bin/env python3
"""
Test script to verify model loading works correctly after the batch_shape fix
"""

import sys
import os
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if models can be loaded successfully"""
    try:
        # Import the app module
        from app import load_models, food_nonfood_model, food_classifier_model, segmentation_model, food_labels

        logger.info("Testing model loading...")

        # Load models
        load_models()

        # Check if models are loaded
        if food_nonfood_model is not None:
            logger.info("✓ Food/non-food model loaded successfully")
        else:
            logger.error("✗ Food/non-food model failed to load")
            return False

        if food_classifier_model is not None:
            logger.info("✓ Food classifier model loaded successfully")
        else:
            logger.error("✗ Food classifier model failed to load")
            return False

        if segmentation_model is not None:
            logger.info("✓ Segmentation model loaded successfully")
        else:
            logger.error("✗ Segmentation model failed to load")
            return False

        if food_labels is not None:
            logger.info("✓ Food labels loaded successfully")
        else:
            logger.error("✗ Food labels failed to load")
            return False

        logger.info("✓ All models loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"✗ Model loading test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Model loading test FAILED!")
        sys.exit(1)