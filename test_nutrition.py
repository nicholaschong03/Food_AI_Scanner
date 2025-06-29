#!/usr/bin/env python3
"""
Test script for nutrition analysis functionality
"""

import requests
import json
import os
from typing import List, Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your API URL

def test_nutrition_endpoint():
    """Test the nutrition analysis endpoint"""
    print("🧬 Testing Nutrition Analysis Endpoint")
    print("=" * 50)

    # Test ingredients
    test_ingredients = ["apple", "banana", "chicken breast", "rice"]

    try:
        response = requests.post(
            f"{API_BASE_URL}/nutrition",
            json=test_ingredients,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Nutrition analysis successful!")
            print(f"📊 Ingredients analyzed: {len(test_ingredients)}")

            nutrition_analysis = result.get("nutrition_analysis", {})
            summary = nutrition_analysis.get("summary", {})

            print(f"🎯 Successful lookups: {summary.get('successful_lookups', 0)}")
            print(f"🔥 Total calories: {summary.get('total_calories', 0)}")

            main_macros = summary.get("main_macros", {})
            print(f"🥩 Protein: {main_macros.get('protein_g', 0)}g")
            print(f"🍞 Carbs: {main_macros.get('carbs_g', 0)}g")
            print(f"🧈 Fat: {main_macros.get('fat_g', 0)}g")
            print(f"🌾 Fiber: {main_macros.get('fiber_g', 0)}g")
            print(f"🍯 Sugars: {main_macros.get('sugars_g', 0)}g")

            # Show individual ingredient nutrition
            ingredients_nutrition = nutrition_analysis.get("ingredients_nutrition", [])
            print("\n📋 Individual Ingredient Nutrition:")
            for i, nutrition in enumerate(ingredients_nutrition):
                if nutrition.get("success", False):
                    print(f"  {i+1}. {nutrition['food_name']}: {nutrition['calories']} cal, "
                          f"{nutrition['protein']}g protein, {nutrition['total_carbohydrate']}g carbs")
                else:
                    print(f"  {i+1}. {nutrition.get('food_name', 'Unknown')}: {nutrition.get('error', 'Failed to get nutrition')}")

        else:
            print(f"❌ Nutrition analysis failed with status {response.status_code}")
            print(f"Error: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def test_health_check():
    """Test the health check endpoint"""
    print("\n🏥 Testing Health Check")
    print("=" * 30)

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful!")
            print(f"📊 Status: {result.get('status', 'unknown')}")
            print(f"🔧 Models loaded: {result.get('models_loaded', 'unknown')}")
            print(f"🧬 Nutritionix configured: {result.get('nutritionix_configured', 'unknown')}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {str(e)}")

def test_nutritionix_configuration():
    """Test if Nutritionix API is properly configured"""
    print("\n🔧 Testing Nutritionix Configuration")
    print("=" * 40)

    # Check environment variables
    app_id = os.getenv('NUTRITIONIX_APP_ID')
    app_key = os.getenv('NUTRITIONIX_APP_KEY')

    if app_id and app_key:
        print("✅ Nutritionix API credentials found in environment")
        print(f"📝 App ID: {app_id[:8]}...")
        print(f"🔑 App Key: {app_key[:8]}...")
    else:
        print("❌ Nutritionix API credentials not found")
        print("💡 Please set NUTRITIONIX_APP_ID and NUTRITIONIX_APP_KEY environment variables")
        print("   or add them to your .env file")

def test_single_ingredient():
    """Test nutrition analysis for a single ingredient"""
    print("\n🍎 Testing Single Ingredient Analysis")
    print("=" * 40)

    try:
        response = requests.post(
            f"{API_BASE_URL}/nutrition",
            json=["grape"],
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            nutrition_analysis = result.get("nutrition_analysis", {})
            ingredients_nutrition = nutrition_analysis.get("ingredients_nutrition", [])

            if ingredients_nutrition and ingredients_nutrition[0].get("success", False):
                nutrition = ingredients_nutrition[0]
                print("✅ Single ingredient analysis successful!")
                print(f"🍇 Food: {nutrition['food_name']}")
                print(f"⚖️  Serving: {nutrition['serving_qty']} {nutrition['serving_unit']}")
                print(f"🔥 Calories: {nutrition['calories']}")
                print(f"🥩 Protein: {nutrition['protein']}g")
                print(f"🍞 Carbs: {nutrition['total_carbohydrate']}g")
                print(f"🧈 Fat: {nutrition['total_fat']}g")
                print(f"🌾 Fiber: {nutrition['dietary_fiber']}g")
                print(f"🍯 Sugars: {nutrition['sugars']}g")
                print(f"🧂 Sodium: {nutrition['sodium']}mg")
                print(f"🥔 Potassium: {nutrition['potassium']}mg")

                # Show vitamins if available
                if nutrition.get('vitamin_c') is not None:
                    print(f"🍊 Vitamin C: {nutrition['vitamin_c']}mg")
                if nutrition.get('vitamin_a') is not None:
                    print(f"🥕 Vitamin A: {nutrition['vitamin_a']}IU")
                if nutrition.get('calcium') is not None:
                    print(f"🥛 Calcium: {nutrition['calcium']}mg")
                if nutrition.get('iron') is not None:
                    print(f"🔩 Iron: {nutrition['iron']}mg")
            else:
                print("❌ Single ingredient analysis failed")
                if ingredients_nutrition:
                    print(f"Error: {ingredients_nutrition[0].get('error', 'Unknown error')}")
        else:
            print(f"❌ Single ingredient analysis failed with status {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")

def main():
    """Run all nutrition tests"""
    print("🧬 Food AI Scanner - Nutrition Analysis Tests")
    print("=" * 60)

    # Test health check first
    test_health_check()

    # Test Nutritionix configuration
    test_nutritionix_configuration()

    # Test single ingredient analysis
    test_single_ingredient()

    # Test multiple ingredients analysis
    test_nutrition_endpoint()

    print("\n" + "=" * 60)
    print("🎉 Nutrition analysis tests completed!")

if __name__ == "__main__":
    main()