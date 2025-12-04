"""
Test script to verify location extraction and weather service integration.
"""

import asyncio
import json
from utils.location_service import extract_location_with_gemini
from utils.weather_service import fetch_weather_data
from utils.prediction_service import load_model, predict_rainfall


async def test_full_pipeline(user_message: str, model):
    """Test the full pipeline: extract location -> fetch weather data -> predict."""
    
    print(f"User message: '{user_message}'")
    print("=" * 80)
    
    try:
        # Step 1: Extract location from message
        print("\n[1] Extracting location from message...")
        city, latitude, longitude = await extract_location_with_gemini(user_message)
        print(f"✓ Location extracted: {city}")
        print(f"  Coordinates: ({latitude}, {longitude})")
        
        # Step 2: Fetch weather data (automatically uses location's timezone)
        print(f"\n[2] Fetching weather data for {city}...")
        weather_data_2d, df = await fetch_weather_data(latitude, longitude)
        
        print(f"✓ Successfully fetched {len(df)} hours of weather data")
        print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"  Data shape: {len(weather_data_2d)} timesteps × {len(weather_data_2d[0])} features")
        
        # Step 3: Run prediction
        print(f"\n[3] Running rainfall prediction...")
        prediction = predict_rainfall(model, weather_data_2d)
        
        print(f"\n{'='*80}")
        print("PREDICTION RESULT:")
        print('='*80)
        print(f"  Rainfall:        {prediction['rain_mm']} mm/h")
        print(f"  Chance of rain:  {prediction['chance_of_rain']}%")
        print(f"  Log value:       {prediction['rain_log']}")
        
        # Interpretation
        if prediction['rain_mm'] < 0.1:
            status = "No rain expected"
        elif prediction['rain_mm'] < 1.0:
            status = "Light rain possible"
        elif prediction['rain_mm'] < 5.0:
            status = "Moderate rain expected"
        else:
            status = "Heavy rain expected"
        
        print(f"  Status:          {status}")
        print('='*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run tests with different user messages."""
    
    print("\n" + "="*80)
    print("RAINFALL PREDICTION API TEST")
    print("="*80)
    
    # Load model once
    print("\nLoading trained model from checkpoint_best.pt...")
    model = load_model("checkpoint_best.pt")
    print("✓ Model loaded successfully\n")
    
    test_messages = [
        "Will it rain in Delhi next hour?",
        "What's the weather in Mumbai?",
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}/{len(test_messages)}")
        print(f"{'=' * 80}")
        await test_full_pipeline(message, model)
    
    print(f"\n{'=' * 80}")
    print("ALL TESTS COMPLETE")
    print('='*80 + "\n")
        


if __name__ == "__main__":
    asyncio.run(main())
