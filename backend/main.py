from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from timezonefinder import TimezoneFinder

from utils.nlp_service import parse_user_query, calculate_target_time
from utils.weather_service import fetch_weather_data
from utils.prediction_service import load_model, predict_rainfall
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Weather Prediction Chat Backend")

# Configure CORS for localhost origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model("checkpoint_best.pt")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load model: {e}")


class ChatRequest(BaseModel):
    message: str
    current_location: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    location: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None


@app.get("/")
async def root():
    return {"message": "Weather Prediction Chat Backend API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for weather predictions with NLP parsing.
    
    Handles natural language queries like:
    - "What's the weather in Tokyo?"
    - "Will it rain tomorrow in Paris?"
    - "Show me yesterday's weather in London"
    """
    try:
        # Parse user query with NLP
        parsed = await parse_user_query(request.message, request.current_location)
        
        intent = parsed["intent"]
        location = parsed["location"]
        time_offset = parsed["time_offset_hours"]
        needs_location = parsed["needs_location"]
        
        # Handle different intents
        if intent == "greeting":
            return ChatResponse(
                response="Hello! I'm your rainfall prediction assistant. Ask me about weather and rainfall for any city, including future or past predictions. For example: 'Will it rain tomorrow in Tokyo?' or 'What was the weather yesterday in Paris?'"
            )
        
        if intent == "help":
            return ChatResponse(
                response="I can help you with:\n\n• Rainfall predictions for any city\n• Weather forecasts (next hour, tomorrow, etc.)\n• Historical weather data (yesterday, last week, etc.)\n• Location-based predictions\n\nJust ask naturally! Examples:\n- 'Will it rain in London tomorrow?'\n- 'What's the weather in Tokyo?'\n- 'Show me yesterday's rainfall in Paris'"
            )
        
        if intent == "info":
            # Handle general weather information questions with Gemini
            return await _handle_info_query(request.message)
        
        # Check if we need location
        if needs_location and not location:
            return ChatResponse(
                response="I'd be happy to help with weather predictions! Which city would you like to know about?"
            )
        
        # Use current location if no new location specified
        if not location and request.current_location:
            location = request.current_location
        
        if not location:
            return ChatResponse(
                response="Please specify a city for the weather prediction. For example: 'What's the weather in New York?'"
            )
        
        # Get timezone for the location
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=location["latitude"], lng=location["longitude"])
        if not timezone_str:
            timezone_str = "UTC"
        
        # Calculate target time
        target_time = calculate_target_time(time_offset, timezone_str)
        
        # Check if model is loaded
        if model is None:
            return ChatResponse(
                response="Sorry, the prediction model is not available at the moment. Please try again later.",
                location=location
            )
        
        # Fetch weather data
        try:
            weather_data, weather_df = await fetch_weather_data(
                latitude=location["latitude"],
                longitude=location["longitude"],
                end_time=target_time,
                hours=24
            )
        except Exception as e:
            return ChatResponse(
                response=f"Sorry, I couldn't fetch weather data for {location['city']}. The location might not have historical data available, or the requested time period is too far in the past. Error: {str(e)}",
                location=location
            )
        
        # Make prediction
        prediction = predict_rainfall(model, weather_data)
        
        # Format response based on time context
        time_context = _format_time_context(time_offset, target_time)
        response_text = _format_prediction_response(
            location["city"],
            prediction,
            time_context,
            weather_df
        )
        
        return ChatResponse(
            response=response_text,
            location=location,
            prediction=prediction
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return ChatResponse(
            response=f"Sorry, I encountered an error processing your request: {str(e)}"
        )


def _format_time_context(time_offset: int, target_time: datetime) -> str:
    """Format time context for response."""
    if time_offset == 0:
        return "for the next hour"
    elif time_offset > 0:
        if time_offset == 1:
            return "in the next hour"
        elif time_offset <= 24:
            return f"in the next {time_offset} hours"
        else:
            days = time_offset // 24
            return f"in {days} day{'s' if days > 1 else ''}"
    else:
        abs_offset = abs(time_offset)
        if abs_offset == 1:
            return "an hour ago"
        elif abs_offset <= 24:
            return f"{abs_offset} hours ago"
        else:
            days = abs_offset // 24
            return f"{days} day{'s' if days > 1 else ''} ago"


async def _handle_info_query(message: str) -> ChatResponse:
    """Handle general weather information questions using Gemini."""
    if not GEMINI_API_KEY:
        return ChatResponse(
            response="I can answer general weather questions, but I need a Gemini API key configured."
        )
    
    try:
        info_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""You are a helpful weather assistant. Answer this weather-related question in a clear, conversational way.
Keep your response concise (2-4 sentences) and informative.

Question: {message}

Provide a friendly, accurate answer about weather concepts, meteorology, or general weather information."""

        response = info_model.generate_content(prompt)
        return ChatResponse(response=response.text.strip())
    
    except Exception as e:
        print(f"Error in _handle_info_query: {e}")
        return ChatResponse(
            response=f"I'm having trouble answering that right now. Error: {str(e)}"
        )


def _format_prediction_response(
    city: str,
    prediction: Dict[str, Any],
    time_context: str,
    weather_df: Any
) -> str:
    """Format the prediction into a natural language response."""
    rain_mm = prediction["rain_mm"]
    chance = prediction["chance_of_rain"]
    
    # Get latest weather conditions
    latest = weather_df.iloc[-1]
    temp = latest["temperature_2m"]
    humidity = latest["relative_humidity_2m"]
    wind = latest["wind_speed_10m"]
    
    # Build response
    response = f"**Weather Prediction for {city}** {time_context}\n\n"
    
    # Rainfall prediction
    if chance >= 70:
        response += f"🌧️ **High chance of rain** ({chance}%)\n"
        response += f"Expected rainfall: {rain_mm:.2f} mm/h\n\n"
    elif chance >= 40:
        response += f"🌦️ **Moderate chance of rain** ({chance}%)\n"
        response += f"Expected rainfall: {rain_mm:.2f} mm/h\n\n"
    elif chance >= 20:
        response += f"⛅ **Low chance of rain** ({chance}%)\n"
        response += f"Expected rainfall: {rain_mm:.2f} mm/h\n\n"
    else:
        response += f"☀️ **Very low chance of rain** ({chance}%)\n"
        response += f"Expected rainfall: {rain_mm:.2f} mm/h\n\n"
    
    # Weather conditions
    response += f"**Current Conditions:**\n"
    response += f"• Temperature: {temp:.1f}°C\n"
    response += f"• Humidity: {humidity:.0f}%\n"
    response += f"• Wind Speed: {wind:.1f} km/h\n"
    
    # Recommendation
    if chance >= 60:
        response += f"\n💡 Recommendation: Bring an umbrella!"
    elif chance >= 30:
        response += f"\n💡 Recommendation: You might want to carry an umbrella just in case."
    
    return response
