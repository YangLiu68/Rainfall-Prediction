"""
NLP service for parsing user queries about weather and rainfall predictions.

This module uses Gemini AI to understand natural language queries and extract:
- Location (city name and coordinates)
- Time reference (next hour, previous date, specific time)
- Query intent (prediction, general info, location change)
"""

import os
import json
import re
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class NLPParseError(Exception):
    """Raised when query cannot be parsed."""
    pass


async def parse_user_query(message: str, current_location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse user's natural language query to extract intent, location, and time.
    
    Args:
        message: User's natural language message
        current_location: Current location context (city, lat, lon) if available
        
    Returns:
        Dictionary with:
        - intent: "prediction" | "info" | "location_change" | "greeting" | "help"
        - location: {"city": str, "latitude": float, "longitude": float} or None
        - time_offset_hours: int (positive for future, negative for past, 0 for now)
        - original_message: str
        
    Examples:
        "What's the weather in Tokyo?" -> intent: prediction, location: Tokyo, time: 0
        "Will it rain tomorrow in Paris?" -> intent: prediction, location: Paris, time: 24
        "Show me yesterday's weather" -> intent: prediction, location: current, time: -24
        "Change location to London" -> intent: location_change, location: London
    """
    if not GEMINI_API_KEY:
        raise NLPParseError("No GEMINI_API_KEY configured")
    
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    # Build context about current location
    location_context = ""
    if current_location:
        location_context = f"\nCurrent location context: {current_location['city']} ({current_location['latitude']}, {current_location['longitude']})"
    
    prompt = f"""Parse this weather query and extract structured information.
{location_context}

Query: "{message}"

Respond with ONLY a JSON object in this exact format:
{{
  "intent": "prediction" | "info" | "location_change" | "greeting" | "help",
  "location": {{"city": "City Name", "latitude": 00.0000, "longitude": -00.0000}} or null,
  "time_offset_hours": 0,
  "needs_location": true | false
}}

Intent types:
- "prediction": User wants rainfall/weather prediction for a specific location (e.g., "Will it rain in Tokyo?", "Weather in Paris")
- "info": General weather knowledge question that doesn't need prediction (e.g., "What is rainfall?", "How does rain form?", "What causes humidity?")
- "location_change": User explicitly wants to change location
- "greeting": Hello, hi, hey
- "help": User needs help or instructions

IMPORTANT: Only use "prediction" if the user is asking for actual weather/rainfall data for a location. Use "info" for general knowledge questions about weather concepts.

Location rules:
- Extract city name and coordinates if mentioned in query
- If no location mentioned but current location exists, set location to null and needs_location to false
- If no location mentioned and no current location, set location to null and needs_location to true

Time offset rules:
- "next hour" or "in an hour" = 1
- "tomorrow" = 24
- "yesterday" = -24
- "in 2 days" = 48
- "2 days ago" = -48
- "now" or no time mentioned = 0
- Extract any relative time reference and convert to hours

Respond with ONLY the JSON object, no other text."""

    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Extract JSON from response
    json_text = _extract_json_from_response(response_text)
    data = json.loads(json_text)
    
    # Validate and structure the response
    result = {
        "intent": data.get("intent", "prediction"),
        "location": data.get("location"),
        "time_offset_hours": int(data.get("time_offset_hours", 0)),
        "needs_location": data.get("needs_location", False),
        "original_message": message
    }
    
    return result


def _extract_json_from_response(response_text: str) -> str:
    """Extract JSON from Gemini response, handling markdown code blocks."""
    # Remove markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find JSON object directly
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return response_text


def calculate_target_time(time_offset_hours: int, timezone_str: str = "UTC") -> datetime:
    """
    Calculate target datetime based on offset from current time.
    
    Args:
        time_offset_hours: Hours offset (positive=future, negative=past)
        timezone_str: Timezone string (e.g., "America/New_York")
        
    Returns:
        Target datetime in specified timezone
    """
    tz = ZoneInfo(timezone_str)
    current_time = datetime.now(tz)
    target_time = current_time + timedelta(hours=time_offset_hours)
    return target_time
