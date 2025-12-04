"""
Location extraction service using Gemini AI.

This module provides functionality to extract city names and coordinates
from natural language user messages using Google's Gemini AI.
"""

import os
import json
import re
from typing import Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class LocationExtractionError(Exception):
    """Raised when location cannot be extracted from message."""
    pass


async def extract_location_with_gemini(message: str) -> Tuple[str, float, float]:
    """
    Extract city name and coordinates from user message using Gemini AI.
    
    Args:
        message: User's natural language message about weather
        
    Returns:
        Tuple of (city_name, latitude, longitude)
        
    Requirements: 1.1, 1.2
    """
    if not GEMINI_API_KEY:
        raise LocationExtractionError("No GEMINI_API_KEY configured")
    
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    prompt = f"""Extract the city name from this message and provide its coordinates.
Respond with ONLY a JSON object in this format:
{{"city": "City Name", "latitude": 00.0000, "longitude": -00.0000}}

Message: "{message}"

Respond with ONLY the JSON object, no other text."""

    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Extract JSON from response (handle markdown code blocks)
    json_text = _extract_json_from_response(response_text)
    
    data = json.loads(json_text)
    
    city = data["city"]
    lat = float(data["latitude"])
    lon = float(data["longitude"])
    
    return (city, lat, lon)


def _extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON from Gemini response, handling markdown code blocks.
    
    Args:
        response_text: Raw response from Gemini
        
    Returns:
        Clean JSON string
    """
    # Remove markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find JSON object directly
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return response_text
