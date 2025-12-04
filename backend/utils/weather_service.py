"""
Weather data service for fetching historical weather data from Open-Meteo API.
"""

import httpx
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from typing import Optional
from timezonefinder import TimezoneFinder

# Feature normalization constants (mean and std for each feature)
# Order: temperature_2m, relative_humidity_2m, surface_pressure, 
#        wind_speed_10m, wind_direction_10m, cloud_cover,
#        hour, month, hour_sin, hour_cos, month_sin, month_cos
FEATURE_MEAN = np.array([
    15.0,   # temperature_2m (°C)
    60.0,   # relative_humidity_2m (%)
    1013.0, # surface_pressure (hPa)
    10.0,   # wind_speed_10m (km/h)
    180.0,  # wind_direction_10m (degrees)
    50.0,   # cloud_cover (%)
    11.5,   # hour (0-23)
    6.5,    # month (1-12)
    0.0,    # hour_sin
    0.0,    # hour_cos
    0.0,    # month_sin
    0.0     # month_cos
])

FEATURE_STD = np.array([
    10.0,   # temperature_2m
    25.0,   # relative_humidity_2m
    10.0,   # surface_pressure
    5.0,    # wind_speed_10m
    100.0,  # wind_direction_10m
    35.0,   # cloud_cover
    6.9,    # hour
    3.5,    # month
    0.7,    # hour_sin
    0.7,    # hour_cos
    0.7,    # month_sin
    0.7     # month_cos
])


async def fetch_weather_data(
    latitude: float,
    longitude: float,
    end_time: datetime = None,
    hours: int = 24
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo Historical Weather API.
    
    Args:
        latitude: Location latitude (-90 to 90)
        longitude: Location longitude (-180 to 180)
        end_time: End of the time window (if None, uses current time in location's timezone)
        hours: Number of hours to fetch (default: 24)
    
    Returns:
        DataFrame with columns: time, temperature_2m, relative_humidity_2m,
        surface_pressure, wind_speed_10m, wind_direction_10m, cloud_cover
    
    Raises:
        httpx.HTTPError: If the API request fails
        ValueError: If the response contains insufficient data
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    # If no end_time provided, get current time in the location's timezone
    if end_time is None:
        # Find the timezone for the given coordinates
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
        
        if timezone_str:
            # Get current time in the location's timezone
            location_tz = ZoneInfo(timezone_str)
            end_time = datetime.now(location_tz)
        else:
            # Fallback to UTC if timezone cannot be determined
            end_time = datetime.now(ZoneInfo("UTC"))
    
    # Calculate start time (hours before end_time)
    start_time = end_time - timedelta(hours=hours)
    
    # Format dates for API (YYYY-MM-DD)
    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")
    
    # Determine which API to use based on the date
    # Archive API only has data up to ~5 days ago
    # For recent/current data, use the forecast API
    now_utc = datetime.now(ZoneInfo("UTC"))
    days_ago = (now_utc.date() - end_time.date()).days
    
    if days_ago > 5:
        # Use historical archive API for older data
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        # Use forecast API for recent/current data (includes historical forecast data)
        url = "https://api.open-meteo.com/v1/forecast"
    
    # Required weather parameters (Requirements 2.1)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover"
        ]),
        "timezone": "auto"
    }
    
    # Add date parameters based on API type
    if days_ago > 5:
        # Archive API uses start_date and end_date
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        # Forecast API uses past_days and forecast_days
        # Calculate how many past days we need
        past_days = max(0, (now_utc.date() - start_time.date()).days)
        params["past_days"] = min(past_days + 1, 92)  # API limit is 92 days
        params["forecast_days"] = 1  # Include today
    
    # Make async HTTP request
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise httpx.HTTPError(f"Open-Meteo API request failed: {str(e)}")
    
    # Parse JSON response
    data = response.json()
    
    # Extract hourly data
    if "hourly" not in data:
        raise ValueError("API response missing 'hourly' data")
    
    hourly = data["hourly"]
    
    # Validate all required fields are present
    required_fields = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover"
    ]
    
    for field in required_fields:
        if field not in hourly:
            raise ValueError(f"API response missing required field: {field}")
    
    # Create DataFrame
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "relative_humidity_2m": hourly["relative_humidity_2m"],
        "surface_pressure": hourly["surface_pressure"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "wind_direction_10m": hourly["wind_direction_10m"],
        "cloud_cover": hourly["cloud_cover"]
    })
    
    # Convert start_time and end_time to timezone-naive for comparison
    # (Open-Meteo returns timezone-naive timestamps in local time)
    start_time_naive = start_time.replace(tzinfo=None) if hasattr(start_time, 'tzinfo') else start_time
    end_time_naive = end_time.replace(tzinfo=None) if hasattr(end_time, 'tzinfo') else end_time
    
    # Filter to exact time window (end_time - hours to end_time)
    df = df[
        (df["time"] > start_time_naive - timedelta(hours=1)) &
        (df["time"] <= end_time_naive)
    ]
    
    # Validate we have enough data (Requirements 2.5)
    # Allow some flexibility - require at least 80% of requested hours
    min_required_hours = int(hours * 0.8)
    if len(df) < min_required_hours:
        raise ValueError(
            f"Insufficient weather data: expected at least {min_required_hours} hours, got {len(df)} hours"
        )
    
    # Sort by time to ensure chronological order (oldest to newest)
    df = df.sort_values('time').reset_index(drop=True)
    
    # Take the last 'hours' rows if we have more than requested
    if len(df) > hours:
        df = df.tail(hours).reset_index(drop=True)
    elif len(df) < hours:
        # If we have fewer hours, pad with the last available values
        # This ensures the model gets the expected input shape
        rows_to_add = hours - len(df)
        last_row = df.iloc[-1:].copy()
        for i in range(rows_to_add):
            # Increment time for each padded row
            padded_row = last_row.copy()
            padded_row['time'] = last_row['time'].iloc[0] + timedelta(hours=i+1)
            df = pd.concat([df, padded_row], ignore_index=True)
    
    # Add time-based features for the model
    t = pd.to_datetime(df["time"], errors="coerce")
    df["hour"] = t.dt.hour.fillna(0).astype(int)
    df["month"] = t.dt.month.fillna(1).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Convert to 2D list format for JSON serialization and model input
    # Format: [[feat1_hour_0, feat2_hour_0, ..., featF_hour_0], ...]
    # Each row represents one hour (from oldest to newest)
    # Each column represents one feature
    # Feature order: temperature_2m, relative_humidity_2m, surface_pressure, 
    #                wind_speed_10m, wind_direction_10m, cloud_cover,
    #                hour, month, hour_sin, hour_cos, month_sin, month_cos
    feature_columns = [
        "temperature_2m",
        "relative_humidity_2m", 
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
        "hour",
        "month",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos"
    ]
    
    # Extract feature values as numpy array
    weather_data_array = df[feature_columns].values
    
    # Normalize the data: (x - mean) / (std + epsilon)
    weather_data_normalized = (weather_data_array - FEATURE_MEAN) / (FEATURE_STD + 1e-6)
    
    # Convert to 2D list (JSON-serializable)
    weather_data_2d = weather_data_normalized.tolist()
    
    return weather_data_2d, df
