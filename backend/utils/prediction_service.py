"""
Prediction service for rainfall forecasting using the trained model.
"""

import torch
import torch.nn as nn
from typing import List


class MyRainModel(nn.Module):
    """
    Rain prediction model architecture using Transformer.
    Must match the architecture used during training.
    """
    def __init__(self, input_size=12, d_model=256, nhead=4, num_layers=4, seq_len=24):
        super(MyRainModel, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),         # head.0
            nn.Linear(d_model, d_model),   # head.1
            nn.ReLU(),                     # head.2
            nn.Dropout(0.1),               # head.3
            nn.Linear(d_model, 1)          # head.4
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.encoder(x)
        # Take mean across sequence
        x = x.mean(dim=1)
        # Predict log1p(rainfall)
        prediction = self.head(x)
        return prediction


def load_model(checkpoint_path: str = "checkpoint_best.pt") -> MyRainModel:
    """
    Load the trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
    
    Returns:
        Loaded model in evaluation mode
    """
    model = MyRainModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_rainfall(model: MyRainModel, weather_sequence: List[List[float]]) -> dict:
    """
    Predict rainfall from weather sequence.
    
    Args:
        model: Trained rainfall prediction model
        weather_sequence: 2D list of shape (24, 12) with weather features
    
    Returns:
        Dictionary with prediction results:
        - rain_mm: Predicted rainfall in mm/h
        - rain_log: Log-transformed prediction value
        - chance_of_rain: Probability of rain (0-100%)
    """
    # Convert to tensor: shape (1, 24, 12)
    x = torch.tensor(weather_sequence, dtype=torch.float32).unsqueeze(0)
    
    # Run prediction (no gradient needed)
    with torch.no_grad():
        y_log = model(x).item()
    
    # Convert log prediction to actual rainfall (mm/h)
    rain_mm = torch.expm1(torch.tensor(y_log)).item()
    
    # Calculate chance of rain (%)
    # Cap at 100% for values above threshold
    chance_of_rain = min(rain_mm * 20, 100)
    
    return {
        "rain_mm": round(rain_mm, 3),
        "rain_log": round(y_log, 3),
        "chance_of_rain": round(chance_of_rain, 1)
    }
