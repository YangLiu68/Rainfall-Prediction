import pandas as pd
import torch
import numpy as np
from daily_dataset import DailyWeatherDataset
from train_daily_model import RainfallTransformer, get_device


MODEL_PATH = "daily_transformer_best.pt"


def load_model_and_daily_ds():
    device = get_device()
    daily_ds = DailyWeatherDataset(
        csv_path="./dataset_global/weather_daily_global.csv",
        cities=("San Francisco", "New York"),
        date_col="time",
    )

    seq_len = 30
    input_dim = len(daily_ds.feature_cols)

    model = RainfallTransformer(input_dim=input_dim, seq_len=seq_len).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, device, daily_ds, seq_len


def build_input_by_city_date(daily_ds, city: str, target_date: pd.Timestamp, lookback: int):
    df = daily_ds.df
    df_city = df[df["city"] == city].sort_values("date").reset_index(drop=True)

    if target_date not in df_city["date"].values:
        raise ValueError(f"在 {city} 的历史数据里找不到日期 {target_date.date()}")

    idx_list = df_city.index[df_city["date"] == target_date].tolist()
    idx = idx_list[0]

    if idx < lookback:
        raise ValueError(f"{city} 在 {target_date.date()} 之前不足 {lookback} 天历史记录")

    start = idx - lookback
    end = idx

    window = df_city.loc[start:end-1, daily_ds.feature_cols].values.astype("float32")  # [30, 18]
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # [1, 30, 18]
    return x


def predict_by_city_date(city: str, date_str: str):
    model, device, daily_ds, lookback = load_model_and_daily_ds()
    target_date = pd.Timestamp(date_str)

    x = build_input_by_city_date(daily_ds, city, target_date, lookback).to(device)

    with torch.no_grad():
        pred_log = model(x)[0].item()
        pred_mm = float(np.expm1(pred_log))

    print(f"📅 {city} {date_str} 预测降雨量: {pred_mm:.3f} mm")


if __name__ == "__main__":
    predict_by_city_date("San Francisco", "2019-01-15")
