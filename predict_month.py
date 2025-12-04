import torch
import numpy as np
import pandas as pd

from datetime import datetime
from daily_dataset import DailyWeatherDataset, DailySequenceDataset
from train_daily_model import RainfallTransformer, get_device


MODEL_PATH = "daily_transformer_best.pt"


# =========================================================
# 1. 加载模型 + 数据
# =========================================================
def load_model_and_data():
    """
    加载：
      - 日级数据集（SF + NY）
      - 训练好的 Transformer 模型
    """
    device = get_device()

    daily_ds = DailyWeatherDataset(
        csv_path="./dataset_global/weather_daily_global.csv",
        cities=("San Francisco", "New York"),
        date_col="time",
    )
    seq_len = 30
    _ = DailySequenceDataset(daily_ds, lookback=seq_len)  # 主要是为了确认数据没问题

    input_dim = len(daily_ds.feature_cols)
    model = RainfallTransformer(input_dim=input_dim, seq_len=seq_len)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("✅ 已加载训练好的模型和数据")
    return model, device, daily_ds, seq_len


# =========================================================
# 2. 构造某天的输入序列：过去 lookback 天 → 目标日
# =========================================================
def build_input_sequence_for_date(daily_ds, city: str, target_date: pd.Timestamp, lookback: int):
    """
    给定城市 + 某一天（在历史数据中），
    使用该城市过去 lookback 天的特征构造模型输入。

    返回:
      x: [1, lookback, feature_dim] 的 tensor
    """
    df = daily_ds.df

    # 只取这个城市
    df_city = df[df["city"] == city].sort_values("date").reset_index(drop=True)

    # 确保目标日期存在
    if target_date not in df_city["date"].values:
        raise ValueError(f"在历史数据中找不到 {city} 的日期 {target_date.date()}")

    # 找到目标日期在该城市序列中的位置（local index）
    idx_list = df_city.index[df_city["date"] == target_date].tolist()
    if not idx_list:
        raise ValueError(f"在历史数据中找不到 {city} 的日期 {target_date.date()}")
    idx = idx_list[0]

    if idx < lookback:
        # 前 lookback 天不够，无法构造完整窗口
        raise ValueError(
            f"{city} 在 {target_date.date()} 之前历史不足 {lookback} 天，无法构造序列输入。"
        )

    start = idx - lookback
    end = idx

    window = df_city.loc[start:end-1, daily_ds.feature_cols].values.astype("float32")  # [lookback, feat_dim]
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # [1, lookback, feat_dim]
    return x


# =========================================================
# 3. 预测某天（单天预测）
# =========================================================
def predict_rain_for_day(model, device, daily_ds, city: str, target_date: pd.Timestamp, lookback: int):
    """
    使用已加载的模型 + 数据，对某城市某一天做降雨预测（mm）。
    """
    x = build_input_sequence_for_date(daily_ds, city, target_date, lookback)
    x = x.to(device)

    with torch.no_grad():
        pred_log = model(x)[0].item()    # log1p(mm)
        pred_mm = float(np.expm1(pred_log))

    return pred_mm


# =========================================================
# 4. 预测某个月：返回每天的预测 + 总降雨
# =========================================================
def predict_rain_for_month(city: str, year: int, month: int):
    """
    对指定城市的某一年某一月做日级降雨预测。

    返回:
      dates: [N] 列表，每个元素是 datetime.date
      preds: [N] 列表，对应每天的预测降雨 (mm)
      total_mm: 该月预测总降雨量 (mm)
    """
    model, device, daily_ds, lookback = load_model_and_data()

    # 从数据中筛选出该城市 + 指定年月的所有“实际存在的日期”
    df = daily_ds.df
    df_city = df[df["city"] == city].copy()
    df_city_month = df_city[
        (df_city["date"].dt.year == year) & (df_city["date"].dt.month == month)
    ].sort_values("date")

    if df_city_month.empty:
        raise ValueError(f"在数据中找不到 {city} {year}-{month:02d} 的任何记录。")

    dates = []
    preds = []

    for _, row in df_city_month.iterrows():
        date_ts = row["date"]

        # 如果前面天数不够 lookback，就跳过（比如数据最开头的一个月）
        try:
            pred_mm = predict_rain_for_day(model, device, daily_ds, city, date_ts, lookback)
            dates.append(date_ts.date())
            preds.append(pred_mm)
        except ValueError as e:
            # 可以选择打印一下提示
            print(f"⚠️ 跳过 {city} {date_ts.date()}: {e}")
            continue

    if not dates:
        raise RuntimeError(f"{city} {year}-{month:02d} 没有任何一天能构造完整的 {lookback} 天窗口。")

    total_mm = float(np.sum(preds))
    return dates, preds, total_mm


# =========================================================
# 5. main: 示例调用 & 打印结果
# =========================================================
if __name__ == "__main__":
    # 你可以随便改这三个参数（注意要在历史数据范围内）
    city = "San Francisco"   # 或 "New York"
    year = 2019
    month = 1

    dates, preds, total_mm = predict_rain_for_month(city, year, month)

    print(f"\n📅 {city} {year}-{month:02d} 每日预测降雨量 (mm):")
    for d, p in zip(dates, preds):
        print(f"  {d}: {p:.3f} mm")

    print(f"\n🌧️ 该月预测总降雨量 ≈ {total_mm:.3f} mm")
