import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =========================================================
# 1. 日级天气 Dataset：读取 CSV、筛选城市、清洗 NaN/Inf、标准化
# =========================================================
class DailyWeatherDataset(Dataset):
    """
    读取日级天气数据，只保留 San Francisco 和 New York，
    使用你筛选过的 18 个特征，清洗掉含 NaN/Inf 的行，然后标准化。
    """

    def __init__(
        self,
        csv_path="./dataset_global/weather_daily_global.csv",
        cities=("San Francisco", "New York"),
        date_col="time",   # 你的日期列名
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 找不到日级数据文件: {csv_path}")

        # 1) 读取 CSV
        df = pd.read_csv(csv_path)

        # 2) 检查必须列：city + 日期列
        required_cols = ["city", date_col]
        for c in required_cols:
            if c not in df.columns:
                raise KeyError(f"❌ CSV 中缺少必须列: {c}")

        # 3) 只保留两个城市
        df = df[df["city"].isin(cities)].copy()
        if df.empty:
            raise RuntimeError(f"❌ CSV 中没有找到城市 {cities} 的数据")

        # 4) 处理日期：统一成 df["date"]
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values(["city", "date"]).reset_index(drop=True)

        # 5) 目标列
        target_col = "precipitation_sum_mm"

        # 6) 特征列：你相关性分析后选的 18 个
        feature_cols = [
            "temp_mean_C",
            "temp_max_C",
            "temp_min_C",
            "rh_mean_pct",
            "press_mean_hPa",
            "wind_mean_ms",
            "wind_dir_deg",
            "cloud_mean_pct",
            "dew_point_C",
            "month",
            "month_sin",
            "month_cos",
            "precip_lag_1",
            "precip_lag_3",
            "temp_mean_lag_1",
            "precip_roll7",
            "lat",
            "lon",
        ]

        for c in feature_cols + [target_col]:
            if c not in df.columns:
                raise KeyError(f"❌ CSV 缺少列: {c}")

        # 7) 先处理 NaN / Inf：删除含 NaN/Inf 的样本
        cols_to_check = feature_cols + [target_col]
        df[cols_to_check] = df[cols_to_check].replace([np.inf, -np.inf], np.nan)
        before = len(df)
        df = df.dropna(subset=cols_to_check).reset_index(drop=True)
        after = len(df)
        dropped = before - after
        print(f"🧹 已去除含 NaN/Inf 的行: {dropped} 行，剩余 {after} 行")

        # 8) 保存配置
        self.feature_cols = feature_cols
        self.target_col = target_col

        # 9) 对特征做标准化
        self.mean = df[feature_cols].mean()
        self.std = df[feature_cols].std().replace(0, 1e-6)
        df[feature_cols] = (df[feature_cols] - self.mean) / (self.std + 1e-6)

        self.df = df.reset_index(drop=True)

        print("✅ DailyWeatherDataset 初始化完成（清洗后）")
        print(f"   ✔ 城市: {sorted(self.df['city'].unique())}")
        print(f"   ✔ 总天数: {len(self.df)}")
        print(f"   ✔ 特征维度: {len(self.feature_cols)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.feature_cols].astype(float).values, dtype=torch.float32)
        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        city = row["city"]
        date = row["date"]
        return x, y, city, date


# =========================================================
# 2. 序列 Dataset：过去 lookback 天 → 明天 (log1p)
# =========================================================
class DailySequenceDataset(Dataset):
    """
    把 DailyWeatherDataset 转成序列：
        输入：过去 lookback 天的特征（例如 30 天）
        输出：目标那天的 log1p(降雨量)
    """

    def __init__(self, daily_ds: DailyWeatherDataset, lookback=30):
        self.daily_ds = daily_ds
        self.feature_cols = daily_ds.feature_cols
        self.target_col = daily_ds.target_col
        self.lookback = lookback

        df = daily_ds.df
        self.df = df

        self.samples = []  # (city, start_idx, end_idx)

        for city, df_city in df.groupby("city"):
            idxs = df_city.index.to_list()
            if len(idxs) <= lookback:
                continue
            for i in range(lookback, len(idxs)):
                start = idxs[i - lookback]
                end = idxs[i]
                self.samples.append((city, start, end))

        print("✅ DailySequenceDataset 初始化完成")
        print(f"   ✔ lookback = {lookback} 天")
        print(f"   ✔ 序列样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        city, start, end = self.samples[idx]

        window = self.df.loc[start:end-1, self.feature_cols].values.astype("float32")
        target = float(self.df.loc[end, self.target_col])  # mm

        x = torch.tensor(window, dtype=torch.float32)
        y_log = torch.tensor(np.log1p(target), dtype=torch.float32)

        # 训练时只需要 x, y_log
        return x, y_log


# =========================================================
# 3. 测试 usage
# =========================================================
if __name__ == "__main__":
    print("🧪 正在测试 DailyWeatherDataset 和 DailySequenceDataset ...")

    daily_ds = DailyWeatherDataset(
        csv_path="./dataset_global/weather_daily_global.csv",
        cities=("San Francisco", "New York"),
        date_col="time",
    )

    print("\n🔍 查看一条单天样本：")
    x0, y0, c0, d0 = daily_ds[0]
    print(f"   城市: {c0}, 日期: {d0.date()}, 雨量: {y0.item():.3f} mm")
    print(f"   特征维度: {x0.shape}")

    seq_ds = DailySequenceDataset(daily_ds, lookback=30)

    print("\n🔍 查看一条序列样本：")
    x1, y1 = seq_ds[0]
    print(f"   X 形状: {x1.shape} (应为 [30, 特征数])")
    print(f"   y_log: {y1.item():.4f}")
    print("🎉 Dataset 测试通过！")
