"""
Global sampling + correlation (Concurrent + Rate Limited + Progress)
- asyncio + aiohttp 并发下载
- 令牌桶限速 + 全局并发上限
- tqdm 进度条：并发下载分片、逐城市聚合、全局合并、相关性分析
- 断点续跑（城市-月份分片落盘）+ Ctrl+C 优雅中断

Docs: https://open-meteo.com/en/docs/historical-weather-api
"""

import os, sys, time, math, signal, warnings, asyncio, aiohttp
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ------------------ 路径配置 ------------------
OUT_DIR = "./dataset_global"
RAW_DIR = os.path.join(OUT_DIR, "raw_hourly")         # 分片缓存（城市-月份）
DAILY_DIR = os.path.join(OUT_DIR, "daily_by_city")    # 每城市日频
GLOBAL_DAILY_CSV = os.path.join(OUT_DIR, "weather_daily_global.csv")

# ------------------ 任务配置 ------------------
START_DATE = date(2015, 1, 1)
END_DATE   = date(2024, 12, 31)

# 代表性城市（可扩展）
CITIES = [
    ("San Francisco", 37.77, -122.42), ("New York", 40.71, -74.01),
    ("Mexico City", 19.43, -99.13), ("Sao Paulo", -23.55, -46.63),
    ("Buenos Aires", -34.61, -58.38), ("Lima", -12.05, -77.05),
    ("London", 51.51, -0.13), ("Paris", 48.86, 2.35), ("Berlin", 52.52, 13.41),
    ("Moscow", 55.76, 37.62), ("Madrid", 40.42, -3.70), ("Rome", 41.90, 12.50),
    ("Cairo", 30.04, 31.24), ("Lagos", 6.46, 3.40), ("Nairobi", -1.29, 36.82),
    ("Johannesburg", -26.20, 28.04), ("Tokyo", 35.68, 139.76),
    ("Seoul", 37.57, 126.98), ("Shanghai", 31.23, 121.47),
    ("Singapore", 1.35, 103.82), ("Bangkok", 13.75, 100.50),
    ("Mumbai", 19.08, 72.88), ("Delhi", 28.61, 77.21),
    ("Dubai", 25.20, 55.27), ("Riyadh", 24.71, 46.68),
    ("Sydney", -33.87, 151.21), ("Melbourne", -37.81, 144.96),
    ("Auckland", -36.85, 174.76), ("Jakarta", -6.21, 106.85),
    ("Honolulu", 21.31, -157.86), ("Reykjavik", 64.13, -21.90),
    ("Ulaanbaatar", 47.92, 106.92),
]

# ------------------ 下载与API ------------------
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover", "precipitation"
]

# ------------------ 并发与限速 ------------------
MAX_CONCURRENCY = 8   # 并发上限
MAX_RPS = 4           # 每秒请求数上限
BURST = 4             # 瞬时突发上限
REQUEST_TIMEOUT = 40
MAX_RETRIES = 4
RETRY_BACKOFF = 1.8

# ------------------ 分析配置 ------------------
TARGET_COL = "precipitation_sum_mm"
SAMPLE_FRAC_GLOBAL = 0.10
SAMPLE_FRAC_PER_CITY = 0.20

# ------------------ 中断标记 ------------------
stop_flag = False
def _handle_interrupt(signum, frame):
    global stop_flag
    stop_flag = True
    print("\n⚠️ 捕获到中断信号：停止派发新任务，等待进行中的请求完成。")
signal.signal(signal.SIGINT, _handle_interrupt)

# ------------------ 工具函数 ------------------
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)

def month_chunks(start_d: date, end_d: date):
    cur = date(start_d.year, start_d.month, 1)
    last = date(end_d.year, end_d.month, 1)
    while cur <= last:
        if cur.month == 12:
            month_end = date(cur.year, 12, 31)
        else:
            month_end = date(cur.year, cur.month + 1, 1) - timedelta(days=1)
        s = max(cur, start_d)
        e = min(month_end, end_d)
        yield s, e
        if cur.month == 12: cur = date(cur.year + 1, 1, 1)
        else: cur = date(cur.year, cur.month + 1, 1)

def hourly_json_to_df(payload: dict) -> pd.DataFrame:
    if "hourly" not in payload or "time" not in payload["hourly"]:
        return pd.DataFrame()
    hourly = payload["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for k, v in hourly.items():
        if k == "time": continue
        df[k] = v
    df = df.set_index("time").sort_index()
    return df

def aggregate_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    daily = pd.DataFrame()
    daily[TARGET_COL] = df_hourly["precipitation"].resample("D").sum(min_count=1)

    def maybe(col, how="mean"):
        if col in df_hourly.columns:
            if how == "mean": return df_hourly[col].resample("D").mean()
            if how == "max":  return df_hourly[col].resample("D").max()
            if how == "min":  return df_hourly[col].resample("D").min()
        return pd.Series(dtype=float)

    daily["temp_mean_C"]    = maybe("temperature_2m","mean")
    daily["temp_max_C"]     = maybe("temperature_2m","max")
    daily["temp_min_C"]     = maybe("temperature_2m","min")
    daily["rh_mean_pct"]    = maybe("relative_humidity_2m","mean")
    daily["press_mean_hPa"] = maybe("surface_pressure","mean")
    daily["wind_mean_ms"]   = maybe("wind_speed_10m","mean")
    daily["wind_dir_deg"]   = maybe("wind_direction_10m","mean")
    daily["cloud_mean_pct"] = maybe("cloud_cover","mean")

    # 露点（Magnus）
    a, b = 17.62, 243.12
    T = daily["temp_mean_C"]
    RH = daily["rh_mean_pct"]
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = (a*T/(b+T)) + np.log(np.clip(RH, 1e-6, 100)/100.0)
        daily["dew_point_C"] = (b * gamma) / (a - gamma)

    # 季节循环 + 滞后/滚动
    daily["month"] = daily.index.month
    daily["month_sin"] = np.sin(2*np.pi*daily["month"]/12)
    daily["month_cos"] = np.cos(2*np.pi*daily["month"]/12)
    daily["precip_lag_1"] = daily[TARGET_COL].shift(1)
    daily["precip_lag_3"] = daily[TARGET_COL].shift(3)
    daily["temp_mean_lag_1"] = daily["temp_mean_C"].shift(1)
    daily["precip_roll7"] = daily[TARGET_COL].rolling(7, min_periods=3).mean()
    return daily

# ------------------ 令牌桶限速器 ------------------
class RateLimiter:
    def __init__(self, max_rps: int, burst: int):
        self.max_rps = max(1, int(max_rps))
        self.capacity = max(1, int(burst))
        self.tokens = self.capacity
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            refill = elapsed * self.max_rps
            if refill > 0:
                self.tokens = min(self.capacity, self.tokens + refill)
                self.updated_at = now
            if self.tokens < 1:
                needed = 1 - self.tokens
                wait_s = needed / self.max_rps
                await asyncio.sleep(wait_s)
                now = time.monotonic()
                elapsed = now - self.updated_at
                refill = elapsed * self.max_rps
                self.tokens = min(self.capacity, self.tokens + refill)
                self.updated_at = now
            self.tokens -= 1

# ------------------ 异步下载（带进度） ------------------
async def fetch_hourly_chunk(session: aiohttp.ClientSession, limiter: RateLimiter,
                             lat: float, lon: float, s: date, e: date) -> dict:
    if stop_flag:
        return {}
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": s.isoformat(), "end_date": e.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    url = BASE_URL
    attempt = 0
    while attempt <= MAX_RETRIES and not stop_flag:
        attempt += 1
        try:
            await limiter.acquire()
            async with session.get(url, params=params, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status in (429, 500, 502, 503, 504):
                    await asyncio.sleep((RETRY_BACKOFF ** (attempt - 1)) * 0.7)
                else:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status} {text[:200]}")
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep((RETRY_BACKOFF ** (attempt - 1)) * 0.7)
    return {}

async def concurrent_download_with_progress():
    ensure_dirs()

    # 统计总分片数（用于总进度）
    all_jobs = []
    for name, lat, lon in CITIES:
        for s, e in month_chunks(START_DATE, END_DATE):
            shard = os.path.join(RAW_DIR, f"{name.replace(' ','_')}_{s}_{e}.parquet")
            all_jobs.append((name, lat, lon, s, e, shard))

    # 过滤掉已存在的分片（已完成即视为进度已达成）
    pending_jobs = [(n, lat, lon, s, e, shard) for (n, lat, lon, s, e, shard) in all_jobs if not os.path.exists(shard)]
    completed = len(all_jobs) - len(pending_jobs)

    limiter = RateLimiter(MAX_RPS, BURST)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=REQUEST_TIMEOUT)

    pbar = tqdm(total=len(all_jobs), initial=completed, desc="并发下载分片", unit="shard")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def run_job(name, lat, lon, s, e, shard):
            if stop_flag:
                return
            async with sem:
                payload = await fetch_hourly_chunk(session, limiter, lat, lon, s, e)
                if payload:
                    df = hourly_json_to_df(payload)
                    if df is not None and not df.empty:
                        df.to_parquet(shard)
                # 无论成功或失败都推进进度（失败的下次会重试/补上）
                pbar.update(1)

        tasks = [run_job(n, lat, lon, s, e, shard) for (n, lat, lon, s, e, shard) in pending_jobs]
        # 顺序消费 as_completed，便于 Ctrl+C 尽快停
        for coro in asyncio.as_completed(tasks):
            if stop_flag:
                break
            try:
                await coro
            except Exception as ex:
                # 单分片异常不影响整体，进度已在 run_job 中推进
                print(f"⚠️ 分片异常：{ex}")

    pbar.close()

# ------------------ 聚合 + 进度 ------------------
def build_city_daily_with_progress():
    files = os.listdir(RAW_DIR) if os.path.exists(RAW_DIR) else []
    city_to_shards = {}
    for fn in files:
        if not fn.endswith(".parquet"): continue
        city = fn.split("_")[0]
        city_to_shards.setdefault(city, 0)
        city_to_shards[city] += 1

    pbar = tqdm(total=len(CITIES), desc="逐城市聚合(日频)", unit="city")
    for name, lat, lon in CITIES:
        out_path = os.path.join(DAILY_DIR, f"{name.replace(' ','_')}.parquet")
        if os.path.exists(out_path):
            pbar.update(1)
            continue
        prefix = f"{name.replace(' ','_')}_"
        shards = [os.path.join(RAW_DIR, fn) for fn in os.listdir(RAW_DIR)
                  if fn.startswith(prefix) and fn.endswith(".parquet")]
        if not shards:
            # 即使没有分片也推进进度，避免卡住
            pbar.update(1)
            continue
        try:
            parts = [pd.read_parquet(p) for p in shards]
            hourly = pd.concat(parts).sort_index()
            daily = aggregate_to_daily(hourly)
            daily["city"] = name
            daily["lat"] = lat
            daily["lon"] = lon
            daily.to_parquet(out_path)
        except Exception as ex:
            print(f"⚠️ {name} 聚合失败：{ex}")
        pbar.update(1)
    pbar.close()

def build_global_csv_with_progress():
    if os.path.exists(GLOBAL_DAILY_CSV):
        print(f"✅ 已有全局 CSV：{GLOBAL_DAILY_CSV}")
        return
    files = [fn for fn in os.listdir(DAILY_DIR) if fn.endswith(".parquet")]
    if not files:
        print("❌ 未找到任何城市日频数据。")
        sys.exit(1)

    pbar = tqdm(total=len(files), desc="合并全局CSV", unit="city")
    parts = []
    for fn in files:
        df = pd.read_parquet(os.path.join(DAILY_DIR, fn))
        parts.append(df.reset_index().rename(columns={"index":"date"}))
        pbar.update(1)
    pbar.close()

    g = pd.concat(parts, ignore_index=True)
    g.to_csv(GLOBAL_DAILY_CSV, index=False)
    print(f"🧩 全局 CSV 已生成：{GLOBAL_DAILY_CSV}")

# ------------------ 相关性 + 进度 ------------------
def correlation_tables(df_num: pd.DataFrame, target: str):
    corr_p = df_num.corr(method="pearson")[target].sort_values(ascending=False)
    corr_s = df_num.corr(method="spearman")[target].sort_values(ascending=False)
    X = df_num.drop(columns=[target]).fillna(0)
    y = df_num[target].values
    mi = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    out = pd.DataFrame({
        "pearson": corr_p.reindex(X.columns),
        "spearman": corr_s.reindex(X.columns),
        "mutual_info": mi_series
    }).sort_values(["mutual_info","pearson"], ascending=False)
    return out

def plot_heatmap(df_num: pd.DataFrame, title: str, out_png: str, method="pearson"):
    plt.figure(figsize=(11,8))
    cmap = "coolwarm" if method == "pearson" else "BrBG"
    sns.heatmap(df_num.corr(method=method), cmap=cmap, center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def run_correlation_with_progress():
    g = pd.read_csv(GLOBAL_DAILY_CSV, parse_dates=["time"])
    num_cols = g.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL not in num_cols:
        raise ValueError(f"目标列 {TARGET_COL} 不在全局数据中。")

    # 全局抽样
    g_sample = g.sample(frac=SAMPLE_FRAC_GLOBAL, random_state=42)
    g_num = g_sample[num_cols].dropna()
    out_global = correlation_tables(g_num, TARGET_COL)
    out_global.to_csv(os.path.join(OUT_DIR, "correlation_global.csv"), index=True)
    plot_heatmap(g_num, "Global Pearson Correlation", os.path.join(OUT_DIR, "global_corr_pearson.png"), "pearson")
    plot_heatmap(g_num, "Global Spearman Correlation", os.path.join(OUT_DIR, "global_corr_spearman.png"), "spearman")
    print("🌍 全局相关性已输出。")

    # 分城市进度
    pbar = tqdm(total=g["city"].nunique(), desc="分城市相关性", unit="city")
    for city, dfc in g.groupby("city"):
        sample = dfc.sample(frac=min(SAMPLE_FRAC_PER_CITY, 1.0), random_state=42)
        num = sample.select_dtypes(include=[np.number]).dropna()
        if len(num) >= 20 and TARGET_COL in num.columns:
            out = correlation_tables(num, TARGET_COL)
            out.to_csv(os.path.join(OUT_DIR, f"correlation_{city.replace(' ','_')}.csv"), index=True)
        pbar.update(1)
    pbar.close()

# ------------------ 主流程 ------------------
async def _download_phase():
    await concurrent_download_with_progress()

def main():
    try:
        ensure_dirs()
        # 1) 并发下载（带进度）
        asyncio.run(_download_phase())
        if stop_flag:
            print("⏹️ 已中断：保留已完成分片；下次运行会自动跳过。")
            return
        # 2) 聚合为日频（带进度）
        build_city_daily_with_progress()
        if stop_flag:
            print("⏹️ 已中断：日频聚合阶段提前结束。")
            return
        # 3) 合并全局 CSV（带进度）
        build_global_csv_with_progress()
        # 4) 全局 + 分城市相关性（带进度）
        run_correlation_with_progress()
        print("✅ 全流程完成。输出目录：", OUT_DIR)
    except KeyboardInterrupt:
        print("\n🛑 用户中断，已尽量保存当前成果。")
    except Exception as e:
        print(f"❌ 异常：{e}")
        raise

if __name__ == "__main__":
    main()
