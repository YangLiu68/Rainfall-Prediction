import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Config ===
CSV_PATH = "./dataset_global/weather_daily_global.csv"
TARGET_COL = "precipitation_sum_mm"
SAMPLE_FRAC_GLOBAL = 0.10
SAMPLE_FRAC_PER_CITY = 0.20
OUT_DIR = "./dataset_global"

def load_dataset(path):
    """Load CSV safely; handle missing 'date' column."""
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # If no 'date', try to infer from index or any datetime-like column
    if "date" not in df.columns:
        for col in df.columns:
            if "time" in col.lower() or "Unnamed: 0" in col:
                df = df.rename(columns={col: "date"})
                break
        else:
            df["date"] = pd.RangeIndex(len(df))  # fallback
    # Parse date safely
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception:
        pass
    return df

def correlation_tables(df_num: pd.DataFrame, target: str):
    """Return Pearson, Spearman, Mutual-Info correlation table."""
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
    }).sort_values(["mutual_info", "pearson"], ascending=False)
    return out

def plot_heatmap(df_num, title, out_png, method="pearson"):
    plt.figure(figsize=(11, 8))
    cmap = "coolwarm" if method == "pearson" else "BrBG"
    sns.heatmap(df_num.corr(method=method), cmap=cmap, center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def run_correlation(csv_path=CSV_PATH):
    df = load_dataset(csv_path)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL not in num_cols:
        raise ValueError(f"Target column '{TARGET_COL}' not found in numeric columns.")

    # === Global correlation ===
    g_sample = df.sample(frac=SAMPLE_FRAC_GLOBAL, random_state=42)
    g_num = g_sample[num_cols].dropna()
    print(f"Global sample: {len(g_num)} rows")

    out_global = correlation_tables(g_num, TARGET_COL)
    out_global.to_csv(f"{OUT_DIR}/correlation_global.csv", index=True)
    plot_heatmap(g_num, "Global Pearson Correlation",
                 f"{OUT_DIR}/global_corr_pearson.png", "pearson")
    plot_heatmap(g_num, "Global Spearman Correlation",
                 f"{OUT_DIR}/global_corr_spearman.png", "spearman")
    print("🌍 Global correlation saved.")

    # === Per-city correlation ===
    if "city" in df.columns:
        print("🏙️  Per-city correlation:")
        for city, group in tqdm(df.groupby("city"), total=df["city"].nunique()):
            num = group.select_dtypes(include=[np.number]).dropna()
            if len(num) < 20 or TARGET_COL not in num.columns:
                continue
            out = correlation_tables(num, TARGET_COL)
            out.to_csv(f"{OUT_DIR}/correlation_{city.replace(' ', '_')}.csv", index=True)

    print("✅ Done. Results saved to:", OUT_DIR)

if __name__ == "__main__":
    run_correlation()
