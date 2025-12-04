import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt  # ✅ 新增：用于画图

from daily_dataset import DailyWeatherDataset, DailySequenceDataset


# =========================================================
# 1. 设备选择
# =========================================================
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac M1/M2
    else:
        device = torch.device("cpu")
    print(f"🧠 使用设备: {device}")
    return device


# =========================================================
# 2. 日级 Transformer 模型：过去 30 天 → 明天的 log1p(降雨)
# =========================================================
class RainfallTransformer(nn.Module):
    def __init__(self, input_dim, seq_len=30, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        输出: [batch]，表示 log1p(mm)
        """
        if x.size(1) > self.seq_len:
            x = x[:, -self.seq_len:, :]

        x_proj = self.input_proj(x) + self.pos_embedding[:, : x.size(1)]
        enc = self.encoder(x_proj)          # [B, seq_len, d_model]
        pooled = enc.mean(dim=1)            # [B, d_model]
        out = self.head(pooled).squeeze(-1) # [B]
        return out


# =========================================================
# 3. 训练与验证函数（Dataset 现在只返回 xb, yb）
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for xb, yb in tqdm(loader, desc="👟 训练", unit="batch"):
        xb = xb.to(device)    # [B, 30, 18]
        yb = yb.to(device)    # [B], log1p(mm)

        optimizer.zero_grad()
        pred = model(xb)      # [B]
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_pred_log = []
    all_true_log = []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="🔎 验证", unit="batch"):
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)

            all_pred_log.append(pred.cpu())
            all_true_log.append(yb.cpu())

    avg_loss = total_loss / len(loader.dataset)

    # 拼成数组（log 空间）
    pred_log = torch.cat(all_pred_log).numpy()  # log1p(mm)
    true_log = torch.cat(all_true_log).numpy()

    # log 空间 RMSE
    rmse_log = float(np.sqrt(np.mean((pred_log - true_log) ** 2)))

    # 还原到 mm 空间
    pred_mm = np.expm1(pred_log)
    true_mm = np.expm1(true_log)

    rmse_mm = float(np.sqrt(np.mean((pred_mm - true_mm) ** 2)))
    mae_mm = float(np.mean(np.abs(pred_mm - true_mm)))

    return avg_loss, rmse_log, rmse_mm, mae_mm


# =========================================================
# 4. main：准备数据、模型、训练循环
# =========================================================
def main():
    device = get_device()

    # ---- 1) 构建 Dataset ----
    daily_ds = DailyWeatherDataset(
        csv_path="./dataset_global/weather_daily_global.csv",
        cities=("San Francisco", "New York"),
        date_col="time",
    )
    seq_ds = DailySequenceDataset(daily_ds, lookback=30)

    # ---- 2) 划分训练 / 验证 ----
    val_ratio = 0.2
    val_size = int(len(seq_ds) * val_ratio)
    train_size = len(seq_ds) - val_size

    train_ds, val_ds = random_split(
        seq_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    # ---- 3) 初始化模型、损失、优化器 ----
    input_dim = len(daily_ds.feature_cols)  # 18
    seq_len = 30

    model = RainfallTransformer(input_dim=input_dim, seq_len=seq_len).to(device)

    criterion = nn.HuberLoss(delta=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    best_val_loss = float("inf")
    best_path = "daily_transformer_best.pt"

    # ✅ 新增：用来记录每个 epoch 的 loss
    train_losses = []
    val_losses = []

    # ---- 4) 训练若干 epoch ----
    num_epochs = 15

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== 🌀 Epoch {epoch}/{num_epochs} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"📉 训练集平均损失: {train_loss:.6f}")

        val_loss, rmse_log, rmse_mm, mae_mm = evaluate(model, val_loader, criterion, device)
        print(f"✅ 验证集平均损失: {val_loss:.6f}")
        print(f"   ↳ RMSE(log空间): {rmse_log:.4f}")
        print(f"   ↳ RMSE(mm):     {rmse_mm:.4f}")
        print(f"   ↳ MAE(mm):      {mae_mm:.4f}")

        # ✅ 记录 loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存当前最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"💾 发现更好的模型，已保存到: {best_path}")

    print("\n🎉 训练完成！")
    print(f"⭐ 最佳验证损失: {best_val_loss:.6f}")
    print(f"⭐ 最优模型权重保存在: {best_path}")

    # ✅ 训练结束后画 loss 曲线
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Daily Transformer Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("📈 已保存 loss 曲线到 loss_curve.png")
    # 如果你在本地跑，并且想弹出窗口看图，可以取消下一行注释
    # plt.show()


if __name__ == "__main__":
    main()
