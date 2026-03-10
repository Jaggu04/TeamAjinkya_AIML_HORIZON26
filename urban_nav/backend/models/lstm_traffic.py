"""
backend/models/lstm_traffic.py
MODULE 1 — Predictive Traffic Forecasting (LSTM)

Architecture:
  Input  → [batch, seq_len=24, features=7]
  LSTM   → hidden_size=128, num_layers=2, dropout=0.2
  FC     → 128 → 64 → 3   (outputs: 1h, 3h, 6h congestion)

Run training: python -m backend.models.lstm_traffic
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# ─── Paths ────────────────────────────────────────────────────────
SAVE_DIR    = "backend/models/saved"
MODEL_PATH  = f"{SAVE_DIR}/lstm_traffic.pt"
SCALER_PATH = f"{SAVE_DIR}/scaler.pkl"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────
SEQ_LEN     = 24      # last 24 hours as input
FEATURES    = 7       # number of input features
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 0.001
HORIZONS    = [1, 3, 6]   # hours ahead to predict


# ══════════════════════════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════════════════════════

class TrafficDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

class TrafficLSTM(nn.Module):
    """
    Multi-horizon LSTM for traffic congestion forecasting.
    Predicts congestion % at 1h, 3h, and 6h ahead simultaneously.
    """
    def __init__(self, input_size=FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, num_horizons=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            dropout      = dropout if num_layers > 1 else 0,
            batch_first  = True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_horizons),   # 3 outputs: 1h, 3h, 6h
            nn.Sigmoid(),                  # output in [0,1] → ×100 for %
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)              # [batch, seq_len, hidden]

        # Attention over time steps
        attn_w   = self.attention(lstm_out)     # [batch, seq_len, 1]
        context  = (attn_w * lstm_out).sum(1)  # [batch, hidden]

        out = self.head(context)                # [batch, 3]
        return out * 100                        # scale to 0–100


# ══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all features from raw traffic data."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    # Cyclical time encoding (captures 23→0 continuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Weather encoding
    if "weather_code" not in df.columns:
        weather_map = {"clear": 0, "rain": 1, "heavy_rain": 2, "fog": 3}
        df["weather_code"] = df.get("weather", "clear").map(weather_map).fillna(0)

    # Lag features (rolling context)
    df["cong_lag1"]  = df.groupby("route_id")["congestion_pct"].shift(1)
    df["cong_lag3"]  = df.groupby("route_id")["congestion_pct"].shift(3)
    df["cong_lag24"] = df.groupby("route_id")["congestion_pct"].shift(24)  # yesterday same hour
    df["speed_norm"] = df["current_speed"] / df["freeflow_speed"]

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "weather_code",
    "has_event",
    "speed_norm",
]

TARGET_COL = "congestion_pct"


def create_sequences(df: pd.DataFrame, scaler=None, fit_scaler=False):
    """Create sliding window sequences per route."""
    X_all, y_all = [], []

    for route_id in df["route_id"].unique():
        route_df = df[df["route_id"] == route_id].reset_index(drop=True)
        feats    = route_df[FEATURE_COLS].values
        targets  = route_df[TARGET_COL].values

        if fit_scaler and scaler is not None:
            feats = scaler.fit_transform(feats)
        elif scaler is not None:
            feats = scaler.transform(feats)

        # Multi-horizon targets: [1h, 3h, 6h] ahead
        for i in range(SEQ_LEN, len(route_df) - max(HORIZONS)):
            X_all.append(feats[i - SEQ_LEN : i])
            y_all.append([
                targets[i + h - 1] for h in HORIZONS
            ])

    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# 4. TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════

def train():
    print("🚦 Starting LSTM Training Pipeline\n")

    # ── Load data ──────────────────────────────────────────────────
    csv = "data/processed/mumbai_traffic_history.csv"
    if not os.path.exists(csv):
        print("⚠️  No dataset found. Generating synthetic data...")
        import sys; sys.path.insert(0, ".")
        from data.generate_synthetic_data import generate_dataset
        generate_dataset(days=60)

    df = pd.read_csv(csv)
    print(f"📊 Loaded {len(df):,} rows from {csv}")

    # ── Feature engineering ────────────────────────────────────────
    df = build_features(df)
    print(f"📐 After feature engineering: {len(df):,} rows")

    # ── Train / val split (time-based, not random!) ────────────────
    split_idx  = int(len(df) * 0.85)
    train_df   = df.iloc[:split_idx]
    val_df     = df.iloc[split_idx:]

    scaler     = StandardScaler()
    X_train, y_train = create_sequences(train_df, scaler, fit_scaler=True)
    X_val,   y_val   = create_sequences(val_df,   scaler, fit_scaler=False)
    print(f"✂️  Train: {X_train.shape} | Val: {X_val.shape}")

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # ── DataLoaders ────────────────────────────────────────────────
    train_loader = DataLoader(TrafficDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TrafficDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    # ── Model, loss, optimizer ─────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = TrafficLSTM().to(device)
    criterion = nn.HuberLoss()        # robust to outliers vs MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"\n🧠 Model on: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Training loop ──────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_pred, all_true = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred     = model(Xb)
                val_loss += criterion(pred, yb).item()
                all_pred.append(pred.cpu().numpy())
                all_true.append(yb.cpu().numpy())
        val_loss /= len(val_loader)

        preds = np.vstack(all_pred)
        trues = np.vstack(all_true)
        mae_1h = mean_absolute_error(trues[:, 0], preds[:, 0])
        mae_3h = mean_absolute_error(trues[:, 1], preds[:, 1])
        mae_6h = mean_absolute_error(trues[:, 2], preds[:, 2])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            saved = "✅ saved"
        else:
            saved = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_loss:.3f} | Val: {val_loss:.3f} | "
                  f"MAE 1h: {mae_1h:.1f} 3h: {mae_3h:.1f} 6h: {mae_6h:.1f} {saved}")

    print(f"\n🏁 Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"💾 Model saved to: {MODEL_PATH}")
    return model, scaler


# ══════════════════════════════════════════════════════════════════
# 5. PREDICTION SERVICE
# ══════════════════════════════════════════════════════════════════

class TrafficPredictor:
    """
    Inference wrapper — call predict() to get congestion forecasts.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = TrafficLSTM().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        print("✅ TrafficPredictor loaded.")

    def predict(self, route_id: str, recent_data: pd.DataFrame) -> dict:
        """
        recent_data: DataFrame with last 24 hours of data for the route.
        Returns: dict with 1h, 3h, 6h congestion predictions.
        """
        df = build_features(recent_data)
        if len(df) < SEQ_LEN:
            raise ValueError(f"Need at least {SEQ_LEN} rows of history.")

        feats = df[FEATURE_COLS].values[-SEQ_LEN:]
        feats = self.scaler.transform(feats)
        x     = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(x).squeeze(0).cpu().numpy()

        return {
            "route_id":       route_id,
            "forecast_1h":    round(float(preds[0]), 1),
            "forecast_3h":    round(float(preds[1]), 1),
            "forecast_6h":    round(float(preds[2]), 1),
            "level_1h":       _level(preds[0]),
            "level_3h":       _level(preds[1]),
            "level_6h":       _level(preds[2]),
        }


def _level(score: float) -> str:
    if score < 30:   return "LOW"
    if score < 60:   return "MODERATE"
    if score < 80:   return "HIGH"
    return "SEVERE"


if __name__ == "__main__":
    train()
