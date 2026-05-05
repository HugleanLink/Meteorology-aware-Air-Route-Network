import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent


def _load_config():
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Task 1: Generate Labels ──────────────────────────────────────────

def generate_labels(df_samples, config):
    """
    Generate status labels for edge weather samples.

    Labels:
      2 = failed  (wind >= fail threshold OR rain >= fail threshold)
      1 = affected (wind >= affected threshold, not failed)
      0 = normal   (otherwise)

    Returns df_samples with added 'status' column.
    """
    cfg = config["weather"]
    wind_fail = cfg["wind_fail_threshold_mps"]
    wind_aff = cfg["wind_affected_threshold_mps"]
    rain_fail = cfg["rain_fail_threshold_mm_day"]

    df = df_samples.copy()
    failed = (df["wind_speed"] >= wind_fail) | (df["tp_mm_day"] >= rain_fail)
    affected = (df["wind_speed"] >= wind_aff) & ~failed

    df["status"] = 0
    df.loc[affected, "status"] = 1
    df.loc[failed, "status"] = 2

    n = len(df)
    for label, name in [(0, "正常"), (1, "受影响"), (2, "失效")]:
        count = (df["status"] == label).sum()
        print(f"  {name} (status={label}): {count:>8d}  ({count/n*100:.2f}%)")

    return df


# ── Task 2: Build Feature Matrix ─────────────────────────────────────

def build_features(df_labeled):
    """
    Build normalized feature matrix X and label vector y.

    Features (per row = one edge x time-step sample):
      - wind_speed / 20.0
      - tp_mm_day / 300.0
      - weather_pool encoded: 常规=0, 极端风=1, 极端雨=2, 风雨复合=3, then /3
      - edge_id LabelEncoder, then / n_edges
      - edge_type: trunk=0, branch=1 (normalized by /1)
    """
    df = df_labeled.copy()

    X = np.zeros((len(df), 5), dtype=np.float32)

    X[:, 0] = df["wind_speed"].values / 20.0
    X[:, 1] = df["tp_mm_day"].values / 300.0

    pool_map = {"常规": 0, "极端风": 1, "极端雨": 2, "风雨复合": 3}
    pool_codes = df["weather_pool"].map(pool_map).fillna(0).values.astype(np.float32)
    X[:, 2] = pool_codes / 3.0

    le = LabelEncoder()
    edge_codes = le.fit_transform(df["edge_id"]).astype(np.float32)
    n_edges = len(le.classes_)
    X[:, 3] = edge_codes / n_edges

    if "edge_type" in df.columns:
        X[:, 4] = df["edge_type"].values.astype(np.float32)
    else:
        X[:, 4] = 0.0

    y = df["status"].values.astype(np.int64)

    print(f"\n特征矩阵: X shape={X.shape}, y shape={y.shape}")
    print(f"  边数: {n_edges}")
    print(f"  特征: wind_speed_norm, tp_mm_day_norm, weather_pool_norm, edge_id_norm")
    print(f"  标签分布: {np.bincount(y)}")

    return X, y


# ── Task 3: MLP Model ────────────────────────────────────────────────

class MeteoRiskMLP(nn.Module):
    """3-layer MLP for edge weather risk classification."""

    def __init__(self, input_dim=5, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_model(X, y, config):
    """
    Train MeteoRiskMLP with class-balanced sampling.

    Splits X, y into 80/10/10 train/val/test (random_state=42).
    Uses WeightedRandomSampler to handle class imbalance.
    Saves best checkpoint (by val accuracy) to config's model_path.
    Returns (model, X_test, y_test).
    """
    rs = config["project"].get("random_state", 42)
    epochs = config.get("meteo_model", {}).get("epochs", 50)
    batch_size = config.get("meteo_model", {}).get("batch_size", 1024)
    lr = config.get("meteo_model", {}).get("lr", 1e-3)

    # Subsample if dataset is too large for PyTorch (limit: 2^24 categories)
    max_samples = 500_000  # balanced subset for practical training speed
    if len(X) > max_samples:
        print(f"\n数据集过大 ({len(X):,}), 进行分层降采样...")
        indices = np.arange(len(X))
        keep_idx = []
        for label in [0, 1, 2]:
            label_idx = indices[y == label]
            if len(label_idx) > max_samples // 3:
                label_idx = np.random.default_rng(rs).choice(
                    label_idx, max_samples // 3, replace=False)
            keep_idx.append(label_idx)
        keep_idx = np.concatenate(keep_idx)
        np.random.default_rng(rs).shuffle(keep_idx)
        X = X[keep_idx]
        y = y[keep_idx]
        print(f"降采样后: {len(X):,} 样本 (标签分布: {np.bincount(y)})")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=rs, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=rs, stratify=y_temp
    )
    print(f"\n数据划分: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    try:
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(y_train),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    except RuntimeError:
        print(f"  (数据集过大 {len(y_train):,}, 使用普通随机采样)")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MeteoRiskMLP(input_dim=X.shape[1], num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    model_path = _PROJECT_ROOT / config["meteo_model"]["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n开始训练: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(y_train)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val, dtype=torch.float32))
                val_preds = val_logits.argmax(dim=1).numpy()
                val_acc = (val_preds == y_val).mean()

            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print(f"    -> 保存最佳模型 (val_acc={best_val_acc:.4f})")

    model.load_state_dict(torch.load(model_path))
    print(f"\n训练完成，最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存至: {model_path}")

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.  Prints accuracy, classification report,
    and confusion matrix.  Returns metrics dict.
    """
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        y_pred = probs.argmax(axis=1)

    acc = (y_pred == y_test).mean()
    labels = [0, 1, 2]
    target_names = ["正常", "受影响", "失效"]

    print(f"\n{'='*60}")
    print("测试集评估")
    print(f"{'='*60}")
    print(f"测试集准确率: {acc:.4f}")

    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"混淆矩阵 (行=真实, 列=预测):")
    print(f"{'':>6}  预测0  预测1  预测2")
    for i, name in enumerate(target_names):
        print(f"  {name:>4}:  {cm[i,0]:>5}  {cm[i,1]:>5}  {cm[i,2]:>5}")

    return {"accuracy": acc, "confusion_matrix": cm,
            "y_true": y_test, "y_pred": y_pred}


# ── Task 4: Inference ────────────────────────────────────────────────

def _load_resources():
    """Load model and label encoder for inference."""
    config = _load_config()
    model = MeteoRiskMLP(input_dim=5, num_classes=3)
    model_path = _PROJECT_ROOT / config["meteo_model"]["model_path"]
    model.load_state_dict(torch.load(model_path))
    model.eval()

    samples_path = _PROJECT_ROOT / config["meteo_model"].get(
        "samples_path", "data/processed/edge_weather_samples.parquet")
    df = pd.read_parquet(samples_path)
    le = LabelEncoder()
    le.fit(df["edge_id"])
    n_edges = len(le.classes_)

    return model, le, n_edges


def predict_edge_status(model, edge_id, wind_speed, tp_mm_day,
                        weather_pool, n_edges):
    """
    Predict status for a single edge given weather conditions.

    Returns dict with status, label, and per-class probabilities.
    """
    pool_map = {"常规": 0, "极端风": 1, "极端雨": 2, "风雨复合": 3}
    pool_code = pool_map.get(weather_pool, 0)

    le = LabelEncoder()
    le.fit(np.unique([edge_id]))
    edge_code = le.transform([edge_id])[0]

    x = np.array([[
        wind_speed / 20.0,
        tp_mm_day / 300.0,
        pool_code / 3.0,
        edge_code / n_edges,
        0.0,  # edge_type: default trunk
    ]], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x))
        probs = torch.softmax(logits, dim=1).numpy()[0]

    status = int(probs.argmax())
    labels = {0: "正常", 1: "受影响", 2: "失效"}

    return {
        "status": status,
        "status_label": labels[status],
        "prob_normal": float(probs[0]),
        "prob_affected": float(probs[1]),
        "prob_failed": float(probs[2]),
    }


def batch_predict(model, df_edge_weather, n_edges):
    """
    Batch inference on edge weather samples.

    Adds columns: pred_status, prob_normal, prob_affected, prob_failed.
    Saves to config's prediction_path and returns the DataFrame.
    """
    config = _load_config()

    pool_map = {"常规": 0, "极端风": 1, "极端雨": 2, "风雨复合": 3}

    df = df_edge_weather.copy()
    le = LabelEncoder()
    edge_codes = le.fit_transform(df["edge_id"]).astype(np.float32)

    chunk_size = 500_000
    n_total = len(df)
    preds_all = np.zeros(n_total, dtype=np.int32)
    probs_all = np.zeros((n_total, 3), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk = df.iloc[start:end]
            X_chunk = np.zeros((len(chunk), 5), dtype=np.float32)
            X_chunk[:, 0] = chunk["wind_speed"].values / 20.0
            X_chunk[:, 1] = chunk["tp_mm_day"].values / 300.0
            X_chunk[:, 2] = chunk["weather_pool"].map(pool_map).fillna(0).values.astype(np.float32) / 3.0
            X_chunk[:, 3] = edge_codes[start:end] / n_edges
            X_chunk[:, 4] = chunk["edge_type"].values.astype(np.float32) if "edge_type" in chunk.columns else 0.0

            logits = model(torch.tensor(X_chunk))
            probs = torch.softmax(logits, dim=1).numpy()
            preds_all[start:end] = probs.argmax(axis=1)
            probs_all[start:end] = probs
            if start % (chunk_size * 5) == 0:
                print(f"  推理进度: {end}/{n_total}")

    df["pred_status"] = preds_all
    df["prob_normal"] = probs_all[:, 0]
    df["prob_affected"] = probs_all[:, 1]
    df["prob_failed"] = probs_all[:, 2]

    output_path = _PROJECT_ROOT / config["meteo_model"]["prediction_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n批量推理完成: {len(df)} 条")
    print(f"保存至: {output_path}")

    for label, name in [(0, "正常"), (1, "受影响"), (2, "失效")]:
        count = (df["pred_status"] == label).sum()
        print(f"  预测 {name}: {count:>8d}  ({count/len(df)*100:.2f}%)")

    return df


# ── Task 5: Main Training Pipeline ───────────────────────────────────

if __name__ == "__main__":
    import sys
    config = _load_config()

    # Check for --all flag to use all-edge data
    use_all = "--all" in sys.argv
    if use_all:
        samples_path = _PROJECT_ROOT / config["meteo_model"].get(
            "samples_path_all", "data/processed/edge_weather_samples_all.parquet")
        config["meteo_model"]["model_path"] = config["meteo_model"].get(
            "model_path_all", "models/meteo_edge_model_all.pt")
        config["meteo_model"]["prediction_path"] = config["meteo_model"].get(
            "prediction_path_all", "data/processed/meteo_edge_predictions_all.parquet")
        mode_label = "(主干+支线全边)"
    else:
        samples_path = _PROJECT_ROOT / "data" / "processed" / "edge_weather_samples.parquet"
        mode_label = "(仅主干边)"

    print("=" * 60)
    print(f"阶段3: 气象风险模型训练 {mode_label}")
    print("=" * 60)

    # ── 1. Generate labels ──
    print("\n" + "=" * 60)
    print("任务1: 生成训练标签")
    print("=" * 60)
    df_samples = pd.read_parquet(samples_path)
    n_edges = df_samples["edge_id"].nunique()
    n_trunk = (df_samples["edge_type"] == 0).sum() if "edge_type" in df_samples.columns else len(df_samples)
    n_branch = (df_samples["edge_type"] == 1).sum() if "edge_type" in df_samples.columns else 0
    print(f"加载样本数据: {len(df_samples)} 条, {n_edges} 条边 "
          f"(主干: {n_trunk//n_edges if n_edges else 0} 条样本, "
          f"支线: {n_branch//n_edges if n_edges else 0} 条样本)")
    df_labeled = generate_labels(df_samples, config)

    # ── 2. Build features ──
    print("\n" + "=" * 60)
    print("任务2: 构建特征矩阵 (5维)")
    print("=" * 60)
    X, y = build_features(df_labeled)

    # ── 3. Train ──
    print("\n" + "=" * 60)
    print("任务3: 训练 MLP 模型")
    print("=" * 60)
    model, X_test, y_test = train_model(X, y, config)

    # ── 4. Evaluate ──
    evaluate_model(model, X_test, y_test)

    # ── 5. Batch predict ──
    print("\n" + "=" * 60)
    print("任务4: 批量推理")
    print("=" * 60)
    df_pred = batch_predict(model, df_samples, n_edges)
