import os
import json
import math
import random
import csv
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

LOW_THETA_MIN = 0.0
LOW_THETA_MAX = 55.0
HIGH_THETA_MIN = 45.0
HIGH_THETA_MAX = 85.0
ALPHA_ABS_MAX = 15.0

try:
    from generate_dataset import COLUMN_INDEX
except ImportError:
    COLUMN_INDEX = {}


class AngleDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.12):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AngleHead(nn.Module):
    def __init__(self, hidden: int, theta_min: float, theta_max: float, alpha_abs_max: float):
        super().__init__()
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.alpha_abs_max = float(alpha_abs_max)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 2),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(feats)
        theta = torch.sigmoid(raw[:, 0:1]) * (self.theta_max - self.theta_min) + self.theta_min
        alpha = torch.tanh(raw[:, 1:2]) * self.alpha_abs_max
        return torch.cat([theta, alpha], dim=1)


class SingleBranchDNN(nn.Module):
    def __init__(
        self,
        in_dim: int = 14,
        hidden: int = 192,
        num_blocks: int = 3,
        dropout: float = 0.12,
        theta_min: float = LOW_THETA_MIN,
        theta_max: float = LOW_THETA_MAX,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.backbone = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(num_blocks)])
        self.head = AngleHead(hidden, theta_min, theta_max, ALPHA_ABS_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(self.stem(x))
        return self.head(feats)


class SmoothPhysicsLoss(nn.Module):
    def __init__(self, mass: float = 0.04828, caliber: float = 0.01295, dt: float = 0.05):
        super().__init__()
        self.mass = float(mass)
        self.area = math.pi * (caliber / 2.0) ** 2
        self.dt = float(dt)
        self.g0 = 9.80665
        self.z0 = 0.05
        self.h_ref = 2.0
        self.sound_coeff = 20.05

    def smooth_cd_g7(self, v_rel: torch.Tensor, T_kelvin: torch.Tensor) -> torch.Tensor:
        c = self.sound_coeff * torch.sqrt(torch.clamp(T_kelvin, min=1.0))
        ma = v_rel / torch.clamp(c, min=1e-6)

        cd_sub = torch.full_like(ma, 0.12)
        cd_mid = 0.12 + (ma - 0.9) * (0.28 / 0.30)
        cd_sup = 0.40 * torch.pow(torch.clamp(1.2 / torch.clamp(ma, min=1e-6), min=1e-6), 0.5)

        w1 = torch.sigmoid((ma - 0.9) / 0.03)
        w2 = torch.sigmoid((ma - 1.2) / 0.03)
        return cd_sub * (1.0 - w1) + cd_mid * w1 * (1.0 - w2) + cd_sup * w2

    def muzzle_wind_scale(self, device, dtype):
        scale = math.log((self.z0 + 0.01) / self.z0) / math.log(self.h_ref / self.z0)
        scale = max(0.0, min(scale, 5.0))
        return torch.tensor(scale, device=device, dtype=dtype)

    def forward(self, pred_angles: torch.Tensor, X_raw: torch.Tensor):
        v0 = X_raw[:, 3:4]
        wind_x = X_raw[:, 6:7]
        wind_y = X_raw[:, 7:8]
        wind_z = X_raw[:, 8:9]
        cant_rad = X_raw[:, 9:10] * (math.pi / 180.0)
        T0_C = X_raw[:, 11:12]
        P0_Pa = X_raw[:, 12:13]

        theta_rad = pred_angles[:, 0:1] * (math.pi / 180.0)
        alpha_rad = pred_angles[:, 1:2] * (math.pi / 180.0)

        vx = v0 * torch.cos(theta_rad) * torch.cos(alpha_rad)
        vy = v0 * (
            torch.sin(theta_rad) * torch.cos(cant_rad)
            - torch.cos(theta_rad) * torch.sin(alpha_rad) * torch.sin(cant_rad)
        )
        vz = v0 * (
            torch.sin(theta_rad) * torch.sin(cant_rad)
            + torch.cos(theta_rad) * torch.sin(alpha_rad) * torch.cos(cant_rad)
        )

        scale = self.muzzle_wind_scale(X_raw.device, X_raw.dtype)
        vx_rel = vx - wind_x * scale
        vy_rel = vy - wind_y * scale
        vz_rel = vz - wind_z * scale
        v_rel = torch.sqrt(vx_rel ** 2 + vy_rel ** 2 + vz_rel ** 2 + 1e-8)

        T_kelvin = T0_C + 273.15
        rho = (P0_Pa * 0.0289644) / (8.3144598 * torch.clamp(T_kelvin, min=1.0))
        cd = self.smooth_cd_g7(v_rel, T_kelvin)
        k = 0.5 * rho * cd * self.area

        ax = -(k / self.mass) * v_rel * vx_rel
        ay = -self.g0 - (k / self.mass) * v_rel * vy_rel
        az = -(k / self.mass) * v_rel * vz_rel

        dx = vx * self.dt + 0.5 * ax * self.dt ** 2
        dy = vy * self.dt + 0.5 * ay * self.dt ** 2
        dz = vz * self.dt + 0.5 * az * self.dt ** 2
        return dx, dy, dz


def minmax_fit(X: np.ndarray) -> dict:
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    span = np.where((xmax - xmin) < 1e-12, 1.0, (xmax - xmin))
    return {"xmin": xmin.tolist(), "xmax": xmax.tolist(), "span": span.tolist()}


def minmax_transform(X: np.ndarray, scaler: dict) -> np.ndarray:
    xmin = np.array(scaler["xmin"], dtype=np.float32)
    span = np.array(scaler["span"], dtype=np.float32)
    return (X - xmin) / span


def build_trajectory_groups(arr: np.ndarray, round_decimals: int = 4):
    theta = np.round(arr[:, -1].astype(np.float64), round_decimals)
    alpha = np.round(arr[:, -2].astype(np.float64), round_decimals)
    v0 = np.round(arr[:, 3].astype(np.float64), round_decimals)
    rho = np.round(arr[:, 4].astype(np.float64), round_decimals)
    wind_x = np.round(arr[:, 6].astype(np.float64), round_decimals)
    wind_y = np.round(arr[:, 7].astype(np.float64), round_decimals)
    wind_z = np.round(arr[:, 8].astype(np.float64), round_decimals)
    cant = np.round(arr[:, 9].astype(np.float64), round_decimals)
    T0_C = np.round(arr[:, 12].astype(np.float64), round_decimals)
    P0_Pa = np.round(arr[:, 13].astype(np.float64), round_decimals)
    alt_gun = np.round(arr[:, 14].astype(np.float64), round_decimals)

    keys = list(
        zip(
            theta.tolist(),
            alpha.tolist(),
            v0.tolist(),
            rho.tolist(),
            wind_x.tolist(),
            wind_y.tolist(),
            wind_z.tolist(),
            cant.tolist(),
            T0_C.tolist(),
            P0_Pa.tolist(),
            alt_gun.tolist(),
        )
    )

    unique_map = {}
    group_ids = np.empty(len(keys), dtype=np.int64)
    group_keys = []
    gid = 0
    for i, key in enumerate(keys):
        if key not in unique_map:
            unique_map[key] = gid
            group_keys.append(key)
            gid += 1
        group_ids[i] = unique_map[key]
    return group_ids, group_keys


def trajectory_group_split(group_ids: np.ndarray, seed: int = 42, ratios=(0.70, 0.15, 0.15)) -> dict:
    uniq = np.unique(group_ids).tolist()
    rnd = random.Random(seed)
    rnd.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    return {
        "train_groups": sorted([int(x) for x in uniq[:n_train]]),
        "val_groups": sorted([int(x) for x in uniq[n_train:n_train + n_val]]),
        "test_groups": sorted([int(x) for x in uniq[n_train + n_val:]]),
    }


def _per_elem_smooth_l1(err: torch.Tensor, delta: float = 0.2) -> torch.Tensor:
    abs_err = torch.abs(err)
    delta_t = torch.full_like(abs_err, float(delta))
    quadratic = torch.minimum(abs_err, delta_t)
    linear = abs_err - quadratic
    return 0.5 * quadratic ** 2 + delta_t * linear


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def _sample_importance(yb: torch.Tensor, X_raw: torch.Tensor) -> torch.Tensor:
    theta = yb[:, 0:1]
    overlap = ((theta >= 42.0) & (theta <= 58.0)).float()
    extreme_theta = ((theta <= 8.0) | (theta >= 68.0)).float()

    wind_norm = torch.sqrt(
        (X_raw[:, 6:7] / 20.0) ** 2 +
        (X_raw[:, 7:8] / 5.0) ** 2 +
        (X_raw[:, 8:9] / 15.0) ** 2
    )
    wind_norm = torch.clamp(wind_norm, 0.0, 2.0)

    cant_norm = torch.clamp(torch.abs(X_raw[:, 9:10]) / 15.0, 0.0, 1.0)
    weights = 1.0 + 0.60 * overlap + 0.20 * extreme_theta + 0.15 * wind_norm + 0.10 * cant_norm
    return weights.detach()


def _per_sample_angle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    theta_err = pred[:, 0:1] - target[:, 0:1]
    alpha_err = pred[:, 1:2] - target[:, 1:2]
    theta_loss = _per_elem_smooth_l1(theta_err, delta=0.35)
    alpha_loss = _per_elem_smooth_l1(alpha_err, delta=0.25)
    return 1.35 * theta_loss + 1.00 * alpha_loss


def _physics_residual_loss(
    phys_engine: SmoothPhysicsLoss,
    pred_angles: torch.Tensor,
    y_true: torch.Tensor,
    X_raw: torch.Tensor,
):
    pred_dx, pred_dy, pred_dz = phys_engine(pred_angles, X_raw)
    true_dx, true_dy, true_dz = phys_engine(y_true, X_raw)

    sx = torch.clamp(torch.abs(true_dx).detach(), min=1.0)
    sy = torch.clamp(torch.abs(true_dy).detach(), min=1.0)
    sz = torch.clamp(torch.abs(true_dz).detach(), min=1.0)

    lx = _per_elem_smooth_l1((pred_dx - true_dx) / sx, delta=0.05)
    ly = _per_elem_smooth_l1((pred_dy - true_dy) / sy, delta=0.05)
    lz = _per_elem_smooth_l1((pred_dz - true_dz) / sz, delta=0.05)
    return lx + ly + lz


def _scheduled_lambda_phys(epoch: int, lambda_phys: float) -> float:
    warm_phys = min(1.0, epoch / 12.0)
    return float(lambda_phys) * warm_phys


def _forward_loss_dict(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    phys_engine: SmoothPhysicsLoss,
    xmin_tensor: torch.Tensor,
    span_tensor: torch.Tensor,
    epoch: int,
    lambda_phys: float,
) -> Dict[str, torch.Tensor]:
    X_raw = xb * span_tensor + xmin_tensor
    sample_w = _sample_importance(yb, X_raw)
    pred = model(xb)

    loss_data = _weighted_mean(_per_sample_angle_loss(pred, yb), sample_w)
    loss_phys = _weighted_mean(_physics_residual_loss(phys_engine, pred, yb, X_raw), sample_w)
    total_loss = loss_data + _scheduled_lambda_phys(epoch, lambda_phys) * loss_phys

    return {
        "loss_data": loss_data,
        "loss_phys": loss_phys,
        "total_loss": total_loss,
        "mae_theta": torch.mean(torch.abs(pred[:, 0] - yb[:, 0])),
        "mae_alpha": torch.mean(torch.abs(pred[:, 1] - yb[:, 1])),
    }


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = None
        for name, value in model.state_dict().items():
            self.shadow[name] = value.detach().clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[name].copy_(value.detach())

    def apply_to(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    phys_engine: SmoothPhysicsLoss,
    xmin_tensor: torch.Tensor,
    span_tensor: torch.Tensor,
    epoch: int,
    lambda_phys: float,
) -> Dict[str, float]:
    model.eval()

    vals_total = []
    vals_data = []
    vals_phys = []
    vals_mae_theta = []
    vals_mae_alpha = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            out = _forward_loss_dict(
                model=model,
                xb=xb,
                yb=yb,
                phys_engine=phys_engine,
                xmin_tensor=xmin_tensor,
                span_tensor=span_tensor,
                epoch=epoch,
                lambda_phys=lambda_phys,
            )

            vals_total.append(out["total_loss"].item())
            vals_data.append(out["loss_data"].item())
            vals_phys.append(out["loss_phys"].item())
            vals_mae_theta.append(out["mae_theta"].item())
            vals_mae_alpha.append(out["mae_alpha"].item())

    return {
        "loss": float(np.mean(vals_total)) if vals_total else float("nan"),
        "loss_data": float(np.mean(vals_data)) if vals_data else float("nan"),
        "loss_phys": float(np.mean(vals_phys)) if vals_phys else float("nan"),
        "mae_theta": float(np.mean(vals_mae_theta)) if vals_mae_theta else float("nan"),
        "mae_alpha": float(np.mean(vals_mae_alpha)) if vals_mae_alpha else float("nan"),
    }


@dataclass
class TrainConfig:
    out_dir: str = "artifacts"
    dataset_npy: str = "artifacts/dataset.npy"
    model_name: str = "dual_model"
    scaler_name: str = "dual_scaler"
    history_name: str = "train_history"
    plot_name: str = "loss_curve"
    seed: int = 42
    batch_size: int = 1536
    lr: float = 2.2e-4
    max_epochs: int = 80
    patience: int = 8
    hidden: int = 256
    dropout: float = 0.12
    device: Optional[str] = None
    plot_losses: bool = True
    lambda_phys: float = 0.0025
    trajectory_mode: int = 0


def _save_history_csv(history: Dict[str, Any], csv_path: str):
    epoch_count = len(history["train_total"])
    fieldnames = [
        "epoch",
        "train_total",
        "train_data",
        "train_phys",
        "train_mae_theta",
        "train_mae_alpha",
        "val_loss",
        "val_data",
        "val_phys",
        "val_mae_theta",
        "val_mae_alpha",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(epoch_count):
            writer.writerow({
                "epoch": idx + 1,
                "train_total": history["train_total"][idx],
                "train_data": history["train_data"][idx],
                "train_phys": history["train_phys"][idx],
                "train_mae_theta": history["train_mae_theta"][idx],
                "train_mae_alpha": history["train_mae_alpha"][idx],
                "val_loss": history["val_loss"][idx],
                "val_data": history["val_data"][idx],
                "val_phys": history["val_phys"][idx],
                "val_mae_theta": history["val_mae_theta"][idx],
                "val_mae_alpha": history["val_mae_alpha"][idx],
            })


def _read_ammo_meta(out_dir: str):
    meta_path = os.path.join(out_dir, "dataset_meta.json")
    mass, caliber = 0.04828, 0.01295
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            mass = float(meta.get("mass_kg", mass))
            caliber = float(meta.get("caliber_m", caliber))
    return mass, caliber


def _make_adamw(params, lr: float, weight_decay: float, use_fused: bool):
    if use_fused:
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _make_grad_scaler(use_cuda: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_cuda)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=use_cuda)


def train(
    out_dir: str = "artifacts",
    dataset_npy: str = "artifacts/dataset.npy",
    model_name: str = "dual_model",
    scaler_name: str = "dual_scaler",
    history_name: str = "train_history",
    plot_name: str = "loss_curve",
    seed: int = 42,
    batch_size: int = 1536,
    lr: float = 2.2e-4,
    max_epochs: int = 80,
    patience: int = 8,
    hidden: int = 256,
    dropout: float = 0.12,
    device: Optional[str] = None,
    plot_losses: bool = True,
    lambda_phys: float = 0.0025,
    trajectory_mode: int = 0,
) -> Dict[str, Any]:

    cfg = TrainConfig(
        out_dir=out_dir,
        dataset_npy=dataset_npy,
        model_name=model_name,
        scaler_name=scaler_name,
        history_name=history_name,
        plot_name=plot_name,
        seed=seed,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        hidden=hidden,
        dropout=dropout,
        device=device,
        plot_losses=plot_losses,
        lambda_phys=lambda_phys,
        trajectory_mode=int(trajectory_mode),
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    arr = np.load(cfg.dataset_npy).astype(np.float32)
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {cfg.device}")

    if cfg.trajectory_mode not in (0, 1):
        raise ValueError("trajectory_mode must be 0 (low) or 1 (high).")

    in_low_col = COLUMN_INDEX["in_low_branch"]
    in_high_col = COLUMN_INDEX["in_high_branch"]
    if cfg.trajectory_mode == 0:
        mode_mask = arr[:, in_low_col] == 1.0
    else:
        mode_mask = arr[:, in_high_col] == 1.0
    arr = arr[mode_mask]
    mode_name = "low" if cfg.trajectory_mode == 0 else "high"
    print(f"Trajectory mode: {mode_name} ({cfg.trajectory_mode})")
    print(f"Filtered samples: {len(arr)}")
    if len(arr) == 0:
        raise RuntimeError(f"No samples found for trajectory_mode={cfg.trajectory_mode}.")

    y_theta = arr[:, -1:]
    y_alpha = arr[:, -2:-1]
    y = np.concatenate([y_theta, y_alpha], axis=1)

    X = np.delete(arr[:, :-2], [in_low_col, in_high_col], axis=1)
    print(f"Feature dim: {X.shape[1]}, Label dim: {y.shape[1]}")

    group_ids, _ = build_trajectory_groups(arr)
    split = trajectory_group_split(group_ids, seed=cfg.seed)

    mask_train = np.isin(group_ids, split["train_groups"])
    mask_val = np.isin(group_ids, split["val_groups"])
    mask_test = np.isin(group_ids, split["test_groups"])
    print(f"Samples -> Train: {mask_train.sum()}, Val: {mask_val.sum()}, Test: {mask_test.sum()}")

    scaler = minmax_fit(X[mask_train])
    X_train_n = minmax_transform(X[mask_train], scaler)
    X_val_n = minmax_transform(X[mask_val], scaler)
    X_test_n = minmax_transform(X[mask_test], scaler)

    train_dataset = AngleDataset(X_train_n, y[mask_train])
    val_dataset = AngleDataset(X_val_n, y[mask_val])
    test_dataset = AngleDataset(X_test_n, y[mask_test])

    use_cuda = str(cfg.device).lower() == "cuda"
    num_workers = 0 if os.name == "nt" else (4 if len(train_dataset) > cfg.batch_size * 2 else 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=use_cuda, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=use_cuda, num_workers=0)

    if cfg.trajectory_mode == 0:
        theta_min, theta_max = LOW_THETA_MIN, LOW_THETA_MAX
    else:
        theta_min, theta_max = HIGH_THETA_MIN, HIGH_THETA_MAX

    model = SingleBranchDNN(
        in_dim=X.shape[1],
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        theta_min=theta_min,
        theta_max=theta_max,
    ).to(cfg.device)
    ema = ModelEMA(model, decay=0.999)
    mass, caliber = _read_ammo_meta(cfg.out_dir)
    phys_engine = SmoothPhysicsLoss(mass=mass, caliber=caliber).to(cfg.device)

    xmin_tensor = torch.tensor(scaler["xmin"], dtype=torch.float32, device=cfg.device)
    span_tensor = torch.tensor(scaler["span"], dtype=torch.float32, device=cfg.device)

    optim = _make_adamw(model.parameters(), lr=cfg.lr, weight_decay=1e-2, use_fused=use_cuda)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=4, min_lr=8e-7)
    scaler_amp = _make_grad_scaler(use_cuda)

    best_val = math.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    history = {
        "train_total": [],
        "train_data": [],
        "train_phys": [],
        "train_mae_theta": [],
        "train_mae_alpha": [],
        "val_loss": [],
        "val_data": [],
        "val_phys": [],
        "val_mae_theta": [],
        "val_mae_alpha": [],
        "test_loss": None,
        "test_mae_theta": None,
        "test_mae_alpha": None,
        "best_epoch": None,
        "best_val_loss": None,
        "trajectory_mode": int(cfg.trajectory_mode),
    }

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tr_total, tr_data, tr_phys, tr_mae_th, tr_mae_al = [], [], [], [], []

        for xb, yb in train_loader:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            if model.training:
                xb = xb + torch.randn_like(xb) * 0.001

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = _forward_loss_dict(
                    model=model,
                    xb=xb,
                    yb=yb,
                    phys_engine=phys_engine,
                    xmin_tensor=xmin_tensor,
                    span_tensor=span_tensor,
                    epoch=epoch,
                    lambda_phys=cfg.lambda_phys,
                )
                total_loss = out["total_loss"]

            scaler_amp.scale(total_loss).backward()
            scaler_amp.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler_amp.step(optim)
            scaler_amp.update()
            ema.update(model)

            tr_total.append(out["total_loss"].item())
            tr_data.append(out["loss_data"].item())
            tr_phys.append(out["loss_phys"].item())
            tr_mae_th.append(out["mae_theta"].item())
            tr_mae_al.append(out["mae_alpha"].item())

        ema.apply_to(model)
        val_info = evaluate_model(
            model=model,
            loader=val_loader,
            device=cfg.device,
            phys_engine=phys_engine,
            xmin_tensor=xmin_tensor,
            span_tensor=span_tensor,
            epoch=epoch,
            lambda_phys=cfg.lambda_phys,
        )
        ema.restore(model)

        scheduler.step(val_info["loss"])

        history["train_total"].append(float(np.mean(tr_total)))
        history["train_data"].append(float(np.mean(tr_data)))
        history["train_phys"].append(float(np.mean(tr_phys)))
        history["train_mae_theta"].append(float(np.mean(tr_mae_th)))
        history["train_mae_alpha"].append(float(np.mean(tr_mae_al)))
        history["val_loss"].append(val_info["loss"])
        history["val_data"].append(val_info["loss_data"])
        history["val_phys"].append(val_info["loss_phys"])
        history["val_mae_theta"].append(val_info["mae_theta"])
        history["val_mae_alpha"].append(val_info["mae_alpha"])

        print(
            f"[{mode_name.upper()}] Epoch {epoch:03d} | "
            f"TrainTotal:{history['train_total'][-1]:.4f} "
            f"[D:{history['train_data'][-1]:.4f} P:{history['train_phys'][-1]:.4f}] | "
            f"ValTotal:{val_info['loss']:.4f} "
            f"[D:{val_info['loss_data']:.4f} P:{val_info['loss_phys']:.4f}] | "
            f"Theta MAE:{val_info['mae_theta']:.4f} deg Alpha MAE:{val_info['mae_alpha']:.4f} deg | "
            f"LR:{optim.param_groups[0]['lr']:.2e}"
        )

        if val_info["loss"] < best_val:
            best_val = val_info["loss"]
            best_epoch = epoch

            ema.apply_to(model)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ema.restore(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_info = evaluate_model(
        model=model,
        loader=test_loader,
        device=cfg.device,
        phys_engine=phys_engine,
        xmin_tensor=xmin_tensor,
        span_tensor=span_tensor,
        epoch=best_epoch if best_epoch > 0 else 1,
        lambda_phys=cfg.lambda_phys,
    )

    history["test_loss"] = test_info["loss"]
    history["test_mae_theta"] = test_info["mae_theta"]
    history["test_mae_alpha"] = test_info["mae_alpha"]
    history["best_epoch"] = int(best_epoch)
    history["best_val_loss"] = float(best_val)

    model_path = os.path.join(cfg.out_dir, f"{cfg.model_name}.pt")
    scaler_path = os.path.join(cfg.out_dir, f"{cfg.scaler_name}.json")
    history_path = os.path.join(cfg.out_dir, f"{cfg.history_name}.json")
    history_csv_path = os.path.join(cfg.out_dir, f"{cfg.history_name}.csv")
    plot_path = os.path.join(cfg.out_dir, f"{cfg.plot_name}.png")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    _save_history_csv(history, history_csv_path)

    if cfg.plot_losses and len(history["train_total"]) > 0:
        epochs = np.arange(1, len(history["train_total"]) + 1)
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        ax.plot(epochs, history["train_total"], linewidth=2.0, label="Train Loss")
        ax.plot(epochs, history["val_loss"], linewidth=2.0, linestyle="--", label="Val Loss")
        ax.set_title(f"Training and Validation Loss ({mode_name})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=400, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_path}")
    print(f"Saved metrics CSV: {history_csv_path}")

    print("=" * 60)
    print(f"Mode            : {mode_name}")
    print(f"Best epoch      : {best_epoch}")
    print(f"Best val loss   : {best_val:.6f}")
    print(f"Test loss       : {test_info['loss']:.6f}")
    print(f"Test Theta MAE  : {test_info['mae_theta']:.4f} deg")
    print(f"Test Alpha MAE  : {test_info['mae_alpha']:.4f} deg")
    print("=" * 60)

    return history


if __name__ == "__main__":
    train(
        out_dir="artifacts_127",
        dataset_npy="artifacts_127/dataset.npy",
        model_name="dual_model_low",
        scaler_name="dual_scaler_low",
        history_name="train_history_low",
        plot_name="loss_curve_low",
        trajectory_mode=0,
    )
