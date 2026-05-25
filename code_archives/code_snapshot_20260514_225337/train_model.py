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

from model_architecture import (
    LOW_THETA_MIN, LOW_THETA_MAX, HIGH_THETA_MIN, HIGH_THETA_MAX, ALPHA_ABS_MAX,
    ResidualBlock, AngleHead, SingleBranchDNN, ModelEMA, build_single_branch_model,
)
from physics_loss import (
    SmoothPhysicsLoss,
    _per_elem_smooth_l1,
    _weighted_mean,
    _sample_importance,
    _per_sample_angle_loss,
    _physics_residual_loss,
    _scheduled_lambda_phys,
    _forward_loss_dict,
)

try:
    from generate_dataset import COLUMN_INDEX
except ImportError:
    _FEATURE_COLS = [
        "x", "y", "z", "v0_actual", "rho", "slant_range",
        "wind_x", "wind_y", "wind_z", "cant_angle", "T_powder_C",
        "in_low_branch", "in_high_branch", "T0_C", "P0_Pa", "alt_gun",
    ]
    _LABEL_COLS = ["alpha_deg", "theta_deg"]
    COLUMN_INDEX = {name: i for i, name in enumerate(_FEATURE_COLS + _LABEL_COLS)}


class AngleDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
    theta = np.round(arr[:, COLUMN_INDEX["theta_deg"]].astype(np.float64), round_decimals)
    alpha = np.round(arr[:, COLUMN_INDEX["alpha_deg"]].astype(np.float64), round_decimals)
    v0 = np.round(arr[:, COLUMN_INDEX["v0_actual"]].astype(np.float64), round_decimals)
    rho = np.round(arr[:, COLUMN_INDEX["rho"]].astype(np.float64), round_decimals)
    wind_x = np.round(arr[:, COLUMN_INDEX["wind_x"]].astype(np.float64), round_decimals)
    wind_y = np.round(arr[:, COLUMN_INDEX["wind_y"]].astype(np.float64), round_decimals)
    wind_z = np.round(arr[:, COLUMN_INDEX["wind_z"]].astype(np.float64), round_decimals)
    cant = np.round(arr[:, COLUMN_INDEX["cant_angle"]].astype(np.float64), round_decimals)
    T0_C = np.round(arr[:, COLUMN_INDEX["T0_C"]].astype(np.float64), round_decimals)
    P0_Pa = np.round(arr[:, COLUMN_INDEX["P0_Pa"]].astype(np.float64), round_decimals)
    alt_gun = np.round(arr[:, COLUMN_INDEX["alt_gun"]].astype(np.float64), round_decimals)

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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    phys_engine: SmoothPhysicsLoss,
    xmin_tensor: torch.Tensor,
    span_tensor: torch.Tensor,
    epoch: int,
    lambda_phys: float,
    eval_lambda_phys: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    use_cuda = str(device).lower() == "cuda"
    loss_lambda = lambda_phys if eval_lambda_phys is None else float(eval_lambda_phys)

    sum_total = sum_data = sum_phys = sum_mae_theta = sum_mae_alpha = None
    sample_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = _forward_loss_dict(
                    model=model,
                    xb=xb,
                    yb=yb,
                    phys_engine=phys_engine,
                    xmin_tensor=xmin_tensor,
                    span_tensor=span_tensor,
                    epoch=epoch,
                    lambda_phys=loss_lambda,
                )

            batch_n = xb.shape[0]
            sample_count += batch_n
            if sum_total is None:
                sum_total = out["total_loss"].detach() * batch_n
                sum_data = out["loss_data"].detach() * batch_n
                sum_phys = out["loss_phys"].detach() * batch_n
                sum_mae_theta = out["mae_theta"].detach() * batch_n
                sum_mae_alpha = out["mae_alpha"].detach() * batch_n
            else:
                sum_total += out["total_loss"].detach() * batch_n
                sum_data += out["loss_data"].detach() * batch_n
                sum_phys += out["loss_phys"].detach() * batch_n
                sum_mae_theta += out["mae_theta"].detach() * batch_n
                sum_mae_alpha += out["mae_alpha"].detach() * batch_n

    return {
        "loss": (sum_total / sample_count).item() if sample_count else float("nan"),
        "loss_data": (sum_data / sample_count).item() if sample_count else float("nan"),
        "loss_phys": (sum_phys / sample_count).item() if sample_count else float("nan"),
        "mae_theta": (sum_mae_theta / sample_count).item() if sample_count else float("nan"),
        "mae_alpha": (sum_mae_alpha / sample_count).item() if sample_count else float("nan"),
    }


def _array_bytes(*arrays: np.ndarray) -> int:
    return int(sum(arr.nbytes for arr in arrays))


def _can_cache_on_gpu(required_bytes: int, device: str, max_total_frac: float, max_free_frac: float) -> tuple[bool, str]:
    if str(device).lower() != "cuda" or not torch.cuda.is_available():
        return False, "CUDA is not available."
    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    free, _ = torch.cuda.mem_get_info(0)
    total_limit = int(total * float(max_total_frac))
    free_limit = int(free * float(max_free_frac))
    limit = min(total_limit, free_limit)
    if required_bytes > limit:
        return (
            False,
            f"required {required_bytes / 1024 ** 2:.1f} MB exceeds safe cache limit {limit / 1024 ** 2:.1f} MB",
        )
    return True, f"required {required_bytes / 1024 ** 2:.1f} MB, safe limit {limit / 1024 ** 2:.1f} MB"


def _gpu_batch_iter(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
    n = X.shape[0]
    if shuffle:
        order = torch.randperm(n, device=X.device)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            yield X.index_select(0, idx), y.index_select(0, idx)
    else:
        for start in range(0, n, batch_size):
            yield X[start:start + batch_size], y[start:start + batch_size]


def evaluate_tensor_batches(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    batch_size: int,
    phys_engine: SmoothPhysicsLoss,
    xmin_tensor: torch.Tensor,
    span_tensor: torch.Tensor,
    epoch: int,
    lambda_phys: float,
    eval_lambda_phys: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    use_cuda = X_tensor.device.type == "cuda"
    loss_lambda = lambda_phys if eval_lambda_phys is None else float(eval_lambda_phys)

    sum_total = sum_data = sum_phys = sum_mae_theta = sum_mae_alpha = None
    sample_count = 0

    with torch.no_grad():
        for xb, yb in _gpu_batch_iter(X_tensor, y_tensor, batch_size, shuffle=False):
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = _forward_loss_dict(
                    model=model,
                    xb=xb,
                    yb=yb,
                    phys_engine=phys_engine,
                    xmin_tensor=xmin_tensor,
                    span_tensor=span_tensor,
                    epoch=epoch,
                    lambda_phys=loss_lambda,
                )

            batch_n = xb.shape[0]
            sample_count += batch_n
            if sum_total is None:
                sum_total = out["total_loss"].detach() * batch_n
                sum_data = out["loss_data"].detach() * batch_n
                sum_phys = out["loss_phys"].detach() * batch_n
                sum_mae_theta = out["mae_theta"].detach() * batch_n
                sum_mae_alpha = out["mae_alpha"].detach() * batch_n
            else:
                sum_total += out["total_loss"].detach() * batch_n
                sum_data += out["loss_data"].detach() * batch_n
                sum_phys += out["loss_phys"].detach() * batch_n
                sum_mae_theta += out["mae_theta"].detach() * batch_n
                sum_mae_alpha += out["mae_alpha"].detach() * batch_n

    return {
        "loss": (sum_total / sample_count).item() if sample_count else float("nan"),
        "loss_data": (sum_data / sample_count).item() if sample_count else float("nan"),
        "loss_phys": (sum_phys / sample_count).item() if sample_count else float("nan"),
        "mae_theta": (sum_mae_theta / sample_count).item() if sample_count else float("nan"),
        "mae_alpha": (sum_mae_alpha / sample_count).item() if sample_count else float("nan"),
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
    model_type: str = "mlp"
    device: Optional[str] = None
    plot_losses: bool = True
    lambda_phys: float = 0.0025
    trajectory_mode: int = 0
    physics_steps: int = 6
    cache_data_on_gpu: bool = True
    gpu_cache_max_total_frac: float = 0.20
    gpu_cache_max_free_frac: float = 0.50
    train_noise_x_rel_std: float = 0.0
    train_noise_y_abs_std: float = 0.0
    train_noise_z_abs_std: float = 0.0
    train_noise_v0_std: float = 0.0
    train_noise_rho_rel_std: float = 0.0
    train_noise_wind_x_std: float = 0.0
    train_noise_wind_y_std: float = 0.0
    train_noise_wind_z_std: float = 0.0
    train_noise_cant_std: float = 0.0
    train_noise_theta_std: float = 0.0
    train_noise_alpha_std: float = 0.0
    eval_phys_loss: bool = True


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


def _feature_col_after_branch_drop(name: str) -> int:
    raw_idx = COLUMN_INDEX[name]
    dropped = int(COLUMN_INDEX["in_low_branch"] < raw_idx) + int(COLUMN_INDEX["in_high_branch"] < raw_idx)
    return raw_idx - dropped


def _add_gaussian_noise(rng: np.random.Generator, values: np.ndarray, std: float) -> np.ndarray:
    if float(std) <= 0.0:
        return values
    return values + rng.normal(0.0, float(std), size=values.shape).astype(np.float32)


def _apply_train_noise(X_train: np.ndarray, y_train: np.ndarray, cfg: TrainConfig) -> dict:
    rng = np.random.default_rng(cfg.seed + 7919)
    Xn = X_train.copy()
    yn = y_train.copy()

    noise_summary = {
        "train_noise_x_rel_std": float(cfg.train_noise_x_rel_std),
        "train_noise_y_abs_std": float(cfg.train_noise_y_abs_std),
        "train_noise_z_abs_std": float(cfg.train_noise_z_abs_std),
        "train_noise_v0_std": float(cfg.train_noise_v0_std),
        "train_noise_rho_rel_std": float(cfg.train_noise_rho_rel_std),
        "train_noise_wind_x_std": float(cfg.train_noise_wind_x_std),
        "train_noise_wind_y_std": float(cfg.train_noise_wind_y_std),
        "train_noise_wind_z_std": float(cfg.train_noise_wind_z_std),
        "train_noise_cant_std": float(cfg.train_noise_cant_std),
        "train_noise_theta_std": float(cfg.train_noise_theta_std),
        "train_noise_alpha_std": float(cfg.train_noise_alpha_std),
    }

    x_col = _feature_col_after_branch_drop("x")
    y_col = _feature_col_after_branch_drop("y")
    z_col = _feature_col_after_branch_drop("z")
    slant_col = _feature_col_after_branch_drop("slant_range")

    if cfg.train_noise_x_rel_std > 0.0:
        scale = np.maximum(np.abs(Xn[:, x_col:x_col + 1]), 1.0)
        Xn[:, x_col:x_col + 1] += (
            rng.normal(0.0, cfg.train_noise_x_rel_std, size=(len(Xn), 1)).astype(np.float32) * scale
        )
        Xn[:, x_col:x_col + 1] = np.maximum(Xn[:, x_col:x_col + 1], 1.0)

    Xn[:, y_col:y_col + 1] = _add_gaussian_noise(rng, Xn[:, y_col:y_col + 1], cfg.train_noise_y_abs_std)
    Xn[:, z_col:z_col + 1] = _add_gaussian_noise(rng, Xn[:, z_col:z_col + 1], cfg.train_noise_z_abs_std)
    Xn[:, slant_col] = np.sqrt(
        Xn[:, x_col] ** 2 + Xn[:, y_col] ** 2 + Xn[:, z_col] ** 2
    ).astype(np.float32)

    v0_col = _feature_col_after_branch_drop("v0_actual")
    rho_col = _feature_col_after_branch_drop("rho")
    wind_x_col = _feature_col_after_branch_drop("wind_x")
    wind_y_col = _feature_col_after_branch_drop("wind_y")
    wind_z_col = _feature_col_after_branch_drop("wind_z")
    cant_col = _feature_col_after_branch_drop("cant_angle")

    Xn[:, v0_col:v0_col + 1] = _add_gaussian_noise(rng, Xn[:, v0_col:v0_col + 1], cfg.train_noise_v0_std)
    Xn[:, v0_col:v0_col + 1] = np.maximum(Xn[:, v0_col:v0_col + 1], 1.0)

    if cfg.train_noise_rho_rel_std > 0.0:
        rho_scale = np.maximum(np.abs(Xn[:, rho_col:rho_col + 1]), 1e-6)
        Xn[:, rho_col:rho_col + 1] += (
            rng.normal(0.0, cfg.train_noise_rho_rel_std, size=(len(Xn), 1)).astype(np.float32) * rho_scale
        )
        Xn[:, rho_col:rho_col + 1] = np.maximum(Xn[:, rho_col:rho_col + 1], 1e-6)

    Xn[:, wind_x_col:wind_x_col + 1] = _add_gaussian_noise(rng, Xn[:, wind_x_col:wind_x_col + 1], cfg.train_noise_wind_x_std)
    Xn[:, wind_y_col:wind_y_col + 1] = _add_gaussian_noise(rng, Xn[:, wind_y_col:wind_y_col + 1], cfg.train_noise_wind_y_std)
    Xn[:, wind_z_col:wind_z_col + 1] = _add_gaussian_noise(rng, Xn[:, wind_z_col:wind_z_col + 1], cfg.train_noise_wind_z_std)
    Xn[:, cant_col:cant_col + 1] = _add_gaussian_noise(rng, Xn[:, cant_col:cant_col + 1], cfg.train_noise_cant_std)

    yn[:, 0:1] = _add_gaussian_noise(rng, yn[:, 0:1], cfg.train_noise_theta_std)
    yn[:, 1:2] = _add_gaussian_noise(rng, yn[:, 1:2], cfg.train_noise_alpha_std)

    if any(v > 0.0 for v in noise_summary.values()):
        print("Training noise enabled:", noise_summary)
    return {"X_train": Xn.astype(np.float32), "y_train": yn.astype(np.float32), "summary": noise_summary}


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
    model_type: str = "mlp",
    device: Optional[str] = None,
    plot_losses: bool = True,
    lambda_phys: float = 0.0025,
    trajectory_mode: int = 0,
    physics_steps: int = 6,
    cache_data_on_gpu: bool = True,
    gpu_cache_max_total_frac: float = 0.20,
    gpu_cache_max_free_frac: float = 0.50,
    train_noise_x_rel_std: float = 0.0,
    train_noise_y_abs_std: float = 0.0,
    train_noise_z_abs_std: float = 0.0,
    train_noise_v0_std: float = 0.0,
    train_noise_rho_rel_std: float = 0.0,
    train_noise_wind_x_std: float = 0.0,
    train_noise_wind_y_std: float = 0.0,
    train_noise_wind_z_std: float = 0.0,
    train_noise_cant_std: float = 0.0,
    train_noise_theta_std: float = 0.0,
    train_noise_alpha_std: float = 0.0,
    eval_phys_loss: bool = True,
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
        model_type=str(model_type),
        device=device,
        plot_losses=plot_losses,
        lambda_phys=lambda_phys,
        trajectory_mode=int(trajectory_mode),
        physics_steps=int(physics_steps),
        cache_data_on_gpu=cache_data_on_gpu,
        gpu_cache_max_total_frac=gpu_cache_max_total_frac,
        gpu_cache_max_free_frac=gpu_cache_max_free_frac,
        train_noise_x_rel_std=float(train_noise_x_rel_std),
        train_noise_y_abs_std=float(train_noise_y_abs_std),
        train_noise_z_abs_std=float(train_noise_z_abs_std),
        train_noise_v0_std=float(train_noise_v0_std),
        train_noise_rho_rel_std=float(train_noise_rho_rel_std),
        train_noise_wind_x_std=float(train_noise_wind_x_std),
        train_noise_wind_y_std=float(train_noise_wind_y_std),
        train_noise_wind_z_std=float(train_noise_wind_z_std),
        train_noise_cant_std=float(train_noise_cant_std),
        train_noise_theta_std=float(train_noise_theta_std),
        train_noise_alpha_std=float(train_noise_alpha_std),
        eval_phys_loss=bool(eval_phys_loss),
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
    if str(cfg.device).lower() == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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

    train_noise = _apply_train_noise(X[mask_train], y[mask_train], cfg)
    X_train_raw = train_noise["X_train"]
    y_train = train_noise["y_train"]

    scaler = minmax_fit(X_train_raw)
    X_train_n = minmax_transform(X_train_raw, scaler)
    X_val_n = minmax_transform(X[mask_val], scaler)
    X_test_n = minmax_transform(X[mask_test], scaler)
    y_val = y[mask_val]
    y_test = y[mask_test]

    use_cuda = str(cfg.device).lower() == "cuda"
    train_loader = val_loader = test_loader = None
    X_train_gpu = y_train_gpu = X_val_gpu = y_val_gpu = X_test_gpu = y_test_gpu = None
    cache_on_gpu = False
    if cfg.cache_data_on_gpu and use_cuda:
        cache_bytes = _array_bytes(X_train_n, y_train, X_val_n, y_val, X_test_n, y_test)
        cache_on_gpu, cache_reason = _can_cache_on_gpu(
            cache_bytes,
            cfg.device,
            cfg.gpu_cache_max_total_frac,
            cfg.gpu_cache_max_free_frac,
        )
        if cache_on_gpu:
            X_train_gpu = torch.tensor(X_train_n, dtype=torch.float32, device=cfg.device)
            y_train_gpu = torch.tensor(y_train, dtype=torch.float32, device=cfg.device)
            X_val_gpu = torch.tensor(X_val_n, dtype=torch.float32, device=cfg.device)
            y_val_gpu = torch.tensor(y_val, dtype=torch.float32, device=cfg.device)
            X_test_gpu = torch.tensor(X_test_n, dtype=torch.float32, device=cfg.device)
            y_test_gpu = torch.tensor(y_test, dtype=torch.float32, device=cfg.device)
            print(f"GPU data cache: enabled ({cache_reason})")
        else:
            print(f"GPU data cache: disabled ({cache_reason})")

    if not cache_on_gpu:
        train_dataset = AngleDataset(X_train_n, y_train)
        val_dataset = AngleDataset(X_val_n, y_val)
        test_dataset = AngleDataset(X_test_n, y_test)
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

    model = build_single_branch_model(
        model_type=cfg.model_type,
        in_dim=X.shape[1],
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        theta_min=theta_min,
        theta_max=theta_max,
    ).to(cfg.device)
    print(f"Model type: {cfg.model_type}")
    ema = ModelEMA(model, decay=0.999)
    mass, caliber = _read_ammo_meta(cfg.out_dir)
    phys_engine = SmoothPhysicsLoss(mass=mass, caliber=caliber, steps=cfg.physics_steps).to(cfg.device)
    print(f"Physics loss steps: {cfg.physics_steps}")

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
        "test_data_loss": None,
        "test_phys_loss": None,
        "test_phys_eval_loss": None,
        "test_mae_theta": None,
        "test_mae_alpha": None,
        "best_epoch": None,
        "best_val_loss": None,
        "best_val_data_loss": None,
        "best_val_phys_loss": None,
        "trajectory_mode": int(cfg.trajectory_mode),
        "model_type": cfg.model_type,
        "noise": train_noise["summary"],
    }

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_sample_count = 0
        sum_tr_total = sum_tr_data = sum_tr_phys = sum_tr_mae_th = sum_tr_mae_al = None

        if cache_on_gpu:
            train_batches = _gpu_batch_iter(X_train_gpu, y_train_gpu, cfg.batch_size, shuffle=True)
        else:
            train_batches = train_loader

        for xb, yb in train_batches:
            if not cache_on_gpu:
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

            batch_n = xb.shape[0]
            train_sample_count += batch_n
            if sum_tr_total is None:
                sum_tr_total = out["total_loss"].detach() * batch_n
                sum_tr_data = out["loss_data"].detach() * batch_n
                sum_tr_phys = out["loss_phys"].detach() * batch_n
                sum_tr_mae_th = out["mae_theta"].detach() * batch_n
                sum_tr_mae_al = out["mae_alpha"].detach() * batch_n
            else:
                sum_tr_total += out["total_loss"].detach() * batch_n
                sum_tr_data += out["loss_data"].detach() * batch_n
                sum_tr_phys += out["loss_phys"].detach() * batch_n
                sum_tr_mae_th += out["mae_theta"].detach() * batch_n
                sum_tr_mae_al += out["mae_alpha"].detach() * batch_n

        ema.apply_to(model)
        if cache_on_gpu:
            val_info = evaluate_tensor_batches(
                model=model,
                X_tensor=X_val_gpu,
                y_tensor=y_val_gpu,
                batch_size=cfg.batch_size,
                phys_engine=phys_engine,
                xmin_tensor=xmin_tensor,
                span_tensor=span_tensor,
                epoch=epoch,
                lambda_phys=cfg.lambda_phys,
            )
        else:
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

        history["train_total"].append((sum_tr_total / train_sample_count).item())
        history["train_data"].append((sum_tr_data / train_sample_count).item())
        history["train_phys"].append((sum_tr_phys / train_sample_count).item())
        history["train_mae_theta"].append((sum_tr_mae_th / train_sample_count).item())
        history["train_mae_alpha"].append((sum_tr_mae_al / train_sample_count).item())
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

    if cache_on_gpu:
        test_info = evaluate_tensor_batches(
            model=model,
            X_tensor=X_test_gpu,
            y_tensor=y_test_gpu,
            batch_size=cfg.batch_size,
            phys_engine=phys_engine,
            xmin_tensor=xmin_tensor,
            span_tensor=span_tensor,
            epoch=best_epoch if best_epoch > 0 else 1,
            lambda_phys=cfg.lambda_phys,
        )
        test_phys_eval_info = evaluate_tensor_batches(
            model=model,
            X_tensor=X_test_gpu,
            y_tensor=y_test_gpu,
            batch_size=cfg.batch_size,
            phys_engine=phys_engine,
            xmin_tensor=xmin_tensor,
            span_tensor=span_tensor,
            epoch=12,
            lambda_phys=cfg.lambda_phys,
            eval_lambda_phys=1.0,
        ) if cfg.eval_phys_loss else None
    else:
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
        test_phys_eval_info = evaluate_model(
            model=model,
            loader=test_loader,
            device=cfg.device,
            phys_engine=phys_engine,
            xmin_tensor=xmin_tensor,
            span_tensor=span_tensor,
            epoch=12,
            lambda_phys=cfg.lambda_phys,
            eval_lambda_phys=1.0,
        ) if cfg.eval_phys_loss else None

    history["test_loss"] = test_info["loss"]
    history["test_data_loss"] = test_info["loss_data"]
    history["test_phys_loss"] = test_info["loss_phys"]
    history["test_phys_eval_loss"] = None if test_phys_eval_info is None else test_phys_eval_info["loss_phys"]
    history["test_mae_theta"] = test_info["mae_theta"]
    history["test_mae_alpha"] = test_info["mae_alpha"]
    history["best_epoch"] = int(best_epoch)
    history["best_val_loss"] = float(best_val)
    if best_epoch > 0:
        best_idx = best_epoch - 1
        history["best_val_data_loss"] = float(history["val_data"][best_idx])
        history["best_val_phys_loss"] = float(history["val_phys"][best_idx])

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
    print(f"Test data loss  : {test_info['loss_data']:.6f}")
    print(f"Test phys loss  : {test_info['loss_phys']:.6f}")
    if test_phys_eval_info is not None:
        print(f"Test phys eval  : {test_phys_eval_info['loss_phys']:.6f}")
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
