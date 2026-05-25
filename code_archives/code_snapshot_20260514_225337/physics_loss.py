"""Differentiable ballistics physics loss functions for neural network training."""

import math
from typing import Dict

import torch
import torch.nn as nn


class SmoothPhysicsLoss(nn.Module):
    def __init__(self, mass: float = 0.04828, caliber: float = 0.01295, dt: float = 0.05, steps: int = 10):
        super().__init__()
        self.mass = float(mass)
        self.area = math.pi * (caliber / 2.0) ** 2
        self.dt = float(dt)
        self.steps = int(max(1, steps))
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

    def wind_scale_for_height(self, y: torch.Tensor) -> torch.Tensor:
        h = torch.clamp(y, min=self.z0 + 0.01)
        scale = torch.log(h / self.z0) / math.log(self.h_ref / self.z0)
        return torch.clamp(scale, 0.0, 5.0)

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

        T_kelvin = T0_C + 273.15
        rho = (P0_Pa * 0.0289644) / (8.3144598 * torch.clamp(T_kelvin, min=1.0))
        x = torch.zeros_like(vx)
        y = torch.zeros_like(vy)
        z = torch.zeros_like(vz)
        dt = self.dt

        for _ in range(self.steps):
            scale = self.wind_scale_for_height(y)
            vx_rel = vx - wind_x * scale
            vy_rel = vy - wind_y
            vz_rel = vz - wind_z * scale
            v_rel = torch.sqrt(vx_rel ** 2 + vy_rel ** 2 + vz_rel ** 2 + 1e-8)

            cd = self.smooth_cd_g7(v_rel, T_kelvin)
            k = 0.5 * rho * cd * self.area

            ax = -(k / self.mass) * v_rel * vx_rel
            ay = -self.g0 - (k / self.mass) * v_rel * vy_rel
            az = -(k / self.mass) * v_rel * vz_rel

            x = x + vx * dt + 0.5 * ax * dt ** 2
            y = y + vy * dt + 0.5 * ay * dt ** 2
            z = z + vz * dt + 0.5 * az * dt ** 2
            vx = vx + ax * dt
            vy = vy + ay * dt
            vz = vz + az * dt

        return x, y, z


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
    pred_x, pred_y, pred_z = phys_engine(pred_angles, X_raw)
    with torch.no_grad():
        true_x, true_y, true_z = phys_engine(y_true, X_raw)

    sx = torch.clamp(torch.abs(true_x), min=1.0)
    sy = torch.clamp(torch.abs(true_y), min=1.0)
    sz = torch.clamp(torch.abs(true_z), min=1.0)

    lx = _per_elem_smooth_l1((pred_x - true_x) / sx, delta=0.05)
    ly = _per_elem_smooth_l1((pred_y - true_y) / sy, delta=0.05)
    lz = _per_elem_smooth_l1((pred_z - true_z) / sz, delta=0.05)
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
    lambda_eff = _scheduled_lambda_phys(epoch, lambda_phys)
    if lambda_eff > 0.0:
        loss_phys = _weighted_mean(_physics_residual_loss(phys_engine, pred, yb, X_raw), sample_w)
    else:
        loss_phys = loss_data.new_zeros(())
    total_loss = loss_data + lambda_eff * loss_phys

    return {
        "loss_data": loss_data,
        "loss_phys": loss_phys,
        "total_loss": total_loss,
        "mae_theta": torch.mean(torch.abs(pred[:, 0] - yb[:, 0])),
        "mae_alpha": torch.mean(torch.abs(pred[:, 1] - yb[:, 1])),
    }
