import os
import json
import math
import csv
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ballistics import ProjectileParams, simulate_trajectory, time_and_y_at_x
from train_model import (
    SingleBranchDNN,
    minmax_transform,
    LOW_THETA_MIN,
    LOW_THETA_MAX,
    HIGH_THETA_MIN,
    HIGH_THETA_MAX,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHA_ABS_MAX = 15.0


def load_scaler(scaler_path: str) -> dict:
    if not os.path.isabs(scaler_path):
        scaler_path = os.path.join(BASE_DIR, scaler_path)
    with open(scaler_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_branch_model(
    model_path: str,
    scaler_path: str,
    theta_min: float,
    theta_max: float,
    device: Optional[str] = None,
    in_dim: int = 14,
    hidden: int = 256,
    dropout: float = 0.15,
):
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE_DIR, model_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = load_scaler(scaler_path)
    model = SingleBranchDNN(
        in_dim=in_dim,
        hidden=hidden,
        dropout=dropout,
        theta_min=theta_min,
        theta_max=theta_max,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, device


def infer_model_dims_from_state_dict(model_path: str) -> dict:
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE_DIR, model_path)
    state = torch.load(model_path, map_location="cpu")
    stem_weight = state["stem.0.weight"]
    head0_weight = state["head.mlp.0.weight"]
    return {
        "in_dim": int(stem_weight.shape[1]),
        "hidden": int(stem_weight.shape[0]),
        "dropout": 0.0,
        "head_hidden": int(head0_weight.shape[0]),
    }


def _predict_branch_angles(
    model,
    scaler,
    device,
    X_in: np.ndarray,
):
    X_norm = minmax_transform(X_in, scaler)
    with torch.no_grad():
        xb = torch.from_numpy(X_norm).to(device)
        pred = model(xb)
        return float(pred[0][0]), float(pred[0][1])


def _make_env_params(
    params,
    v0_actual,
    wind_x, wind_y, wind_z,
    cant_angle, T_powder_C,
    alt_gun, T0_C, P0_Pa,
) -> ProjectileParams:
    if params is None:
        params = ProjectileParams()
    p = ProjectileParams(**params.__dict__)
    p.T_powder_C = float(T_powder_C)
    p.v0_base = float(v0_actual) - p.temp_coeff * (float(T_powder_C) - 15.0)
    p.wind_x = float(wind_x)
    p.wind_y = float(wind_y)
    p.wind_z = float(wind_z)
    p.cant_angle_deg = float(cant_angle)
    p.alt_gun = float(alt_gun)
    p.T0_C = float(T0_C)
    p.P0_Pa = float(P0_Pa)
    return p


def _score_angle_at_target(
    th: float, alpha: float,
    x_target: float, y_target: float, z_target: float,
    params_env: ProjectileParams,
    dt: float, t_max: float,
):
    traj = simulate_trajectory(th, alpha_deg=alpha, params=params_env, dt=dt, t_max=t_max)
    if traj["range"] < x_target:
        return None
    t_hit, y_hit, z_hit = time_and_y_at_x(traj, x_target)
    if t_hit is None or (not np.isfinite(y_hit)) or (not np.isfinite(z_hit)):
        return None
    err_3d = math.sqrt((y_hit - y_target) ** 2 + (z_hit - z_target) ** 2)
    return {
        "theta": float(th),
        "alpha": float(alpha),
        "err_3d": float(err_3d),
        "y_err": float(abs(y_hit - y_target)),
        "z_err": float(abs(z_hit - z_target)),
        "t_hit": float(t_hit),
        "y_hit": float(y_hit),
        "z_hit": float(z_hit),
        "trajectory": traj,
    }


def _central_jacobian(
    th: float,
    al: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params_env: ProjectileParams,
    dt: float,
    t_max: float,
    dth: float,
    dal: float,
):
    base = _score_angle_at_target(th, al, x_target, y_target, z_target, params_env, dt, t_max)
    if base is None:
        return None, None

    th_p = _score_angle_at_target(th + dth, al, x_target, y_target, z_target, params_env, dt, t_max)
    th_m = _score_angle_at_target(th - dth, al, x_target, y_target, z_target, params_env, dt, t_max)
    al_p = _score_angle_at_target(th, al + dal, x_target, y_target, z_target, params_env, dt, t_max)
    al_m = _score_angle_at_target(th, al - dal, x_target, y_target, z_target, params_env, dt, t_max)

    if None in (th_p, th_m, al_p, al_m):
        return base, None

    dy_dth = (th_p["y_hit"] - th_m["y_hit"]) / (2.0 * dth)
    dz_dth = (th_p["z_hit"] - th_m["z_hit"]) / (2.0 * dth)
    dy_dal = (al_p["y_hit"] - al_m["y_hit"]) / (2.0 * dal)
    dz_dal = (al_p["z_hit"] - al_m["z_hit"]) / (2.0 * dal)

    J = np.array([[dy_dth, dy_dal], [dz_dth, dz_dal]], dtype=np.float64)
    return base, J


def _grid_search_refine(
    center_th: float,
    center_al: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params_env: ProjectileParams,
    dt: float,
    t_max: float,
    th_min: float,
    th_max: float,
    alpha_min: float,
    alpha_max: float,
    theta_radius: float,
    alpha_radius: float,
    theta_step: float,
    alpha_step: float,
):
    theta_values = np.arange(center_th - theta_radius, center_th + theta_radius + 1e-12, theta_step)
    alpha_values = np.arange(center_al - alpha_radius, center_al + alpha_radius + 1e-12, alpha_step)
    best_sol = None

    for th in theta_values:
        th = float(np.clip(th, th_min, th_max))
        for al in alpha_values:
            al = float(np.clip(al, alpha_min, alpha_max))
            cand = _score_angle_at_target(th, al, x_target, y_target, z_target, params_env, dt, t_max)
            if cand is None:
                continue
            if best_sol is None or cand["err_3d"] < best_sol["err_3d"]:
                best_sol = dict(cand)
    return best_sol


def _coarse_to_fine_refine(
    th_guess: float,
    alpha_guess: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params_env: ProjectileParams,
    dt: float,
    t_max: float,
    th_min: float,
    th_max: float,
    alpha_min: float = -ALPHA_ABS_MAX,
    alpha_max: float = ALPHA_ABS_MAX,
):
    coarse = _grid_search_refine(
        center_th=th_guess,
        center_al=alpha_guess,
        x_target=x_target,
        y_target=y_target,
        z_target=z_target,
        params_env=params_env,
        dt=dt,
        t_max=t_max,
        th_min=th_min,
        th_max=th_max,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        theta_radius=2.0,
        alpha_radius=2.0,
        theta_step=0.5,
        alpha_step=0.5,
    )
    if coarse is None:
        return None

    fine = _grid_search_refine(
        center_th=coarse["theta"],
        center_al=coarse["alpha"],
        x_target=x_target,
        y_target=y_target,
        z_target=z_target,
        params_env=params_env,
        dt=dt,
        t_max=t_max,
        th_min=th_min,
        th_max=th_max,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        theta_radius=0.5,
        alpha_radius=0.5,
        theta_step=0.1,
        alpha_step=0.1,
    )
    return fine if fine is not None else coarse


def _newton_raphson_3d_refine(
    th_guess: float,
    alpha_guess: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params_env: ProjectileParams,
    dt: float,
    t_max: float,
    th_min: float,
    th_max: float,
    y_tol: float,
    z_tol: float,
    alpha_min: float = -ALPHA_ABS_MAX,
    alpha_max: float = ALPHA_ABS_MAX,
    max_iter: int = 15,
):
    th = float(np.clip(th_guess, th_min, th_max))
    al = float(np.clip(alpha_guess, alpha_min, alpha_max))
    dth = 0.05
    dal = 0.05
    best_sol = None

    for _ in range(max_iter):
        base, J = _central_jacobian(
            th, al, x_target, y_target, z_target, params_env, dt, t_max, dth, dal
        )
        if base is None:
            break

        if best_sol is None or base["err_3d"] < best_sol["err_3d"]:
            best_sol = dict(base)

        E_y = base["y_hit"] - y_target
        E_z = base["z_hit"] - z_target
        if abs(E_y) < y_tol and abs(E_z) < z_tol:
            break

        if J is None:
            break

        rhs = np.array([E_y, E_z], dtype=np.float64)
        try:
            delta = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, rhs, rcond=None)[0]

        delta_th = float(np.clip(delta[0], -2.0, 2.0))
        delta_al = float(np.clip(delta[1], -2.0, 2.0))

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            cand_th = float(np.clip(th - scale * delta_th, th_min, th_max))
            cand_al = float(np.clip(al - scale * delta_al, alpha_min, alpha_max))
            cand = _score_angle_at_target(
                cand_th, cand_al, x_target, y_target, z_target, params_env, dt, t_max
            )
            if cand is None:
                continue
            if best_sol is None or cand["err_3d"] <= best_sol["err_3d"]:
                th, al = cand_th, cand_al
                if best_sol is None or cand["err_3d"] < best_sol["err_3d"]:
                    best_sol = dict(cand)
                accepted = True
                break

        if not accepted:
            break

    return best_sol


def _newton_only_refine(
    th_guess: float,
    alpha_guess: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params_env: ProjectileParams,
    th_min: float,
    th_max: float,
    dt: float = 0.05,
    t_max: float = 120.0,
    alpha_min: float = -ALPHA_ABS_MAX,
    alpha_max: float = ALPHA_ABS_MAX,
    y_tol: float = 2.0,
    z_tol: float = 2.0,
    max_iter: int = 15,
    dth: float = 0.05,
    dal: float = 0.05,
):
    th = float(np.clip(th_guess, th_min, th_max))
    al = float(np.clip(alpha_guess, alpha_min, alpha_max))
    best = None
    iters = 0

    for _ in range(max_iter):
        iters += 1
        base = _score_angle_at_target(th, al, x_target, y_target, z_target, params_env, dt, t_max)
        if base is None:
            break
        if best is None or base["err_3d"] < best["err_3d"]:
            best = dict(base)

        E_y = base["y_hit"] - y_target
        E_z = base["z_hit"] - z_target
        if abs(E_y) < y_tol and abs(E_z) < z_tol:
            break

        th_p = _score_angle_at_target(th + dth, al, x_target, y_target, z_target, params_env, dt, t_max)
        th_m = _score_angle_at_target(th - dth, al, x_target, y_target, z_target, params_env, dt, t_max)
        al_p = _score_angle_at_target(th, al + dal, x_target, y_target, z_target, params_env, dt, t_max)
        al_m = _score_angle_at_target(th, al - dal, x_target, y_target, z_target, params_env, dt, t_max)
        if None in (th_p, th_m, al_p, al_m):
            break

        J = np.array([
            [(th_p["y_hit"] - th_m["y_hit"]) / (2 * dth), (al_p["y_hit"] - al_m["y_hit"]) / (2 * dal)],
            [(th_p["z_hit"] - th_m["z_hit"]) / (2 * dth), (al_p["z_hit"] - al_m["z_hit"]) / (2 * dal)],
        ], dtype=np.float64)
        rhs = np.array([E_y, E_z], dtype=np.float64)
        try:
            delta = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, rhs, rcond=None)[0]

        d_th = float(np.clip(delta[0], -2.0, 2.0))
        d_al = float(np.clip(delta[1], -2.0, 2.0))

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            cand_th = float(np.clip(th - scale * d_th, th_min, th_max))
            cand_al = float(np.clip(al - scale * d_al, alpha_min, alpha_max))
            cand = _score_angle_at_target(cand_th, cand_al, x_target, y_target, z_target, params_env, dt, t_max)
            if cand is None:
                continue
            if best is None or cand["err_3d"] <= best["err_3d"] + 1e-6:
                th, al = cand_th, cand_al
                if cand["err_3d"] < best["err_3d"]:
                    best = dict(cand)
                accepted = True
                break
        if not accepted:
            break

    return best, iters


def _refine_candidate(
    th_guess: float,
    alpha_guess: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params_env: ProjectileParams,
    dt: float,
    t_max: float,
    th_min: float,
    th_max: float,
    y_tol: float,
    z_tol: float,
    alpha_min: float = -ALPHA_ABS_MAX,
    alpha_max: float = ALPHA_ABS_MAX,
):
    grid_best = _coarse_to_fine_refine(
        th_guess=th_guess,
        alpha_guess=alpha_guess,
        x_target=x_target,
        y_target=y_target,
        z_target=z_target,
        params_env=params_env,
        dt=dt,
        t_max=t_max,
        th_min=th_min,
        th_max=th_max,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    if grid_best is None:
        return None

    newton_best = _newton_raphson_3d_refine(
        th_guess=grid_best["theta"],
        alpha_guess=grid_best["alpha"],
        x_target=x_target,
        y_target=y_target,
        z_target=z_target,
        params_env=params_env,
        dt=dt,
        t_max=t_max,
        th_min=th_min,
        th_max=th_max,
        y_tol=y_tol,
        z_tol=z_tol,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        max_iter=5,
    )
    if newton_best is None:
        return grid_best
    return newton_best if newton_best["err_3d"] <= grid_best["err_3d"] else grid_best


def _save_solution_plot(
    out_path: str,
    x_target: float,
    y_target: float,
    z_target: float,
    low_sol,
    high_sol,
    chosen,
):
    fig = plt.figure(figsize=(9.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    if low_sol is not None:
        traj = low_sol["trajectory"]
        ax.plot(traj["x"], traj["z"], traj["y"], label="Low Trajectory", linewidth=1.8)
    if high_sol is not None:
        traj = high_sol["trajectory"]
        ax.plot(traj["x"], traj["z"], traj["y"], label="High Trajectory", linewidth=1.8, linestyle="--")
    if chosen is not None:
        traj = chosen["trajectory"]
        ax.plot(
            traj["x"], traj["z"], traj["y"],
            label=f"Chosen ({chosen['mode']})",
            linewidth=2.8,
            color="crimson",
        )

    ax.scatter([x_target], [z_target], [y_target], color="black", s=40, label="Target")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title("Trajectory Refinement Result")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_nn_prediction_plot(
    out_path: str,
    x_target: float,
    y_target: float,
    z_target: float,
    low_traj,
    high_traj,
):
    fig = plt.figure(figsize=(9.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    if low_traj is not None:
        ax.plot(low_traj["x"], low_traj["z"], low_traj["y"], label="NN Low", linewidth=1.8)
    if high_traj is not None:
        ax.plot(high_traj["x"], high_traj["z"], high_traj["y"], label="NN High", linewidth=1.8, linestyle="--")

    ax.scatter([x_target], [z_target], [y_target], color="black", s=40, label="Target")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title("NN Prediction Before Refinement")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_prediction_compare_csv(
    out_path: str,
    x_target: float,
    y_target: float,
    z_target: float,
    nn_prediction: dict,
    low_nn_score,
    high_nn_score,
    low_solution,
    high_solution,
    chosen,
):
    low_nn_err = np.nan if low_nn_score is None else low_nn_score["err_3d"]
    high_nn_err = np.nan if high_nn_score is None else high_nn_score["err_3d"]
    low_refined_err = np.nan if low_solution is None else low_solution["err_3d"]
    high_refined_err = np.nan if high_solution is None else high_solution["err_3d"]
    chosen_err = np.nan if chosen is None else chosen["err_3d"]

    rows = [
        {
            "stage": "nn_low",
            "x_target": x_target,
            "y_target": y_target,
            "z_target": z_target,
            "mode": "low",
            "theta_deg": nn_prediction["low"]["theta"],
            "alpha_deg": nn_prediction["low"]["alpha"],
            "err_3d": np.nan if low_nn_score is None else low_nn_score["err_3d"],
            "y_err": np.nan if low_nn_score is None else low_nn_score["y_err"],
            "z_err": np.nan if low_nn_score is None else low_nn_score["z_err"],
            "t_hit": np.nan if low_nn_score is None else low_nn_score["t_hit"],
            "improve_vs_nn_low_err3d": 0.0 if np.isfinite(low_nn_err) else np.nan,
            "improve_vs_nn_high_err3d": np.nan,
        },
        {
            "stage": "nn_high",
            "x_target": x_target,
            "y_target": y_target,
            "z_target": z_target,
            "mode": "high",
            "theta_deg": nn_prediction["high"]["theta"],
            "alpha_deg": nn_prediction["high"]["alpha"],
            "err_3d": np.nan if high_nn_score is None else high_nn_score["err_3d"],
            "y_err": np.nan if high_nn_score is None else high_nn_score["y_err"],
            "z_err": np.nan if high_nn_score is None else high_nn_score["z_err"],
            "t_hit": np.nan if high_nn_score is None else high_nn_score["t_hit"],
            "improve_vs_nn_low_err3d": np.nan,
            "improve_vs_nn_high_err3d": 0.0 if np.isfinite(high_nn_err) else np.nan,
        },
        {
            "stage": "refined_low",
            "x_target": x_target,
            "y_target": y_target,
            "z_target": z_target,
            "mode": "low",
            "theta_deg": np.nan if low_solution is None else low_solution["theta"],
            "alpha_deg": np.nan if low_solution is None else low_solution["alpha"],
            "err_3d": np.nan if low_solution is None else low_solution["err_3d"],
            "y_err": np.nan if low_solution is None else low_solution["y_err"],
            "z_err": np.nan if low_solution is None else low_solution["z_err"],
            "t_hit": np.nan if low_solution is None else low_solution["t_hit"],
            "improve_vs_nn_low_err3d": np.nan if (not np.isfinite(low_nn_err) or not np.isfinite(low_refined_err)) else float(low_nn_err - low_refined_err),
            "improve_vs_nn_high_err3d": np.nan,
        },
        {
            "stage": "refined_high",
            "x_target": x_target,
            "y_target": y_target,
            "z_target": z_target,
            "mode": "high",
            "theta_deg": np.nan if high_solution is None else high_solution["theta"],
            "alpha_deg": np.nan if high_solution is None else high_solution["alpha"],
            "err_3d": np.nan if high_solution is None else high_solution["err_3d"],
            "y_err": np.nan if high_solution is None else high_solution["y_err"],
            "z_err": np.nan if high_solution is None else high_solution["z_err"],
            "t_hit": np.nan if high_solution is None else high_solution["t_hit"],
            "improve_vs_nn_low_err3d": np.nan,
            "improve_vs_nn_high_err3d": np.nan if (not np.isfinite(high_nn_err) or not np.isfinite(high_refined_err)) else float(high_nn_err - high_refined_err),
        },
        {
            "stage": "chosen",
            "x_target": x_target,
            "y_target": y_target,
            "z_target": z_target,
            "mode": "" if chosen is None else chosen["mode"],
            "theta_deg": np.nan if chosen is None else chosen["theta"],
            "alpha_deg": np.nan if chosen is None else chosen["alpha"],
            "err_3d": np.nan if chosen is None else chosen["err_3d"],
            "y_err": np.nan if chosen is None else chosen["y_err"],
            "z_err": np.nan if chosen is None else chosen["z_err"],
            "t_hit": np.nan if chosen is None else chosen["t_hit"],
            "improve_vs_nn_low_err3d": np.nan if (not np.isfinite(low_nn_err) or not np.isfinite(chosen_err)) else float(low_nn_err - chosen_err),
            "improve_vs_nn_high_err3d": np.nan if (not np.isfinite(high_nn_err) or not np.isfinite(chosen_err)) else float(high_nn_err - chosen_err),
        },
    ]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_error_compare_plot(
    out_path: str,
    low_nn_score,
    high_nn_score,
    low_solution,
    high_solution,
    chosen,
):
    labels = ["nn_low", "nn_high", "refined_low", "refined_high", "chosen"]
    values = [
        np.nan if low_nn_score is None else low_nn_score["err_3d"],
        np.nan if high_nn_score is None else high_nn_score["err_3d"],
        np.nan if low_solution is None else low_solution["err_3d"],
        np.nan if high_solution is None else high_solution["err_3d"],
        np.nan if chosen is None else chosen["err_3d"],
    ]

    x = np.arange(len(labels))
    plot_values = [0.0 if not np.isfinite(v) else float(v) for v in values]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B"]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(x, plot_values, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylabel("3D Error")
    ax.set_title("Prediction Error Comparison")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    for bar, raw in zip(bars, values):
        label = "NA" if not np.isfinite(raw) else f"{raw:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def solve_target_unified(
    x_target: float,
    y_target: float,
    z_target: float = 0.0,
    v0_actual: float = 840.0,
    rho: float = 1.225,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    wind_z: float = 0.0,
    cant_angle: float = 0.0,
    T_powder_C: float = 15.0,
    alt_gun: float = 0.0,
    T0_C: float = 15.0,
    P0_Pa: float = 101325.0,
    dir_path: str = "artifacts",
    low_model_path: str | None = None,
    low_scaler_path: str | None = None,
    high_model_path: str | None = None,
    high_scaler_path: str | None = None,
    loaded_low_model=None,
    loaded_low_scaler=None,
    loaded_high_model=None,
    loaded_high_scaler=None,
    params: ProjectileParams | None = None,
    y_tol: float = 2.0,
    z_tol: float = 2.0,
    dt: float = 0.05,
    t_max: float = 120.0,
    use_grid: bool = False,
    grid_fallback: bool = True,
    save_plot: bool = True,
    plot_path: str | None = None,
):
    params_env = _make_env_params(
        params, v0_actual,
        wind_x, wind_y, wind_z,
        cant_angle, T_powder_C,
        alt_gun, T0_C, P0_Pa,
    )
    slant_range = float(np.sqrt(x_target ** 2 + y_target ** 2 + z_target ** 2))

    X_in = np.array(
        [[x_target, y_target, z_target, v0_actual, rho, slant_range,
          wind_x, wind_y, wind_z, cant_angle, T_powder_C,
          T0_C, P0_Pa, alt_gun]],
        dtype=np.float32,
    )

    if loaded_low_model is not None and loaded_low_scaler is not None:
        low_model, low_scaler = loaded_low_model, loaded_low_scaler
        low_device = next(low_model.parameters()).device
    else:
        if low_model_path is None:
            low_model_path = os.path.join(dir_path, "dual_model_low.pt")
        if low_scaler_path is None:
            low_scaler_path = os.path.join(dir_path, "dual_scaler_low.json")
        low_model, low_scaler, low_device = load_branch_model(
            low_model_path, low_scaler_path,
            theta_min=LOW_THETA_MIN, theta_max=LOW_THETA_MAX, in_dim=14
        )

    if loaded_high_model is not None and loaded_high_scaler is not None:
        high_model, high_scaler = loaded_high_model, loaded_high_scaler
        high_device = next(high_model.parameters()).device
    else:
        if high_model_path is None:
            high_model_path = os.path.join(dir_path, "dual_model_high.pt")
        if high_scaler_path is None:
            high_scaler_path = os.path.join(dir_path, "dual_scaler_high.json")
        high_model, high_scaler, high_device = load_branch_model(
            high_model_path, high_scaler_path,
            theta_min=HIGH_THETA_MIN, theta_max=HIGH_THETA_MAX, in_dim=14
        )

    th_low_pred, al_low_pred = _predict_branch_angles(low_model, low_scaler, low_device, X_in)
    th_high_pred, al_high_pred = _predict_branch_angles(high_model, high_scaler, high_device, X_in)
    low_nn_score = _score_angle_at_target(
        th_low_pred, al_low_pred, x_target, y_target, z_target, params_env, dt, t_max
    )
    high_nn_score = _score_angle_at_target(
        th_high_pred, al_high_pred, x_target, y_target, z_target, params_env, dt, t_max
    )
    low_nn_traj = None if low_nn_score is None else low_nn_score["trajectory"]
    high_nn_traj = None if high_nn_score is None else high_nn_score["trajectory"]

    if use_grid:
        low_sol = _refine_candidate(
            th_guess=th_low_pred, alpha_guess=al_low_pred,
            x_target=x_target, y_target=y_target, z_target=z_target,
            params_env=params_env, dt=dt, t_max=t_max,
            th_min=LOW_THETA_MIN, th_max=LOW_THETA_MAX,
            y_tol=y_tol, z_tol=z_tol,
        )
        high_sol = _refine_candidate(
            th_guess=th_high_pred, alpha_guess=al_high_pred,
            x_target=x_target, y_target=y_target, z_target=z_target,
            params_env=params_env, dt=dt, t_max=t_max,
            th_min=HIGH_THETA_MIN, th_max=HIGH_THETA_MAX,
            y_tol=y_tol, z_tol=z_tol,
        )
    else:
        low_sol, _ = _newton_only_refine(
            th_low_pred, al_low_pred,
            x_target, y_target, z_target, params_env,
            LOW_THETA_MIN, LOW_THETA_MAX,
            dt=dt, t_max=t_max, y_tol=y_tol, z_tol=z_tol,
        )
        high_sol, _ = _newton_only_refine(
            th_high_pred, al_high_pred,
            x_target, y_target, z_target, params_env,
            HIGH_THETA_MIN, HIGH_THETA_MAX,
            dt=dt, t_max=t_max, y_tol=y_tol, z_tol=z_tol,
        )
        if grid_fallback:
            tol = max(y_tol, z_tol)
            if low_sol is None or low_sol["err_3d"] > tol:
                low_sol = _refine_candidate(
                    th_guess=th_low_pred, alpha_guess=al_low_pred,
                    x_target=x_target, y_target=y_target, z_target=z_target,
                    params_env=params_env, dt=dt, t_max=t_max,
                    th_min=LOW_THETA_MIN, th_max=LOW_THETA_MAX,
                    y_tol=y_tol, z_tol=z_tol,
                )
            if high_sol is None or high_sol["err_3d"] > tol:
                high_sol = _refine_candidate(
                    th_guess=th_high_pred, alpha_guess=al_high_pred,
                    x_target=x_target, y_target=y_target, z_target=z_target,
                    params_env=params_env, dt=dt, t_max=t_max,
                    th_min=HIGH_THETA_MIN, th_max=HIGH_THETA_MAX,
                    y_tol=y_tol, z_tol=z_tol,
                )

    def _valid(sol):
        return sol is not None and sol["y_err"] <= y_tol and sol["z_err"] <= z_tol

    if _valid(low_sol):
        low_sol["mode"] = "low"
        low_sol["theta_nn"] = th_low_pred
        low_sol["alpha_nn"] = al_low_pred
    if _valid(high_sol):
        high_sol["mode"] = "high"
        high_sol["theta_nn"] = th_high_pred
        high_sol["alpha_nn"] = al_high_pred

    solutions = [sol for sol in (low_sol, high_sol) if _valid(sol)]

    out = {
        "reachable": len(solutions) > 0,
        "nn_prediction": {
            "low": {"theta": th_low_pred, "alpha": al_low_pred},
            "high": {"theta": th_high_pred, "alpha": al_high_pred},
        },
        "solutions": solutions,
        "low_solution": low_sol if _valid(low_sol) else None,
        "high_solution": high_sol if _valid(high_sol) else None,
    }

    if solutions:
        chosen = min(
            solutions,
            key=lambda s: (
                s["err_3d"],
                s["t_hit"],
            ),
        )
        out.update({
            "chosen": chosen,
            "theta_chosen": chosen["theta"],
            "alpha_chosen": chosen["alpha"],
            "trajectory": chosen["trajectory"],
        })
    else:
        out.update({
            "chosen": None,
            "theta_chosen": None,
            "alpha_chosen": None,
            "trajectory": None,
        })

    if save_plot:
        if plot_path is None:
            plot_path = os.path.join(dir_path, "prediction_refine_plot.png")
        nn_plot_path = os.path.join(dir_path, "prediction_before_refine_plot.png")
        compare_csv_path = os.path.join(dir_path, "prediction_compare.csv")
        error_plot_path = os.path.join(dir_path, "prediction_error_compare.png")
        _save_nn_prediction_plot(
            out_path=nn_plot_path,
            x_target=x_target,
            y_target=y_target,
            z_target=z_target,
            low_traj=low_nn_traj,
            high_traj=high_nn_traj,
        )
        _save_solution_plot(
            out_path=plot_path,
            x_target=x_target,
            y_target=y_target,
            z_target=z_target,
            low_sol=out["low_solution"],
            high_sol=out["high_solution"],
            chosen=out["chosen"],
        )
        _save_prediction_compare_csv(
            out_path=compare_csv_path,
            x_target=x_target,
            y_target=y_target,
            z_target=z_target,
            nn_prediction=out["nn_prediction"],
            low_nn_score=low_nn_score,
            high_nn_score=high_nn_score,
            low_solution=out["low_solution"],
            high_solution=out["high_solution"],
            chosen=out["chosen"],
        )
        _save_error_compare_plot(
            out_path=error_plot_path,
            low_nn_score=low_nn_score,
            high_nn_score=high_nn_score,
            low_solution=out["low_solution"],
            high_solution=out["high_solution"],
            chosen=out["chosen"],
        )
        out["plot_path"] = plot_path
        out["nn_plot_path"] = nn_plot_path
        out["compare_csv_path"] = compare_csv_path
        out["error_plot_path"] = error_plot_path
    else:
        out["plot_path"] = None
        out["nn_plot_path"] = None
        out["compare_csv_path"] = None
        out["error_plot_path"] = None

    return out


if __name__ == "__main__":
    DEMO_CONFIG = {
        "dir_path": "artifacts_127",
        "x_target": 800.0,
        "y_target": 5.0,
        "z_target": 0.0,
        "v0_actual": 840.0,
        "rho": 1.225,
        "wind_x": 0.0,
        "wind_y": 0.0,
        "wind_z": 2.0,
        "cant_angle": 0.0,
        "T_powder_C": 15.0,
        "alt_gun": 0.0,
        "T0_C": 15.0,
        "P0_Pa": 101325.0,
        "save_plot": True,
    }

    params = ProjectileParams(mass=0.04828, caliber=0.01295)
    result = solve_target_unified(
        x_target=DEMO_CONFIG["x_target"],
        y_target=DEMO_CONFIG["y_target"],
        z_target=DEMO_CONFIG["z_target"],
        v0_actual=DEMO_CONFIG["v0_actual"],
        rho=DEMO_CONFIG["rho"],
        wind_x=DEMO_CONFIG["wind_x"],
        wind_y=DEMO_CONFIG["wind_y"],
        wind_z=DEMO_CONFIG["wind_z"],
        cant_angle=DEMO_CONFIG["cant_angle"],
        T_powder_C=DEMO_CONFIG["T_powder_C"],
        alt_gun=DEMO_CONFIG["alt_gun"],
        T0_C=DEMO_CONFIG["T0_C"],
        P0_Pa=DEMO_CONFIG["P0_Pa"],
        dir_path=DEMO_CONFIG["dir_path"],
        params=params,
        save_plot=DEMO_CONFIG["save_plot"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
