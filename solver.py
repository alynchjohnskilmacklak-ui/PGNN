"""Numerical solvers for refining ballistic angle predictions."""

import math

import numpy as np

from ballistics import ALPHA_ABS_MAX, ProjectileParams, simulate_trajectory, time_and_y_at_x


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

    newton_best, _ = _newton_only_refine(
        th_guess=grid_best["theta"],
        alpha_guess=grid_best["alpha"],
        x_target=x_target,
        y_target=y_target,
        z_target=z_target,
        params_env=params_env,
        th_min=th_min,
        th_max=th_max,
        dt=dt,
        t_max=t_max,
        y_tol=y_tol,
        z_tol=z_tol,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        max_iter=5,
    )
    if newton_best is None:
        return grid_best
    return newton_best if newton_best["err_3d"] <= grid_best["err_3d"] else grid_best
