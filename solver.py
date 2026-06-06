"""Numerical solvers for refining ballistic angle predictions."""

import math

import numpy as np

from ballistics import ALPHA_ABS_MAX, ProjectileParams, simulate_trajectory, time_and_y_at_x

# =============================================================================
# Configuration constants
# =============================================================================

DEFAULT_MAX_REFINE_ITER = 5
DEFAULT_DTH = 0.05
DEFAULT_DAL = 0.05
DEFAULT_STEP_CLIP = 2.0
DEFAULT_DAMPING_SCALES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)

GRID_COARSE_THETA_RADIUS = 2.0
GRID_COARSE_ALPHA_RADIUS = 2.0
GRID_COARSE_THETA_STEP = 0.5
GRID_COARSE_ALPHA_STEP = 0.5

GRID_FINE_THETA_RADIUS = 0.5
GRID_FINE_ALPHA_RADIUS = 0.5
GRID_FINE_THETA_STEP = 0.1
GRID_FINE_ALPHA_STEP = 0.1

# =============================================================================
# 1. 基础评分函数
# =============================================================================

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


# =============================================================================
# 通用小工具
# =============================================================================

def _is_solution_within_tol(sol: dict | None, y_tol: float, z_tol: float) -> bool:
    """Return True if *sol* satisfies both y and z tolerances."""
    if sol is None:
        return False
    return sol["y_err"] <= y_tol and sol["z_err"] <= z_tol


def _tag_solution(
    sol: dict | None, source: str, iters: int = 0, used_grid: bool = False,
) -> dict | None:
    """Return a shallow copy of *sol* with diagnostic keys attached."""
    if sol is None:
        return None
    tagged = dict(sol)
    tagged["refine_source"] = source
    tagged["refine_iters"] = int(iters)
    tagged["used_grid"] = bool(used_grid)
    return tagged


def _best_by_error(candidates: list[dict | None]) -> dict | None:
    """Return the candidate with the smallest err_3d, or None if all invalid."""
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None
    return min(valid, key=lambda c: c["err_3d"])


# =============================================================================
# 2. 网格搜索函数
# =============================================================================

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
        theta_radius=GRID_COARSE_THETA_RADIUS,
        alpha_radius=GRID_COARSE_ALPHA_RADIUS,
        theta_step=GRID_COARSE_THETA_STEP,
        alpha_step=GRID_COARSE_ALPHA_STEP,
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
        theta_radius=GRID_FINE_THETA_RADIUS,
        alpha_radius=GRID_FINE_ALPHA_RADIUS,
        theta_step=GRID_FINE_THETA_STEP,
        alpha_step=GRID_FINE_ALPHA_STEP,
    )
    return fine if fine is not None else coarse


# =============================================================================
# 3. Newton 精修函数
# =============================================================================

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
    dth: float = DEFAULT_DTH,
    dal: float = DEFAULT_DAL,
    step_clip: float = DEFAULT_STEP_CLIP,
    damping_scales: tuple[float, ...] = DEFAULT_DAMPING_SCALES,
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

        d_th = float(np.clip(delta[0], -step_clip, step_clip))
        d_al = float(np.clip(delta[1], -step_clip, step_clip))

        accepted = False
        for scale in damping_scales:
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


# =============================================================================
# 4. Broyden 精修函数
# =============================================================================

def _broyden_refine(
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
    max_iter: int = DEFAULT_MAX_REFINE_ITER,
    dth: float = DEFAULT_DTH,
    dal: float = DEFAULT_DAL,
    step_clip: float = DEFAULT_STEP_CLIP,
    damping_scales: tuple[float, ...] = DEFAULT_DAMPING_SCALES,
):
    """Broyden quasi-Newton refinement.
    Uses one initial central-difference Jacobian and then updates it with good
    Broyden updates.  This avoids recomputing the full Jacobian at every
    iteration, reducing repeated trajectory simulations.
    Returns (best, iters)."""
    th = float(np.clip(th_guess, th_min, th_max))
    al = float(np.clip(alpha_guess, alpha_min, alpha_max))

    base = _score_angle_at_target(th, al, x_target, y_target, z_target, params_env, dt, t_max)
    if base is None:
        return None, 0

    best = dict(base)
    iters = 1

    F_old = np.array([base["y_hit"] - y_target, base["z_hit"] - z_target], dtype=np.float64)

    if abs(F_old[0]) < y_tol and abs(F_old[1]) < z_tol:
        return best, iters

    th_p = _score_angle_at_target(th + dth, al, x_target, y_target, z_target, params_env, dt, t_max)
    th_m = _score_angle_at_target(th - dth, al, x_target, y_target, z_target, params_env, dt, t_max)
    al_p = _score_angle_at_target(th, al + dal, x_target, y_target, z_target, params_env, dt, t_max)
    al_m = _score_angle_at_target(th, al - dal, x_target, y_target, z_target, params_env, dt, t_max)
    if None in (th_p, th_m, al_p, al_m):
        return best, iters

    J = np.array([
        [(th_p["y_hit"] - th_m["y_hit"]) / (2 * dth), (al_p["y_hit"] - al_m["y_hit"]) / (2 * dal)],
        [(th_p["z_hit"] - th_m["z_hit"]) / (2 * dth), (al_p["z_hit"] - al_m["z_hit"]) / (2 * dal)],
    ], dtype=np.float64)

    for _ in range(max_iter):
        if abs(F_old[0]) < y_tol and abs(F_old[1]) < z_tol:
            break

        try:
            delta = np.linalg.solve(J, F_old)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, F_old, rcond=None)[0]

        delta = np.clip(delta, -step_clip, step_clip)

        accepted = False
        for scale in damping_scales:
            cand_th = float(np.clip(th - scale * delta[0], th_min, th_max))
            cand_al = float(np.clip(al - scale * delta[1], alpha_min, alpha_max))

            cand = _score_angle_at_target(
                cand_th, cand_al, x_target, y_target, z_target, params_env, dt, t_max,
            )
            if cand is None:
                continue

            if cand["err_3d"] <= best["err_3d"] + 1e-6:
                x_old = np.array([th, al], dtype=np.float64)
                x_new = np.array([cand_th, cand_al], dtype=np.float64)

                F_new = np.array([
                    cand["y_hit"] - y_target,
                    cand["z_hit"] - z_target,
                ], dtype=np.float64)

                s = x_new - x_old
                y_vec = F_new - F_old

                denom = float(s @ s)
                if denom > 1e-14:
                    J = J + np.outer((y_vec - J @ s), s) / denom

                th = cand_th
                al = cand_al
                base = cand
                F_old = F_new
                iters += 1

                if cand["err_3d"] < best["err_3d"]:
                    best = dict(cand)

                accepted = True
                break

        if not accepted:
            break

    return best, iters


# =============================================================================
# 5. 精修策略辅助函数
# =============================================================================

def _try_direct_broyden(
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
    alpha_min: float,
    alpha_max: float,
) -> dict | None:
    """Run Broyden refinement directly from the initial guess (fast path)."""
    broyden_best, broyden_iters = _broyden_refine(
        th_guess=th_guess,
        alpha_guess=alpha_guess,
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
        max_iter=DEFAULT_MAX_REFINE_ITER,
        dth=DEFAULT_DTH,
        dal=DEFAULT_DAL,
        step_clip=DEFAULT_STEP_CLIP,
        damping_scales=DEFAULT_DAMPING_SCALES,
    )
    return _tag_solution(
        broyden_best,
        source="direct_broyden",
        iters=broyden_iters,
        used_grid=False,
    )


def _try_grid_broyden_fallback(
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
    alpha_min: float,
    alpha_max: float,
) -> tuple[dict | None, dict | None]:
    """Run coarse-to-fine grid search, then Broyden from the grid result.

    Returns (tagged_grid_best, tagged_grid_broyden).  If grid finds nothing
    the second element is None.
    """
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

    tagged_grid = _tag_solution(grid_best, source="grid_best", iters=0, used_grid=True)

    if grid_best is None:
        return tagged_grid, None

    grid_broyden_best, grid_broyden_iters = _broyden_refine(
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
        max_iter=DEFAULT_MAX_REFINE_ITER,
        dth=DEFAULT_DTH,
        dal=DEFAULT_DAL,
        step_clip=DEFAULT_STEP_CLIP,
        damping_scales=DEFAULT_DAMPING_SCALES,
    )

    tagged_grid_broyden = _tag_solution(
        grid_broyden_best,
        source="grid_broyden",
        iters=grid_broyden_iters,
        used_grid=True,
    )

    return tagged_grid, tagged_grid_broyden


# =============================================================================
# 6. 统一入口
# =============================================================================

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
    refine_mode: str = "broyden_fast",
):
    """Adaptive refinement with selectable strategy.

    refine_mode values:
      - "newton":       Newton refinement directly from input guess (no grid).
      - "broyden":      Broyden refinement directly from input guess (no grid).
      - "grid_newton":  coarse-to-fine grid search, then Newton.
      - "grid_broyden": coarse-to-fine grid search, then Broyden.
      - "broyden_fast": direct Broyden → if converged return; else grid + Broyden.

    Every returned dict includes diagnostic keys:
      refine_source, refine_iters, used_grid.
    """
    # ── newton ──────────────────────────────────────────────────────────
    if refine_mode == "newton":
        best, iters = _newton_only_refine(
            th_guess=th_guess, alpha_guess=alpha_guess,
            x_target=x_target, y_target=y_target, z_target=z_target,
            params_env=params_env, th_min=th_min, th_max=th_max,
            dt=dt, t_max=t_max, y_tol=y_tol, z_tol=z_tol,
            alpha_min=alpha_min, alpha_max=alpha_max,
            max_iter=DEFAULT_MAX_REFINE_ITER,
        )
        return _tag_solution(best, source="newton", iters=iters, used_grid=False)

    # ── broyden ────────────────────────────────────────────────────────
    if refine_mode == "broyden":
        best, iters = _broyden_refine(
            th_guess=th_guess, alpha_guess=alpha_guess,
            x_target=x_target, y_target=y_target, z_target=z_target,
            params_env=params_env, th_min=th_min, th_max=th_max,
            dt=dt, t_max=t_max, y_tol=y_tol, z_tol=z_tol,
            alpha_min=alpha_min, alpha_max=alpha_max,
        )
        return _tag_solution(best, source="direct_broyden", iters=iters, used_grid=False)

    # ── grid_newton ────────────────────────────────────────────────────
    if refine_mode == "grid_newton":
        grid_best = _coarse_to_fine_refine(
            th_guess=th_guess, alpha_guess=alpha_guess,
            x_target=x_target, y_target=y_target, z_target=z_target,
            params_env=params_env, dt=dt, t_max=t_max,
            th_min=th_min, th_max=th_max,
            alpha_min=alpha_min, alpha_max=alpha_max,
        )
        if grid_best is None:
            return None
        newton_best, iters = _newton_only_refine(
            th_guess=grid_best["theta"], alpha_guess=grid_best["alpha"],
            x_target=x_target, y_target=y_target, z_target=z_target,
            params_env=params_env, th_min=th_min, th_max=th_max,
            dt=dt, t_max=t_max, y_tol=y_tol, z_tol=z_tol,
            alpha_min=alpha_min, alpha_max=alpha_max,
            max_iter=DEFAULT_MAX_REFINE_ITER,
        )
        if newton_best is not None and newton_best["err_3d"] < grid_best["err_3d"]:
            return _tag_solution(newton_best, source="grid_newton", iters=iters, used_grid=True)
        return _tag_solution(grid_best, source="grid_best", iters=0, used_grid=True)

    # ── grid_broyden ───────────────────────────────────────────────────
    if refine_mode == "grid_broyden":
        grid_best = _coarse_to_fine_refine(
            th_guess=th_guess, alpha_guess=alpha_guess,
            x_target=x_target, y_target=y_target, z_target=z_target,
            params_env=params_env, dt=dt, t_max=t_max,
            th_min=th_min, th_max=th_max,
            alpha_min=alpha_min, alpha_max=alpha_max,
        )
        if grid_best is None:
            return None
        grid_broyden_best, iters = _broyden_refine(
            th_guess=grid_best["theta"], alpha_guess=grid_best["alpha"],
            x_target=x_target, y_target=y_target, z_target=z_target,
            params_env=params_env, th_min=th_min, th_max=th_max,
            dt=dt, t_max=t_max, y_tol=y_tol, z_tol=z_tol,
            alpha_min=alpha_min, alpha_max=alpha_max,
        )
        if grid_broyden_best is not None and grid_broyden_best["err_3d"] < grid_best["err_3d"]:
            return _tag_solution(grid_broyden_best, source="grid_broyden", iters=iters, used_grid=True)
        return _tag_solution(grid_best, source="grid_best", iters=0, used_grid=True)

    # ── broyden_fast (default) ─────────────────────────────────────────
    # Fast path:
    # Try Broyden directly from the NN-predicted angles.
    # If it satisfies y/z tolerances, skip grid search.
    direct_broyden = _try_direct_broyden(
        th_guess=th_guess, alpha_guess=alpha_guess,
        x_target=x_target, y_target=y_target, z_target=z_target,
        params_env=params_env, dt=dt, t_max=t_max,
        th_min=th_min, th_max=th_max,
        y_tol=y_tol, z_tol=z_tol,
        alpha_min=alpha_min, alpha_max=alpha_max,
    )

    if _is_solution_within_tol(direct_broyden, y_tol, z_tol):
        return direct_broyden

    # Fallback path:
    # Run coarse-to-fine grid search, then apply Broyden from the grid result.
    # Return the lowest-error candidate.
    grid_best, grid_broyden = _try_grid_broyden_fallback(
        th_guess=th_guess, alpha_guess=alpha_guess,
        x_target=x_target, y_target=y_target, z_target=z_target,
        params_env=params_env, dt=dt, t_max=t_max,
        th_min=th_min, th_max=th_max,
        y_tol=y_tol, z_tol=z_tol,
        alpha_min=alpha_min, alpha_max=alpha_max,
    )

    # If grid also failed, return the direct result (tagged as grid-attempted).
    if grid_best is None:
        if direct_broyden is not None:
            direct_broyden["used_grid"] = True
        return direct_broyden

    return _best_by_error([direct_broyden, grid_best, grid_broyden])
