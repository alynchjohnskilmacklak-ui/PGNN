"""benchmark_newton_vs_broyden_solver.py

用途：
    Newton vs Broyden 求解器离线对比 benchmark。
    在相同初始条件下对比两种迭代方法的耗时、迭代次数、
    仿真调用次数和最终 3D 误差。

适用场景：
    评估 Broyden 是否比 Newton 更高效（更少的仿真调用）。

输入：
    不依赖 NN 模型，使用正向仿真生成测试样本。

输出：
    benchmark_newton_vs_broyden_solver.csv（逐样本结果）
    benchmark_newton_vs_broyden_summary.csv（按方法汇总）

运行方式：
    python scripts/benchmarks/benchmark_newton_vs_broyden_solver.py --n_samples 100 --seed 20260606
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional, Callable

import numpy as np
import pandas as pd

from ballistics import (
    ALPHA_ABS_MAX,
    ProjectileParams,
    simulate_trajectory,
    time_and_y_at_x,
    isa_pressure,
    get_atmosphere,
)
from generate_dataset import AMMO_CONFIGS, CURRENT_AMMO
from model_architecture import LOW_THETA_MIN, LOW_THETA_MAX, HIGH_THETA_MIN, HIGH_THETA_MAX
import solver  # noqa: F401  (referenced for doc/cross-check; not called directly)


# =============================================================================
# Counter
# =============================================================================

@dataclass
class SimCounter:
    """Counts simulate_trajectory calls."""
    n: int = 0

    def reset(self) -> None:
        self.n = 0

    def inc(self) -> None:
        self.n += 1


# =============================================================================
# Counted score function (wraps simulate_trajectory)
# =============================================================================

def counted_score_angle_at_target(
    th: float,
    alpha: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params_env: ProjectileParams,
    dt: float,
    t_max: float,
    counter: SimCounter,
) -> Optional[dict]:
    """Score a (theta, alpha) pair against a target, counting the sim call.

    Returns None if the trajectory does not reach *x_target*.
    """
    traj = simulate_trajectory(th, alpha_deg=alpha, params=params_env, dt=dt, t_max=t_max)
    counter.inc()
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
# Newton benchmark (replicates solver._newton_only_refine with counting)
# =============================================================================

def benchmark_newton_refine(
    th_guess: float,
    alpha_guess: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params_env: ProjectileParams,
    th_min: float,
    th_max: float,
    dt: float,
    t_max: float,
    alpha_min: float,
    alpha_max: float,
    y_tol: float,
    z_tol: float,
    max_iter: int,
    dth: float,
    dal: float,
    step_clip: float,
    damping_scales: tuple[float, ...],
) -> dict:
    """Newton refinement with central-difference Jacobian and full call counting.

    Replicates the logic of solver._newton_only_refine() but routes every
    simulation through counted_score_angle_at_target for accurate metrics.
    """
    counter = SimCounter()
    jacobian_calls = 0
    t_start = time.perf_counter()

    th = float(np.clip(th_guess, th_min, th_max))
    al = float(np.clip(alpha_guess, alpha_min, alpha_max))
    best = None
    iters = 0

    for _ in range(max_iter):
        iters += 1
        base = counted_score_angle_at_target(
            th, al, x_target, y_target, z_target, params_env, dt, t_max, counter,
        )
        if base is None:
            break
        if best is None or base["err_3d"] < best["err_3d"]:
            best = dict(base)

        E_y = base["y_hit"] - y_target
        E_z = base["z_hit"] - z_target
        if abs(E_y) < y_tol and abs(E_z) < z_tol:
            break

        th_p = counted_score_angle_at_target(
            th + dth, al, x_target, y_target, z_target, params_env, dt, t_max, counter,
        )
        th_m = counted_score_angle_at_target(
            th - dth, al, x_target, y_target, z_target, params_env, dt, t_max, counter,
        )
        al_p = counted_score_angle_at_target(
            th, al + dal, x_target, y_target, z_target, params_env, dt, t_max, counter,
        )
        al_m = counted_score_angle_at_target(
            th, al - dal, x_target, y_target, z_target, params_env, dt, t_max, counter,
        )
        jacobian_calls += 1
        if None in (th_p, th_m, al_p, al_m):
            break

        J = np.array(
            [
                [(th_p["y_hit"] - th_m["y_hit"]) / (2 * dth), (al_p["y_hit"] - al_m["y_hit"]) / (2 * dal)],
                [(th_p["z_hit"] - th_m["z_hit"]) / (2 * dth), (al_p["z_hit"] - al_m["z_hit"]) / (2 * dal)],
            ],
            dtype=np.float64,
        )
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
            cand = counted_score_angle_at_target(
                cand_th, cand_al, x_target, y_target, z_target, params_env, dt, t_max, counter,
            )
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

    wall_ms = (time.perf_counter() - t_start) * 1000
    simulate_calls = counter.n
    function_calls = counter.n

    return {
        "best": best,
        "iterations": iters,
        "wall_ms": wall_ms,
        "simulate_calls": simulate_calls,
        "function_calls": function_calls,
        "jacobian_calls": jacobian_calls,
    }


# =============================================================================
# Broyden benchmark
# =============================================================================

def benchmark_broyden_refine(
    th_guess: float,
    alpha_guess: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params_env: ProjectileParams,
    th_min: float,
    th_max: float,
    dt: float,
    t_max: float,
    alpha_min: float,
    alpha_max: float,
    y_tol: float,
    z_tol: float,
    max_iter: int,
    dth: float,
    dal: float,
    step_clip: float,
    damping_scales: tuple[float, ...],
) -> dict:
    """Broyden (quasi-Newton) refinement for ballistic angle solving.

    Uses a central-difference Jacobian only on the first iteration; subsequent
    iterations apply the good Broyden rank-1 update to approximate the Jacobian.
    """
    counter = SimCounter()
    jacobian_calls = 0
    t_start = time.perf_counter()

    th = float(np.clip(th_guess, th_min, th_max))
    al = float(np.clip(alpha_guess, alpha_min, alpha_max))
    x_vec = np.array([th, al], dtype=np.float64)

    # ---- initial residual ----
    base = counted_score_angle_at_target(
        th, al, x_target, y_target, z_target, params_env, dt, t_max, counter,
    )
    if base is None:
        wall_ms = (time.perf_counter() - t_start) * 1000
        return {
            "best": None, "iterations": 0, "wall_ms": wall_ms,
            "simulate_calls": counter.n, "function_calls": counter.n,
            "jacobian_calls": 0,
        }

    best = dict(base)
    F = np.array([base["y_hit"] - y_target, base["z_hit"] - z_target], dtype=np.float64)

    if abs(F[0]) < y_tol and abs(F[1]) < z_tol:
        wall_ms = (time.perf_counter() - t_start) * 1000
        return {
            "best": best, "iterations": 0, "wall_ms": wall_ms,
            "simulate_calls": counter.n, "function_calls": counter.n,
            "jacobian_calls": 0,
        }

    # ---- initial Jacobian via central differences ----
    th_p = counted_score_angle_at_target(
        th + dth, al, x_target, y_target, z_target, params_env, dt, t_max, counter,
    )
    th_m = counted_score_angle_at_target(
        th - dth, al, x_target, y_target, z_target, params_env, dt, t_max, counter,
    )
    al_p = counted_score_angle_at_target(
        th, al + dal, x_target, y_target, z_target, params_env, dt, t_max, counter,
    )
    al_m = counted_score_angle_at_target(
        th, al - dal, x_target, y_target, z_target, params_env, dt, t_max, counter,
    )
    jacobian_calls += 1
    if None in (th_p, th_m, al_p, al_m):
        wall_ms = (time.perf_counter() - t_start) * 1000
        return {
            "best": best, "iterations": 0, "wall_ms": wall_ms,
            "simulate_calls": counter.n, "function_calls": counter.n,
            "jacobian_calls": jacobian_calls,
        }

    J = np.array(
        [
            [(th_p["y_hit"] - th_m["y_hit"]) / (2 * dth), (al_p["y_hit"] - al_m["y_hit"]) / (2 * dal)],
            [(th_p["z_hit"] - th_m["z_hit"]) / (2 * dth), (al_p["z_hit"] - al_m["z_hit"]) / (2 * dal)],
        ],
        dtype=np.float64,
    )

    # ---- Broyden iterations ----
    iters = 0
    for _ in range(max_iter):
        iters += 1

        # solve J * delta = F
        try:
            delta = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, F, rcond=None)[0]

        delta = np.clip(delta, -step_clip, step_clip)

        accepted = False
        for scale in damping_scales:
            x_new = x_vec - scale * delta
            cand_th = float(np.clip(x_new[0], th_min, th_max))
            cand_al = float(np.clip(x_new[1], alpha_min, alpha_max))
            cand = counted_score_angle_at_target(
                cand_th, cand_al, x_target, y_target, z_target, params_env, dt, t_max, counter,
            )
            if cand is None:
                continue
            if cand["err_3d"] <= best["err_3d"] + 1e-6:
                x_new_clipped = np.array([cand_th, cand_al], dtype=np.float64)
                F_new = np.array(
                    [cand["y_hit"] - y_target, cand["z_hit"] - z_target], dtype=np.float64,
                )

                s = x_new_clipped - x_vec
                y_vec = F_new - F

                # good Broyden rank-1 update
                denom = float(s @ s)
                if denom > 1e-14:
                    J = J + np.outer(y_vec - J @ s, s) / denom

                x_vec = x_new_clipped
                F = F_new
                th, al = cand_th, cand_al
                if cand["err_3d"] < best["err_3d"]:
                    best = dict(cand)
                accepted = True
                break

        if not accepted:
            break

        if abs(F[0]) < y_tol and abs(F[1]) < z_tol:
            break

    wall_ms = (time.perf_counter() - t_start) * 1000
    simulate_calls = counter.n
    function_calls = counter.n

    return {
        "best": best,
        "iterations": iters,
        "wall_ms": wall_ms,
        "simulate_calls": simulate_calls,
        "function_calls": function_calls,
        "jacobian_calls": jacobian_calls,
    }


# =============================================================================
# Unified result dataclass
# =============================================================================

@dataclass
class RefineResult:
    method: str
    sample_id: int
    branch: str
    converged: bool
    err_3d: float
    y_err: float
    z_err: float
    theta: float
    alpha: float
    iterations: int
    wall_ms: float
    simulate_calls: int
    function_calls: int
    jacobian_calls: int
    status: str


# =============================================================================
# Test sample generation (no NN dependency)
# =============================================================================

def build_params_from_sample(sample: dict, base_params: ProjectileParams) -> ProjectileParams:
    """Construct a ProjectileParams from a sample dict, mirroring Benchmark.py."""
    p = ProjectileParams(**base_params.__dict__)
    p.T_powder_C = float(sample["T_powder_C"])
    p.v0_base = float(sample["v0_actual"]) - p.temp_coeff * (p.T_powder_C - 15.0)
    p.wind_x = float(sample["wind_x"])
    p.wind_y = float(sample["wind_y"])
    p.wind_z = float(sample["wind_z"])
    p.cant_angle_deg = float(sample["cant_angle"])
    p.alt_gun = float(sample["alt_gun"])
    p.T0_C = float(sample["T0_C"])
    p.P0_Pa = float(sample["P0_Pa"])
    return p


def generate_benchmark_samples(
    n_samples: int, seed: int, base_params: ProjectileParams,
) -> list[dict]:
    """Generate independent test samples via forward simulation (no NN needed).

    Each sample is guaranteed physically reachable: random (env, theta, alpha)
    → simulate → pick x_target in 30%-90% of range → read y_target, z_target.
    """
    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    attempts = 0
    max_attempts = n_samples * 5

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        alt_gun = float(rng.uniform(0.0, 1500.0))
        T0_C = float(rng.uniform(-20.0, 40.0))
        P0_Pa = float(isa_pressure(alt_gun) * rng.uniform(0.95, 1.05))
        wind_x = float(rng.uniform(-15.0, 15.0))
        wind_y = float(rng.uniform(-3.0, 3.0))
        wind_z = float(rng.uniform(-12.0, 12.0))
        cant_angle = float(rng.uniform(-12.0, 12.0))
        T_powder = T0_C + float(rng.normal(0.0, 5.0))
        v0_actual = base_params.v0_base + base_params.temp_coeff * (T_powder - 15.0)

        T0_K = T0_C + 273.15
        rho, _ = get_atmosphere(alt_gun, alt_gun, T0_K, P0_Pa)

        # 70% low, 30% high trajectory
        if rng.random() < 0.7:
            theta = float(rng.uniform(3.0, 45.0))
            trajectory_mode = "low"
        else:
            theta = float(rng.uniform(50.0, 70.0))
            trajectory_mode = "high"
        alpha = float(rng.uniform(-6.0, 6.0))

        sample_dict = {
            "T_powder_C": T_powder,
            "v0_actual": v0_actual,
            "wind_x": wind_x,
            "wind_y": wind_y,
            "wind_z": wind_z,
            "cant_angle": cant_angle,
            "alt_gun": alt_gun,
            "T0_C": T0_C,
            "P0_Pa": P0_Pa,
        }
        p = build_params_from_sample(sample_dict, base_params)
        traj = simulate_trajectory(
            theta_deg=theta, alpha_deg=alpha, params=p, dt=0.05, t_max=200.0,
        )
        max_x = traj["range"]
        if max_x < 200.0:
            continue

        x_frac = float(rng.uniform(0.30, 0.90))
        x_target = max_x * x_frac
        t_hit, y_target, z_target = time_and_y_at_x(traj, x_target)
        if t_hit is None or y_target is None or y_target < 0.5:
            continue

        sample_dict.update({
            "rho": float(rho),
            "x_target": float(x_target),
            "y_target": float(y_target),
            "z_target": float(z_target),
            "true_theta": float(theta),
            "true_alpha": float(alpha),
            "trajectory_mode": trajectory_mode,
        })
        samples.append(sample_dict)

    if len(samples) < n_samples:
        print(f"Warning: only generated {len(samples)}/{n_samples} samples "
              f"(attempts={attempts})")
    return samples


# =============================================================================
# Result helper
# =============================================================================

def _build_result(
    method: str,
    sample_id: int,
    branch: str,
    res: dict,
    y_tol: float,
    z_tol: float,
) -> RefineResult:
    """Convert a raw refine result dict into a RefineResult."""
    best = res["best"]
    if best is None:
        return RefineResult(
            method=method, sample_id=sample_id, branch=branch,
            converged=False, err_3d=float("nan"), y_err=float("nan"),
            z_err=float("nan"), theta=float("nan"), alpha=float("nan"),
            iterations=res["iterations"], wall_ms=res["wall_ms"],
            simulate_calls=res["simulate_calls"], function_calls=res["function_calls"],
            jacobian_calls=res["jacobian_calls"], status="no_reach",
        )
    conv = bool(best["y_err"] <= y_tol and best["z_err"] <= z_tol)
    return RefineResult(
        method=method, sample_id=sample_id, branch=branch,
        converged=conv,
        err_3d=float(best["err_3d"]), y_err=float(best["y_err"]),
        z_err=float(best["z_err"]), theta=float(best["theta"]),
        alpha=float(best["alpha"]),
        iterations=res["iterations"], wall_ms=res["wall_ms"],
        simulate_calls=res["simulate_calls"], function_calls=res["function_calls"],
        jacobian_calls=res["jacobian_calls"],
        status="converged" if conv else "not_converged",
    )


# =============================================================================
# Main benchmark
# =============================================================================

DAMPING_SCALES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)


def run_benchmark(args) -> pd.DataFrame:
    """Run Newton-vs-Broyden comparison across all benchmark samples."""
    cfg = AMMO_CONFIGS[CURRENT_AMMO]
    base_params = ProjectileParams(mass=cfg["mass"], caliber=cfg["caliber"])
    base_params.v0_base = cfg["v0_base"]

    print(f"Ammo: {CURRENT_AMMO}, v0_base={cfg['v0_base']}")
    samples = generate_benchmark_samples(args.n_samples, args.seed, base_params)
    print(f"Generated {len(samples)} benchmark samples")

    rows: list[dict] = []
    for i, s in enumerate(samples):
        params_env = build_params_from_sample(s, base_params)
        branch = s["trajectory_mode"]

        if branch == "low":
            th_min = LOW_THETA_MIN
            th_max = LOW_THETA_MAX
        else:
            th_min = HIGH_THETA_MIN
            th_max = HIGH_THETA_MAX

        theta_guess = float(np.clip(
            s["true_theta"] + args.initial_theta_offset, th_min, th_max,
        ))
        alpha_guess = float(np.clip(
            s["true_alpha"] + args.initial_alpha_offset, -ALPHA_ABS_MAX, ALPHA_ABS_MAX,
        ))

        common = dict(
            th_guess=theta_guess,
            alpha_guess=alpha_guess,
            x_target=s["x_target"],
            y_target=s["y_target"],
            z_target=s["z_target"],
            params_env=params_env,
            th_min=th_min,
            th_max=th_max,
            dt=args.dt,
            t_max=args.t_max,
            alpha_min=-ALPHA_ABS_MAX,
            alpha_max=ALPHA_ABS_MAX,
            y_tol=args.y_tol,
            z_tol=args.z_tol,
            max_iter=args.max_iter,
            dth=args.dth,
            dal=args.dal,
            step_clip=args.step_clip,
            damping_scales=DAMPING_SCALES,
        )

        newton_res = benchmark_newton_refine(**common)
        broyden_res = benchmark_broyden_refine(**common)

        rows.append(asdict(_build_result("Newton", i, branch, newton_res, args.y_tol, args.z_tol)))
        rows.append(asdict(_build_result("Broyden", i, branch, broyden_res, args.y_tol, args.z_tol)))

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# Summary & output
# =============================================================================

def print_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print per-method summary statistics and return the summary DataFrame."""
    grouped = df.groupby("method").agg(
        converged=("converged", "mean"),
        err_3d_mean=("err_3d", "mean"),
        err_3d_median=("err_3d", "median"),
        err_3d_max=("err_3d", "max"),
        y_err_mean=("y_err", "mean"),
        y_err_median=("y_err", "median"),
        y_err_max=("y_err", "max"),
        z_err_mean=("z_err", "mean"),
        z_err_median=("z_err", "median"),
        z_err_max=("z_err", "max"),
        iterations_mean=("iterations", "mean"),
        iterations_median=("iterations", "median"),
        iterations_max=("iterations", "max"),
        wall_ms_mean=("wall_ms", "mean"),
        wall_ms_median=("wall_ms", "median"),
        wall_ms_max=("wall_ms", "max"),
        simulate_calls_mean=("simulate_calls", "mean"),
        simulate_calls_median=("simulate_calls", "median"),
        simulate_calls_max=("simulate_calls", "max"),
        function_calls_mean=("function_calls", "mean"),
        function_calls_median=("function_calls", "median"),
        function_calls_max=("function_calls", "max"),
        jacobian_calls_mean=("jacobian_calls", "mean"),
        jacobian_calls_median=("jacobian_calls", "median"),
        jacobian_calls_max=("jacobian_calls", "max"),
    ).reset_index()

    print("\n" + "=" * 80)
    print("  Per-Method Summary")
    print("=" * 80)
    print(grouped.to_string(index=False))

    print("\n  converged = convergence rate (fraction within tolerance)")
    return grouped


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Newton vs Broyden solver on offline ballistic samples.",
    )
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t_max", type=float, default=120.0)
    parser.add_argument("--y_tol", type=float, default=2.0)
    parser.add_argument("--z_tol", type=float, default=2.0)
    parser.add_argument("--max_iter", type=int, default=5)
    parser.add_argument("--dth", type=float, default=0.05)
    parser.add_argument("--dal", type=float, default=0.05)
    parser.add_argument("--step_clip", type=float, default=2.0)
    parser.add_argument("--initial_theta_offset", type=float, default=1.0)
    parser.add_argument("--initial_alpha_offset", type=float, default=0.5)
    parser.add_argument("--out_csv", type=str, default="benchmark_newton_vs_broyden_solver.csv")
    parser.add_argument("--summary_csv", type=str, default="benchmark_newton_vs_broyden_summary.csv")
    args = parser.parse_args()

    df = run_benchmark(args)

    # per-sample detail
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved per-sample results: {args.out_csv}")

    # summary
    summary = print_summary(df)
    summary.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
