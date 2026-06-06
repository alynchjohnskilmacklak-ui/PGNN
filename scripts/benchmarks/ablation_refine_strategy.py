"""ablation_refine_strategy.py

用途：
    消融实验：对比 solver.py 中不同 refine_mode 策略的性能。
    测试 nn_only / newton / broyden / grid_newton / grid_broyden / broyden_fast。

适用场景：
    确定哪种精修策略在给定条件下最优（精度 vs 速度 vs 仿真成本）。

输入：
    不依赖 NN 模型，使用正向仿真生成独立测试样本。

输出：
    ablation_refine_strategy_results.csv（逐样本结果）
    ablation_refine_strategy_summary.csv（按策略汇总）

运行方式：
    python scripts/benchmarks/ablation_refine_strategy.py --n 100 --seed 20260606
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
    python ablation_refine_strategy.py --n 100 --seed 20260606
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

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
import solver


# =============================================================================
# Simulation counter (monkey-patch for counting simulate_trajectory calls)
# =============================================================================

class _SimCounter:
    def __init__(self):
        self.n = 0

    def reset(self):
        self.n = 0

    def inc(self):
        self.n += 1


COUNTER = _SimCounter()
_orig_simulate = simulate_trajectory


def _counted_simulate(*args, **kwargs):
    COUNTER.inc()
    return _orig_simulate(*args, **kwargs)


# Monkey-patch so that solver._score_angle_at_target counts every call.
import ballistics as _ballistics_mod
_ballistics_mod.simulate_trajectory = _counted_simulate
solver.simulate_trajectory = _counted_simulate


# =============================================================================
# Sample generation (mirrors Benchmark.py generate_test_samples)
# =============================================================================

def build_params(sample: dict, base_params: ProjectileParams) -> ProjectileParams:
    """Construct a ProjectileParams from a sample dict."""
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


def generate_test_samples(
    n_samples: int, base_params: ProjectileParams, seed: int = 99999,
) -> list[dict]:
    """Generate independent test samples via forward simulation (no NN needed)."""
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
        p = build_params(sample_dict, base_params)
        traj = _orig_simulate(theta_deg=theta, alpha_deg=alpha, params=p, dt=0.05, t_max=200.0)
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

    return samples


# =============================================================================
# Initial guess (simulates NN prediction without requiring a trained model)
# =============================================================================

def make_initial_guess(
    sample: dict, theta_offset: float = 1.0, alpha_offset: float = 0.5,
) -> tuple[float, float]:
    """Return (theta_guess, alpha_guess) by perturbing the true angles."""
    th = sample["true_theta"] + theta_offset
    al = sample["true_alpha"] + alpha_offset
    return float(th), float(al)


# =============================================================================
# Strategy runner
# =============================================================================

STRATEGIES = ["nn_only", "newton", "broyden", "grid_newton", "grid_broyden", "broyden_fast"]


def run_one_strategy(
    strategy: str,
    sample: dict,
    params_env: ProjectileParams,
    theta_guess: float,
    alpha_guess: float,
    th_min: float,
    th_max: float,
    y_tol: float,
    z_tol: float,
    dt: float,
    t_max: float,
) -> dict:
    """Run a single refine strategy on a single sample, return metrics dict."""
    COUNTER.reset()
    t0 = time.perf_counter()

    if strategy == "nn_only":
        # Evaluate initial guess without any refinement.
        best = solver._score_angle_at_target(
            theta_guess, alpha_guess,
            sample["x_target"], sample["y_target"], sample["z_target"],
            params_env, dt, t_max,
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        n_sim = COUNTER.n
        if best is None:
            return _failed(strategy, wall_ms, n_sim)
        conv = bool(best["y_err"] <= y_tol and best["z_err"] <= z_tol)
        return {
            "strategy": strategy,
            "converged": conv,
            "err_3d": float(best["err_3d"]),
            "y_err": float(best["y_err"]),
            "z_err": float(best["z_err"]),
            "wall_ms": wall_ms,
            "n_sim": n_sim,
            "used_grid": False,
            "refine_iters": 0,
            "refine_source": "none",
        }

    # All other strategies go through solver._refine_candidate.
    refine_mode = strategy  # maps 1:1 for newton/broyden/grid_newton/grid_broyden/broyden_fast
    best = solver._refine_candidate(
        th_guess=theta_guess,
        alpha_guess=alpha_guess,
        x_target=sample["x_target"],
        y_target=sample["y_target"],
        z_target=sample["z_target"],
        params_env=params_env,
        dt=dt,
        t_max=t_max,
        th_min=th_min,
        th_max=th_max,
        y_tol=y_tol,
        z_tol=z_tol,
        alpha_min=-ALPHA_ABS_MAX,
        alpha_max=ALPHA_ABS_MAX,
        refine_mode=refine_mode,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    n_sim = COUNTER.n

    if best is None:
        return _failed(strategy, wall_ms, n_sim)

    conv = bool(best["y_err"] <= y_tol and best["z_err"] <= z_tol)
    return {
        "strategy": strategy,
        "converged": conv,
        "err_3d": float(best["err_3d"]),
        "y_err": float(best["y_err"]),
        "z_err": float(best["z_err"]),
        "wall_ms": wall_ms,
        "n_sim": n_sim,
        "used_grid": bool(best.get("used_grid", False)),
        "refine_iters": int(best.get("refine_iters", 0)),
        "refine_source": str(best.get("refine_source", "unknown")),
    }


def _failed(strategy: str, wall_ms: float, n_sim: int) -> dict:
    return {
        "strategy": strategy,
        "converged": False,
        "err_3d": float("nan"),
        "y_err": float("nan"),
        "z_err": float("nan"),
        "wall_ms": wall_ms,
        "n_sim": n_sim,
        "used_grid": False,
        "refine_iters": 0,
        "refine_source": "failed",
    }


# =============================================================================
# Main ablation
# =============================================================================

def run_ablation(args) -> pd.DataFrame:
    cfg = AMMO_CONFIGS[CURRENT_AMMO]
    base_params = ProjectileParams(mass=cfg["mass"], caliber=cfg["caliber"])
    base_params.v0_base = cfg["v0_base"]

    print(f"Ammo: {CURRENT_AMMO}, v0_base={cfg['v0_base']}")
    samples = generate_test_samples(args.n, base_params, seed=args.seed)
    print(f"Generated {len(samples)} samples\n")

    rows: list[dict] = []
    for i, s in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f"  progress {i + 1}/{len(samples)}")

        params_env = build_params(s, base_params)
        branch = s["trajectory_mode"]
        th_min = LOW_THETA_MIN if branch == "low" else HIGH_THETA_MIN
        th_max = LOW_THETA_MAX if branch == "low" else HIGH_THETA_MAX
        th_guess, al_guess = make_initial_guess(s)

        for strategy in STRATEGIES:
            result = run_one_strategy(
                strategy=strategy,
                sample=s,
                params_env=params_env,
                theta_guess=th_guess,
                alpha_guess=al_guess,
                th_min=th_min,
                th_max=th_max,
                y_tol=args.y_tol,
                z_tol=args.z_tol,
                dt=args.dt,
                t_max=args.t_max,
            )
            result["sample_id"] = i
            rows.append(result)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-strategy summary."""
    grouped = df.groupby("strategy").agg(
        n=("strategy", "count"),
        converged_rate=("converged", "mean"),
        err_3d_mean=("err_3d", "mean"),
        err_3d_median=("err_3d", "median"),
        err_3d_max=("err_3d", "max"),
        wall_ms_mean=("wall_ms", "mean"),
        wall_ms_median=("wall_ms", "median"),
        wall_ms_max=("wall_ms", "max"),
        n_sim_mean=("n_sim", "mean"),
        n_sim_median=("n_sim", "median"),
        n_sim_max=("n_sim", "max"),
        used_grid_rate=("used_grid", "mean"),
        refine_iters_mean=("refine_iters", "mean"),
        refine_iters_median=("refine_iters", "median"),
        refine_iters_max=("refine_iters", "max"),
    ).reset_index()

    # Reorder columns for readability
    cols = [
        "strategy", "n", "converged_rate",
        "err_3d_mean", "err_3d_median", "err_3d_max",
        "wall_ms_mean", "wall_ms_median", "wall_ms_max",
        "n_sim_mean", "n_sim_median", "n_sim_max",
        "used_grid_rate",
        "refine_iters_mean", "refine_iters_median", "refine_iters_max",
    ]
    return grouped[[c for c in cols if c in grouped.columns]]


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: compare refine strategies from solver.py.",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=20260606, help="Random seed")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation time step")
    parser.add_argument("--t_max", type=float, default=120.0, help="Max simulation time")
    parser.add_argument("--y_tol", type=float, default=2.0, help="Y tolerance (m)")
    parser.add_argument("--z_tol", type=float, default=2.0, help="Z tolerance (m)")
    args = parser.parse_args()

    df = run_ablation(args)

    detail_csv = "ablation_refine_strategy_results.csv"
    summary_csv = "ablation_refine_strategy_summary.csv"

    df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved per-sample results: {detail_csv}")

    summary = summarize(df)
    print("\n" + "=" * 90)
    print("  Ablation Summary")
    print("=" * 90)
    print(summary.to_string(index=False))

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved summary: {summary_csv}")

    # refine_source distribution per strategy
    if "refine_source" in df.columns:
        print("\n  refine_source distribution per strategy:")
        for strat in STRATEGIES:
            sub = df[df["strategy"] == strat]
            if len(sub) == 0:
                continue
            counts = sub["refine_source"].value_counts()
            parts = [f"{src}: {cnt}" for src, cnt in counts.items()]
            print(f"    {strat}:  {', '.join(parts)}")


if __name__ == "__main__":
    main()
