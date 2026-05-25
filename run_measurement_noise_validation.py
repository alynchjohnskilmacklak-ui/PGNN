import argparse
import csv
import json
import math
import os
from typing import Iterable

import numpy as np
import pandas as pd

from ballistics import ProjectileParams, get_atmosphere, time_and_y_at_x, simulate_trajectory
from generate_dataset import AMMO_CONFIGS, CURRENT_AMMO, get_ammo_config
from main import _generate_verify_samples
from predict import load_branch_model, solve_target_unified, minmax_transform
from solver import _newton_only_refine, _score_angle_at_target
from train_model import train
from feature_schema import build_inference_input


REALISTIC_MEASUREMENT_NOISE = {
    "x_rel_std": 0.002,
    "y_abs_std": 0.75,
    "z_abs_std": 0.75,
    "v0_abs_std": 2.0,
    "rho_rel_std": 0.01,
    "wind_x_abs_std": 1.0,
    "wind_y_abs_std": 0.4,
    "wind_z_abs_std": 1.0,
    "cant_abs_std": 0.2,
    "T_powder_abs_std": 1.5,
    "T0_abs_std": 1.0,
    "P0_rel_std": 0.005,
    "alt_abs_std": 5.0,
}


CONFIGS = [
    {
        "name": "data_only",
        "lambda_phys": 0.0,
        "physics_loss_mode": "consistency",
    },
    {
        "name": "pgnn_consistency",
        "lambda_phys": 0.004,
        "physics_loss_mode": "consistency",
    },
    {
        "name": "pgnn_target_intercept",
        "lambda_phys": 0.004,
        "physics_loss_mode": "target_intercept",
    },
]


TRAIN_NOISE = {
    "train_noise_x_rel_std": REALISTIC_MEASUREMENT_NOISE["x_rel_std"],
    "train_noise_y_abs_std": REALISTIC_MEASUREMENT_NOISE["y_abs_std"],
    "train_noise_z_abs_std": REALISTIC_MEASUREMENT_NOISE["z_abs_std"],
    "train_noise_v0_std": REALISTIC_MEASUREMENT_NOISE["v0_abs_std"],
    "train_noise_rho_rel_std": REALISTIC_MEASUREMENT_NOISE["rho_rel_std"],
    "train_noise_wind_x_std": REALISTIC_MEASUREMENT_NOISE["wind_x_abs_std"],
    "train_noise_wind_y_std": REALISTIC_MEASUREMENT_NOISE["wind_y_abs_std"],
    "train_noise_wind_z_std": REALISTIC_MEASUREMENT_NOISE["wind_z_abs_std"],
    "train_noise_cant_std": REALISTIC_MEASUREMENT_NOISE["cant_abs_std"],
}


def _run_name(config_name: str, mode_name: str, seed: int) -> str:
    return f"measnoise_{config_name}_{mode_name}_seed{int(seed)}"


def _save_rows(rows: list[dict], json_path: str, csv_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_measurement_noise_models(
    out_dir: str | None = None,
    ammo_id: str | None = None,
    seed: int = 42,
    configs: Iterable[dict] = CONFIGS,
    max_epochs: int = 80,
    patience: int = 8,
    batch_size: int = 16384,
    model_type: str = "kan_mlp",
    physics_steps: int = 6,
) -> list[dict]:
    ammo_id, cfg = get_ammo_config(ammo_id)
    if out_dir is None:
        out_dir = cfg["out_dir"]
    dataset_npy = os.path.join(out_dir, "dataset.npy")
    if not os.path.exists(dataset_npy):
        raise FileNotFoundError(f"Dataset not found: {dataset_npy}")

    rows = []
    for config in configs:
        for trajectory_mode, mode_name in ((0, "low"), (1, "high")):
            run_name = _run_name(config["name"], mode_name, seed)
            print("=" * 80)
            print(
                f"Training {run_name}: ammo={ammo_id}, model_type={model_type}, "
                f"lambda_phys={config['lambda_phys']}, physics_loss_mode={config['physics_loss_mode']}"
            )
            print("=" * 80)
            history = train(
                out_dir=out_dir,
                dataset_npy=dataset_npy,
                model_name=run_name,
                scaler_name=f"scaler_{run_name}",
                history_name=f"hist_{run_name}",
                plot_name=f"loss_{run_name}",
                seed=seed,
                batch_size=batch_size,
                lr=1.6e-4,
                max_epochs=max_epochs,
                patience=patience,
                hidden=256,
                dropout=0.15,
                model_type=model_type,
                lambda_phys=float(config["lambda_phys"]),
                physics_loss_mode=str(config["physics_loss_mode"]),
                physics_steps=physics_steps,
                plot_losses=False,
                gpu_cache_max_total_frac=0.45,
                gpu_cache_max_free_frac=0.65,
                trajectory_mode=trajectory_mode,
                **TRAIN_NOISE,
            )
            row = {
                "config": config["name"],
                "ammo_id": ammo_id,
                "seed": int(seed),
                "mode": mode_name,
                "trajectory_mode": trajectory_mode,
                "model_type": model_type,
                "lambda_phys": float(config["lambda_phys"]),
                "physics_loss_mode": str(config["physics_loss_mode"]),
                "physics_steps": int(physics_steps),
                "best_epoch": history["best_epoch"],
                "best_val_loss": history["best_val_loss"],
                "test_loss": history["test_loss"],
                "test_mae_theta": history["test_mae_theta"],
                "test_mae_alpha": history["test_mae_alpha"],
                "model_path": os.path.join(out_dir, f"{run_name}.pt"),
                "scaler_path": os.path.join(out_dir, f"scaler_{run_name}.json"),
            }
            row.update(history.get("noise", {}))
            rows.append(row)
            _save_rows(
                rows,
                os.path.join(out_dir, f"measurement_noise_train_seed{seed}.json"),
                os.path.join(out_dir, f"measurement_noise_train_seed{seed}.csv"),
            )
    return rows


def _noisy_value(rng: np.random.Generator, value: float, abs_std: float = 0.0, rel_std: float = 0.0) -> float:
    noisy = float(value)
    if rel_std > 0.0:
        noisy += float(rng.normal(0.0, rel_std)) * max(abs(float(value)), 1e-9)
    if abs_std > 0.0:
        noisy += float(rng.normal(0.0, abs_std))
    return noisy


def perturb_observation(sample: dict, rng: np.random.Generator, noise: dict) -> dict:
    obs = dict(sample)
    obs["x_target"] = max(1.0, _noisy_value(rng, sample["x_target"], rel_std=noise["x_rel_std"]))
    obs["y_target"] = _noisy_value(rng, sample["y_target"], abs_std=noise["y_abs_std"])
    obs["z_target"] = _noisy_value(rng, sample["z_target"], abs_std=noise["z_abs_std"])
    obs["v0_actual"] = max(1.0, _noisy_value(rng, sample["v0_actual"], abs_std=noise["v0_abs_std"]))
    obs["rho"] = max(1e-6, _noisy_value(rng, sample["rho"], rel_std=noise["rho_rel_std"]))
    obs["wind_x"] = _noisy_value(rng, sample["wind_x"], abs_std=noise["wind_x_abs_std"])
    obs["wind_y"] = _noisy_value(rng, sample["wind_y"], abs_std=noise["wind_y_abs_std"])
    obs["wind_z"] = _noisy_value(rng, sample["wind_z"], abs_std=noise["wind_z_abs_std"])
    obs["cant_angle"] = _noisy_value(rng, sample["cant_angle"], abs_std=noise["cant_abs_std"])
    obs["T_powder_C"] = _noisy_value(rng, sample["T_powder_C"], abs_std=noise["T_powder_abs_std"])
    obs["T0_C"] = _noisy_value(rng, sample["T0_C"], abs_std=noise["T0_abs_std"])
    obs["P0_Pa"] = max(1.0, _noisy_value(rng, sample["P0_Pa"], rel_std=noise["P0_rel_std"]))
    obs["alt_gun"] = max(0.0, _noisy_value(rng, sample["alt_gun"], abs_std=noise["alt_abs_std"]))
    return obs


def _params_from_sample(sample: dict, base_params: ProjectileParams) -> ProjectileParams:
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


def _score_solution_in_true_environment(chosen: dict | None, sample: dict, base_params: ProjectileParams) -> dict:
    if chosen is None:
        return {
            "true_reachable": False,
            "true_err_3d": math.nan,
            "true_y_err": math.nan,
            "true_z_err": math.nan,
        }
    true_params = _params_from_sample(sample, base_params)
    traj = simulate_trajectory(
        theta_deg=float(chosen["theta"]),
        alpha_deg=float(chosen["alpha"]),
        params=true_params,
        dt=0.01,
        t_max=200.0,
    )
    if traj["range"] < sample["x_target"]:
        return {
            "true_reachable": False,
            "true_err_3d": math.nan,
            "true_y_err": math.nan,
            "true_z_err": math.nan,
        }
    t_hit, y_hit, z_hit = time_and_y_at_x(traj, sample["x_target"])
    if t_hit is None or y_hit is None or z_hit is None:
        return {
            "true_reachable": False,
            "true_err_3d": math.nan,
            "true_y_err": math.nan,
            "true_z_err": math.nan,
        }
    y_err = abs(float(y_hit) - float(sample["y_target"]))
    z_err = abs(float(z_hit) - float(sample["z_target"]))
    return {
        "true_reachable": True,
        "true_err_3d": float(math.sqrt(y_err ** 2 + z_err ** 2)),
        "true_y_err": float(y_err),
        "true_z_err": float(z_err),
    }


def _predict_branch(model, scaler: dict, obs: dict) -> tuple[float, float]:
    X = build_inference_input(
        x_target=obs["x_target"], y_target=obs["y_target"], z_target=obs["z_target"],
        v0_actual=obs["v0_actual"], rho=obs["rho"],
        wind_x=obs["wind_x"], wind_y=obs["wind_y"], wind_z=obs["wind_z"],
        cant_angle=obs["cant_angle"], T_powder_C=obs["T_powder_C"],
        T0_C=obs["T0_C"], P0_Pa=obs["P0_Pa"], alt_gun=obs["alt_gun"],
    )
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        xb = torch.from_numpy(minmax_transform(X, scaler)).to(device)
        pred = model(xb)
    return float(pred[0, 0]), float(pred[0, 1])


def evaluate_initialization_quality(
    out_dir: str | None = None,
    ammo_id: str | None = None,
    seed: int = 42,
    sample_n: int = 500,
    configs: Iterable[dict] = CONFIGS,
    noise: dict = REALISTIC_MEASUREMENT_NOISE,
    newton_max_iter: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ammo_id, cfg = get_ammo_config(ammo_id)
    if out_dir is None:
        out_dir = cfg["out_dir"]

    base_params = ProjectileParams(mass=cfg["mass"], caliber=cfg["caliber"])
    base_params.v0_base = cfg["v0_base"]
    samples = _generate_verify_samples(sample_n, base_params, seed=seed + 100000)
    rng = np.random.default_rng(seed + 200000)
    noisy_observations = [perturb_observation(sample, rng, noise) for sample in samples]

    rows = []
    for config in configs:
        low_name = _run_name(config["name"], "low", seed)
        high_name = _run_name(config["name"], "high", seed)
        low_model, low_scaler, _ = load_branch_model(
            os.path.join(out_dir, f"{low_name}.pt"),
            os.path.join(out_dir, f"scaler_{low_name}.json"),
            theta_min=0.0,
            theta_max=55.0,
        )
        high_model, high_scaler, _ = load_branch_model(
            os.path.join(out_dir, f"{high_name}.pt"),
            os.path.join(out_dir, f"scaler_{high_name}.json"),
            theta_min=45.0,
            theta_max=85.0,
        )

        for idx, (sample, obs) in enumerate(zip(samples, noisy_observations)):
            obs_params = _params_from_sample(obs, base_params)
            branch_preds = {
                "low": (*_predict_branch(low_model, low_scaler, obs), 0.0, 55.0),
                "high": (*_predict_branch(high_model, high_scaler, obs), 45.0, 85.0),
            }
            for branch, (theta_pred, alpha_pred, theta_min, theta_max) in branch_preds.items():
                raw_score = _score_angle_at_target(
                    theta_pred,
                    alpha_pred,
                    obs["x_target"],
                    obs["y_target"],
                    obs["z_target"],
                    obs_params,
                    dt=0.05,
                    t_max=120.0,
                )
                refined, iters = _newton_only_refine(
                    theta_pred,
                    alpha_pred,
                    obs["x_target"],
                    obs["y_target"],
                    obs["z_target"],
                    obs_params,
                    theta_min,
                    theta_max,
                    dt=0.05,
                    t_max=120.0,
                    y_tol=2.0,
                    z_tol=2.0,
                    max_iter=newton_max_iter,
                )
                raw_true_score = _score_solution_in_true_environment(
                    None if raw_score is None else {"theta": theta_pred, "alpha": alpha_pred},
                    sample,
                    base_params,
                )
                refined_true_score = _score_solution_in_true_environment(refined, sample, base_params)
                rows.append({
                    "config": config["name"],
                    "sample_id": idx,
                    "branch": branch,
                    "theta_pred_deg": theta_pred,
                    "alpha_pred_deg": alpha_pred,
                    "theta_true_deg": sample["true_theta"],
                    "alpha_true_deg": sample["true_alpha"],
                    "theta_abs_err_deg": abs(theta_pred - sample["true_theta"]),
                    "alpha_abs_err_deg": abs(alpha_pred - sample["true_alpha"]),
                    "raw_obs_reachable": raw_score is not None,
                    "raw_obs_err_3d": math.nan if raw_score is None else raw_score["err_3d"],
                    "raw_true_reachable": raw_true_score["true_reachable"],
                    "raw_true_err_3d": raw_true_score["true_err_3d"],
                    "newton_limited_iters": int(iters),
                    "newton_limited_converged_obs": (
                        refined is not None and refined["y_err"] <= 2.0 and refined["z_err"] <= 2.0
                    ),
                    "newton_limited_obs_err_3d": math.nan if refined is None else refined["err_3d"],
                    "newton_limited_true_reachable": refined_true_score["true_reachable"],
                    "newton_limited_true_err_3d": refined_true_score["true_err_3d"],
                })

    detail = pd.DataFrame(rows)
    summary_rows = []
    for (config_name, branch), sub in detail.groupby(["config", "branch"]):
        raw_ok = sub[sub["raw_true_reachable"] & np.isfinite(sub["raw_true_err_3d"])]
        newton_ok = sub[sub["newton_limited_true_reachable"] & np.isfinite(sub["newton_limited_true_err_3d"])]
        summary_rows.append({
            "config": config_name,
            "branch": branch,
            "sample_size": int(len(sub)),
            "theta_abs_err_median": float(sub["theta_abs_err_deg"].median()),
            "theta_abs_err_p95": float(sub["theta_abs_err_deg"].quantile(0.95)),
            "alpha_abs_err_median": float(sub["alpha_abs_err_deg"].median()),
            "alpha_abs_err_p95": float(sub["alpha_abs_err_deg"].quantile(0.95)),
            "raw_obs_reachable_rate": float(sub["raw_obs_reachable"].mean()),
            "raw_true_reachable_rate": float(sub["raw_true_reachable"].mean()),
            "raw_true_err_3d_median": float(raw_ok["raw_true_err_3d"].median()) if len(raw_ok) else math.nan,
            "raw_true_err_3d_p95": float(raw_ok["raw_true_err_3d"].quantile(0.95)) if len(raw_ok) else math.nan,
            "newton_limited_converged_obs_rate": float(sub["newton_limited_converged_obs"].mean()),
            "newton_limited_iters_mean": float(sub["newton_limited_iters"].mean()),
            "newton_limited_true_reachable_rate": float(sub["newton_limited_true_reachable"].mean()),
            "newton_limited_true_err_3d_median": (
                float(newton_ok["newton_limited_true_err_3d"].median()) if len(newton_ok) else math.nan
            ),
            "newton_limited_true_err_3d_p95": (
                float(newton_ok["newton_limited_true_err_3d"].quantile(0.95)) if len(newton_ok) else math.nan
            ),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["branch", "config"]).reset_index(drop=True)

    detail_path = os.path.join(out_dir, f"measurement_noise_init_quality_seed{seed}.csv")
    summary_path = os.path.join(out_dir, f"measurement_noise_init_quality_summary_seed{seed}.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    return detail, summary


def evaluate_measurement_noise_models(
    out_dir: str | None = None,
    ammo_id: str | None = None,
    seed: int = 42,
    sample_n: int = 500,
    configs: Iterable[dict] = CONFIGS,
    noise: dict = REALISTIC_MEASUREMENT_NOISE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ammo_id, cfg = get_ammo_config(ammo_id)
    if out_dir is None:
        out_dir = cfg["out_dir"]

    base_params = ProjectileParams(mass=cfg["mass"], caliber=cfg["caliber"])
    base_params.v0_base = cfg["v0_base"]
    samples = _generate_verify_samples(sample_n, base_params, seed=seed + 100000)
    rng = np.random.default_rng(seed + 200000)
    noisy_observations = [perturb_observation(sample, rng, noise) for sample in samples]

    rows = []
    for config in configs:
        low_name = _run_name(config["name"], "low", seed)
        high_name = _run_name(config["name"], "high", seed)
        low_model, low_scaler, _ = load_branch_model(
            os.path.join(out_dir, f"{low_name}.pt"),
            os.path.join(out_dir, f"scaler_{low_name}.json"),
            theta_min=0.0,
            theta_max=55.0,
        )
        high_model, high_scaler, _ = load_branch_model(
            os.path.join(out_dir, f"{high_name}.pt"),
            os.path.join(out_dir, f"scaler_{high_name}.json"),
            theta_min=45.0,
            theta_max=85.0,
        )

        for idx, (sample, obs) in enumerate(zip(samples, noisy_observations)):
            obs_params = _params_from_sample(obs, base_params)
            out = solve_target_unified(
                x_target=obs["x_target"],
                y_target=obs["y_target"],
                z_target=obs["z_target"],
                v0_actual=obs["v0_actual"],
                rho=obs["rho"],
                wind_x=obs["wind_x"],
                wind_y=obs["wind_y"],
                wind_z=obs["wind_z"],
                cant_angle=obs["cant_angle"],
                T_powder_C=obs["T_powder_C"],
                alt_gun=obs["alt_gun"],
                T0_C=obs["T0_C"],
                P0_Pa=obs["P0_Pa"],
                dir_path=out_dir,
                loaded_low_model=low_model,
                loaded_low_scaler=low_scaler,
                loaded_high_model=high_model,
                loaded_high_scaler=high_scaler,
                params=obs_params,
                y_tol=2.0,
                z_tol=2.0,
                use_grid=False,
                save_plot=False,
            )
            chosen = out.get("chosen")
            true_score = _score_solution_in_true_environment(chosen, sample, base_params)
            rows.append({
                "config": config["name"],
                "sample_id": idx,
                "solver_reachable_on_noisy_obs": bool(out.get("reachable")),
                "chosen_mode": "" if chosen is None else chosen["mode"],
                "obs_err_3d": math.nan if chosen is None else chosen["err_3d"],
                "theta_chosen_deg": math.nan if chosen is None else chosen["theta"],
                "alpha_chosen_deg": math.nan if chosen is None else chosen["alpha"],
                "theta_true_deg": sample["true_theta"],
                "alpha_true_deg": sample["true_alpha"],
                **true_score,
            })

    detail = pd.DataFrame(rows)
    summary_rows = []
    for config_name, sub in detail.groupby("config"):
        ok = sub[sub["true_reachable"] & np.isfinite(sub["true_err_3d"])]
        summary_rows.append({
            "config": config_name,
            "sample_size": int(len(sub)),
            "true_reachable_count": int(len(ok)),
            "true_reachable_rate": float(len(ok) / max(len(sub), 1)),
            "true_err_3d_median": float(ok["true_err_3d"].median()) if len(ok) else math.nan,
            "true_err_3d_mean": float(ok["true_err_3d"].mean()) if len(ok) else math.nan,
            "true_err_3d_p90": float(ok["true_err_3d"].quantile(0.90)) if len(ok) else math.nan,
            "true_err_3d_p95": float(ok["true_err_3d"].quantile(0.95)) if len(ok) else math.nan,
            "true_err_3d_p99": float(ok["true_err_3d"].quantile(0.99)) if len(ok) else math.nan,
            "true_err_3d_max": float(ok["true_err_3d"].max()) if len(ok) else math.nan,
        })
    summary = pd.DataFrame(summary_rows).sort_values("config").reset_index(drop=True)

    detail_path = os.path.join(out_dir, f"measurement_noise_detail_seed{seed}.csv")
    summary_path = os.path.join(out_dir, f"measurement_noise_summary_seed{seed}.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    return detail, summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate data-only vs PGNN under realistic measurement noise.")
    parser.add_argument("--stage", choices=("train", "eval", "init", "all"), default="all")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--ammo_id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_n", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--newton_max_iter", type=int, default=3)
    args = parser.parse_args()

    if args.stage in ("train", "all"):
        train_measurement_noise_models(
            out_dir=args.out_dir,
            ammo_id=args.ammo_id,
            seed=args.seed,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
    if args.stage in ("eval", "all"):
        evaluate_measurement_noise_models(
            out_dir=args.out_dir,
            ammo_id=args.ammo_id,
            seed=args.seed,
            sample_n=args.sample_n,
        )
    if args.stage in ("init", "all"):
        evaluate_initialization_quality(
            out_dir=args.out_dir,
            ammo_id=args.ammo_id,
            seed=args.seed,
            sample_n=args.sample_n,
            newton_max_iter=args.newton_max_iter,
        )


if __name__ == "__main__":
    main()
