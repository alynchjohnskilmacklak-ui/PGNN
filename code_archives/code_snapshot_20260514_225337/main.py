import os
import json
import numpy as np
import pandas as pd

from generate_dataset import generate_dataset, AMMO_CONFIGS, CURRENT_AMMO
from train_model import train
from predict import load_branch_model, solve_target_unified, infer_model_dims_from_state_dict
from ballistics import ProjectileParams, get_atmosphere, isa_pressure, simulate_trajectory, time_and_y_at_x


PIPELINE_OUTPUT_FILES = (
    "dataset.csv",
    "dataset.npy",
    "dataset_meta.json",
    "dual_model_low.pt",
    "dual_scaler_low.json",
    "train_history_low.json",
    "train_history_low.csv",
    "loss_curve_low.png",
    "dual_model_high.pt",
    "dual_scaler_high.json",
    "train_history_high.json",
    "train_history_high.csv",
    "loss_curve_high.png",
    "prediction_before_refine_plot.png",
    "prediction_refine_plot.png",
    "prediction_compare.csv",
    "prediction_error_compare.png",
    "verify_predictions.csv",
    "verify_summary.csv",
    "pipeline_result_refined.json",
)

TRAIN_BATCH_SIZE = 16384
TRAIN_LR = 1.6e-4
TRAIN_MAX_EPOCHS = 120
TRAIN_PATIENCE = 12
TRAIN_HIDDEN = 256
TRAIN_DROPOUT = 0.15
TRAIN_MODEL_TYPE = "kan_mlp"
TRAIN_LAMBDA_PHYS = 0.004
TRAIN_PHYSICS_STEPS = 6
TRAIN_GPU_CACHE_MAX_TOTAL_FRAC = 0.45
TRAIN_GPU_CACHE_MAX_FREE_FRAC = 0.65


def _remove_pipeline_outputs(out_dir):
    removed = []
    for name in PIPELINE_OUTPUT_FILES:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)
    return removed


def _generate_verify_samples(
    n_samples: int,
    base_params: ProjectileParams,
    seed: int = 99999,
) -> list:
    rng = np.random.default_rng(seed)
    samples = []
    attempts = 0
    max_attempts = n_samples * 5

    print(f"[VERIFY] 生成 {n_samples} 个独立测试样本（seed={seed}）...")
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
        else:
            theta = float(rng.uniform(50.0, 70.0))
        alpha = float(rng.uniform(-6.0, 6.0))

        p = ProjectileParams(mass=base_params.mass, caliber=base_params.caliber)
        p.v0_base = float(v0_actual) - p.temp_coeff * (float(T_powder) - 15.0)
        p.T_powder_C = float(T_powder)
        p.wind_x = float(wind_x)
        p.wind_y = float(wind_y)
        p.wind_z = float(wind_z)
        p.cant_angle_deg = float(cant_angle)
        p.alt_gun = float(alt_gun)
        p.T0_C = float(T0_C)
        p.P0_Pa = float(P0_Pa)

        traj = simulate_trajectory(theta_deg=theta, alpha_deg=alpha, params=p, dt=0.01, t_max=200.0)
        max_x = traj["range"]
        if max_x < 200.0:
            continue
        x_frac = float(rng.uniform(0.30, 0.90))
        x_target = max_x * x_frac
        t_hit, y_target, z_target = time_and_y_at_x(traj, x_target)
        if t_hit is None or y_target is None or y_target < 0.5:
            continue

        sample_dict = {
            "x_target": float(x_target),
            "y_target": float(y_target),
            "z_target": float(z_target),
            "v0_actual": float(v0_actual),
            "rho": float(rho),
            "wind_x": float(wind_x),
            "wind_y": float(wind_y),
            "wind_z": float(wind_z),
            "cant_angle": float(cant_angle),
            "T_powder_C": float(T_powder),
            "alt_gun": float(alt_gun),
            "T0_C": float(T0_C),
            "P0_Pa": float(P0_Pa),
            "true_theta": float(theta),
            "true_alpha": float(alpha),
        }
        samples.append(sample_dict)

    print(f"[VERIFY] 完成 {len(samples)}/{n_samples}（尝试 {attempts} 次）")
    return samples


def _verify_unified_model(out_dir, sample_n=500, seed=99999):
    verify_csv_path = os.path.join(out_dir, "verify_predictions.csv")
    summary_csv_path = os.path.join(out_dir, "verify_summary.csv")

    low_model_path = os.path.join(out_dir, "dual_model_low.pt")
    low_scaler_path = os.path.join(out_dir, "dual_scaler_low.json")
    high_model_path = os.path.join(out_dir, "dual_model_high.pt")
    high_scaler_path = os.path.join(out_dir, "dual_scaler_high.json")
    if not all(os.path.exists(p) for p in (low_model_path, low_scaler_path, high_model_path, high_scaler_path)):
        return {
            "sample_size": 0, "converged_count": 0, "convergence_rate": 0.0,
            "err_3d_median": float("nan"), "err_3d_mean": float("nan"), "err_3d_p95": float("nan"),
            "note": "model files missing",
        }

    low_dims = infer_model_dims_from_state_dict(low_model_path)
    high_dims = infer_model_dims_from_state_dict(high_model_path)
    loaded_low_model, loaded_low_scaler, _ = load_branch_model(
        low_model_path, low_scaler_path,
        theta_min=0.0, theta_max=55.0,
        in_dim=low_dims["in_dim"], hidden=low_dims["hidden"], dropout=low_dims["dropout"],
    )
    loaded_high_model, loaded_high_scaler, _ = load_branch_model(
        high_model_path, high_scaler_path,
        theta_min=45.0, theta_max=85.0,
        in_dim=high_dims["in_dim"], hidden=high_dims["hidden"], dropout=high_dims["dropout"],
    )

    meta_path = os.path.join(out_dir, "dataset_meta.json")
    cfg = AMMO_CONFIGS[CURRENT_AMMO]
    mass_kg = cfg["mass"]
    caliber_m = cfg["caliber"]
    v0_base = cfg["v0_base"]
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            mass_kg = meta.get("mass_kg", mass_kg)
            caliber_m = meta.get("caliber_m", caliber_m)
            v0_base = meta.get("v0_base", v0_base)

    weapon_params = ProjectileParams(mass=mass_kg, caliber=caliber_m)
    weapon_params.v0_base = v0_base

    samples = _generate_verify_samples(sample_n, weapon_params, seed=seed)
    if not samples:
        return {
            "sample_size": 0, "converged_count": 0, "convergence_rate": 0.0,
            "err_3d_median": float("nan"), "err_3d_mean": float("nan"), "err_3d_p95": float("nan"),
            "note": "no samples generated",
        }

    errs = []
    converged_count = 0
    rows = []

    for s in samples:
        out = solve_target_unified(
            x_target=s["x_target"], y_target=s["y_target"], z_target=s["z_target"],
            v0_actual=s["v0_actual"], rho=s["rho"],
            wind_x=s["wind_x"], wind_y=s["wind_y"], wind_z=s["wind_z"],
            cant_angle=s["cant_angle"], T_powder_C=s["T_powder_C"],
            alt_gun=s["alt_gun"], T0_C=s["T0_C"], P0_Pa=s["P0_Pa"],
            dir_path=out_dir,
            loaded_low_model=loaded_low_model, loaded_low_scaler=loaded_low_scaler,
            loaded_high_model=loaded_high_model, loaded_high_scaler=loaded_high_scaler,
            params=weapon_params,
            y_tol=2.0, z_tol=2.0,
            use_grid=False,
            save_plot=False,
        )
        chosen = out.get("chosen")
        err = np.nan if chosen is None else chosen["err_3d"]
        converged = chosen is not None

        if converged:
            converged_count += 1
            errs.append(err)

        rows.append({
            "x": s["x_target"], "y": s["y_target"], "z": s["z_target"],
            "theta_true_deg": s["true_theta"], "alpha_true_deg": s["true_alpha"],
            "theta_pred_low_deg": out["nn_prediction"]["low"]["theta"],
            "alpha_pred_low_deg": out["nn_prediction"]["low"]["alpha"],
            "theta_pred_high_deg": out["nn_prediction"]["high"]["theta"],
            "alpha_pred_high_deg": out["nn_prediction"]["high"]["alpha"],
            "chosen_converged": int(converged),
            "chosen_mode": "" if chosen is None else chosen["mode"],
            "theta_chosen_deg": np.nan if chosen is None else chosen["theta"],
            "alpha_chosen_deg": np.nan if chosen is None else chosen["alpha"],
            "err_3d": err,
        })

    errs_arr = np.array(errs)
    n_total = len(samples)

    result = {
        "sample_size": int(n_total),
        "converged_count": int(converged_count),
        "convergence_rate": float(converged_count / max(n_total, 1)),
        "err_3d_median": float(np.median(errs_arr)) if len(errs_arr) else float("nan"),
        "err_3d_mean": float(np.mean(errs_arr)) if len(errs_arr) else float("nan"),
        "err_3d_p95": float(np.percentile(errs_arr, 95)) if len(errs_arr) else float("nan"),
        "verify_csv_path": verify_csv_path,
        "summary_csv_path": summary_csv_path,
    }

    pd.DataFrame(rows).to_csv(verify_csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    return result


def full_pipeline_dual_task(seed=42, out_dir=None, rebuild_dataset=False):
    if out_dir is None:
        out_dir = AMMO_CONFIGS[CURRENT_AMMO]["out_dir"]

    if rebuild_dataset and os.path.isdir(out_dir):
        removed_files = _remove_pipeline_outputs(out_dir)
        print(f"Removed old files: {len(removed_files)}")

    dataset_npy = os.path.join(out_dir, "dataset.npy")
    dataset_csv = os.path.join(out_dir, "dataset.csv")
    if (not rebuild_dataset) and os.path.exists(dataset_npy):
        ds_info = {
            "out_dir": out_dir,
            "csv_path": dataset_csv,
            "npy_path": dataset_npy,
            "num_samples": int(np.load(dataset_npy, mmap_mode="r").shape[0]),
        }
        print("Using existing dataset:", ds_info)
    else:
        ds_info = generate_dataset(
            out_dir=out_dir,
            seed=seed,
            theta_max=75,
            theta_step=1.0,
            dt=0.05,
            x_step=10.0,
            t_max=200.0,
            plot_examples=False,
            samples_per_angle=120,
            keep_prob=0.50,
            wind_x_range=(-20.0, 20.0),
            wind_y_range=(-5.0, 5.0),
            wind_z_range=(-15.0, 15.0),
            cant_angle_range=(-15.0, 15.0),
            alpha_range=(-10.0, 10.0),
        )
    print("Dataset:", ds_info)

    train_low_info = train(
        out_dir=ds_info["out_dir"],
        dataset_npy=os.path.join(ds_info["out_dir"], "dataset.npy"),
        model_name="dual_model_low",
        scaler_name="dual_scaler_low",
        history_name="train_history_low",
        plot_name="loss_curve_low",
        seed=seed,
        batch_size=TRAIN_BATCH_SIZE,
        lr=TRAIN_LR,
        max_epochs=TRAIN_MAX_EPOCHS,
        patience=TRAIN_PATIENCE,
        hidden=TRAIN_HIDDEN,
        dropout=TRAIN_DROPOUT,
        model_type=TRAIN_MODEL_TYPE,
        lambda_phys=TRAIN_LAMBDA_PHYS,
        physics_steps=TRAIN_PHYSICS_STEPS,
        gpu_cache_max_total_frac=TRAIN_GPU_CACHE_MAX_TOTAL_FRAC,
        gpu_cache_max_free_frac=TRAIN_GPU_CACHE_MAX_FREE_FRAC,
        plot_losses=True,
        trajectory_mode=0,
    )
    train_high_info = train(
        out_dir=ds_info["out_dir"],
        dataset_npy=os.path.join(ds_info["out_dir"], "dataset.npy"),
        model_name="dual_model_high",
        scaler_name="dual_scaler_high",
        history_name="train_history_high",
        plot_name="loss_curve_high",
        seed=seed,
        batch_size=TRAIN_BATCH_SIZE,
        lr=TRAIN_LR,
        max_epochs=TRAIN_MAX_EPOCHS,
        patience=TRAIN_PATIENCE,
        hidden=TRAIN_HIDDEN,
        dropout=TRAIN_DROPOUT,
        model_type=TRAIN_MODEL_TYPE,
        lambda_phys=TRAIN_LAMBDA_PHYS,
        physics_steps=TRAIN_PHYSICS_STEPS,
        gpu_cache_max_total_frac=TRAIN_GPU_CACHE_MAX_TOTAL_FRAC,
        gpu_cache_max_free_frac=TRAIN_GPU_CACHE_MAX_FREE_FRAC,
        plot_losses=True,
        trajectory_mode=1,
    )
    print(
        "Train Low:",
        {
            k: train_low_info[k]
            for k in (
                "best_epoch",
                "best_val_loss",
                "best_val_data_loss",
                "test_loss",
                "test_data_loss",
                "test_phys_loss",
                "test_phys_eval_loss",
                "test_mae_theta",
                "test_mae_alpha",
            )
        },
    )
    print(
        "Train High:",
        {
            k: train_high_info[k]
            for k in (
                "best_epoch",
                "best_val_loss",
                "best_val_data_loss",
                "test_loss",
                "test_data_loss",
                "test_phys_loss",
                "test_phys_eval_loss",
                "test_mae_theta",
                "test_mae_alpha",
            )
        },
    )

    verify_info = _verify_unified_model(out_dir=ds_info["out_dir"], sample_n=500, seed=seed)
    print("Verify:", verify_info)

    result = {
        "dataset": ds_info,
        "training_low": train_low_info,
        "training_high": train_high_info,
        "verify": verify_info,
    }
    result_path = os.path.join(ds_info["out_dir"], "pipeline_result_refined.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Saved:", result_path)
    return result


if __name__ == "__main__":
    info = full_pipeline_dual_task(seed=42, rebuild_dataset=False)
    print("All done:", info)
