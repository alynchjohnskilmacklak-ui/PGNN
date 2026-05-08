import os
import json
import numpy as np
import pandas as pd

from generate_dataset import generate_dataset, AMMO_CONFIGS, CURRENT_AMMO
from train_model import train
from predict import load_branch_model, solve_target_unified, infer_model_dims_from_state_dict
from ballistics import ProjectileParams


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


def _remove_pipeline_outputs(out_dir):
    removed = []
    for name in PIPELINE_OUTPUT_FILES:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)
    return removed


def _verify_unified_model(out_dir, sample_n=700, seed=42):
    verify_csv_path = os.path.join(out_dir, "verify_predictions.csv")
    summary_csv_path = os.path.join(out_dir, "verify_summary.csv")
    csv_path = os.path.join(out_dir, "dataset.csv")
    if not os.path.exists(csv_path):
        return {
            "sample_size": 0,
            "reachable_count": 0,
            "angle_mae_theta_deg": float("nan"),
            "angle_mae_alpha_deg": float("nan"),
            "verify_csv_path": verify_csv_path,
            "summary_csv_path": summary_csv_path,
        }

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {
            "sample_size": 0,
            "reachable_count": 0,
            "angle_mae_theta_deg": float("nan"),
            "angle_mae_alpha_deg": float("nan"),
            "verify_csv_path": verify_csv_path,
            "summary_csv_path": summary_csv_path,
        }

    sample_n = min(sample_n, len(df))
    sample_df = df.sample(n=sample_n, random_state=seed).copy()

    low_model_path = os.path.join(out_dir, "dual_model_low.pt")
    low_scaler_path = os.path.join(out_dir, "dual_scaler_low.json")
    high_model_path = os.path.join(out_dir, "dual_model_high.pt")
    high_scaler_path = os.path.join(out_dir, "dual_scaler_high.json")
    low_dims = infer_model_dims_from_state_dict(low_model_path)
    high_dims = infer_model_dims_from_state_dict(high_model_path)
    loaded_low_model, loaded_low_scaler, _ = load_branch_model(
        low_model_path,
        low_scaler_path,
        theta_min=0.0,
        theta_max=55.0,
        in_dim=low_dims["in_dim"],
        hidden=low_dims["hidden"],
        dropout=low_dims["dropout"],
    )
    loaded_high_model, loaded_high_scaler, _ = load_branch_model(
        high_model_path,
        high_scaler_path,
        theta_min=45.0,
        theta_max=85.0,
        in_dim=high_dims["in_dim"],
        hidden=high_dims["hidden"],
        dropout=high_dims["dropout"],
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

    preds_theta, trues_theta = [], []
    preds_alpha, trues_alpha = [], []
    reachable_count = 0
    rows = []

    for _, row in sample_df.iterrows():
        out = solve_target_unified(
            x_target=float(row["x"]),
            y_target=float(row["y"]),
            z_target=float(row["z"]),
            v0_actual=float(row["v0_actual"]),
            rho=float(row["rho"]),
            wind_x=float(row["wind_x"]),
            wind_y=float(row["wind_y"]),
            wind_z=float(row["wind_z"]),
            cant_angle=float(row["cant_angle"]),
            T_powder_C=float(row["T_powder_C"]),
            alt_gun=float(row["alt_gun"]),
            T0_C=float(row["T0_C"]),
            P0_Pa=float(row["P0_Pa"]),
            dir_path=out_dir,
            loaded_low_model=loaded_low_model,
            loaded_low_scaler=loaded_low_scaler,
            loaded_high_model=loaded_high_model,
            loaded_high_scaler=loaded_high_scaler,
            params=weapon_params,
            y_tol=2.0,
            z_tol=2.0,
            save_plot=False,
        )
        chosen = out.get("chosen")
        rows.append({
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "theta_true_deg": float(row["theta_deg"]),
            "alpha_true_deg": float(row["alpha_deg"]),
            "theta_pred_low_deg": float(out["nn_prediction"]["low"]["theta"]),
            "alpha_pred_low_deg": float(out["nn_prediction"]["low"]["alpha"]),
            "theta_pred_high_deg": float(out["nn_prediction"]["high"]["theta"]),
            "alpha_pred_high_deg": float(out["nn_prediction"]["high"]["alpha"]),
            "chosen_reachable": int(chosen is not None),
            "chosen_mode": "" if chosen is None else chosen["mode"],
            "theta_chosen_deg": np.nan if chosen is None else float(chosen["theta"]),
            "alpha_chosen_deg": np.nan if chosen is None else float(chosen["alpha"]),
            "chosen_err_3d": np.nan if chosen is None else float(chosen["err_3d"]),
        })
        if chosen is None:
            continue

        reachable_count += 1
        preds_theta.append(float(chosen["theta"]))
        preds_alpha.append(float(chosen["alpha"]))
        trues_theta.append(float(row["theta_deg"]))
        trues_alpha.append(float(row["alpha_deg"]))

    mae_theta = float(np.mean(np.abs(np.array(preds_theta) - np.array(trues_theta)))) if preds_theta else float("nan")
    mae_alpha = float(np.mean(np.abs(np.array(preds_alpha) - np.array(trues_alpha)))) if preds_alpha else float("nan")
    pd.DataFrame(rows).to_csv(verify_csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "sample_size": int(sample_n),
        "reachable_count": int(reachable_count),
        "angle_mae_theta_deg": mae_theta,
        "angle_mae_alpha_deg": mae_alpha,
    }]).to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    return {
        "sample_size": int(sample_n),
        "reachable_count": int(reachable_count),
        "angle_mae_theta_deg": mae_theta,
        "angle_mae_alpha_deg": mae_alpha,
        "verify_csv_path": verify_csv_path,
        "summary_csv_path": summary_csv_path,
    }


def full_pipeline_dual_task(seed=42, out_dir=None, rebuild_dataset=False):
    if out_dir is None:
        out_dir = AMMO_CONFIGS[CURRENT_AMMO]["out_dir"]

    if rebuild_dataset and os.path.isdir(out_dir):
        removed_files = _remove_pipeline_outputs(out_dir)
        print(f"Removed old files: {len(removed_files)}")

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
        batch_size=2048,
        lr=1.6e-4,
        max_epochs=80,
        patience=8,
        hidden=256,
        dropout=0.15,
        lambda_phys=0.004,
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
        batch_size=2048,
        lr=1.6e-4,
        max_epochs=80,
        patience=8,
        hidden=256,
        dropout=0.15,
        lambda_phys=0.004,
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
