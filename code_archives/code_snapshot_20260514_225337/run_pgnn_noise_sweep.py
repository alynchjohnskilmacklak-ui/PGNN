import csv
import json
import os
from typing import Iterable

from generate_dataset import AMMO_CONFIGS, CURRENT_AMMO
from train_model import train


INPUT_NOISE = {
    "train_noise_x_rel_std": 0.005,
    "train_noise_y_abs_std": 1.0,
    "train_noise_z_abs_std": 1.0,
    "train_noise_v0_std": 2.0,
    "train_noise_rho_rel_std": 0.01,
    "train_noise_wind_x_std": 1.0,
    "train_noise_wind_y_std": 0.5,
    "train_noise_wind_z_std": 1.0,
    "train_noise_cant_std": 0.25,
}

LABEL_NOISE = {
    "train_noise_theta_std": 0.2,
    "train_noise_alpha_std": 0.1,
}


def _merged_noise(*items: dict) -> dict:
    merged = {}
    for item in items:
        merged.update(item)
    return merged


def default_configs() -> list[dict]:
    return [
        {"name": "clean_data_only", "lambda_phys": 0.0, "noise": {}},
        {"name": "clean_pgnn", "lambda_phys": 0.004, "noise": {}},
        {"name": "input_noise_data_only", "lambda_phys": 0.0, "noise": INPUT_NOISE},
        {"name": "input_noise_pgnn", "lambda_phys": 0.004, "noise": INPUT_NOISE},
        {"name": "label_noise_data_only", "lambda_phys": 0.0, "noise": LABEL_NOISE},
        {"name": "label_noise_pgnn", "lambda_phys": 0.004, "noise": LABEL_NOISE},
        {"name": "mixed_noise_data_only", "lambda_phys": 0.0, "noise": _merged_noise(INPUT_NOISE, LABEL_NOISE)},
        {"name": "mixed_noise_pgnn", "lambda_phys": 0.004, "noise": _merged_noise(INPUT_NOISE, LABEL_NOISE)},
    ]


def run_noise_sweep(
    configs: Iterable[dict] | None = None,
    out_dir: str | None = None,
    seed: int = 42,
    trajectory_mode: int = 0,
    physics_steps: int = 6,
    batch_size: int = 16384,
) -> list[dict]:
    if out_dir is None:
        out_dir = AMMO_CONFIGS[CURRENT_AMMO]["out_dir"]
    if configs is None:
        configs = default_configs()

    dataset_npy = os.path.join(out_dir, "dataset.npy")
    if not os.path.exists(dataset_npy):
        raise FileNotFoundError(f"Dataset not found: {dataset_npy}")

    mode_name = "low" if int(trajectory_mode) == 0 else "high"
    results = []

    for cfg in configs:
        name = str(cfg["name"])
        lambda_phys = float(cfg["lambda_phys"])
        noise = dict(cfg.get("noise", {}))
        run_name = f"noise_{name}_{mode_name}"

        print("=" * 80)
        print(f"Running noise sweep: {name}, mode={mode_name}, lambda_phys={lambda_phys}")
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
            max_epochs=80,
            patience=8,
            hidden=256,
            dropout=0.15,
            lambda_phys=lambda_phys,
            physics_steps=physics_steps,
            plot_losses=False,
            gpu_cache_max_total_frac=0.45,
            gpu_cache_max_free_frac=0.65,
            trajectory_mode=trajectory_mode,
            **noise,
        )

        row = {
            "config": name,
            "trajectory_mode": int(trajectory_mode),
            "mode": mode_name,
            "lambda_phys": lambda_phys,
            "physics_steps": int(physics_steps),
            "batch_size": int(batch_size),
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "best_val_data_loss": history.get("best_val_data_loss"),
            "best_val_phys_loss": history.get("best_val_phys_loss"),
            "test_loss": history["test_loss"],
            "test_data_loss": history.get("test_data_loss"),
            "test_phys_loss": history.get("test_phys_loss"),
            "test_phys_eval_loss": history.get("test_phys_eval_loss"),
            "test_mae_theta": history["test_mae_theta"],
            "test_mae_alpha": history["test_mae_alpha"],
            "history_csv": os.path.join(out_dir, f"hist_{run_name}.csv"),
            "model_path": os.path.join(out_dir, f"{run_name}.pt"),
        }
        row.update(history.get("noise", {}))
        results.append(row)

        json_path = os.path.join(out_dir, f"pgnn_noise_sweep_{mode_name}.json")
        csv_path = os.path.join(out_dir, f"pgnn_noise_sweep_{mode_name}.csv")
        _save_results(results, json_path, csv_path)

    return results


def _save_results(rows: list[dict], json_path: str, csv_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved noise sweep results: {csv_path}")
    print(f"Saved noise sweep results: {json_path}")


if __name__ == "__main__":
    run_noise_sweep()
