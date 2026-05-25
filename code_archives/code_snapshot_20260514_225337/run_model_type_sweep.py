import csv
import json
import os
import time
from typing import Iterable

from generate_dataset import AMMO_CONFIGS, CURRENT_AMMO
from train_model import train


def default_configs() -> list[dict]:
    return [
        {"name": "mlp_data_only", "model_type": "mlp", "hidden": 256, "lambda_phys": 0.0},
        {"name": "kan_data_only", "model_type": "kan", "hidden": 128, "lambda_phys": 0.0},
        {"name": "kan_mlp_data_only", "model_type": "kan_mlp", "hidden": 256, "lambda_phys": 0.0},
    ]


def run_model_type_sweep(
    configs: Iterable[dict] | None = None,
    out_dir: str | None = None,
    seed: int = 42,
    trajectory_mode: int = 0,
    physics_steps: int = 6,
    batch_size: int = 16384,
    max_epochs: int = 80,
    patience: int = 8,
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
    json_path = os.path.join(out_dir, f"model_type_sweep_{mode_name}.json")
    csv_path = os.path.join(out_dir, f"model_type_sweep_{mode_name}.csv")

    for cfg in configs:
        name = str(cfg["name"])
        model_type = str(cfg["model_type"])
        hidden = int(cfg.get("hidden", 256))
        lambda_phys = float(cfg.get("lambda_phys", 0.0))
        run_name = f"modeltype_{name}_{mode_name}_seed{int(seed)}"

        print("=" * 80)
        print(
            f"Running model type sweep: name={name}, model_type={model_type}, "
            f"hidden={hidden}, seed={int(seed)}, mode={mode_name}, lambda_phys={lambda_phys}"
        )
        print("=" * 80)

        start = time.perf_counter()
        history = train(
            out_dir=out_dir,
            dataset_npy=dataset_npy,
            model_name=run_name,
            scaler_name=f"scaler_{run_name}",
            history_name=f"hist_{run_name}",
            plot_name=f"loss_{run_name}",
            seed=int(seed),
            batch_size=batch_size,
            lr=1.6e-4,
            max_epochs=max_epochs,
            patience=patience,
            hidden=hidden,
            dropout=0.15,
            model_type=model_type,
            lambda_phys=lambda_phys,
            physics_steps=physics_steps,
            plot_losses=False,
            gpu_cache_max_total_frac=0.45,
            gpu_cache_max_free_frac=0.65,
            trajectory_mode=trajectory_mode,
        )
        elapsed_sec = time.perf_counter() - start

        row = {
            "name": name,
            "model_type": model_type,
            "hidden": hidden,
            "seed": int(seed),
            "trajectory_mode": int(trajectory_mode),
            "mode": mode_name,
            "lambda_phys": lambda_phys,
            "physics_steps": int(physics_steps),
            "batch_size": int(batch_size),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "elapsed_sec": float(elapsed_sec),
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
        results.append(row)
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

    print(f"Saved model type sweep results: {csv_path}")
    print(f"Saved model type sweep results: {json_path}")


if __name__ == "__main__":
    run_model_type_sweep()
