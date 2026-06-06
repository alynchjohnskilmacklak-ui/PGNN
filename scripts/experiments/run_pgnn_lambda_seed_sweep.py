"""
用途：
    PGNN lambda 参数 + seed 联合扫描。在不同 lambda_phys 值和随机种子下训练，
    评估物理损失权重对精度的影响。

适用场景：
    确定最佳 lambda_phys 超参数。

输入：
    需要先运行 generate_dataset.py 生成数据集。
    依赖 scripts/experiments/run_pgnn_noise_sweep.py 中的 LABEL_NOISE。

输出：
    各 lambda/seed 组合的训练结果 CSV。

运行方式：
    python scripts/experiments/run_pgnn_lambda_seed_sweep.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
import os
from typing import Iterable

from generate_dataset import AMMO_CONFIGS, CURRENT_AMMO
from scripts.experiments.run_pgnn_noise_sweep import LABEL_NOISE
from train_model import train


DEFAULT_SEEDS = [42, 123, 2026]
DEFAULT_LAMBDAS = [0.0, 0.004, 0.008, 0.015, 0.03]


def default_configs() -> list[dict]:
    configs = []
    for scenario, noise in [
        ("clean", {}),
        ("label_noise", LABEL_NOISE),
    ]:
        for lambda_phys in DEFAULT_LAMBDAS:
            method = "data_only" if float(lambda_phys) == 0.0 else f"pgnn_lam{_lambda_tag(lambda_phys)}"
            configs.append({
                "scenario": scenario,
                "method": method,
                "lambda_phys": float(lambda_phys),
                "noise": dict(noise),
            })
    return configs


def run_lambda_seed_sweep(
    configs: Iterable[dict] | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    out_dir: str | None = None,
    trajectory_mode: int = 0,
    physics_steps: int = 6,
    batch_size: int = 16384,
    max_epochs: int = 120,
    patience: int = 12,
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
    json_path = os.path.join(out_dir, f"pgnn_lambda_seed_sweep_{mode_name}.json")
    csv_path = os.path.join(out_dir, f"pgnn_lambda_seed_sweep_{mode_name}.csv")

    for seed in seeds:
        for cfg in configs:
            scenario = str(cfg["scenario"])
            method = str(cfg["method"])
            lambda_phys = float(cfg["lambda_phys"])
            noise = dict(cfg.get("noise", {}))
            run_name = f"lambda_seed_{scenario}_{method}_seed{int(seed)}_{mode_name}"

            print("=" * 80)
            print(
                f"Running lambda/seed sweep: scenario={scenario}, method={method}, "
                f"seed={int(seed)}, mode={mode_name}, lambda_phys={lambda_phys}"
            )
            print("=" * 80)

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
                "scenario": scenario,
                "method": method,
                "seed": int(seed),
                "trajectory_mode": int(trajectory_mode),
                "mode": mode_name,
                "lambda_phys": lambda_phys,
                "physics_steps": int(physics_steps),
                "batch_size": int(batch_size),
                "max_epochs": int(max_epochs),
                "patience": int(patience),
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
            _save_results(results, json_path, csv_path)

    return results


def _lambda_tag(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _save_results(rows: list[dict], json_path: str, csv_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved lambda/seed sweep results: {csv_path}")
    print(f"Saved lambda/seed sweep results: {json_path}")


if __name__ == "__main__":
    run_lambda_seed_sweep()
