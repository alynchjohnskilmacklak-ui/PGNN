"""
用途：
    PGNN 物理步数扫描实验。在固定 lambda_phys 下对比不同 physics rollout steps
    对训练精度的影响。

适用场景：
    确定最佳 physics_steps 参数（计算成本 vs 物理精度权衡）。

输入：
    需要先运行 generate_dataset.py 生成数据集。

输出：
    各 steps 配置的训练结果 CSV。

运行方式：
    python scripts/experiments/run_pgnn_steps_sweep.py
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
from train_model import train


def run_steps_sweep(
    steps_list: Iterable[int] = (1, 3, 6, 12),
    out_dir: str | None = None,
    seed: int = 42,
    trajectory_mode: int = 0,
) -> list[dict]:
    """Run the first PGNN sweep: fixed lambda, varied physics rollout steps."""
    if out_dir is None:
        out_dir = AMMO_CONFIGS[CURRENT_AMMO]["out_dir"]

    dataset_npy = os.path.join(out_dir, "dataset.npy")
    if not os.path.exists(dataset_npy):
        raise FileNotFoundError(f"Dataset not found: {dataset_npy}")

    mode_name = "low" if int(trajectory_mode) == 0 else "high"
    results = []

    for steps in steps_list:
        steps = int(steps)
        run_name = f"step_sweep_s{steps}_{mode_name}"
        print("=" * 80)
        print(f"Running PGNN step sweep: mode={mode_name}, physics_steps={steps}")
        print("=" * 80)

        history = train(
            out_dir=out_dir,
            dataset_npy=dataset_npy,
            model_name=run_name,
            scaler_name=f"scaler_{run_name}",
            history_name=f"hist_{run_name}",
            plot_name=f"loss_{run_name}",
            seed=seed,
            batch_size=8192,
            lr=1.6e-4,
            max_epochs=80,
            patience=8,
            hidden=256,
            dropout=0.15,
            lambda_phys=0.004,
            physics_steps=steps,
            plot_losses=True,
            trajectory_mode=trajectory_mode,
        )

        row = {
            "trajectory_mode": int(trajectory_mode),
            "mode": mode_name,
            "physics_steps": steps,
            "lambda_phys": 0.004,
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "test_loss": history["test_loss"],
            "test_mae_theta": history["test_mae_theta"],
            "test_mae_alpha": history["test_mae_alpha"],
            "history_csv": os.path.join(out_dir, f"hist_{run_name}.csv"),
            "model_path": os.path.join(out_dir, f"{run_name}.pt"),
        }
        results.append(row)

        json_path = os.path.join(out_dir, f"pgnn_steps_sweep_{mode_name}.json")
        csv_path = os.path.join(out_dir, f"pgnn_steps_sweep_{mode_name}.csv")
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

    print(f"Saved sweep results: {csv_path}")
    print(f"Saved sweep results: {json_path}")


if __name__ == "__main__":
    run_steps_sweep()
